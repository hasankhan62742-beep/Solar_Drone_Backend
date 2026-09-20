"""
Solar Panel Cleaning Drone — Inspection Backend API
-----------------------------------------------------
Serves the trained CV classifier and simulation logic over a REST API
so the live dashboard can trigger on-demand inspections.

Run locally:
    uvicorn main:app --reload

Docs (auto-generated):
    http://127.0.0.1:8000/docs
"""

import os
import glob
import random
import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("solar-drone-api")

# ---------------------------------------------------------------------------
# Configuration (env vars override these — no secrets hardcoded)
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "best_model_finetuned.h5")
NUM_ROWS = int(os.environ.get("GRID_ROWS", 3))
NUM_COLS = int(os.environ.get("GRID_COLS", 4))
PANEL_SPACING = float(os.environ.get("PANEL_SPACING", 2.0))
IMG_SIZE = (224, 224)

# Populated at startup
model = None
clean_samples = []
dusty_samples = []
prediction_cache = {}  # img_path -> (label, confidence), computed once at startup


def build_panel_grid(rows: int, cols: int, spacing: float):
    """Same panel layout used in the PyBullet simulation."""
    positions = []
    for row in range(rows):
        for col in range(cols):
            positions.append((col * spacing, row * spacing, 0.3))
    return positions


def generate_coverage_path(panel_positions, rows: int, cols: int, flight_height: float = 1.5):
    """Boustrophedon (zigzag) coverage path — identical logic to the simulation notebook."""
    grid = np.array(panel_positions).reshape(rows, cols, 3)
    path = []
    for row in range(rows):
        row_panels = grid[row]
        if row % 2 == 1:
            row_panels = row_panels[::-1]
        for panel in row_panels:
            path.append([float(panel[0]), float(panel[1]), float(panel[2] + flight_height)])
    return path


def load_sample_images(dataset_dir: str):
    """Loads reference sample image paths for simulated camera capture."""
    clean = glob.glob(os.path.join(dataset_dir, "Clean", "*"))
    dusty = glob.glob(os.path.join(dataset_dir, "Dusty", "*"))
    if not clean or not dusty:
        raise FileNotFoundError(
            f"Sample images not found under {dataset_dir}. "
            "Set DATASET_DIR env var to your local Clean/Dusty folders."
        )
    return clean, dusty


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the model and sample images once, at server startup."""
    global model, clean_samples, dusty_samples

    from tensorflow.keras.models import load_model as keras_load_model

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file not found at '{MODEL_PATH}'. "
            "Download final_model.keras from Google Drive and place it in the models/ folder."
        )

    logger.info("Loading model from %s ...", MODEL_PATH)
    model = keras_load_model(MODEL_PATH)
    logger.info("Model loaded successfully.")

    dataset_dir = os.environ.get("DATASET_DIR", ".")
    try:
        clean_samples, dusty_samples = load_sample_images(dataset_dir)
        logger.info("Loaded %d clean / %d dusty sample images.", len(clean_samples), len(dusty_samples))

        all_paths = clean_samples + dusty_samples
        logger.info("Pre-computing classifications for all %d sample images (one-time cost)...", len(all_paths))
        all_results = classify_batch(all_paths)
        for path, result in zip(all_paths, all_results):
            prediction_cache[path] = result
        logger.info("Prediction cache ready — inspections will now be instant.")
    except FileNotFoundError as e:
        logger.warning("%s — /run-inspection will fail until this is fixed.", e)

    yield  # app runs here
    logger.info("Shutting down.")


app = FastAPI(title="Solar Panel Drone Inspection API", version="1.0.0", lifespan=lifespan)

# CORS: restrict to your dashboard's real domain in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


class PanelResult(BaseModel):
    panel_num: int
    position: list
    predicted: str
    confidence: float
    action: str


class InspectionResponse(BaseModel):
    panels: list[PanelResult]
    total_panels: int
    dusty_count: int
    clean_count: int


def load_image_array(img_path: str) -> np.ndarray:
    from tensorflow.keras.preprocessing import image as keras_image

    img = keras_image.load_img(img_path, target_size=IMG_SIZE)
    return keras_image.img_to_array(img) / 255.0


def classify_batch(img_paths: list[str]) -> list[tuple[str, float]]:
    """Runs all images through the model in a single batched call — far faster
    than calling model.predict once per image, which is what was causing
    requests to exceed the platform's timeout on free-tier CPU."""
    batch = np.stack([load_image_array(p) for p in img_paths], axis=0)
    preds = model.predict(batch, verbose=0).flatten()
    results = []
    for pred in preds:
        pred = float(pred)
        label = "Dusty" if pred > 0.5 else "Clean"
        confidence = pred if pred > 0.5 else 1 - pred
        results.append((label, confidence))
    return results


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/run-inspection", response_model=InspectionResponse)
def run_inspection():
    """Runs a full simulated flight + inspection pass and returns results."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    if not clean_samples or not dusty_samples:
        raise HTTPException(status_code=503, detail="Sample images are not available on the server.")

    panel_positions = build_panel_grid(NUM_ROWS, NUM_COLS, PANEL_SPACING)
    flight_path = generate_coverage_path(panel_positions, NUM_ROWS, NUM_COLS)

    if not prediction_cache:
        raise HTTPException(status_code=503, detail="Prediction cache is not ready yet.")

    # Pick one sample image per waypoint (simulated camera captures) and
    # look up its already-computed classification — no live inference here,
    # so this responds instantly regardless of server CPU speed.
    img_paths = [
        random.choice(dusty_samples if random.choice([True, False]) else clean_samples)
        for _ in flight_path
    ]
    classifications = [prediction_cache[p] for p in img_paths]

    results = []
    dusty_count = 0

    for i, (waypoint, (label, confidence)) in enumerate(zip(flight_path, classifications)):
        if label == "Dusty":
            dusty_count += 1
            action = "SPRAY (cleaning triggered)"
        else:
            action = "No action needed"

        results.append(PanelResult(
            panel_num=i + 1,
            position=[waypoint[0], waypoint[1]],
            predicted=label,
            confidence=round(confidence, 3),
            action=action,
        ))

    return InspectionResponse(
        panels=results,
        total_panels=len(results),
        dusty_count=dusty_count,
        clean_count=len(results) - dusty_count,
    )
