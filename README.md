# Backend Setup Guide

## Step 1 — Project folder banayen (VS Code mein)

1. VS Code mein ek naya folder open karein: `solar_drone_backend`
2. Isme `main.py` aur `requirements.txt` copy-paste kar dein (jo diye gaye hain)
3. Do sub-folders banayein: `models/` aur `data/`

## Step 2 — Apna trained model download karein

Google Drive se `final_model.keras` (jo humne Colab mein save kiya tha) download karein
aur `models/final_model.keras` path pe rakh dein.

## Step 3 — Sample images ka data folder

Kaggle se dataset download karein (`hemanthsai7/solar-panel-dust-detection`) aur
`data/Detect_solar_dust/Clean/` aur `data/Detect_solar_dust/Dusty/` folders
apne project ke `data/Detect_solar_dust/` folder mein copy kar dein.

## Step 4 — Virtual environment + install (VS Code terminal mein)

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

## Step 5 — Server chalayein

```bash
uvicorn main:app --reload
```

Browser mein open karein: **http://127.0.0.1:8000/docs**
Yahan se aap `/run-inspection` endpoint test kar sakte hain — "Try it out" button click karein.

Agar successful JSON response aaye (panels ki list ke sath), matlab backend perfectly kaam kar raha hai.

## Step 6 — Deployment (taake ye 24/7 live rahe, sirf laptop pe hi nahi)

**Important honest note:** TensorFlow + MobileNetV2 ko kaam karne ke liye kam se kam
1GB+ RAM chahiye hoti hai. Free hosting tiers (Render free, Railway free) mein
kabhi-kabhi ye limit tight ho sakti hai — agar deploy karte waqt memory error aaye,
to paid starter tier ($5-7/month) ya lighter model conversion (TensorFlow Lite) ki
zaroorat par sakti hai.

**Recommended: Render.com**
1. Is poore backend folder ko GitHub repo mein push karein
2. Render.com pe account banayein → "New Web Service" → GitHub repo connect karein
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Environment variables mein `MODEL_PATH` aur `DATASET_DIR` set karein (agar different paths use kar rahe hain)
6. Deploy hone ke baad aapko ek live URL milega (jaise `https://your-app.onrender.com`)

Ye live URL hi hum dashboard ke "Run Inspection" button mein use karenge.
