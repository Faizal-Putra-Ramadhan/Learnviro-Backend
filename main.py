import os
from datetime import datetime, timezone
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import io
from PIL import Image

def rotate_image_bytes(image_bytes: bytes, degrees: int = -90) -> bytes:
    """Koreksi rotasi kamera. degrees=-90 -> putar 90° searah jarum jam (ke kanan)."""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.rotate(degrees, expand=True)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()

# Load env variables first
load_dotenv()

from emotion_detector import detect_emotion
from supabase_client import insert_emotion_record, upload_image_to_storage

app = FastAPI(title="Emotion Detection API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/status")
def health_check():
    return {"status": "ok", "message": "Server is running"}

import json
import base64

import requests
from fastapi import Request
from starlette.concurrency import run_in_threadpool

@app.post("/predict")
async def predict_endpoint(request: Request):
    try:
        image_bytes = await request.body()
        image_bytes = rotate_image_bytes(image_bytes)
        
        detection_result = detect_emotion(image_bytes)
        emosi = detection_result.get("emosi", "unknown")
        confidence = detection_result.get("confidence", 0.0)
        
        if detection_result.get("error"):
            return [{"label": "Error", "score": 0.0, "error": detection_result.get("error")}]
        
        # Kembalikan gambar ber-landmark (base64) supaya bisa disimpan learnviro-tk
        annotated_b64 = ""
        if detection_result.get("image_bytes"):
            annotated_b64 = base64.b64encode(detection_result["image_bytes"]).decode("utf-8")
            
        return [{"label": emosi, "score": confidence, "annotated_image": annotated_b64}]
    except Exception as e:
        print(f"Error in /predict: {e}")
        return [{"label": "Error", "score": 0.0, "error": str(e)}]

@app.post("/capture")
async def capture_image(
    image: UploadFile = File(...),
    device_id: str = Form(...),
    timestamp: str = Form(...),
    mode: str = Form(...)
):
    try:
        # 1. Forward the image directly to learnviro-tk
        # learnviro-tk will then call our /predict endpoint above!
        url = "http://localhost/learnviro-tk/emotions/classify"
        
        image_bytes = await image.read()
        files = {'image': (image.filename, image_bytes, image.content_type)}
        data = {'device_id': device_id, 'timestamp': timestamp, 'mode': mode}
        
        # Jalankan panggilan blocking di threadpool supaya event loop tetap bebas
        # melayani /predict yang dipanggil balik oleh learnviro-tk (hindari deadlock).
        response = await run_in_threadpool(requests.post, url, data=data, files=files)
        
        if response.status_code >= 400:
            print(f"learnviro-tk returned error: {response.text}")
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
        try:
            return response.json()
        except:
            return {"status": "success", "message": response.text}
            
    except Exception as e:
        print(f"Error in /capture forwarding: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
