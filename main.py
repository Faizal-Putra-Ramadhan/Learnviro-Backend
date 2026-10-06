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

@app.post("/capture")
async def capture_image(
    image: UploadFile = File(...),
    device_id: str = Form(...),
    timestamp: str = Form(...),
    mode: str = Form(...)
):
    try:
        # Check Attendance State
        state_file = os.path.join(os.path.dirname(__file__), "..", "backend_ci4", "writable", f"attendance_state_{device_id}.json")
        state = {"is_active": False}
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                try:
                    state = json.load(f)
                except:
                    pass
        
        if not state.get("is_active"):
            raise HTTPException(status_code=403, detail="Absensi belum dimulai")
            
        current_nama = state.get("current_student", "Unknown")
        current_mode = state.get("mode", mode)
        # Read image bytes
        image_bytes = await image.read()
        image_bytes = rotate_image_bytes(image_bytes) 
        
        # 1. Detect emotion
        detection_result = detect_emotion(image_bytes)
        emosi = detection_result.get("emosi", "unknown")
        confidence = detection_result.get("confidence", 0.0)
        
        if detection_result.get("error"):
            print(f"Detection warning: {detection_result.get('error')}")

        # Ambil gambar yang sudah ditambahkan landmark dan teks
        annotated_image_bytes = detection_result.get("image_bytes", image_bytes)

        # 2. Upload to Supabase Storage (menggantikan Google Drive)
        try:
            dt = datetime.fromtimestamp(int(timestamp))
        except ValueError:
            dt = datetime.now() # Fallback
            
        file_name = f"{device_id}_{dt.strftime('%Y%m%d_%H%M%S')}_{emosi}.jpg"
        
        gambar_url = upload_image_to_storage(annotated_image_bytes, file_name)
        if not gambar_url:
            gambar_url = "" # Fallback if upload fails

        # 3. Insert into Supabase
        # Prepare ISO format time
        iso_time = datetime.now(timezone.utc).isoformat()
        
        db_result = insert_emotion_record(
            device_id=device_id,
            emosi=emosi,
            confidence=confidence,
            mode=current_mode,
            gambar_url=gambar_url,
            waktu=iso_time,
            nama=current_nama
        )
        
        if db_result.get("error"):
            raise HTTPException(status_code=500, detail=f"Database error: {db_result.get('error')}")

        # Auto-advance student index
        state['current_index'] = state.get('current_index', 0) + 1
        if state['current_index'] >= len(state.get('student_list', [])):
            state['is_active'] = False
            state['current_student'] = None
        else:
            state['current_student'] = state['student_list'][state['current_index']]
            
        with open(state_file, 'w') as f:
            json.dump(state, f)

        return {
            "status": "success",
            "emosi": emosi,
            "confidence": confidence,
            "gambar_url": gambar_url,
            "waktu": iso_time,
            "nama": current_nama
        }

    except Exception as e:
        print(f"Error in /capture: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
