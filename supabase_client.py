import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

STORAGE_BUCKET = "emotion-images"

def get_supabase_client() -> Client:
    """Returns a Supabase client."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Warning: Supabase credentials not found in environment variables.")
        return None
        
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Error initializing Supabase client: {e}")
        return None

def upload_image_to_storage(image_bytes, file_name):
    """Uploads an image to Supabase Storage and returns the public URL."""
    supabase = get_supabase_client()
    if not supabase:
        print("Warning: Supabase client not initialized for storage upload.")
        return None

    try:
        # Upload file to Supabase Storage
        file_path = f"captures/{file_name}"
        response = supabase.storage.from_(STORAGE_BUCKET).upload(
            file_path,
            image_bytes,
            file_options={"content-type": "image/jpeg"}
        )
        
        # Get public URL
        public_url = supabase.storage.from_(STORAGE_BUCKET).get_public_url(file_path)
        print(f"Image uploaded to Supabase Storage: {public_url}")
        return public_url
        
    except Exception as e:
        print(f"Error uploading to Supabase Storage: {e}")
        return None

def insert_emotion_record(device_id, emosi, confidence, mode, gambar_url, waktu, nama="Unknown"):
    """Inserts a new emotion record into Supabase."""
    supabase = get_supabase_client()
    if not supabase:
        return {"error": "Supabase client not initialized"}

    try:
        data = {
            "device_id": device_id,
            "nama": nama,
            "hasil_deteksi": emosi,
            "confidence_score": confidence,
            "mode": mode,
            "gambar_url": gambar_url,
            "waktu": waktu
        }
        
        # In supabase-py v2, it returns an APIResponse object where the data is in .data
        response = supabase.table("emotion_records").insert(data).execute()
        return {"success": True, "data": response.data}
    except Exception as e:
        print(f"Error inserting into Supabase: {e}")
        return {"error": str(e)}
