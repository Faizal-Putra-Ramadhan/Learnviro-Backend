import os
import io
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseUpload

# Get credentials from .env
SCOPES = ['https://www.googleapis.com/auth/drive.file']
SERVICE_ACCOUNT_FILE = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'service_account.json')

def get_drive_service():
    """Authenticate and return Google Drive service."""
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        print(f"Warning: Service account file '{SERVICE_ACCOUNT_FILE}' not found.")
        return None
        
    try:
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        service = build('drive', 'v3', credentials=creds)
        return service
    except Exception as e:
        print(f"Error authenticating with Google Drive: {e}")
        return None

def upload_image_to_drive(image_bytes, file_name):
    """Uploads an image to Google Drive and returns the public URL."""
    service = get_drive_service()
    if not service:
        return None

    try:
        # Pindahkan pengambilan env variable ke dalam fungsi untuk memastikan dotenv sudah di-load
        parent_folder_id = os.getenv('GDRIVE_FOLDER_ID')
        
        # File metadata
        file_metadata = {'name': file_name}
        if parent_folder_id:
            file_metadata['parents'] = [parent_folder_id]
        else:
            print("Warning: GDRIVE_FOLDER_ID tidak ditemukan di .env!")

        # Upload file
        media = MediaIoBaseUpload(io.BytesIO(image_bytes), mimetype='image/jpeg', resumable=True)
        file = service.files().create(
            body=file_metadata, 
            media_body=media, 
            fields='id, webViewLink',
            supportsAllDrives=True
        ).execute()
        
        file_id = file.get('id')
        
        # Make the file publicly accessible (view link)
        permission = {
            'type': 'anyone',
            'role': 'reader'
        }
        service.permissions().create(fileId=file_id, body=permission).execute()
        
        return file.get('webViewLink')
        
    except Exception as e:
        print(f"Error uploading to Google Drive: {e}")
        return None
