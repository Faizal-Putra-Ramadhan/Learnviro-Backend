import cv2
import mediapipe as mp
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
import joblib
import numpy as np
import os

# Memastikan jalur model benar dengan mencari di parent directory jika tidak ada di backend
MODEL_PATH = 'emotion_model.pkl'
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = '../emotion_model.pkl'

# Initialize model
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load model from {MODEL_PATH}. Error: {e}")
    model = None

mp_face_mesh = mp.solutions.face_mesh

def detect_emotion(image_bytes):
    """
    Menerima byte gambar, mendeteksi landmark wajah, 
    mengekstrak fitur, dan mengembalikan prediksi emosi.
    """
    if model is None:
        return {"error": "Model not loaded", "emosi": "unknown", "confidence": 0.0, "image_bytes": image_bytes}

    # Convert bytes to cv2 image
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"error": "Invalid image data", "emosi": "unknown", "confidence": 0.0, "image_bytes": image_bytes}

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return {"error": "No face detected", "emosi": "unknown", "confidence": 0.0, "image_bytes": image_bytes}

        # Extract features for the first face
        face_landmarks = results.multi_face_landmarks[0]
        coords = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
        
        # Normalization (must match train_model.py & extract_features.py)
        nose_tip = coords[1]
        coords_centered = coords - nose_tip
        
        max_dist = np.max(np.linalg.norm(coords_centered, axis=1))
        if max_dist > 0:
            coords_normalized = coords_centered / max_dist
        else:
            coords_normalized = coords_centered
            
        features = coords_normalized.flatten()
        
        # Prediction
        try:
            # Check if predict_proba is available
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba([features])[0]
                max_prob_index = np.argmax(probabilities)
                prediction = model.classes_[max_prob_index]
                confidence = float(probabilities[max_prob_index])
            else:
                prediction = model.predict([features])[0]
                confidence = 1.0 # Default confidence if predict_proba not supported
            
            # Gambar Facial Landmarks di gambar aslinya (image)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                
            # Gambar teks emosi di pojok kiri atas
            text = f"{prediction} ({confidence*100:.1f}%)"
            cv2.putText(image, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Encode gambar yang sudah digambar kembali menjadi bytes
            _, encoded_img = cv2.imencode('.jpg', image)
            annotated_image_bytes = encoded_img.tobytes()
            
            return {
                "error": None,
                "emosi": prediction,
                "confidence": confidence,
                "image_bytes": annotated_image_bytes
            }
        except Exception as e:
            return {"error": str(e), "emosi": "unknown", "confidence": 0.0, "image_bytes": image_bytes}
