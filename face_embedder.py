import base64
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis

class FaceEmbedder:
    def __init__(self, model_name='buffalo_l'):
        print("[Init] Initializing FaceAnalysis model...")
        try:
            self.app = FaceAnalysis(name=model_name)
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            print("[Init] Model initialized successfully.")
        except Exception as e:
            print(f"[Init] Failed to initialize model: {e}")
            raise RuntimeError(f"Failed to initialize model: {e}")

    def base64_to_image(self, base64_string):
        print("[base64_to_image] Starting image decode...")
        if not base64_string:
            print("[base64_to_image] No image string provided.")
            return None, {
                "resultStatus": {
                    "status": "FAILED",
                    "errorCode": "9998",
                    "errorMessage": "Invalid Request | Missing image in base64"
                },
                "transRefNo": ""
            }
        try:
            image_data = base64.b64decode(base64_string)
            print(f"[base64_to_image] Decoded {len(image_data)} bytes of image data.")
            np_array = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            if img is None:
                print("[base64_to_image] Failed to decode image from np_array.")
                return None, {
                    "resultStatus": {
                        "status": "FAILED",
                        "errorCode": "4211",
                        "errorMessage": "Failed to convert image from base64 to image."
                    },
                    "transRefNo": ""
                }
            print("[base64_to_image] Image conversion successful.")
            return img, None
        except Exception as e:
            print(f"[base64_to_image] Exception: {e}")
            return None, {
                "resultStatus": {
                    "status": "FAILED",
                    "errorCode": "4211",
                    "errorMessage": "Failed to convert image from base64 to image."
                },
                "transRefNo": ""
            }

    def get_embedding(self, transRefNo, base64_string):
        print(f"[get_embedding] Processing transRefNo: {transRefNo}")
        img, error_response = self.base64_to_image(base64_string)
        if error_response:
            print("[get_embedding] Error in image conversion:", error_response)
            error_response["transRefNo"] = transRefNo
            return error_response
        
        try:
            print("[get_embedding] Running face detection...")
            faces = self.app.get(img)
            print(f"[get_embedding] Number of faces detected: {len(faces)}")
            
            if len(faces) == 0:
                return {
                    "resultStatus": {
                        "status": "FAILED",
                        "errorCode": "4212",
                        "errorMessage": "No face detected in image."
                    },
                    "transRefNo": transRefNo
                }
            
            embedding = (faces[0].embedding / np.linalg.norm(faces[0].embedding)).tolist()
            print("[get_embedding] Embedding generated.")
            return {
                "resultStatus": {
                    "status": "SUCCESS"
                },
                "transRefNo": transRefNo,
                "image_embedding": embedding
            }
        except Exception as e:
            print(f"[get_embedding] Exception during embedding: {e}")
            return {
                "resultStatus": {
                    "status": "FAILED",
                    "errorCode": "9999",
                    "errorMessage": "Unknown Exception."
                },
                "transRefNo": transRefNo
            }
