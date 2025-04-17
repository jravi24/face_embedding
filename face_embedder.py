import base64
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
import uuid

class FaceEmbedder:
    def __init__(self, model_name='buffalo_l'):
        """
        Initializes the FaceAnalysis model from insightface.
        """
        try:
            self.app = FaceAnalysis(name=model_name)
            self.app.prepare(ctx_id=0, det_size=(640, 640))
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {e}")

    def base64_to_image(self, base64_string):
        """
        Converts a base64 encoded image to an OpenCV image.
        """
        try:
            if not base64_string:
                return None, {
                    "resultStatus": {
                        "status": "FAILED",
                        "errorCode": "9998",
                        "errorMessage": "Invalid Request | Missing image in base64"
                    },
                    "transRefNo": ""
                }
            image_data = base64.b64decode(base64_string)
            np_array = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            if img is None:
                return None, {
                    "resultStatus": {
                        "status": "FAILED",
                        "errorCode": "4211",
                        "errorMessage": "Failed to convert image from base64 to image."
                    },
                    "transRefNo": ""
                }
            return img, None
        except Exception:
            return None, {
                "resultStatus": {
                    "status": "FAILED",
                    "errorCode": "4211",
                    "errorMessage": "Failed to convert image from base64 to image."
                },
                "transRefNo": ""
            }

    def get_embedding(self, transRefNo, base64_string):
        """
        Takes a base64 encoded image and returns its 512D embedding in API response format.
        """
        # if not transRefNo:
            # transRefNo = str(uuid.uuid4())  # Generate a unique transaction reference number if missing
        
        img, error_response = self.base64_to_image(base64_string)
        if error_response:
            error_response["transRefNo"] = transRefNo
            return error_response
        
        try:
            faces = self.app.get(img)
            
            if len(faces) == 0:
                return {
                    "resultStatus": {
                        "status": "FAILED",
                        "errorCode": "4212",
                        "errorMessage": "No face detected in image."
                    },
                    "transRefNo": transRefNo
                }
            
            return {
                "resultStatus": {
                    "status": "SUCCESS"
                },
                "transRefNo": transRefNo,
                "image_embedding": (faces[0].embedding / np.linalg.norm(faces[0].embedding)).tolist()
            }
        except Exception:
            return {
                "resultStatus": {
                    "status": "FAILED",
                    "errorCode": "9999",
                    "errorMessage": "Unknown Exception."
                },
                "transRefNo": transRefNo
            }
