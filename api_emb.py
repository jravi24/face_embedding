from flask import Flask, request, jsonify
from face_embedder import FaceEmbedder  # Import the FaceComparer class

app = Flask(__name__)
face_emb = FaceEmbedder()  # Initialize FaceCompare object
app.json.sort_keys = False

@app.route('/generateEmbedding', methods=['POST'])
def compare():
    try:
        data = request.get_json()
        transRefNo = data.get('transRefNo')
        image_base64 = data.get('image')

        emb_res = face_emb.get_embedding(transRefNo, image_base64)
        return jsonify(emb_res), (200 if emb_res['resultStatus']['status'] == 'SUCCESS' else 400)
    
    except Exception as e:
        return jsonify({
                "resultStatus": {
                    "status": "FAILED",
                    "errorCode": "9999",
                    "errorMessage": "Internal Server Error"
                },
                "transRefNo": transRefNo
            }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000, debug=True)
