from flask import Flask, request, jsonify
import logging
import os
from soundcloud_pipeline import SoundCloudPipeline

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route("/extract_features", methods=["POST"])
def extract_features():
    data = request.get_json()
    url = data.get("url")
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    pipeline = SoundCloudPipeline()
    try:
        features = pipeline.extract_features(url)
    except Exception as e:
        logging.exception("Feature extraction failed")
        return jsonify({"error": str(e)}), 500

    # TODO: Send `features` to model inference service and return its prediction
    return jsonify({"features": features})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
