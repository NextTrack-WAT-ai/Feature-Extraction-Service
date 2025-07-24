import os
import logging
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor
from soundcloud_pipeline import SoundCloudScraper, YTDLPDownloader, SoundCloudPipeline
import requests

# ── Setup ────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
app = Flask(__name__)
DOWNLOAD_FOLDER = Path("/tmp/downloads")
DOWNLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

pipeline = SoundCloudPipeline(download_folder=DOWNLOAD_FOLDER)
analyzer = pipeline.analyzer

INFERENCE_URL = os.getenv("INFERENCE_URL")  # e.g. http://model-inference-service:8081/predict

# ── Core Track Processor ─────────────────────────
def _process_track(artist, track_name, debug=False, return_dict=False):
    track_key = f"{artist} - {track_name}"
    audio_file = None

    try:
        logging.info(f"Processing: {track_key}")
        scraper = SoundCloudScraper(browserless_api_key=os.environ["BROWSERLESS_API_KEY"])
        html = scraper.search(track_name, artist)
        results = scraper.parse_results(html)
        best_match = scraper.find_best_match(results, track_name, artist)
        if not best_match:
            raise ValueError(f"No match found for '{track_key}'")

        downloader = YTDLPDownloader(DOWNLOAD_FOLDER)
        audio_path, success = downloader.download_track(best_match["url"], artist, track_name)
        if not success or not audio_path:
            raise ValueError(f"Download failed for '{track_key}'")

        audio_file = Path(audio_path)
        logging.info(f"Downloaded: {audio_file}")

        base_feats = analyzer.precompute_base_features(str(audio_file))
        if base_feats is None:
            raise ValueError("Feature extraction failed")

        if debug:
            import pprint
            logging.info("=== Base Features ===")
            pprint.pprint(base_feats)

        # Send to inference service
        resp = requests.post(INFERENCE_URL, json={"track": track_key, "features": base_feats})
        return resp.json() if return_dict else (jsonify(resp.json()), resp.status_code)

    except Exception as e:
        logging.exception(f"Error in track: {track_key}")
        error_resp = {"error": str(e), "track": track_key}
        return error_resp if return_dict else (jsonify(error_resp), 500)

    finally:
        if audio_file and audio_file.exists():
            try:
                audio_file.unlink()
                logging.info(f"Deleted temp file: {audio_file}")
            except Exception as e:
                logging.warning(f"Failed to delete file: {e}")

# ── Endpoints ─────────────────────────────────────
@app.route('/extract_features', methods=['POST'])
def extract_features():
    data = request.get_json()
    artist = data.get("artist")
    track_name = data.get("track_name")
    debug = request.args.get("debug") == "1"

    if not artist or not track_name:
        return jsonify({"error": "Missing artist or track_name"}), 400

    return _process_track(artist, track_name, debug)

@app.route('/extract_features_batch', methods=['POST'])
def extract_features_batch():
    data = request.get_json()
    tracks = data.get("tracks")
    debug = request.args.get("debug") == "1"

    if not tracks or not isinstance(tracks, list):
        return jsonify({"error": "Missing or invalid 'tracks' list"}), 400

    def handle_track(entry):
        artist = entry.get("artist")
        name = entry.get("track_name")
        if not artist or not name:
            return {"error": "Invalid entry", "track": entry}
        return _process_track(artist, name, debug, return_dict=True)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(handle_track, tracks))

    return jsonify({"results": results})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
