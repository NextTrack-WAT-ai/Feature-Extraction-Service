
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from soundcloud_pipeline import SoundCloudScraper, YTDLPDownloader, SoundCloudPipeline
from run_pipeline import load_pipelines, vector_from_feats, bounded_targets

# ── Config ──────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DOWNLOAD_FOLDER = Path("/tmp/downloads")
DOWNLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
MODEL_DIR = BASE_DIR / "models"

# METADATA_CSV = BASE_DIR / "music_info_cleaned.csv"
pipeline = SoundCloudPipeline(start_index=0, end_index=0, download_folder=DOWNLOAD_FOLDER)
analyzer = pipeline.analyzer
models = load_pipelines(model_dir=MODEL_DIR)

# df = pd.read_csv(METADATA_CSV)
# name_to_tid = {f"{r['artist']} - {r['name']}": r["track_id"] for _, r in df.iterrows()}
# targets = {r["track_id"]: r.drop(["track_id", "artist", "name"]).to_dict() for _, r in df.iterrows()}

# ── Single Track Endpoint ───────────────────────
@app.route('/extract_features', methods=['POST'])
def extract_features():
    data = request.get_json()
    debug = request.args.get("debug") == "1"

    artist = data.get('artist')
    track_name = data.get('track_name')

    if not artist or not track_name:
        return jsonify({'error': 'Both "artist" and "track_name" must be provided'}), 400

    response = _process_track(artist, track_name, debug)
    if not response:
        logging.error(f"Empty response for track: {artist} - {track_name}")
        return jsonify({'error': 'Unexpected empty response'}), 500
    return response


# ── Batch Track Endpoint ────────────────────────
@app.route('/extract_features_batch', methods=['POST'])
def extract_features_batch():
    try:
        data = request.get_json()
        debug = request.args.get("debug") == "1"
        tracks = data.get("tracks")

        if not tracks or not isinstance(tracks, list):
            return jsonify({'error': 'Missing or invalid "tracks" list'}), 400

        def process_single(entry):
            artist = entry.get("artist")
            name = entry.get("track_name")
            if not artist or not name:
                return {"error": "Missing artist or track_name", "track": entry}
            with app.app_context():
                return _process_track(artist, name, debug, return_dict=True)

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_single, tracks))

        return jsonify({"results": results})
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Shared Core Function ─────────────────────────
def _process_track(artist, track_name, debug=False, return_dict=False):
    track_key = f"{artist} - {track_name}"
    audio_file = None

    try:
        logging.info(f"Processing: {track_key}")

        # Search and download
        scraper = SoundCloudScraper(browserless_api_key=os.environ["BROWSERLESS_API_KEY"])
        html = scraper.search(track_name, artist)
        results = scraper.parse_results(html)
        best_match = scraper.find_best_match(results, track_name, artist)

        if not best_match:
            raise ValueError(f"No suitable match found for '{track_key}'")

        downloader = YTDLPDownloader(DOWNLOAD_FOLDER)
        audio_path, success = downloader.download_track(best_match["url"], artist, track_name)

        if not success or not audio_path:
            raise ValueError(f"Download failed for '{track_key}'")

        audio_file = Path(audio_path)
        logging.info(f"Downloaded: {audio_file}")

        # Extract base features
        base_feats = analyzer.precompute_base_features(str(audio_file))
        if base_feats is None:
            raise ValueError("Base feature extraction failed")

        if debug:
            import pprint
            logging.info("=== Base Features ===")
            pprint.pprint(base_feats)

        # Predict
        predictions = {}
        for target, pipe in models.items():
            X_raw = vector_from_feats(base_feats, target)

            if X_raw is None or np.isnan(X_raw).any() or X_raw.shape[1] == 0:
                predictions[target] = None
                continue

            pred = pipe.predict(X_raw)[0]
            if target == "key":
                pred = int(round(pred)) % 12
            elif target in bounded_targets:
                pred = float(np.clip(pred, 0, 1))
            else:
                pred = float(pred)

            predictions[target] = pred

        result = {"track": track_key, "features": predictions}
        return result if return_dict else (jsonify(predictions), 200)

    except Exception as e:
        logging.exception(f"Error in track: {track_key}")
        err_response = {"error": str(e), "track": track_key}
        return err_response if return_dict else (jsonify(err_response), 500)

    finally:
        if audio_file and audio_file.exists():
            try:
                audio_file.unlink()
                logging.info(f"Deleted temp file: {audio_file}")
            except Exception as e:
                logging.warning(f"Could not delete {audio_file}: {e}")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
