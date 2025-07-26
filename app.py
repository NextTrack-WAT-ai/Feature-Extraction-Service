import os
import logging
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor
from soundcloud_pipeline import SoundCloudScraper, YTDLPDownloader, PytubeDownloader, SoundCloudPipeline
from youtubesearchpython import VideosSearch
import requests
import librosa

# ── Setup ────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
app = Flask(__name__)
DOWNLOAD_FOLDER = Path("/tmp/downloads")
DOWNLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

pipeline = SoundCloudPipeline(download_folder=DOWNLOAD_FOLDER)
analyzer = pipeline.analyzer

INFERENCE_URL = os.getenv("INFERENCE_URL")  # e.g. http://model-inference-service:8081/predict

def get_audio_duration(path):
    try:
        y, sr = librosa.load(path, sr=None, mono=True, duration=300)
        return librosa.get_duration(y=y, sr=sr)
    except Exception as e:
        logging.warning(f"Failed to get duration for {path}: {e}")
        return 0


def find_and_download_track(artist, track_name, scraper, downloader, pytube_fallback, min_duration_sec=60):
    """Try SoundCloud first, fallback to YouTube if needed or too short."""
    # Try SoundCloud
    try:
        html = scraper.search(track_name, artist)
        results = scraper.parse_results(html)
        best_match = scraper.find_best_match(results, track_name, artist)
        if best_match:
            path, success = downloader.download_track(best_match["url"], artist, track_name)
            if success and path:
                duration = get_audio_duration(path)
                if duration >= min_duration_sec:
                    return path, "soundcloud"
                else:
                    logging.warning(f"Track too short ({duration:.2f}s), falling back to YouTube.")
                    Path(path).unlink(missing_ok=True)
    except Exception as e:
        logging.warning(f"SoundCloud download failed: {e}")

    # Fallback to YouTube with yt-dlp
    yt_url = f"ytsearch1:{artist} {track_name}"
    path, success = downloader.download_track(yt_url, artist, track_name)
    if success:
        duration = get_audio_duration(path)
        if duration >= min_duration_sec:
            return path, "youtube_yt-dlp"
        else:
            logging.warning(f"YouTube fallback also too short: {duration:.2f}s")
            Path(path).unlink(missing_ok=True)

    # # Final fallback to Pytube + youtube-search-python
    # try:
    #     query = f"{artist} {track_name}"
    #     logging.info(f"Trying pytube fallback for query: {query}")

    #     # Search YouTube for the video URL
    #     videos_search = VideosSearch(query, limit=1)
    #     results = videos_search.result()
    #     if not results.get('result'):
    #         raise ValueError("No YouTube results found")
    #     video_id = results['result'][0]['id']
    #     video_url = f"https://www.youtube.com/watch?v={video_id}"
    #     logging.info(f"Found YouTube video: {video_url}")

    #     # Download audio only with pytube
    #     path = pytube_fallback.download_audio_only(video_url)
    #     if path is None:
    #         raise ValueError("Pytube failed to download audio")

    #     duration = get_audio_duration(path)
    #     if duration >= min_duration_sec:
    #         return path, "youtube_pytube"
    #     else:
    #         logging.warning(f"Pytube fallback audio too short: {duration:.2f}s")
    #         Path(path).unlink(missing_ok=True)

    # except Exception as e:
    #     logging.warning(f"Pytube fallback failed: {e}")

    return None, None

# ── Core Track Processor ─────────────────────────
def _process_track(artist, track_name, debug=False, return_dict=False):
    track_key = f"{artist} - {track_name}"
    audio_file = None

    try:
        logging.info(f"Processing: {track_key}")
        scraper = SoundCloudScraper(browserless_api_key=os.environ["BROWSERLESS_API_KEY"])
        downloader = YTDLPDownloader(DOWNLOAD_FOLDER, browserless_api_key=os.environ["BROWSERLESS_API_KEY"])
        pytube_fallback = PytubeDownloader(DOWNLOAD_FOLDER)
        audio_path, source = find_and_download_track(artist, track_name, scraper, downloader, pytube_fallback)
        if not audio_path:
            raise ValueError(f"Failed to get usable audio for '{track_key}' (too short or unavailable)")

        audio_file = Path(audio_path)
        logging.info(f"Downloaded: {audio_file}")

        base_feats = analyzer.precompute_base_features(str(audio_file))
        if base_feats is None:
            raise ValueError("Feature extraction failed")

        if debug:
            import pprint
            logging.info("=== Base Features ===")
            pprint.pprint(base_feats)

        # Convert all numpy types in base_feats to native Python types
        def to_python_type(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: to_python_type(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple, np.ndarray)):
                return [to_python_type(x) for x in obj]
            else:
                return obj

        base_feats_py = to_python_type(base_feats)

        # Send to inference service
        print("Sending post request to inference service at ", INFERENCE_URL)
        resp = requests.post(INFERENCE_URL, json={"track": track_key, "features": base_feats_py})
        try:
            resp_json = resp.json()
        except requests.exceptions.JSONDecodeError:
            logging.error(f"Non-JSON response: {resp.text}")
            resp_json = {"error": "Internal server error during feature extraction"}
            return jsonify(resp_json), 500

        return resp_json if return_dict else (jsonify(resp_json), resp.status_code)

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
