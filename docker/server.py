"""Kangri SpeechT5 TTS webservice.

Wraps the same functions the CLI scripts (src/infer.py, src/infer_file.py) use --
imported directly from the mounted project, not reimplemented -- so behavior stays in
sync with the CLI automatically. See settings.py for configuration and buildDocker.ps1
for how to run this.

Endpoints:
  GET  /                              simple browser UI for manual testing
  GET  /api/v1/tts/health/            liveness check
  GET  /api/v1/tts/characters/        list characters + quality ranking (?variant=a|b)
  POST /api/v1/tts/synthesize/        synthesize text with one character, all characters,
                                       or N validation examples compared against real audio
  POST /api/v1/tts/synthesize-file/   synthesize an entire spkrEmb-marked-up file
"""

import io
import json
import sys
import tempfile
import unicodedata
import zipfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from flask_swagger_ui import get_swaggerui_blueprint
from gevent.pywsgi import WSGIServer
from werkzeug.exceptions import HTTPException

from settings import (
    API_KEY,
    CHARACTER_MAPPING_DIR,
    CHARACTER_MAPPING_FILENAME,
    DEFAULT_VARIANT,
    PORT,
    PROJECT_PATH,
    WAV_DIR,
)

PROJECT_SRC = Path(PROJECT_PATH) / "src"
if not PROJECT_SRC.is_dir():
    raise FileNotFoundError(
        f"Project src directory not found: {PROJECT_SRC} -- check the PROJECT_PATH volume "
        "mount (see buildDocker.ps1)"
    )
sys.path.insert(0, str(PROJECT_SRC))

import numpy as np
import soundfile as sf
import torch
from datasets import DatasetDict, load_from_disk
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

from infer import build_character_embeddings, rank_characters_by_quality, sanitize_filename, synthesize, variant_paths
from infer_file import load_character_mapping, synthesize_file

TARGET_SR = 16000
OPENAPI_SPEC_PATH = Path(__file__).parent / "openapi.yaml"
OPENAPI_SPEC_URL = "/openapi.yaml"
SWAGGER_URL = "/docs"

app = Flask(__name__)
app.register_blueprint(
    get_swaggerui_blueprint(SWAGGER_URL, OPENAPI_SPEC_URL, config={"app_name": "Kangri TTS Webservice"}),
    url_prefix=SWAGGER_URL,
)


@app.route(OPENAPI_SPEC_URL)
def openapi_spec():
    return send_file(OPENAPI_SPEC_PATH, mimetype="text/yaml")


_bundles = {}  # variant -> dict(processor, model, vocoder, dataset, character_embeddings)


def _check_auth():
    return not API_KEY or request.headers.get("Authorization") == API_KEY


def _unauthorized():
    return jsonify({"error": "Unauthorized"}), 401


def get_bundle(variant: str) -> dict:
    if variant not in ("a", "b"):
        raise ValueError(f"invalid variant {variant!r}, must be 'a' or 'b'")
    if variant in _bundles:
        return _bundles[variant]

    paths = variant_paths(variant)
    print(f"[variant={variant}] loading dataset from {paths['dataset_dir']}...")
    dataset = load_from_disk(str(paths["dataset_dir"]))
    assert isinstance(dataset, DatasetDict)
    character_embeddings = build_character_embeddings(dataset)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[variant={variant}] loading model from {paths['model_dir']} (device={device})...")
    processor = SpeechT5Processor.from_pretrained(str(paths["model_dir"]))
    model = SpeechT5ForTextToSpeech.from_pretrained(str(paths["model_dir"])).to(device)
    model.eval()
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

    bundle = {
        "dataset": dataset,
        "character_embeddings": character_embeddings,
        "processor": processor,
        "model": model,
        "vocoder": vocoder,
        "device": device,
    }
    _bundles[variant] = bundle
    print(f"[variant={variant}] ready.")
    return bundle


def _wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, TARGET_SR, format="WAV")
    return buf.getvalue()


@app.errorhandler(Exception)
def handle_unexpected_error(e: Exception):
    # Without this, an unhandled exception falls through to Flask's default HTML error
    # page, which breaks the browser UI's `await res.json()` call with a confusing
    # "Unexpected token '<'" parse error instead of showing the real problem. But
    # HTTPException (404, 405, etc.) already has correct status/behavior -- e.g. Swagger
    # UI's own asset routes rely on a plain 404 for missing files -- so only genuinely
    # unexpected exceptions should be remapped to a JSON 500.
    if isinstance(e, HTTPException):
        return e.get_response()

    import traceback

    traceback.print_exc()
    return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/v1/tts/health/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "loaded_variants": list(_bundles.keys())})


@app.route("/api/v1/tts/characters/", methods=["GET"])
def characters():
    if not _check_auth():
        return _unauthorized()
    variant = request.args.get("variant", DEFAULT_VARIANT)
    try:
        bundle = get_bundle(variant)
    except (ValueError, FileNotFoundError) as e:
        return jsonify({"error": str(e)}), 400

    ranked, total_duration, train_count, val_count = rank_characters_by_quality(bundle["dataset"])
    result = [
        {
            "rank": i,
            "character": name,
            "trainMinutes": round(total_duration[name] / 60, 1),
            "trainUtterances": train_count[name],
            "valUtterances": val_count.get(name, 0),
        }
        for i, name in enumerate(ranked, start=1)
    ]
    return jsonify({"variant": variant, "characters": result})


@app.route("/api/v1/tts/synthesize/", methods=["POST"])
def synthesize_endpoint():
    if not _check_auth():
        return _unauthorized()

    data = request.get_json(silent=True) or {}
    variant = data.get("variant", DEFAULT_VARIANT)
    character = data.get("character") or "Jesus"
    compare = int(data.get("compare", 0))
    seed = int(data.get("seed", 0))
    text = data.get("text")

    try:
        bundle = get_bundle(variant)
    except (ValueError, FileNotFoundError) as e:
        return jsonify({"error": str(e)}), 400

    dataset = bundle["dataset"]
    character_embeddings = bundle["character_embeddings"]
    model, processor, vocoder, device = bundle["model"], bundle["processor"], bundle["vocoder"], bundle["device"]

    if character != "all" and character not in character_embeddings:
        return jsonify({"error": f"unknown character {character!r}. See GET /api/v1/tts/characters/."}), 400

    if compare:
        if not WAV_DIR:
            return jsonify({"error": "compare requires WAV_DIR to be configured/mounted on the server"}), 400
        wav_dir = Path(WAV_DIR)
        val = dataset["test"].filter(lambda ex: ex["character"] == character)
        if len(val) == 0:
            return jsonify({"error": f"no validation examples for character {character!r}"}), 400
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(val), size=min(compare, len(val)), replace=False)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, idx in enumerate(idxs):
                ex = val[int(idx)]
                line_text = ex["text"]
                speech = synthesize(line_text, character_embeddings[character], model, processor, vocoder, device)
                zf.writestr(f"compare_{i}_synth.wav", _wav_bytes(speech))
                zf.writestr(f"compare_{i}_reference.wav", (wav_dir / ex["audio_file"]).read_bytes())
                zf.writestr(f"compare_{i}_text.txt", line_text.encode("utf-8"))
        buf.seek(0)
        return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=f"compare_{character}.zip")

    if character == "all":
        ranked, *_ = rank_characters_by_quality(dataset)
        text_to_use = unicodedata.normalize("NFC", text) if text else dataset["test"][0]["text"]

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            manifest = [{"text": text_to_use, "rank": i, "character": name} for i, name in enumerate(ranked, start=1)]
            for i, char_name in enumerate(ranked, start=1):
                speech = synthesize(text_to_use, character_embeddings[char_name], model, processor, vocoder, device)
                fname = f"{i:02d}_{sanitize_filename(char_name)}.wav"
                zf.writestr(fname, _wav_bytes(speech))
            zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        buf.seek(0)
        return send_file(buf, mimetype="application/zip", as_attachment=True, download_name="all_characters.zip")

    if not text:
        return jsonify({"error": "'text' is required (unless 'compare' or character 'all' is used)"}), 400

    speech = synthesize(text, character_embeddings[character], model, processor, vocoder, device)
    buf = io.BytesIO(_wav_bytes(speech))
    return send_file(buf, mimetype="audio/wav", as_attachment=True, download_name="synthesized.wav")


@app.route("/api/v1/tts/synthesize-file/", methods=["POST"])
def synthesize_file_endpoint():
    if not _check_auth():
        return _unauthorized()

    if not CHARACTER_MAPPING_DIR:
        return jsonify({"error": "synthesize-file requires CHARACTER_MAPPING_DIR to be configured/mounted on the server"}), 400
    mapping_path = Path(CHARACTER_MAPPING_DIR) / CHARACTER_MAPPING_FILENAME
    if not mapping_path.exists():
        return jsonify({"error": f"character mapping file not found: {mapping_path}"}), 400

    variant = request.form.get("variant") or request.args.get("variant", DEFAULT_VARIANT)

    with tempfile.TemporaryDirectory() as tmpdir:
        if "file" in request.files:
            input_path = Path(tmpdir) / (request.files["file"].filename or "input.txt")
            request.files["file"].save(str(input_path))
        else:
            data = request.get_json(silent=True) or {}
            if not data.get("text"):
                return jsonify({"error": "provide either a multipart 'file' upload or JSON {'text': ...}"}), 400
            variant = data.get("variant", variant)
            input_path = Path(tmpdir) / "input.txt"
            input_path.write_text(data["text"], encoding="utf-8")

        try:
            bundle = get_bundle(variant)
        except (ValueError, FileNotFoundError) as e:
            return jsonify({"error": str(e)}), 400

        character_mapping = load_character_mapping(mapping_path)

        try:
            full_audio, synth_count = synthesize_file(
                input_path,
                character_mapping,
                bundle["character_embeddings"],
                bundle["model"],
                bundle["processor"],
                bundle["vocoder"],
                bundle["device"],
            )
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    buf = io.BytesIO(_wav_bytes(full_audio))
    print(f"synthesized {synth_count} lines from {input_path.name}")
    return send_file(buf, mimetype="audio/wav", as_attachment=True, download_name=f"{input_path.stem}.wav")


if __name__ == "__main__":
    http_server = WSGIServer(("0.0.0.0", PORT), app)
    print(f"listening on 0.0.0.0:{PORT} (project={PROJECT_PATH}, default variant={DEFAULT_VARIANT})")
    http_server.serve_forever()
