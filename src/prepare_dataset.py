"""Build the SpeechT5 training dataset from combined_metadata.csv:
  - resample audio to 16kHz
  - tokenize transcriptionA with our custom Devanagari tokenizer
  - extract target log-mel spectrograms
  - compute per-utterance speaker x-vector embeddings (speechbrain)
  - filter outlier-length examples, split train/val, save to disk

Run src/build_tokenizer.py first (produces tokenizer/speecht5_tokenizer and model_init/).
"""

import os
import sys
import unicodedata
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

import librosa
import numpy as np
import requests
import soundfile as sf
import torch
import torchaudio
import huggingface_hub
from datasets import Dataset

# torchaudio >=2.9 removed list_audio_backends() as part of the new I/O dispatcher;
# speechbrain 1.0.x still probes for it at import time (informational only). Shim it
# BEFORE importing anything from speechbrain, since it runs at speechbrain's import time.
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

import speechbrain.utils.fetching as _sb_fetching


# speechbrain 1.0.x's fetch() was written against an older huggingface_hub that (a)
# accepted a use_auth_token kwarg (removed) and (b) raised requests.exceptions.HTTPError
# on a 404 (huggingface_hub now uses httpx internally and raises HfHubHTTPError, which
# speechbrain's `except HTTPError` in fetching.py does not catch, so a routine "optional
# file not found" 404 becomes an unhandled crash instead of being skipped).
#
# Patched only inside speechbrain.utils.fetching's own module namespace -- NOT the
# global huggingface_hub module -- because transformers' own hf_hub_download call sites
# rely on the real exception types to skip optional files, and a global monkeypatch
# broke that (surfaced as a spurious crash loading microsoft/speecht5_tts's optional
# processor_config.json).
class _HubCompatShim:
    def __getattr__(self, name):
        return getattr(huggingface_hub, name)

    def hf_hub_download(self, *args, **kwargs):
        kwargs.pop("use_auth_token", None)
        try:
            return huggingface_hub.hf_hub_download(*args, **kwargs)
        except huggingface_hub.errors.HfHubHTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 404:
                raise requests.exceptions.HTTPError(f"404 Client Error (compat shim): {e}") from e
            raise


_sb_fetching.huggingface_hub = _HubCompatShim()

sys.path.insert(0, str(Path(__file__).parent))
from load_metadata import parse_metadata

ROOT = Path(__file__).parent.parent
METADATA_CSV = r"C:\Users\pete_\Dropbox\NTprogress\PahariAudio\KangriWordDownloads\FCBH\combined_metadata.csv"
WAV_DIR = Path(r"C:\Users\pete_\Dropbox\NTprogress\PahariAudio\KangriWordDownloads\FCBH\wavs")
TOKENIZER_DIR = ROOT / "tokenizer" / "speecht5_tokenizer"
OUT_DIR = ROOT / "data" / "prepared_dataset"

MAX_TEXT_CHARS = 400   # drops a handful of outlier verse-group clips (p99.5 ~= 343)
MAX_DURATION_S = 25.0
VAL_FRACTION = 0.05
SEED = 42

SPK_MODEL_NAME = "speechbrain/spkrec-xvect-voxceleb"


TARGET_SR = 16000


def load_audio_16k(path: str) -> np.ndarray:
    """Read a wav file and resample to 16kHz mono via soundfile+librosa.

    Deliberately not using datasets' Audio() feature: datasets 5.x requires the
    torchcodec package (with its own FFmpeg dependency) to decode audio, which is
    one more moving part to get working on Windows. soundfile/librosa are already
    proven to work in this environment, so we do the read + resample ourselves.
    """
    waveform, sr = sf.read(path, dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sr != TARGET_SR:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)
    return waveform


def build_raw_dataset() -> Dataset:
    rows = parse_metadata(METADATA_CSV)
    data = {
        "audio_path": [str(WAV_DIR / r.audio_file) for r in rows],
        "text": [unicodedata.normalize("NFC", r.transcription_a) for r in rows],
        "speaker": [r.speaker.strip() for r in rows],
        "character": [r.character.strip() for r in rows],
        "audio_file": [r.audio_file for r in rows],
    }
    return Dataset.from_dict(data)


def main():
    from transformers import SpeechT5Processor, SpeechT5FeatureExtractor, SpeechT5Tokenizer
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    tokenizer = SpeechT5Tokenizer.from_pretrained(str(TOKENIZER_DIR))
    feature_extractor = SpeechT5FeatureExtractor.from_pretrained("microsoft/speecht5_tts")
    processor = SpeechT5Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Windows: symlinks require Developer Mode / elevated privileges, so copy instead.
    speaker_model = EncoderClassifier.from_hparams(
        source=SPK_MODEL_NAME,
        run_opts={"device": device},
        savedir=str(ROOT / "model_cache" / SPK_MODEL_NAME.replace("/", "_")),
        local_strategy=LocalStrategy.COPY,
    )

    def create_speaker_embedding(waveform):
        with torch.no_grad():
            emb = speaker_model.encode_batch(torch.tensor(waveform, device=device).unsqueeze(0))
            emb = torch.nn.functional.normalize(emb, dim=2)
        return emb.squeeze().cpu().numpy()

    def prepare_example(example):
        waveform = load_audio_16k(example["audio_path"])
        out = processor(
            text=example["text"],
            audio_target=waveform,
            sampling_rate=TARGET_SR,
            return_attention_mask=False,
        )
        out["labels"] = out["labels"][0]
        out["speaker_embeddings"] = create_speaker_embedding(waveform)
        out["duration"] = len(waveform) / TARGET_SR
        return out

    print("building raw dataset (metadata + audio decode/resample plan)...")
    ds = build_raw_dataset()
    if len(sys.argv) > 1 and sys.argv[1] == "--limit":
        n = int(sys.argv[2])
        ds = ds.select(range(n))
        print(f"--limit: using first {n} rows only (smoke test)")
    print(f"raw dataset: {len(ds)} rows")

    print("extracting features (log-mel targets, tokenized text, speaker x-vectors)...")
    ds = ds.map(
        prepare_example,
        remove_columns=["audio_path"],
        # No num_proc: datasets.map(num_proc=N) always routes through a
        # multiprocessing.Pool (even at N=1), and on Windows sending the large
        # accumulated result payload back through the pipe crashes with
        # `OSError: [WinError 87] The parameter is incorrect` right at the end
        # of the run (confirmed -- lost a full 32-minute pass this way).
        # Omitting it runs single-process, in-process, no pipe involved, and
        # we're GPU-bound here anyway so there's no parallelism to gain.
    )

    before = len(ds)
    ds = ds.filter(lambda ex: len(ex["input_ids"]) <= MAX_TEXT_CHARS)
    ds = ds.filter(lambda ex: ex["duration"] <= MAX_DURATION_S)
    print(f"filtered {before - len(ds)} outlier rows (text > {MAX_TEXT_CHARS} chars or audio > {MAX_DURATION_S}s)")
    print(f"remaining: {len(ds)} rows")

    ds = ds.train_test_split(test_size=VAL_FRACTION, seed=SEED)
    print(f"train: {len(ds['train'])}, val: {len(ds['test'])}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(OUT_DIR))
    print(f"saved prepared dataset -> {OUT_DIR}")


if __name__ == "__main__":
    main()
