# Devanagari TTS Fine-Tuning (SpeechT5)

Fine-tuning `microsoft/speecht5_tts` for a Devanagari-script language, starting from a metadata
file that pairs example sentences with audio filenames.

## Environment

- **Python 3.11** (`.venv` in this folder) — chosen over 3.12+ because some audio-stack
  dependencies (numba/librosa via speechbrain) have historically lagged on Windows wheel support
  for newer Python releases, and over 3.9 because it's approaching end-of-life for current
  transformers releases.
- **GPU**: NVIDIA RTX 4060 Laptop (8GB VRAM), CUDA 13.0 driver.
- `torch`/`torchaudio` installed from `https://download.pytorch.org/whl/cu130` (see
  `requirements.txt`).

To activate:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Known Windows gotchas (already handled / to keep in mind)

1. **speechbrain model caching uses symlinks by default and this actually fails (not just
   falls back) on Windows without Developer Mode / elevated privileges** — confirmed while
   building this project (`OSError: [WinError 1314] A required privilege is not held...`).
   `HF_HUB_DISABLE_SYMLINKS` does *not* control this — speechbrain's `fetch()` has its own
   `local_strategy` argument, separate from the HF Hub cache. Fix: pass
   `local_strategy=speechbrain.utils.fetching.LocalStrategy.COPY` to `EncoderClassifier.from_hparams(...)`
   (see `src/prepare_dataset.py`). Alternatively enable Developer Mode
   (Settings > Privacy & Security > For Developers) and the default symlink strategy works fine.
2. **`torchaudio`'s `sox_io` backend does not exist on Windows** (no sox wheels). Force the
   `soundfile` backend explicitly in code rather than relying on autodetection.
3. **DataLoader workers**: Windows uses the `spawn` multiprocessing start method (not `fork`), so
   any script using `num_workers > 0` in a `DataLoader` must guard the entry point with
   `if __name__ == "__main__":`, or worker processes will fail to start / re-import the whole script.
4. **Long paths**: Hugging Face Hub cache paths can get deep. If you hit `FileNotFoundError` /
   path-too-long errors, enable Windows long path support (`git config --system core.longpaths true`
   and the `LongPathsEnabled` registry key), or set `HF_HOME` to a short path like `C:\hf`.
5. **`datasets.Dataset.map(num_proc=N)` crashes at the very end on Windows**, even at
   `num_proc=1` — it still routes through a `multiprocessing.Pool`, and Windows named pipes
   fail (`OSError: [WinError 87] The parameter is incorrect`) sending the large accumulated
   result back to the main process once the (otherwise fully successful) work is done. This
   cost a full 32-minute GPU pass while building this project. Fix: don't pass `num_proc` at
   all for GPU-bound `.map()` calls — it runs in-process with no pipe involved, and there's no
   parallelism to gain anyway since the GPU is the bottleneck.
6. **`datasets` 5.x's `Audio()` feature now requires the `torchcodec` package** (with its own
   FFmpeg dependency) to decode audio. Rather than add that dependency chain, `src/prepare_dataset.py`
   reads/resamples audio directly with `soundfile` + `librosa` and never uses `Audio()`/`cast_column`.
7. **speechbrain's pinned dependency versions have drifted from what the very newest
   `transformers`/`huggingface_hub`/`torchaudio` provide**, since we installed current bleeding-edge
   versions of everything: `speechbrain==1.1.0`'s `Xvector` model does a lazy import of an unrelated
   `k2` FSA integration that crashes on import (no `k2` Windows wheels exist) — pin `speechbrain==1.0.3`
   instead. That older speechbrain in turn expects an older `torchaudio` (`list_audio_backends()`,
   removed upstream) and an older `huggingface_hub` (`use_auth_token` kwarg, removed; and it expects
   a 404 to raise `requests.exceptions.HTTPError`, but current `huggingface_hub` uses `httpx` and
   raises `HfHubHTTPError` instead, which speechbrain's `except` clause doesn't catch). All three are
   patched narrowly in `src/prepare_dataset.py` (only inside speechbrain's own module namespace, so
   `transformers`' unrelated use of the same `huggingface_hub` functions isn't affected).

## Project layout

```
data/
  metadata/   # your sentence <-> audio-filename mapping file(s) go here
  audio/      # raw audio clips referenced by the metadata file
notebooks/    # exploratory notebooks
src/          # data prep / training scripts
requirements.txt
```

## Status

- [x] Python 3.11 venv created, CUDA-enabled torch verified working with the RTX 4060.
- [ ] Metadata file format confirmed and a loader script written.
- [ ] Dataset prep (resampling to 16kHz, speaker embeddings via speechbrain x-vector).
- [ ] Fine-tuning script (Seq2SeqTrainer).
- [ ] Inference / listening test.

## Inferencing commands

# see available character voices and how many clips each has
.\.venv\Scripts\python.exe src\infer.py --list-characters

# synthesize custom text as a given character
.\.venv\Scripts\python.exe src\infer.py --text "आपका पाठ यहाँ" --character "Jesus"

# A/B test: synthesize N held-out validation lines + copy the real recording alongside, so you can compare directly
.\.venv\Scripts\python.exe src\infer.py --compare 5 --character "Jesus"
