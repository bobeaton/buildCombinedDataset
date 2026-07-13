"""Inference an entire marked-up text file into a single concatenated wav.

Input file format: plain text, always starting with a line of the form
`spkrEmb: <SFM character name>`. Every following line is either:
  - another `spkrEmb: <name>` line, switching the active voice for subsequent text, or
  - a line of text to synthesize with the currently active voice.

`<SFM character name>` is looked up in CHARACTER_MAPPING_PATH (a flat JSON dict mapping
SFM character names, e.g. "narrator-GEN", to a TTS dataset character name, e.g. "Jesus")
to pick the speaker-embedding voice. The special name "LastSpeaker" instead reactivates
whichever single voice was active immediately before the most recent switch (a two-slot
toggle, not a full stack -- see handle_marker()).

All synthesized lines (regardless of speaker) are concatenated in order into one wav,
written to OUTPUT_DIR/<input file stem>.wav. Any existing file at that path is renamed
to a numbered ".bak" first, so previous attempts aren't lost.

Usage:
  python src/infer_file.py "sampleData\\SpeecheloCleanInLinesNormalized.txt"
  python src/infer_file.py "sampleData\\SpeecheloCleanInLinesNormalized.txt" --variant a
"""

import argparse
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import DatasetDict, load_from_disk
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

sys.path.insert(0, str(Path(__file__).parent))
from infer import build_character_embeddings, synthesize, variant_paths

CHARACTER_MAPPING_PATH = Path(r"C:\My Paratext 9 Projects\xnr\shared\milestone-markers\characterMapping.json")
OUTPUT_DIR = Path(r"C:\Users\pete_\Dropbox\NTprogress\PahariAudio\KangriWordDownloads")

SPEAKER_MARKER_PREFIX = "spkrEmb:"
LAST_SPEAKER_TOKEN = "LastSpeaker"
TARGET_SR = 16000


def load_character_mapping(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        # tolerate a JSON array of single-key objects as well as a flat dict
        mapping = {}
        for entry in raw:
            mapping.update(entry)
        return mapping
    raise ValueError(f"unexpected characterMapping.json structure: {type(raw)}")


def backup_existing_file(path: Path) -> None:
    """Rename an existing file at `path` to '<stem> (n).<ext>.bak' for the first free n."""
    if not path.exists():
        return
    n = 0
    while True:
        candidate = path.parent / f"{path.stem} ({n}){path.suffix}.bak"
        if not candidate.exists():
            break
        n += 1
    path.rename(candidate)
    print(f"backed up existing {path.name} -> {candidate.name}")


def resolve_character(sfm_name: str, character_mapping: dict, character_embeddings: dict) -> str:
    if sfm_name not in character_mapping:
        raise ValueError(
            f"encountered speaker name {sfm_name!r} in a 'spkrEmb:' marker that isn't a key "
            f"in {CHARACTER_MAPPING_PATH}"
        )
    tts_character = character_mapping[sfm_name]
    if tts_character not in character_embeddings:
        raise ValueError(
            f"characterMapping.json maps {sfm_name!r} -> {tts_character!r}, but {tts_character!r} "
            f"isn't a known trained character. Run infer.py --list-characters to see valid options."
        )
    return tts_character


def synthesize_file(input_path, character_mapping, character_embeddings, model, processor, vocoder, device):
    lines = input_path.read_text(encoding="utf-8-sig").splitlines()

    current_character = None  # resolved TTS character name currently active
    last_character = None  # the one active immediately before the most recent switch

    def handle_marker(sfm_name: str):
        nonlocal current_character, last_character
        if sfm_name == LAST_SPEAKER_TOKEN:
            if last_character is None:
                raise ValueError(
                    f"encountered '{SPEAKER_MARKER_PREFIX} {LAST_SPEAKER_TOKEN}' before any "
                    "previous speaker had been recorded"
                )
            current_character, last_character = last_character, current_character
        else:
            resolved = resolve_character(sfm_name, character_mapping, character_embeddings)
            last_character = current_character
            current_character = resolved

    waveforms = []
    synth_count = 0
    for lineno, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith(SPEAKER_MARKER_PREFIX):
            handle_marker(line[len(SPEAKER_MARKER_PREFIX):].strip())
            continue

        if current_character is None:
            raise ValueError(
                f"line {lineno} has text before any '{SPEAKER_MARKER_PREFIX} <name>' marker -- "
                "the file must start with one"
            )

        text = unicodedata.normalize("NFC", line)
        speech = synthesize(text, character_embeddings[current_character], model, processor, vocoder, device)
        waveforms.append(speech)
        synth_count += 1
        preview = text if len(text) <= 60 else text[:60] + "..."
        print(f"[{synth_count}] ({current_character}) {preview}")

    if not waveforms:
        raise ValueError("no text lines found to synthesize")

    return np.concatenate(waveforms), synth_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="path to the marked-up text file to inference")
    parser.add_argument("--variant", choices=["a", "b"], default="b", help="which transcription variant's model/dataset to use (default: b)")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if not CHARACTER_MAPPING_PATH.exists():
        print(f"Error: character mapping file not found: {CHARACTER_MAPPING_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"loading character mapping from {CHARACTER_MAPPING_PATH}...")
    character_mapping = load_character_mapping(CHARACTER_MAPPING_PATH)

    paths = variant_paths(args.variant)
    print(f"loading prepared dataset (variant={args.variant})...")
    dataset = load_from_disk(str(paths["dataset_dir"]))
    assert isinstance(dataset, DatasetDict)
    character_embeddings = build_character_embeddings(dataset)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading model from {paths['model_dir']} (device={device})...")
    processor = SpeechT5Processor.from_pretrained(str(paths["model_dir"]))
    model = SpeechT5ForTextToSpeech.from_pretrained(str(paths["model_dir"])).to(device)
    model.eval()
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

    try:
        full_audio, synth_count = synthesize_file(
            input_path, character_mapping, character_embeddings, model, processor, vocoder, device
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{input_path.stem}.wav"
    backup_existing_file(out_path)
    sf.write(str(out_path), full_audio, TARGET_SR)
    print(f"synthesized {synth_count} lines ({len(full_audio) / TARGET_SR:.1f}s) -> {out_path}")


if __name__ == "__main__":
    main()
