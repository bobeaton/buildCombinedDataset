"""Synthesize speech with the fine-tuned Kangri SpeechT5 model and listen to it.

Voices are keyed by the "character" field (e.g. "Jesus", "Paul", "Luke") rather than
the underlying voice actor's name, since the goal is a consistent character voice
regardless of which actor recorded which verse.

Defaults to the TranscriptionB-trained model/dataset (checkpoints/speecht5_kangri_b,
data/prepared_dataset_b). Pass --variant a to use the TranscriptionA-trained one instead
(checkpoints/speecht5_kangri, data/prepared_dataset).

Examples:
  # synthesize custom text with a given character's averaged voice
  python src/infer.py --text "राजा दाउद्दे दे बंसज, यूसुफ" --character "Jesus"

  # list available characters (from the prepared dataset) and their utterance counts
  python src/infer.py --list-characters

  # pick N held-out validation examples and synthesize them, alongside the
  # original recording, for a direct A/B listen
  python src/infer.py --compare 5 --character "Jesus"

  # same, but against the TranscriptionA-trained model
  python src/infer.py --compare 5 --character "Jesus" --variant a

  # synthesize one line of text in every character's voice, grouped under a per-run
  # ALL_CHARACTERS_ROOT/all_characters/<prefix>/ folder, filenames ranked by data
  # quantity (e.g. 01/01_Jesus.wav, 01/02_Paul.wav, ... 01/52_male_group13.wav)
  python src/infer.py --text "राजा दाउद्दे दे बंसज, यूसुफ" --character
"""

import argparse
import collections
import json
import re
import shutil
import unicodedata
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

ROOT = Path(__file__).parent.parent
WAV_DIR = Path(r"C:\Users\pete_\Dropbox\NTprogress\PahariAudio\KangriWordDownloads\FCBH\wavs")
OUT_DIR = ROOT / "outputs"
TARGET_SR = 16000

# Where --character all/<none> writes its per-character batches (as <ALL_CHARACTERS_ROOT>/all_characters/).
# Separate from OUT_DIR since these are meant to be shared/reviewed outside the repo.
ALL_CHARACTERS_ROOT = Path(r"C:\Users\pete_\Dropbox\NTprogress\PahariAudio\KangriWordDownloads\FCBH")


def variant_paths(variant: str):
    suffix = "" if variant == "a" else "_b"
    return {
        "model_dir": ROOT / "checkpoints" / f"speecht5_kangri{suffix}" / "final",
        "dataset_dir": ROOT / "data" / f"prepared_dataset{suffix}",
        "suffix": suffix,
    }


def build_character_embeddings(dataset) -> dict[str, np.ndarray]:
    """Average all x-vectors for each character into one representative embedding."""
    buckets: dict[str, list] = collections.defaultdict(list)
    for split in dataset.values():
        for character, emb in zip(split["character"], split["speaker_embeddings"]):
            buckets[character].append(emb)

    centroids = {}
    for character, embs in buckets.items():
        mean = np.mean(np.stack(embs), axis=0)
        mean = mean / np.linalg.norm(mean)
        centroids[character] = mean
    return centroids


def rank_characters_by_quality(
    dataset: DatasetDict,
) -> tuple[list[str], dict[str, float], dict[str, int], "collections.Counter[str]"]:
    """Rank characters by total *training-split* audio duration -- the closest
    available proxy for how well the model actually learned that voice, since
    that's the data it was fine-tuned on (validation-split minutes never
    contributed a gradient update).
    """
    train = dataset["train"]
    total_duration: dict[str, float] = collections.defaultdict(float)
    train_count: dict[str, int] = collections.defaultdict(int)
    for character, duration in zip(train["character"], train["duration"]):
        total_duration[character] += duration
        train_count[character] += 1
    val_count = collections.Counter(dataset["test"]["character"])

    ranked = sorted(total_duration, key=lambda c: total_duration[c], reverse=True)
    return ranked, total_duration, train_count, val_count


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name).strip("_")


def cap_silence(
    waveform: np.ndarray,
    max_internal_silence_s: float,
    max_boundary_silence_s: float | None = None,
    sr: int = TARGET_SR,
    top_db: float = 30,
) -> np.ndarray:
    """Cap every silence run in `waveform` -- leading, trailing, AND internal (e.g. a
    comma pause mid-sentence) -- down to at most a target duration. Only trims the
    *excess*; natural gaps shorter than the cap are left untouched.

    Under-trained voices (e.g. "God", ~12 minutes of training data vs. "Jesus"'s ~18.5
    hours) produce noticeably longer pauses throughout a clip than well-trained ones --
    confirmed empirically at ~0.4-0.7s between clauses within a single synthesized
    sentence, vs. a fraction of that for well-trained voices.

    `max_boundary_silence_s` (defaults to `max_internal_silence_s` if not given) applies
    only to the very first/last silence run. Pass a smaller value there when this clip
    will be concatenated with neighbors (see src/infer_file.py), so that an adjacent
    clip's own capped trailing/leading edge can't combine with this one to exceed the
    target -- an *internal* gap has no such neighbor to stack with, so it's always
    capped at the full `max_internal_silence_s` regardless.
    """
    if max_boundary_silence_s is None:
        max_boundary_silence_s = max_internal_silence_s
    if len(waveform) == 0:
        return waveform

    intervals = librosa.effects.split(waveform, top_db=top_db)
    if len(intervals) == 0:
        return waveform

    internal_pad = int(max_internal_silence_s * sr)
    boundary_pad = int(max_boundary_silence_s * sr)

    pieces = []
    first_start = intervals[0][0]
    pieces.append(waveform[first_start - min(first_start, boundary_pad) : intervals[0][1]])
    for i in range(1, len(intervals)):
        prev_end = intervals[i - 1][1]
        cur_start, cur_end = intervals[i]
        pad = min(cur_start - prev_end, internal_pad)
        pieces.append(waveform[cur_start - pad : cur_end])

    last_end = intervals[-1][1]
    trail_pad = min(len(waveform) - last_end, boundary_pad)
    pieces.append(waveform[last_end : last_end + trail_pad])

    return np.concatenate(pieces)


def detect_alignment_issue(
    waveform: np.ndarray, sr: int = TARGET_SR, top_db: float = 30, suspicious_gap_s: float = 1.0
) -> str | None:
    """Flag a probable SpeechT5 encoder-decoder attention-alignment failure: a silence
    gap (leading, trailing, or between words/clauses) longer than `suspicious_gap_s`.

    Confirmed empirically (see project history) that this correlates with the model
    actually failing to voice part of the text -- not just an unusually slow/quiet
    delivery -- for a specific text+speaker-embedding combination. Retrying generation
    does not reliably fix it (tested 10 retries on a known-bad case, all still bad,
    1.15s-6.24s); the practical mitigation is to catch it here for manual review, since
    it isn't preventable through code alone.

    Must be called on the *raw*, uncapped waveform -- cap_silence() trims exactly the
    gaps this function is trying to measure, so checking post-cap audio would never
    detect anything.

    Returns a human-readable description, or None if nothing suspicious was found.
    """
    if len(waveform) == 0:
        return "entire clip is silence"
    intervals = librosa.effects.split(waveform, top_db=top_db)
    if len(intervals) == 0:
        return "entire clip is silence"

    lead = intervals[0][0] / sr
    if lead > suspicious_gap_s:
        return f"leading silence of {lead:.2f}s (> {suspicious_gap_s}s) -- the model may have skipped part of the text"

    trail = (len(waveform) - intervals[-1][1]) / sr
    if trail > suspicious_gap_s:
        return f"trailing silence of {trail:.2f}s (> {suspicious_gap_s}s) -- generation may have run on past the end of the text"

    for i in range(len(intervals) - 1):
        gap = (intervals[i + 1][0] - intervals[i][1]) / sr
        if gap > suspicious_gap_s:
            return (
                f"internal silence gap of {gap:.2f}s (> {suspicious_gap_s}s) around "
                f"{intervals[i][1] / sr:.1f}s -- the model may have skipped part of the text"
            )

    return None


def synthesize(
    text,
    speaker_embedding,
    model,
    processor,
    vocoder,
    device,
    max_silence_s: float = 0.4,
    max_boundary_silence_s: float | None = None,
) -> tuple[np.ndarray, str | None]:
    text = unicodedata.normalize("NFC", text)
    inputs = processor(text=text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    speaker_tensor = torch.tensor(speaker_embedding, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        speech = model.generate_speech(input_ids, speaker_tensor, vocoder=vocoder)
    speech = speech.cpu().numpy()
    warning = detect_alignment_issue(speech)  # check the RAW clip -- capping would hide the gap
    return cap_silence(speech, max_silence_s, max_boundary_silence_s), warning


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["a", "b"], default="b", help="which transcription variant's model/dataset to use (default: b, the TranscriptionB-trained model)")
    parser.add_argument("--text", type=str, default=None, help="custom text to synthesize")
    parser.add_argument(
        "--character",
        type=str,
        default=None,
        nargs="?",
        const="all",
        help="character name (see --list-characters). Pass with no value, or 'all', to "
        "synthesize --text with every character's voice (one file each).",
    )
    parser.add_argument("--list-characters", action="store_true")
    parser.add_argument("--compare", type=int, default=0, help="synthesize N random validation examples for that character, alongside the original recording")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    paths = variant_paths(args.variant)
    model_dir, dataset_dir, suffix = paths["model_dir"], paths["dataset_dir"], paths["suffix"]

    print(f"loading prepared dataset (variant={args.variant}, for character voice embeddings)...")
    dataset = load_from_disk(str(dataset_dir))
    assert isinstance(dataset, DatasetDict)  # save_to_disk always wrote train/test splits
    character_embeddings = build_character_embeddings(dataset)

    if args.list_characters:
        ranked, total_duration, train_count, val_count = rank_characters_by_quality(dataset)

        lines = [
            "# Characters ranked by likely voice quality (descending training-audio minutes).",
            "# Ranking proxy: total duration in the training split -- the actual data the",
            "# model's fine-tuning gradient updates came from. More minutes generally means",
            "# a better-learned, more reliable voice; characters near the bottom were seen",
            "# rarely (or never) during training and may sound unstable or defer to the",
            "# dominant voice.",
            "#",
            f"{'rank':>4}  {'character':<28}{'train_min':>10}{'train_utts':>12}{'val_utts':>10}",
        ]
        for i, character in enumerate(ranked, start=1):
            lines.append(
                f"{i:>4}  {character:<28}{total_duration[character] / 60:>10.1f}"
                f"{train_count[character]:>12}{val_count.get(character, 0):>10}"
            )

        out_path = OUT_DIR / f"characters_by_likely_quality{suffix}.txt"
        OUT_DIR.mkdir(exist_ok=True)
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        print("\n".join(lines))
        print(f"\nwrote ranked list -> {out_path}")
        return

    character = args.character or "Jesus"
    if character != "all" and character not in character_embeddings:
        raise ValueError(f"unknown character {character!r}. Run with --list-characters to see options.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading model from {model_dir} (device={device})...")
    processor = SpeechT5Processor.from_pretrained(str(model_dir))
    model = SpeechT5ForTextToSpeech.from_pretrained(str(model_dir)).to(device)
    model.eval()
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

    OUT_DIR.mkdir(exist_ok=True)

    if character == "all":
        text = args.text or dataset["test"][0]["text"]
        text = unicodedata.normalize("NFC", text)

        ranked, *_ = rank_characters_by_quality(dataset)  # best-data-first order
        all_dir = ALL_CHARACTERS_ROOT / f"all_characters{suffix}"
        all_dir.mkdir(parents=True, exist_ok=True)

        # Each run gets its own <prefix>/ subfolder (e.g. 01/) so batches from different
        # --text values are identifiable and grouped across runs; the prefix counter
        # increments per run and is tracked in manifest.json (kept directly under
        # all_characters/, not per-batch). Within a batch, filenames are numbered by
        # quality rank -- most training data first -- e.g. 01_Jesus.wav, 02_Paul.wav.
        manifest_path = all_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
        prefix = f"{max((int(entry['prefix']) for entry in manifest), default=0) + 1:02d}"
        batch_dir = all_dir / prefix
        batch_dir.mkdir(exist_ok=True)

        print(f"synthesizing for all {len(ranked)} characters (prefix={prefix}): {text!r}")
        for i, char_name in enumerate(ranked, start=1):
            speech, warning = synthesize(text, character_embeddings[char_name], model, processor, vocoder, device)
            fname = f"{i:02d}_{sanitize_filename(char_name)}.wav"
            out_path = batch_dir / fname
            sf.write(str(out_path), speech, 16000)
            print(f"[{i}/{len(ranked)}] {char_name} -> {out_path}")
            if warning:
                print(f"    WARNING: {warning}")

        manifest.append({"text": text, "prefix": prefix})
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {len(ranked)} samples -> {all_dir}")
        print(f"updated manifest -> {manifest_path}")
        return

    if args.compare:
        val = dataset["test"].filter(lambda ex: ex["character"] == character)
        if len(val) == 0:
            raise ValueError(f"no validation examples for character {character!r}")
        rng = np.random.default_rng(args.seed)
        idxs = rng.choice(len(val), size=min(args.compare, len(val)), replace=False)
        for i, idx in enumerate(idxs):
            ex = val[int(idx)]
            text = ex["text"]
            speech, warning = synthesize(text, character_embeddings[character], model, processor, vocoder, device)

            synth_path = OUT_DIR / f"compare_{i}_synth{suffix}.wav"
            ref_path = OUT_DIR / f"compare_{i}_reference{suffix}.wav"
            txt_path = OUT_DIR / f"compare_{i}_text{suffix}.txt"

            sf.write(str(synth_path), speech, 16000)
            shutil.copy(WAV_DIR / ex["audio_file"], ref_path)
            txt_path.write_text(text, encoding="utf-8")

            print(f"[{i}] text: {text}")
            print(f"    synth:     {synth_path}")
            print(f"    reference: {ref_path}")
            if warning:
                print(f"    WARNING: {warning}")
        return

    # Default listening-test sentences: pulled from this variant's own validation text
    # rather than hardcoded, since a hand-written sentence from one transcription variant
    # can contain characters absent from the other's vocab (confirmed: transcriptionA's
    # zero-width joiner isn't in transcriptionB's 85-character vocab).
    if args.text:
        texts = [args.text]
    else:
        texts = [ex["text"] for ex in dataset["test"].select(range(2))]

    for i, text in enumerate(texts):
        speech, warning = synthesize(text, character_embeddings[character], model, processor, vocoder, device)
        out_path = OUT_DIR / f"sample_{i}{suffix}.wav"
        sf.write(str(out_path), speech, 16000)
        print(f"[{i}] text: {text}")
        print(f"    wrote: {out_path}")
        if warning:
            print(f"    WARNING: {warning}")


if __name__ == "__main__":
    main()
