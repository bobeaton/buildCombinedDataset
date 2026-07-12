"""Synthesize speech with the fine-tuned Kangri SpeechT5 model and listen to it.

Voices are keyed by the "character" field (e.g. "Jesus", "Paul", "Luke") rather than
the underlying voice actor's name, since the goal is a consistent character voice
regardless of which actor recorded which verse.

Examples:
  # synthesize custom text with a given character's averaged voice
  python src/infer.py --text "राजा दाउद्दे दे बंसज, यूसुफ" --character "Jesus"

  # list available characters (from the prepared dataset) and their utterance counts
  python src/infer.py --list-characters

  # pick N held-out validation examples and synthesize them, alongside the
  # original recording, for a direct A/B listen
  python src/infer.py --compare 5 --character "Jesus"
"""

import argparse
import collections
import shutil
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "checkpoints" / "speecht5_kangri" / "final"
DATASET_DIR = ROOT / "data" / "prepared_dataset"
WAV_DIR = Path(r"C:\Users\pete_\Dropbox\NTprogress\PahariAudio\KangriWordDownloads\FCBH\wavs")
OUT_DIR = ROOT / "outputs"

DEFAULT_SENTENCES = [
    "राजा दाउद्दे दे बंसज, यूसुफ! दिख, जेह़ड़ा बच्‍चा मरिअमा दे पेट्टे च ह़ै, सैह़ पबित्तर आत्‍मैं दिआ समर्था नैं ह़ै।",
    "परमेसरैं तिज्‍जो पर बड़ी दया कित्तिओ ह़ै! सैह़ तिज्‍जो सौग्‍गी ह़न!",
]


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


def synthesize(text, speaker_embedding, model, processor, vocoder, device) -> np.ndarray:
    text = unicodedata.normalize("NFC", text)
    inputs = processor(text=text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    speaker_tensor = torch.tensor(speaker_embedding, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        speech = model.generate_speech(input_ids, speaker_tensor, vocoder=vocoder)
    return speech.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default=None, help="custom text to synthesize")
    parser.add_argument("--character", type=str, default=None, help="character name (see --list-characters)")
    parser.add_argument("--list-characters", action="store_true")
    parser.add_argument("--compare", type=int, default=0, help="synthesize N random validation examples for that character, alongside the original recording")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("loading prepared dataset (for character voice embeddings)...")
    dataset = load_from_disk(str(DATASET_DIR))
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

        out_path = OUT_DIR / "characters_by_likely_quality.txt"
        OUT_DIR.mkdir(exist_ok=True)
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        print("\n".join(lines))
        print(f"\nwrote ranked list -> {out_path}")
        return

    character = args.character or "Jesus"
    if character not in character_embeddings:
        raise ValueError(f"unknown character {character!r}. Run with --list-characters to see options.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading model from {MODEL_DIR} (device={device})...")
    processor = SpeechT5Processor.from_pretrained(str(MODEL_DIR))
    model = SpeechT5ForTextToSpeech.from_pretrained(str(MODEL_DIR)).to(device)
    model.eval()
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

    OUT_DIR.mkdir(exist_ok=True)

    if args.compare:
        val = dataset["test"].filter(lambda ex: ex["character"] == character)
        if len(val) == 0:
            raise ValueError(f"no validation examples for character {character!r}")
        rng = np.random.default_rng(args.seed)
        idxs = rng.choice(len(val), size=min(args.compare, len(val)), replace=False)
        for i, idx in enumerate(idxs):
            ex = val[int(idx)]
            text = ex["text"]
            speech = synthesize(text, character_embeddings[character], model, processor, vocoder, device)

            synth_path = OUT_DIR / f"compare_{i}_synth.wav"
            ref_path = OUT_DIR / f"compare_{i}_reference.wav"
            txt_path = OUT_DIR / f"compare_{i}_text.txt"

            sf.write(str(synth_path), speech, 16000)
            shutil.copy(WAV_DIR / ex["audio_file"], ref_path)
            txt_path.write_text(text, encoding="utf-8")

            print(f"[{i}] text: {text}")
            print(f"    synth:     {synth_path}")
            print(f"    reference: {ref_path}")
        return

    texts = [args.text] if args.text else DEFAULT_SENTENCES
    for i, text in enumerate(texts):
        speech = synthesize(text, character_embeddings[character], model, processor, vocoder, device)
        out_path = OUT_DIR / f"sample_{i}.wav"
        sf.write(str(out_path), speech, 16000)
        print(f"[{i}] text: {text}")
        print(f"    wrote: {out_path}")


if __name__ == "__main__":
    main()
