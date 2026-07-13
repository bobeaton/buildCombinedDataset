"""Build a character-level SentencePiece tokenizer for the Kangri (Devanagari) corpus
and swap it into a fresh SpeechT5ForTextToSpeech, resizing the text embedding to match.

The stock microsoft/speecht5_tts tokenizer has a 79-token English/Latin vocab and
cannot represent Devanagari at all (every character comes out <unk>). Character-level
SentencePiece guarantees full coverage of whatever appears in the corpus, mirroring
the granularity of the original English tokenizer (which is itself near-character-level).
"""

import argparse
import sys
import unicodedata
from pathlib import Path

import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).parent))
from load_metadata import parse_metadata

METADATA_CSV = r"C:\Users\pete_\Dropbox\NTprogress\PahariAudio\KangriWordDownloads\FCBH\combined_metadata.csv"

# Must match the special-token ID layout of the original SpeechT5 tokenizer
# (bos=0, pad=1, eos=2, unk=3) so downstream code that assumes those IDs still works.
BOS_ID, PAD_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["a", "b"], default="a", help="which transcription column to build the vocab from")
    args = parser.parse_args()

    suffix = "" if args.variant == "a" else "_b"
    out_dir = Path(__file__).parent.parent / f"tokenizer{suffix}"
    corpus_txt = out_dir / "corpus.txt"
    model_prefix = str(out_dir / "kangri_char")

    out_dir.mkdir(exist_ok=True)

    rows = parse_metadata(METADATA_CSV)
    texts = [
        unicodedata.normalize("NFC", r.transcription_a if args.variant == "a" else r.transcription_b)
        for r in rows
    ]

    distinct_chars = set()
    for t in texts:
        distinct_chars.update(t)
    print(f"{len(rows)} utterances, {len(distinct_chars)} distinct characters")

    corpus_txt.write_text("\n".join(texts), encoding="utf-8")

    vocab_size = len(distinct_chars) + 4  # + bos/pad/eos/unk
    spm.SentencePieceTrainer.train(
        input=str(corpus_txt),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="char",
        character_coverage=1.0,
        bos_id=BOS_ID,
        pad_id=PAD_ID,
        eos_id=EOS_ID,
        unk_id=UNK_ID,
        bos_piece="<s>",
        pad_piece="<pad>",
        eos_piece="</s>",
        unk_piece="<unk>",
    )
    print(f"trained sentencepiece model -> {model_prefix}.model (vocab_size={vocab_size})")

    from transformers import SpeechT5Tokenizer, SpeechT5ForTextToSpeech

    tokenizer = SpeechT5Tokenizer(vocab_file=f"{model_prefix}.model")
    tokenizer.add_special_tokens({"mask_token": "<mask>"})

    # sanity check: no <unk> on real corpus text, and round-trip is lossless
    sample = texts[0]
    ids = tokenizer(sample).input_ids
    unk_count = sum(1 for i in ids if i == tokenizer.unk_token_id)
    decoded = tokenizer.decode(ids, skip_special_tokens=True).replace(" ", "")
    original_nospace = sample.replace(" ", "")
    print(f"sample: {sample!r}")
    print(f"unk count in sample: {unk_count}")
    print(f"round-trip match (ignoring spaces): {decoded == original_nospace}")
    assert unk_count == 0, "tokenizer produced <unk> on training corpus text"

    full_unk_check = sum(
        1 for t in texts if tokenizer.unk_token_id in tokenizer(t).input_ids
    )
    print(f"utterances containing <unk> across full corpus: {full_unk_check} / {len(texts)}")

    tokenizer_dir = out_dir / "speecht5_tokenizer"
    tokenizer.save_pretrained(str(tokenizer_dir))
    print(f"saved tokenizer -> {tokenizer_dir}")

    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    old_size = model.get_input_embeddings().num_embeddings
    model.resize_token_embeddings(len(tokenizer))
    model.config.vocab_size = len(tokenizer)
    print(f"resized text embedding: {old_size} -> {len(tokenizer)}")

    model_dir = out_dir.parent / f"model_init{suffix}"
    model.save_pretrained(str(model_dir))
    print(f"saved resized model init checkpoint -> {model_dir}")


if __name__ == "__main__":
    main()
