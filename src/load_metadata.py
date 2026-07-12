"""Parse combined_metadata.csv, which mixes two row formats:

  - comma-delimited, proper CSV quoting, audio filenames prefixed "FCBH_"
    header: audio_file,character,speakerName,transcriptionA,transcriptionB
  - pipe-delimited ("|"), no CSV quoting (occasionally wrapped in a stray
    pair of double quotes), audio filenames with no prefix
    audio_file|character|speakerName|transcriptionA|transcriptionB

Pipe-formatted rows can contain literal commas inside the Devanagari text,
so they cannot be parsed with a comma-based CSV reader. Rows are told apart
by the presence of "|" anywhere in the raw line.
"""

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Utterance:
    audio_file: str
    character: str
    speaker: str
    transcription_a: str
    transcription_b: str
    source_format: str  # "comma" or "pipe"


def _parse_pipe_row(line: str) -> Utterance:
    line = line.rstrip("\n").rstrip("\r")
    if line.endswith("|"):
        line = line[:-1]
    fields = line.split("|")
    if len(fields) != 5:
        raise ValueError(f"expected 5 pipe-delimited fields, got {len(fields)}: {line!r}")
    audio_file, character, speaker, trans_a, trans_b = (f.strip() for f in fields)
    trans_a = trans_a.strip('"')
    trans_b = trans_b.strip('"')
    return Utterance(audio_file, character, speaker, trans_a, trans_b, "pipe")


def parse_metadata(csv_path: str | Path) -> list[Utterance]:
    csv_path = Path(csv_path)
    utterances: list[Utterance] = []
    errors: list[tuple[int, str]] = []

    with open(csv_path, encoding="utf-8") as f:
        header = f.readline()
        assert header.strip().split(",")[:3] == ["audio_file", "character", "speakerName"], (
            f"unexpected header: {header!r}"
        )
        for lineno, line in enumerate(f, start=2):
            if not line.strip():
                continue
            if "|" in line:
                try:
                    utterances.append(_parse_pipe_row(line))
                except ValueError as e:
                    errors.append((lineno, str(e)))
                continue
            row = next(csv.reader([line]))
            if len(row) != 5:
                errors.append((lineno, f"expected 5 comma-delimited fields, got {len(row)}: {line!r}"))
                continue
            audio_file, character, speaker, trans_a, trans_b = row
            utterances.append(Utterance(audio_file, character, speaker, trans_a, trans_b, "comma"))

    if errors:
        raise ValueError(f"{len(errors)} unparseable rows, first few:\n" + "\n".join(
            f"  line {ln}: {msg}" for ln, msg in errors[:10]
        ))

    return utterances


if __name__ == "__main__":
    import sys
    from collections import Counter

    path = sys.argv[1] if len(sys.argv) > 1 else (
        r"C:\Users\pete_\Dropbox\NTprogress\PahariAudio\KangriWordDownloads\FCBH\combined_metadata.csv"
    )
    rows = parse_metadata(path)
    print(f"parsed {len(rows)} rows")
    print("by source format:", Counter(r.source_format for r in rows))
    speakers = Counter(r.speaker for r in rows)
    print(f"distinct speakers: {len(speakers)}")
    print("top 15 speakers:", speakers.most_common(15))
    print("distinct characters:", len(Counter(r.character for r in rows)))
    a_ne_b = sum(1 for r in rows if r.transcription_a != r.transcription_b)
    print(f"rows where transcriptionA != transcriptionB: {a_ne_b}")
