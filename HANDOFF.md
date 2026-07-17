# Handoff: Kangri (XNR) Devanagari TTS Project

This document exists so a new session/model can pick up this project without re-running
the investigation that produced it. Read this before touching code. It covers: what's
built and working, what the core unsolved problem is (with the actual evidence, not just
a summary), what was tried and ruled out, and a recommended next direction.

**Context**: this is for SIL/Bible-translation audio production in Kangri (ISO 639-3
`xnr`, Devanagari script), a low-resource language with no pretrained TTS or phonemizer.
The end consumer is a C#/.NET client ("SpeecheloHelper") that calls a Dockerized HTTP
webservice to turn marked-up SFM-derived text files into narrated audio, book by book.

## 1. What's built and working

**Fine-tuned model**: `microsoft/speecht5_tts` (English base), with a from-scratch
character-level SentencePiece tokenizer for Devanagari (the stock English tokenizer
can't represent Devanagari at all) and a resized text-embedding table. Fine-tuned via
`Seq2SeqTrainer` on ~16k utterances (~40 hours of audio), speaker-conditioned via
per-utterance x-vectors (speechbrain `spkrec-xvect-voxceleb`), averaged per named
"character" for voice selection at inference time.

**Environment**: Windows 11, Python 3.11 (chosen for wheel compatibility with
numba/librosa/speechbrain on Windows -- newer Python versions had gaps here), NVIDIA
RTX 4060 Laptop (8GB VRAM), `torch==2.13.0+cu130`, `transformers==5.13.1`,
`speechbrain==1.0.3` (1.1.0 crashes on Windows -- lazy `k2` import with no Windows
wheel), `datasets==5.0.0`.

**Project layout** (`C:\vscode\buildCombinedDataset\`):
- `src/load_metadata.py` -- parses `combined_metadata.csv` (now fully pipe-delimited,
  originally a mix of comma/pipe rows with inconsistent quoting).
- `src/build_tokenizer.py` -- builds the char-level SentencePiece tokenizer + resizes
  the base model's embeddings; outputs to `tokenizer/` (or `tokenizer_b/`) and
  `model_init/` (or `model_init_b/`).
- `src/prepare_dataset.py` -- audio decode/resample to 16kHz mono, per-utterance x-vector
  extraction, filters (`MAX_DURATION_S=25.0`, `MAX_TEXT_CHARS=400`), train/test split
  (seed=42). Outputs to `data/prepared_dataset` (or `_b`).
- `src/train.py` -- the actual fine-tuning script.
- `src/infer.py` -- single-text/all-characters/compare-vs-reference synthesis, plus the
  silence-capping and alignment-failure-detection logic (see below).
- `src/infer_file.py` -- synthesizes a whole `spkrEmb:`-marked-up file (character
  switches mid-file) into one concatenated wav; includes a pre-flight `validate_file()`.
- `docker/` -- a Flask + gevent webservice (`server.py`) wrapping the above, with an
  OpenAPI spec (`openapi.yaml`), Swagger UI at `/docs`, and `buildDocker.ps1` to build/run
  the container. This is the stable API contract the C# client talks to -- **whatever
  replaces the TTS model, keep this HTTP contract the same or a superset** so the client
  needs no changes.

**Two transcription variants exist** (`--variant a`/`b` throughout): the source metadata
has two transcription columns (`transcriptionA`/`transcriptionB`) with genuinely
different text and different character vocabularies (88 vs 85 distinct characters,
not a subset relationship). Variant `b` is the one actually in use (`DEFAULT_VARIANT=b`
everywhere) -- historically preferred by the user. Both a full parallel pipeline exists
for `a` if ever needed, but there's no reason to touch it.

**External file locations** (outside this repo):
- `C:\Users\pete_\Dropbox\NTprogress\PahariAudio\KangriWordDownloads\FCBH\combined_metadata.csv`
  -- the metadata (audio_file, character, speakerName, transcriptionA, transcriptionB).
  Actively curated by the user (see section 4).
- `...\FCBH\wavs\` -- the raw audio recordings.
- `...\FCBH\characterMapping.json` -- maps SFM milestone-marker character names (e.g.
  `"narrator-GEN"`) to a trained TTS character name (e.g. `"Jesus"`). Actively curated
  chapter-by-chapter by the user; most entries are still self-mapped placeholders
  awaiting real assignment. 
- `C:\Users\pete_\Dropbox\NTprogress\PahariAudio\KangriWordDownloads\` -- where
  `infer_file.py` writes its output wav (CLI only; the webservice returns bytes directly
  instead of writing to disk).

## 2. The core unsolved problem

**Symptom**: for certain (text, voice) combinations, the model generates an anomalously
long stretch of genuine digital silence (not just quiet -- confirmed via RMS energy
envelope, near-zero for 1-7+ seconds) and then jumps into normal speech partway through
the sentence, having apparently never voiced the skipped words at all.

**Root cause**: this is a known, named limitation of the whole class of architecture
SpeechT5's TTS decoder belongs to (Tacotron-style autoregressive TTS with *soft, learned*
encoder-decoder cross-attention). Nothing in the architecture forces that attention to
move monotonically or cover every input token -- it's free to stall or skip. SpeechT5
does use guided-attention loss during training (a soft regularizer nudging attention
toward monotonic alignment) but it's not a hard constraint, and it clearly doesn't fully
prevent the failure. **This is very likely not fixable by data curation or code changes
alone within this architecture** -- see the evidence below.

### What was tried, with actual measurements

1. **Retrying generation** (exploiting GPU floating-point non-determinism, which does
   cause real run-to-run variance): tested 10 independent retries on one known-bad
   (text, voice) pair -- leading silence ranged 1.15s-6.24s, **never once clean**
   (clean/normal is ~0.3-0.5s). Retrying is not a reliable fix for the worst cases,
   though it might help borderline cases with a non-trivial success rate.

2. **Splitting the sentence at a comma** (hypothesis: sentence length was the trigger):
   disproved. Even the *minimal, fully isolated opening clause alone* (~80 characters)
   failed 8/8 independent trials with one voice. Length is not the primary driver.

3. **Separating a blended speaker-embedding average into narrower, more stylistically
   consistent groups**: this **did help, but only partially and inconsistently**.
   Original problem: the "Jesus" character's x-vector average blended ~9600 utterances
   spanning both direct dialogue (NT Gospels) and third-person OT/narrative content
   (Genesis narration, same voice actor "Pawan Kumar", historically mislabeled as
   "Jesus"). Splitting this into a narrower `Jesus` (dialogue-only, ~1400 utterances) and
   a separate `narrator-Pawan` (narrative-only, ~8600 utterances) made the narrower
   `Jesus` **completely reliable** for the previously-failing sentence (0/5 failures
   across 5 trials, was ~100% failure before). But `narrator-Pawan` -- the voice actually
   needed for OT narration -- **remained unreliable** (5/5 failures, similar severity to
   before).

4. **Curating out acoustically-inconsistent recording sessions from the narrator-Pawan
   average**: also tested and **did not help**. Identified a cluster of books (1TH, 1TI,
   2TH, EPH, TIT, 2TI) with a consistently different (2-20x higher) high-frequency
   spectral ratio than the "core" batch (GEN, 1CO, JHN, MAT), strongly suggesting a
   different recording session -- but excluding them from the average barely moved it
   (cosine similarity between full and curated averages: **0.9997**, i.e. essentially
   identical) and empirical results were statistically indistinguishable before/after.
   The x-vector space is dominated by underlying voice identity; recording-session-level
   acoustic variation doesn't move it enough to matter.

5. **Breadth check**: for one specific difficult sentence, ran it through *every* trained
   character (47 voices) in one batch. **30 of 47 (64%) were flagged** by the detector
   below -- including well-trained voices like Luke (187 min), John (153 min), Matthew
   (67 min), Mark (62 min). This is not narrow to one or two under-trained voices; for
   at least some sentences, this is a broad failure mode across the model.

### Conclusion

Given (3) and (4) both showed genuine but *bounded* improvement -- narrower, more
consistent speaker-embedding averages measurably help, but don't come close to
eliminating the problem, and a large fraction of even well-trained voices can still be
affected for specific difficult sentences -- **this looks like an architecture
limitation, not a data-quality or data-curation problem.** More/cleaner data would very
likely help somewhat further but is unlikely to fully close the gap.

## 3. Mitigations already built (working, in production)

Since the failure isn't reliably preventable, the practical approach taken was
detect-and-warn rather than block-and-fail, to fit the user's actual workflow (an
automated per-chapter pipeline that must run unattended start to finish):

- **`cap_silence()`** (`src/infer.py`): trims *excess* silence (leading, trailing, and
  internal) down to a target duration after generation -- cosmetic cleanup, not a fix for
  the underlying issue, but prevents unnaturally long pauses in otherwise-fine clips
  (confirmed this is a *different*, more common issue than the alignment failure: normal
  well-trained voices have ~0.3-0.5s natural boundary silence; under-trained voices can
  have much more, ~0.4-0.7s+ between clauses even in a *clean* generation).
- **`detect_alignment_issue()`** (`src/infer.py`): flags (doesn't block) any clip where
  a silence gap exceeds 1.0s, based on the empirical observation that this threshold
  reliably separates normal generations from probable skipped-word failures. Must run on
  the *raw* pre-cap waveform (capping would hide exactly the gap it's trying to measure).
  `synthesize()` returns `(audio, warning_or_None)`.
- **`validate_file()`** (`src/infer_file.py`): a *separate*, unrelated pre-flight check
  -- verifies every `spkrEmb:` name in a file resolves to a trained character before any
  synthesis starts, collecting *all* problems in one pass rather than failing at the
  first. Exposed via `infer_file.py --check-only` (skips model loading entirely, fast)
  and the webservice's `dryRun` request field.
- **Warnings surfaced everywhere**: `infer.py` CLI (console + `manifest.json` for
  `--character all` batches), `infer_file.py` (per-line console warnings + a returned
  list + CLI summary), and the webservice (`warnings` array in job status JSON,
  `X-Synthesis-Warning` response header for single-text synthesis, warning fields in
  zip manifests for compare/all-characters). The job/request always completes; nothing
  blocks on a warning.

**This detect-and-warn system should be considered permanent infrastructure regardless
of what TTS architecture is used going forward** -- it's a reasonable safety net even if
the underlying model becomes more reliable, and the webservice contract already expects
it (`warnings` field is part of the documented API).

## 4. Metadata curation done during this project

`combined_metadata.csv` has been actively cleaned up; a new session should NOT be
surprised by these changes or assume the original raw export structure:
- Delimiter consistency: originally a mix of comma-CSV-quoted rows and pipe-delimited
  rows; now fully pipe-delimited (backup of the pre-change file exists alongside it as
  `combined_metadata (0).csv.bak`).
- The single blended "Jesus"/narrator voice actor (Pawan Kumar) has been split by
  content type into: `Jesus` (direct dialogue), and narrator content further split into
  `epistles-Pawan`, `hist-narr-Pawan`, `stories-Pawan` (genre-based, done by the user
  after the initial Jesus/narrator-Pawan split).
- `male_groupN`/`female_groupN` (originally arbitrary groupings mixing multiple
  unrelated speakers per group, not by actual voice identity) were regrouped so each
  group represents one actual speaker (by `speakerName`), with low-count speakers (< 20
  utterances) merged into one shared group per gender. Current state: `male_group1`
  =Pankaj Kumar, `male_group2`=Vinay Kumar, `male_group3`=Rajesh Kumar, `male_group4`
  =Vijay, `male_group5`=(Sammi Kumar, Pawan Kumar [1 stray/mislabeled row], Mehek),
  `female_group1`=(Jasmin, Chandresh Kumari, Alyssa).
- `characterMapping.json`'s structure changed to `{"AlphabeticOrder": [...],
  "InFirstOccurrenceOrder": {...}}` (the user uses this structure as a working checklist
  while curating chapter by chapter) -- `load_character_mapping()` in `infer_file.py`
  already handles this, extracting the flat list from `AlphabeticOrder`.
- **Important**: whenever `combined_metadata.csv` changes, `prepare_dataset.py` needs to
  be rerun (~30 min GPU pass) to regenerate `data/prepared_dataset_b` -- but **the model
  does NOT need retraining**. Confirmed: `character` is purely a post-hoc inference-time
  grouping label (used only by `build_character_embeddings()`); the actual fine-tuning
  in `train.py` only ever consumes per-utterance x-vectors, never the character label.

## 5. Recommended next direction

Move to a TTS architecture that uses an **explicit duration predictor** (Monotonic
Alignment Search during training) instead of soft learned attention -- this makes
word-skipping structurally impossible rather than just less likely, which directly
targets the root cause in section 2 rather than working around it.

**Recommendation: YourTTS**, via the community-maintained Coqui TTS fork
(`idiap/coqui-ai-TTS` on GitHub, `coqui-tts` on PyPI -- the original Coqui AI company
shut down Jan 2024, this fork keeps it current). YourTTS = VITS + external
speaker-embedding conditioning ("d-vectors" -- conceptually identical to the x-vectors
already computed via speechbrain in this project) + multilingual support, purpose-built
for exactly this scenario: many voices, each represented by a continuous embedding
rather than a fixed discrete speaker-ID lookup table. This means the existing
`prepare_dataset.py` audio-decode + x-vector-extraction step is likely reusable/
adaptable largely as-is; the tokenizer and training loop are what differ substantially.

Confirmed compatible: coqui-tts requires Python >=3.10,<3.15 and PyTorch >=2.2 -- this
project's existing Python 3.11 / torch 2.13 both satisfy that, so no forced Python
version change, though a fresh venv is still advisable to avoid dependency conflicts.

**Do not reuse or resume `C:\vscode\vits_mms\finetune-hf-vits\`** -- this is an earlier,
unrelated attempt (by the user, using MMS-VITS via a different toolkit) that was never
actually fine-tuned (no training logs/checkpoints exist; the `mms-tts-xnr-train` folder
is just a locally-cached pretrained `facebook/mms-tts-xnr` checkpoint, file-dated April
2024, prepared for fine-tuning but abandoned before any training happened). It was based
on a much smaller single-speaker dataset from colleagues and an older Python/torch
environment. The user has explicitly asked for a fresh approach based on the current,
much larger, actively-curated dataset in this project, not a resumption of that one.

**Suggested approach**: build the new pipeline in a separate sibling project directory
(not nested inside `buildCombinedDataset`), with its own venv, so the two efforts can't
get tangled and the working SpeechT5 setup is never put at risk. Reuse (copy, don't
share) `load_metadata.py`'s parsing logic and the general audio-processing pattern from
`prepare_dataset.py` as a starting point. Keep the Docker webservice's HTTP contract
(`docker/openapi.yaml` is the source of truth) stable across the swap.
