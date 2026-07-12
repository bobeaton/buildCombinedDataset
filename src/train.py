"""Fine-tune SpeechT5 TTS on the prepared Kangri (Devanagari) dataset.

Run src/build_tokenizer.py and src/prepare_dataset.py first.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from datasets import load_from_disk
from transformers import (
    SpeechT5FeatureExtractor,
    SpeechT5ForTextToSpeech,
    SpeechT5Processor,
    SpeechT5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

ROOT = Path(__file__).parent.parent
MODEL_INIT_DIR = ROOT / "model_init"
TOKENIZER_DIR = ROOT / "tokenizer" / "speecht5_tokenizer"
DATASET_DIR = ROOT / "data" / "prepared_dataset"
OUTPUT_DIR = ROOT / "checkpoints" / "speecht5_kangri"


@dataclass
class TTSDataCollatorWithPadding:
    processor: Any
    reduction_factor: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_ids = [{"input_ids": f["input_ids"]} for f in features]
        label_features = [{"input_values": f["labels"]} for f in features]
        speaker_features = [f["speaker_embeddings"] for f in features]

        batch = self.processor.pad(input_ids=input_ids, labels=label_features, return_tensors="pt")

        # ignore padded portions of the target spectrogram in the loss
        batch["labels"] = batch["labels"].masked_fill(batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100)
        del batch["decoder_attention_mask"]

        # SpeechT5's decoder predicts `reduction_factor` mel frames per step, so target
        # lengths must be a multiple of it
        if self.reduction_factor > 1:
            target_lengths = torch.tensor([len(f["input_values"]) for f in label_features])
            target_lengths = target_lengths.new(
                [length - length % self.reduction_factor for length in target_lengths]
            )
            max_length = max(target_lengths)
            batch["labels"] = batch["labels"][:, :max_length]

        batch["speaker_embeddings"] = torch.tensor(speaker_features)
        return batch


def build_trainer(max_steps: int, output_dir: Path):
    tokenizer = SpeechT5Tokenizer.from_pretrained(str(TOKENIZER_DIR))
    feature_extractor = SpeechT5FeatureExtractor.from_pretrained("microsoft/speecht5_tts")
    processor = SpeechT5Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    model = SpeechT5ForTextToSpeech.from_pretrained(str(MODEL_INIT_DIR))
    model.config.use_cache = False  # incompatible with gradient checkpointing

    # SpeechT5's decoder LayerDrop (config default 0.1) independently skips each of the
    # 6 decoder layers per forward pass. If ALL 6 happen to be skipped in the same step
    # (~1e-6 chance per step, but non-negligible over a 6000-step run -- it happened at
    # step 1244 on the first run of this project), cross_attentions comes back empty and
    # the guided-attention loss crashes on torch.cat([]) in modeling_speecht5.py. Setting
    # config.*_layerdrop doesn't help post-construction -- each module cached its own
    # `self.layerdrop` at __init__ time -- so patch the module attributes directly.
    for module in model.modules():
        if hasattr(module, "layerdrop"):
            module.layerdrop = 0.0  # type: ignore[assignment]

    dataset = load_from_disk(str(DATASET_DIR))

    data_collator = TTSDataCollatorWithPadding(
        processor=processor, reduction_factor=model.config.reduction_factor
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,  # effective batch size 32
        learning_rate=1e-4,  # higher than a same-script fine-tune since the text
                              # embedding table is freshly initialized (vocab swap)
        warmup_steps=500,
        max_steps=max_steps,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        save_steps=500,
        eval_steps=500,
        logging_steps=25,
        save_total_limit=3,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        greater_is_better=False,
        label_names=["labels"],
        remove_unused_columns=False,
        dataloader_num_workers=0,  # Windows: avoid multiprocessing worker-spawn issues
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        processing_class=processor,
    )
    return trainer, processor


def main():
    import sys

    max_steps = 6000
    resume_from = None
    args = sys.argv[1:]
    while args:
        if args[0] == "--max-steps":
            max_steps = int(args[1])
            args = args[2:]
        elif args[0] == "--resume":
            # pass a checkpoint dir, or "auto" to resume from the latest checkpoint
            # under OUTPUT_DIR
            resume_from = args[1]
            args = args[2:]
        else:
            raise ValueError(f"unrecognized argument: {args[0]}")

    trainer, processor = build_trainer(max_steps=max_steps, output_dir=OUTPUT_DIR)
    trainer.train(resume_from_checkpoint=(True if resume_from == "auto" else resume_from))

    final_dir = OUTPUT_DIR / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"saved final model -> {final_dir}")


if __name__ == "__main__":
    main()
