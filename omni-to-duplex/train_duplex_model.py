import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
from datasets import Dataset
from swift.llm import (
    Model,
    ModelGroup,
    ModelInfo,
    ModelMeta,
    TemplateMeta,
    get_model_tokenizer,
    get_template,
    register_model,
    register_template,
)
from swift.trainers import Trainer, TrainingArguments
from transformers import AutoTokenizer
from transformers.hf_argparser import HfArgumentParser

from model import QwenWithCausalAudioEncoderAndParallelInputStreamsForCausalLM, QwenWithCausalAudioEncoderConfig

register_template(
    TemplateMeta(
        template_type="custom",
        prefix=["<extra_id_0>System\n{{SYSTEM}}\n"],
        prompt=["<extra_id_1>User\n{{QUERY}}\n<extra_id_1>Assistant\n"],
        chat_sep=["\n"],
    )
)


def get_function(
    model_dir: str,
    model_info: ModelInfo,
    model_kwargs: dict[str, Any],
    load_model: bool = True,
    **kwargs,
) -> tuple[Optional[QwenWithCausalAudioEncoderAndParallelInputStreamsForCausalLM], Any]:
    model_config = QwenWithCausalAudioEncoderConfig.from_pretrained(model_dir, trust_remote_code=True, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = None
    if load_model:
        model = QwenWithCausalAudioEncoderAndParallelInputStreamsForCausalLM.from_pretrained(
            model_dir,
            config=model_config,
            torch_dtype=model_info.torch_dtype,
            trust_remote_code=True,
            **model_kwargs,
        )

    return model, tokenizer


register_model(
    ModelMeta(
        model_type="custom",
        model_groups=[ModelGroup([Model("local/Qwen2.5WithMimi-3B")])],
        template="custom",
        get_function=get_function,
        ignore_patterns=["nemo"],
        is_multimodal=False,
    )
)


@dataclass
class RunArguments: ...


def parse_args() -> tuple[RunArguments, TrainingArguments]:
    parser = HfArgumentParser((RunArguments, TrainingArguments))  # type: ignore
    run_args, training_args = parser.parse_args_into_dataclasses()  # type: ignore
    training_args.remove_unused_columns = False
    training_args.label_names = ["attention_mask", "labels", "audio_codes", "audio_codes_mask", "audio_codes_labels"]  # type: ignore
    return run_args, training_args


def main() -> None:
    run_args, training_args = parse_args()

    torch.manual_seed(training_args.seed)

    model, tokenizer = get_model_tokenizer("local/Qwen2.5WithMimi-3B")
    assert model is not None and tokenizer is not None

    dataset_size = 1024
    seq_len = 512
    audio_seq_len = math.ceil(seq_len / model.config.adaptor_config.output_time_scale)
    num_quantizers = 8
    train_dataset = Dataset.from_dict(
        {
            "input_ids": torch.randint(0, 1024, size=(dataset_size, seq_len)).tolist(),
            "attention_mask": torch.ones(dataset_size, seq_len, dtype=torch.long).tolist(),
            "labels": torch.randint(0, 1024, size=(dataset_size, seq_len)).tolist(),
            "audio_codes": torch.randint(0, 1024, size=(dataset_size, audio_seq_len, num_quantizers)).tolist(),
            "audio_codes_mask": torch.ones(dataset_size, audio_seq_len, dtype=torch.long).tolist(),
            "audio_codes_labels": torch.randint(0, 1024, size=(dataset_size, audio_seq_len)).tolist(),
        }
    )

    template = get_template("custom", tokenizer, max_length=seq_len)
    template.set_mode("train")

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, template=template)
    trainer.train()

    print("Training complete.")


if __name__ == "__main__":
    main()
