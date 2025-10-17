import gc
import os
import shutil
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
from torch.utils.data import IterableDataset
from transformers.hf_argparser import HfArgumentParser
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from model import EmbeddingAdaptor, EmbeddingAdaptorConfig


def debug_xml(obj, name: str) -> None:
    obj_str = "\n".join(str(obj)[:1024].split("\n")[:16])
    print(f"<{name}>\n{obj_str}\n</{name}>")


@dataclass
class RunArguments:
    data_path: str = field(metadata={"help": "Path containing .pt shards."})
    adaptor_config_path: str = field(
        default="configs/voila-to-qwen3-30b-a3b-adaptor-config.json",
        metadata={"help": "Path to config JSON file."},
    )
    attn_implementation: Optional[str] = field(
        default="flash_attention_2",
        metadata={"help": "Must be 'flash_attention_2' for sliding window attention."},
    )
    max_input_seq_len: int = field(default=256, metadata={"help": "Max input sequence length."})
    max_eval_dataset_size: Optional[int] = field(default=None, metadata={"help": "Max eval dataset size."})
    compile: bool = field(default=False, metadata={"help": "Use torch.compile on the base adaptor."})
    final_filename: str = field(default="adaptor.pt", metadata={"help": "Filename of final saved adaptor weights."})
    checkpoint_path: Optional[str] = field(default=None, metadata={"help": "Optional path model state dict."})

    @property
    def adaptor_config(self) -> EmbeddingAdaptorConfig:
        return EmbeddingAdaptorConfig.from_json_file(self.adaptor_config_path)


class FeatureShardIterableDataset(IterableDataset):
    def __init__(self, shard_paths: list[Path], max_size: Optional[int] = None):
        self.shard_paths = sorted([p for p in shard_paths if p.suffix == ".pt"], key=lambda p: str(p)[::-1])
        self.max_size = max_size
        if not self.shard_paths:
            raise FileNotFoundError("No .pt shards found.")

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        count = 0
        for idx, shard_path in enumerate(self.shard_paths):
            print(f"Loading shard {idx + 1}/{len(self.shard_paths)}.")
            data = torch.load(shard_path, map_location="cpu")
            items = data.get("items", [])
            for item in items:
                yield item
                count += 1
                if self.max_size is not None and count >= self.max_size:
                    return

            del data
            gc.collect()


def collate_fn_alignment(batch: list[dict[str, Any]], max_input_seq_len: int, output_time_scale: float) -> dict[str, Any]:
    max_output_seq_len = int(max_input_seq_len * output_time_scale)

    inputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []

    for b in batch:
        x: torch.Tensor = b["mimi_features"]
        y: torch.Tensor = b["qwen_omni_features"]
        output_mask = torch.ones_like(y[:, :1])

        max_pairs = min(x.shape[0], int(y.shape[0] / output_time_scale), max_input_seq_len)
        if max_pairs == 0:
            continue

        if x.shape[0] > max_input_seq_len:
            x = x[:max_input_seq_len]
        elif x.shape[0] < max_input_seq_len:
            x = F.pad(x, (0, 0, 0, max_input_seq_len - x.shape[0]), value=0)

        if y.shape[0] > max_output_seq_len:
            y = y[:max_output_seq_len]
            output_mask = output_mask[:max_output_seq_len]
        elif y.shape[0] < max_output_seq_len:
            y = F.pad(y, (0, 0, 0, max_output_seq_len - y.shape[0]), value=0)
            output_mask = F.pad(output_mask, (0, 0, 0, max_output_seq_len - output_mask.shape[0]), value=0)

        assert x.shape[0] == max_input_seq_len
        assert y.shape[0] == max_output_seq_len
        assert output_mask.shape[0] == max_output_seq_len

        inputs.append(x)
        targets.append(y)
        masks.append(output_mask)

    return {
        "inputs": torch.stack(inputs, dim=0),
        "targets": torch.stack(targets, dim=0),
        "output_mask": torch.stack(masks, dim=0),
    }


def compute_metrics(eval_pred):
    preds = eval_pred.predictions
    targets, output_mask = eval_pred.label_ids
    assert isinstance(preds, np.ndarray) and isinstance(targets, np.ndarray) and isinstance(output_mask, np.ndarray)

    if preds.shape[1] != targets.shape[1]:
        targets = targets[:, : preds.shape[1]]
        output_mask = output_mask[:, : preds.shape[1]]

    assert preds.shape == targets.shape and preds.shape[:2] == output_mask.shape[:2]

    count = max(output_mask.sum(), 1)
    mse = (np.square(preds - targets) * output_mask).sum() / count
    mean = (targets * output_mask).sum(axis=(0, 1)) / count
    var = (np.square(targets - mean) * output_mask).sum() / count
    return {"r2": 1 - mse / var}


def parse_args() -> tuple[RunArguments, TrainingArguments]:
    parser = HfArgumentParser((RunArguments, TrainingArguments))  # type: ignore
    run_args, training_args = parser.parse_args_into_dataclasses()  # type: ignore
    training_args.remove_unused_columns = False
    training_args.label_names = ["targets", "output_mask"]  # type: ignore
    return run_args, training_args


def main() -> None:
    run_args, training_args = parse_args()

    torch.manual_seed(training_args.seed)

    data_root = Path(run_args.data_path)
    shard_paths = [p for p in data_root.iterdir() if p.is_file() and p.suffix == ".pt"]
    assert len(shard_paths) > 1
    train_dataset = FeatureShardIterableDataset(shard_paths[:-1])
    eval_dataset = FeatureShardIterableDataset(shard_paths[-1:], max_size=run_args.max_eval_dataset_size)

    adaptor_config = run_args.adaptor_config
    model: nn.Module = EmbeddingAdaptor(adaptor_config)

    if run_args.checkpoint_path is not None:
        state_dict = torch.load(run_args.checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)

    if run_args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=partial(
            collate_fn_alignment,
            max_input_seq_len=run_args.max_input_seq_len,
            output_time_scale=adaptor_config.output_time_scale,
        ),
    )

    trainer.train()

    save_path = Path(run_args.final_filename)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict, _ = get_state_dict(
        trainer.model,  # type: ignore
        trainer.optimizer,  # type: ignore
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )

    for _ in range(3):
        if os.path.exists(save_path):
            shutil.rmtree(save_path, ignore_errors=True)

        time.sleep(1)

        try:
            torch.save(state_dict, save_path)
            break
        except Exception as e:
            print(f"Save failed due to the following exception: {e}")
            time.sleep(1)

    print(f"Training complete. Final adaptor saved to `{str(save_path)}`.")


if __name__ == "__main__":
    main()
