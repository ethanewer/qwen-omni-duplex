import gc
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import IterableDataset
from transformers.activations import silu
from transformers.configuration_utils import PretrainedConfig
from transformers.hf_argparser import HfArgumentParser
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments


class AdaptorConfig(PretrainedConfig):
    input_size: int
    output_size: int
    intermediate_size: int
    output_timesteps: int


@dataclass
class AdaptorRunArguments:
    data_path: str = field(metadata={"help": "Path containing .pt shards."})
    input_size: int = field(default=512, metadata={"help": "Adaptor input feature size."})
    output_size: int = field(default=2048, metadata={"help": "Adaptor output feature size."})
    intermediate_size: int = field(default=8192, metadata={"help": "Hidden size for MLP."})
    output_timesteps: int = field(default=2, metadata={"help": "Timesteps produced per input step."})
    max_input_seq_len: int = field(default=512, metadata={"help": "Max input sequence length."})
    max_eval_dataset_size: Optional[int] = field(default=None, metadata={"help": "Max eval dataset size."})
    compile: bool = field(default=False, metadata={"help": "Use torch.compile on the base adaptor."})
    final_filename: str = field(default="adaptor.pt", metadata={"help": "Filename of final saved adaptor weights."})

    @property
    def adaptor_config(self) -> AdaptorConfig:
        return AdaptorConfig(
            input_size=self.input_size,
            output_size=self.output_size,
            intermediate_size=self.intermediate_size,
            output_timesteps=self.output_timesteps,
        )


class Adaptor(nn.Module):
    def __init__(self, config: AdaptorConfig):
        super().__init__()
        self.config = config
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.intermediate_size = config.intermediate_size
        self.output_timesteps = config.output_timesteps
        self.gate_proj = nn.Linear(self.input_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.input_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.output_size * self.output_timesteps, bias=False)
        self.act_fn = silu

    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        down_proj: torch.Tensor = self.down_proj(self.act_fn(self.gate_proj(inputs)) * self.up_proj(inputs))
        preds = down_proj.view(*inputs.shape[:-2], -1, self.output_size)
        outputs = {"logits": preds}
        if targets is not None:
            assert mask is not None
            outputs["loss"] = ((preds - targets).square() * mask).sum() / mask.sum().clamp_min(1.0)

        return outputs


class FeatureShardIterableDataset(IterableDataset):
    def __init__(self, shard_paths: list[Path], max_size: Optional[int] = None):
        self.shard_paths = sorted([p for p in shard_paths if p.suffix == ".pt"])
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


def collate_fn_alignment(batch: list[dict[str, Any]], max_input_seq_len: int) -> dict[str, Any]:
    inputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []

    for b in batch:
        x: torch.Tensor = b["mimi_features"]
        y: torch.Tensor = b["qwen_omni_features"]
        mask = torch.ones_like(y[:, :1])

        max_pairs = min(x.shape[0], y.shape[0] // 2, max_input_seq_len)
        if max_pairs == 0:
            continue
        elif x.shape[0] > max_input_seq_len:
            x = x[:max_input_seq_len]
            y = y[: 2 * max_input_seq_len]
            mask = mask[: 2 * max_input_seq_len]
        elif x.shape[0] < max_input_seq_len:
            x = F.pad(x, (0, 0, 0, max_input_seq_len - x.shape[0]), value=0)
            y = F.pad(y, (0, 0, 0, 2 * max_input_seq_len - y.shape[0]), value=0)
            mask = F.pad(mask, (0, 0, 0, 2 * max_input_seq_len - mask.shape[0]), value=0)

        assert x.shape[0] == max_input_seq_len
        assert y.shape[0] == 2 * max_input_seq_len
        assert mask.shape[0] == 2 * max_input_seq_len

        inputs.append(x)
        targets.append(y)
        masks.append(mask)

    return {
        "inputs": torch.stack(inputs, dim=0),
        "targets": torch.stack(targets, dim=0),
        "mask": torch.stack(masks, dim=0),
    }


def compute_metrics(eval_pred):
    preds = eval_pred.predictions

    if isinstance(eval_pred.label_ids, (tuple, list)):
        targets, mask = eval_pred.label_ids
    else:
        targets = eval_pred.label_ids
        mask = None

    if mask is not None:
        assert isinstance(mask, np.ndarray)
    else:
        mask = np.ones_like(targets[:, :, :1])

    assert isinstance(preds, np.ndarray) and isinstance(targets, np.ndarray) and isinstance(mask, np.ndarray)

    count = max(mask.sum(), 1)
    mse = (np.square(preds - targets) * mask).sum() / count
    mean = (targets * mask).sum(axis=(0, 1)) / count
    var = (np.square(targets - mean) * mask).sum() / count
    return {"r2": 1 - mse / var}


def parse_args() -> tuple[AdaptorRunArguments, TrainingArguments]:
    parser = HfArgumentParser((AdaptorRunArguments, TrainingArguments))  # type: ignore
    run_args, training_args = parser.parse_args_into_dataclasses()  # type: ignore
    training_args.remove_unused_columns = False
    training_args.label_names = ["targets", "mask"]  # type: ignore
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
    model: nn.Module = Adaptor(adaptor_config)
    if run_args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=partial(collate_fn_alignment, max_input_seq_len=run_args.max_input_seq_len),
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    final_dir = Path(run_args.final_filename)
    final_dir.mkdir(parents=True, exist_ok=True)
    state = {k.replace("adaptor.", ""): v for k, v in model.state_dict().items() if k.startswith("adaptor.")}
    torch.save(
        {"adaptor_state_dict": state, "config": adaptor_config.__dict__},
        final_dir / run_args.final_filename,
    )
    print(f"Training complete. Final adaptor saved to {final_dir / run_args.final_filename}.")


if __name__ == "__main__":
    main()
