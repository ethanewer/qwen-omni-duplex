import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from transformers.activations import silu
from transformers.configuration_utils import PretrainedConfig
from transformers.hf_argparser import HfArgumentParser
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments


class AdaptorConfig(PretrainedConfig):
    input_size: int
    output_size: int
    intermediate_size: int = 8192
    output_timesteps: int = 1


@dataclass
class AdaptorRunArguments:
    data_path: str = field(metadata={"help": "Path containing .pt shards."})
    input_size: int = field(default=512, metadata={"help": "Adaptor input feature size."})
    output_size: int = field(default=2048, metadata={"help": "Adaptor output feature size."})
    intermediate_size: int = field(default=8192, metadata={"help": "Hidden size for MLP."})
    output_timesteps: int = field(default=2, metadata={"help": "Timesteps produced per input step."})
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
        pred = down_proj.view(*inputs.shape[:-2], -1, self.output_size)
        outputs = {"logits": pred}
        if targets:
            squared_error = (pred - targets).square()
            if mask:
                outputs["loss"] = (squared_error * mask).sum() / mask.sum().clamp_min(1.0)
            else:
                outputs["loss"] = squared_error.mean()

        return outputs


class FeatureShardIterableDataset(IterableDataset):
    def __init__(self, shard_paths: list[Path]):
        self.shard_paths = sorted([p for p in shard_paths if p.suffix == ".pt"])
        if not self.shard_paths:
            raise FileNotFoundError("No .pt shards found.")

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for shard_path in self.shard_paths:
            print("Loading shard.")
            data = torch.load(shard_path, map_location="cpu")
            print("Completed loading shard.")
            items = data.get("items", [])
            for item in items:
                yield item

            del data
            gc.collect()


def collate_fn_alignment(batch: list[dict[str, Any]]) -> dict[str, Any]:
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for b in batch:
        x: torch.Tensor = b["mimi_features"]
        y: torch.Tensor = b["qwen_omni_features"]
        min_size = min(x.shape[0], y.shape[0] // 2)
        if min_size > 0:
            xs.append(x[:min_size])
            ys.append(y[: 2 * min_size])

    x_batch = pad_sequence(xs, batch_first=True)
    y_batch = pad_sequence(ys, batch_first=True)
    mask = pad_sequence([torch.ones_like(y[..., :1]) for y in ys], batch_first=True)
    return {"x": x_batch, "y": y_batch, "mask": mask}


def compute_metrics(eval_pred):
    print(eval_pred)
    return {"r2": 0}


def parse_args() -> tuple[AdaptorRunArguments, TrainingArguments]:
    parser = HfArgumentParser((AdaptorRunArguments, TrainingArguments))  # type: ignore
    run_args, training_args = parser.parse_args_into_dataclasses()  # type: ignore
    training_args.remove_unused_columns = False
    return run_args, training_args


def main() -> None:
    run_args, training_args = parse_args()

    torch.manual_seed(training_args.seed)

    data_root = Path(run_args.data_path)
    shard_paths = [p for p in data_root.iterdir() if p.is_file() and p.suffix == ".pt"]
    assert len(shard_paths) > 1
    train_dataset = FeatureShardIterableDataset(shard_paths[:-1])
    eval_dataset = FeatureShardIterableDataset(shard_paths[-1:])

    adaptor_config = run_args.adaptor_config
    model: nn.Module = Adaptor(adaptor_config)
    if run_args.compile and hasattr(torch, "compile"):
        model.compile()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn_alignment,
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
