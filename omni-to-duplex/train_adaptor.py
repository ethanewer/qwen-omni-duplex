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
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
from transformers.hf_argparser import HfArgumentParser
from transformers.models.qwen3 import Qwen3Config, Qwen3Model
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments


def debug_xml(obj, name: str) -> None:
    obj_str = "\n".join(str(obj)[:1024].split("\n")[:16])
    print(f"<{name}>\n{obj_str}\n</{name}>")


class AdaptorConfig(PretrainedConfig):
    input_size: int
    output_size: int
    output_time_scale: int
    lag_timesteps: int
    decoder_config: Qwen3Config


@dataclass
class AdaptorRunArguments:
    data_path: str = field(metadata={"help": "Path containing .pt shards."})
    input_size: int = field(default=512, metadata={"help": "Adaptor input feature size."})
    output_size: int = field(default=2048, metadata={"help": "Adaptor output feature size."})
    output_time_scale: int = field(default=2, metadata={"help": "Timesteps produced per input step."})
    lag_timesteps: int = field(default=0, metadata={"help": "Timestep lag between outputs and targets for loss calculation."})
    adaptor_decoder_config_path: str = field(
        default="configs/adaptor_decoder_config.json",
        metadata={"help": "Path to a Qwen3Model config."},
    )
    attn_implementation: Optional[str] = field(
        default="flash_attention_2",
        metadata={"help": "Must be 'flash_attention_2' for sliding window attention."},
    )
    max_input_seq_len: int = field(default=512, metadata={"help": "Max input sequence length."})
    max_eval_dataset_size: Optional[int] = field(default=None, metadata={"help": "Max eval dataset size."})
    compile: bool = field(default=False, metadata={"help": "Use torch.compile on the base adaptor."})
    final_filename: str = field(default="adaptor.pt", metadata={"help": "Filename of final saved adaptor weights."})

    @property
    def adaptor_config(self) -> AdaptorConfig:
        decoder_config = Qwen3Config.from_json_file(self.adaptor_decoder_config_path)
        if self.attn_implementation is not None:
            decoder_config._attn_implementation = self.attn_implementation

        return AdaptorConfig(
            input_size=self.input_size,
            output_size=self.output_size,
            output_time_scale=self.output_time_scale,
            lag_timesteps=self.lag_timesteps,
            decoder_config=decoder_config,
        )


class Adaptor(nn.Module):
    def __init__(self, config: AdaptorConfig):
        super().__init__()
        self.config = config
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.output_time_scale = config.output_time_scale
        self.lag_timesteps = config.lag_timesteps
        self.decoder_config = config.decoder_config

        self.input_proj = nn.Linear(self.input_size, self.decoder_config.hidden_size * self.output_time_scale, bias=False)
        self.decoder = Qwen3Model(self.decoder_config)
        self.output_proj = nn.Linear(self.decoder_config.hidden_size, self.output_size, bias=False)

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        targets: Optional[torch.Tensor] = None,
        output_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        inputs_embeds = self.input_proj(inputs).view(*inputs.shape[:-2], -1, self.decoder_config.hidden_size)
        last_hidden_state = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        ).last_hidden_state
        output_embeds: torch.Tensor = self.output_proj(last_hidden_state)

        if self.lag_timesteps > 0:
            output_embeds = output_embeds[:, self.lag_timesteps :]
            if targets is not None and output_mask is not None:
                targets = targets[:, : -self.lag_timesteps]
                output_mask = output_mask[:, : -self.lag_timesteps]

        outputs = {"logits": output_embeds}
        if targets is not None and output_mask is not None:
            outputs["loss"] = ((output_embeds - targets).square() * output_mask).sum() / output_mask.sum().clamp_min(1.0)

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
        output_mask = torch.ones_like(y[:, :1])

        max_pairs = min(x.shape[0], y.shape[0] // 2, max_input_seq_len)
        if max_pairs == 0:
            continue
        elif x.shape[0] > max_input_seq_len:
            x = x[:max_input_seq_len]
            y = y[: 2 * max_input_seq_len]
            output_mask = output_mask[: 2 * max_input_seq_len]
        elif x.shape[0] < max_input_seq_len:
            x = F.pad(x, (0, 0, 0, max_input_seq_len - x.shape[0]), value=0)
            y = F.pad(y, (0, 0, 0, 2 * max_input_seq_len - y.shape[0]), value=0)
            output_mask = F.pad(output_mask, (0, 0, 0, 2 * max_input_seq_len - output_mask.shape[0]), value=0)

        assert x.shape[0] == max_input_seq_len
        assert y.shape[0] == 2 * max_input_seq_len
        assert output_mask.shape[0] == 2 * max_input_seq_len

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

    if isinstance(eval_pred.label_ids, (tuple, list)):
        targets, output_mask = eval_pred.label_ids
    else:
        targets = eval_pred.label_ids
        output_mask = None

    if output_mask is not None:
        assert isinstance(output_mask, np.ndarray)
    else:
        output_mask = np.ones_like(targets[:, :, :1])

    assert isinstance(preds, np.ndarray) and isinstance(targets, np.ndarray) and isinstance(output_mask, np.ndarray)

    if preds.shape[1] != targets.shape[1]:
        targets = targets[:, -preds.shape[1] :]
        output_mask = output_mask[:, -preds.shape[1] :]

    assert preds.shape == targets.shape and preds.shape[:2] == output_mask.shape[:2]

    count = max(output_mask.sum(), 1)
    mse = (np.square(preds - targets) * output_mask).sum() / count
    mean = (targets * output_mask).sum(axis=(0, 1)) / count
    var = (np.square(targets - mean) * output_mask).sum() / count
    return {"r2": 1 - mse / var}


def parse_args() -> tuple[AdaptorRunArguments, TrainingArguments]:
    parser = HfArgumentParser((AdaptorRunArguments, TrainingArguments))  # type: ignore
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
            trainer.model.module.save_pretrained(  # type: ignore
                save_path,
                state_dict=state_dict,
                safe_serialization=True,
            )
            break
        except Exception as e:
            print(f"Save failed due to the following exception: {e}")
            time.sleep(1)

    print(f"Training complete. Final adaptor saved to `{str(save_path)}`.")


if __name__ == "__main__":
    main()
