import os
import argparse
import torch

from pathlib import Path
from huggingface_hub import hf_hub_download
from torch.nn.utils.rnn import pad_sequence
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor  # type: ignore
from torch import nn
from transformers.activations import silu
from moshi.models import loaders
from dataclasses import dataclass
from data_util import load_audio_samples, AudioSample, get_quantized_mimi_features, get_qwen_omni_features


@dataclass
class AdaptorConfig:
    input_size: int
    output_size: int
    intermediate_size: int = 8192
    output_timesteps: int = 1


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_proj: torch.Tensor = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj.view(*x.shape[:-2], -1, self.output_size)



def parse_args() -> Path:
    parser = argparse.ArgumentParser(description="Extract Mimi & Qwen features (resume-safe, multi-GPU).")
    fs_path = Path("/") / "mnt/efs/fs1"
    parser.add_argument(
        "--data_path",
        type=Path,
        default=fs_path / "extracted_audio_features",
    )
    args = parser.parse_args()
    return args.data_path



if __name__ == "__main__":
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-3B", 
        torch_dtype=torch.bfloat16, 
        device_map="cuda",
    )
    model.disable_talker()

    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device="cuda").to(torch.bfloat16)
    mimi.set_num_codebooks(8)

    adaptor_config = AdaptorConfig(
        input_size=mimi.dimension,
        output_size=model.thinker.config.text_config.hidden_size,
        output_timesteps=2,
    )
    adaptor = Adaptor(adaptor_config).to("cuda", torch.bfloat16)
    
    