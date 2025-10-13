from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from moshi.models import loaders
from torch import nn
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen2_5_omni import Qwen2_5OmniThinkerConfig, Qwen2_5OmniThinkerForConditionalGeneration
from transformers.models.qwen3 import Qwen3Config, Qwen3Model


class MimiToQwenOmniAdaptorConfig(PretrainedConfig):
    input_size: int
    output_size: int
    output_time_scale: int
    lag_timesteps: int
    decoder_config: Qwen3Config


class MimiToQwenOmniAdaptor(nn.Module):
    def __init__(self, config: MimiToQwenOmniAdaptorConfig):
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


class QwenOmniThinkerWithMimiAdaptor(Qwen2_5OmniThinkerForConditionalGeneration):
    def __init__(self, config: Qwen2_5OmniThinkerConfig, audio_adaptor_config: MimiToQwenOmniAdaptorConfig) -> None:
        super().__init__(config)
        self.mimi = loaders.get_mimi(hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME), num_codebooks=8)
        self.adaptor = MimiToQwenOmniAdaptor(audio_adaptor_config)
        assert self.adaptor.lag_timesteps == 0
        del self.audio_tower

    def get_audio_features(
        self,
        input_features: torch.Tensor,
        feature_attention_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        mimi_latent_features = self.mimi.decode_latent(input_features.transpose(1, 2)).transpose(1, 2)
        adaptor_outputs = self.adaptor(inputs=mimi_latent_features, attention_mask=feature_attention_mask)
        return adaptor_outputs["logits"]


def update_qwen2_5_omni_with_adaptor(
    model: Qwen2_5OmniThinkerForConditionalGeneration,
    adaptor: MimiToQwenOmniAdaptor,
) -> QwenOmniThinkerWithMimiAdaptor:
    assert adaptor.lag_timesteps == 0
    model.mimi = loaders.get_mimi(hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME), num_codebooks=8)
    model.adaptor = adaptor
    del model.audio_tower
    model.get_audio_features = QwenOmniThinkerWithMimiAdaptor.get_audio_features.__get__(model, type(model))
    return model  # type: ignore


def load_qwen_omni_thinker_with_mimi_adaptor(
    model_name_or_path: str | Path,
    adaptor_config_path: str | Path,
    adaptor_state_dict_path: Optional[str | Path] = None,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
    unfreeze_modules: list[str] = ["model", "adaptor"],
) -> QwenOmniThinkerWithMimiAdaptor:
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
    )
    adaptor_config: MimiToQwenOmniAdaptorConfig = MimiToQwenOmniAdaptorConfig.from_json_file(adaptor_config_path)  # type: ignore
    adaptor_config.decoder_config = Qwen3Config(**adaptor_config.decoder_config)  # type: ignore
    adaptor_config.decoder_config._attn_implementation = attn_implementation
    adaptor = MimiToQwenOmniAdaptor(adaptor_config)
    if adaptor_state_dict_path is not None:
        adaptor.load_state_dict(torch.load(adaptor_state_dict_path, map_location="cpu"))

    model = update_qwen2_5_omni_with_adaptor(model, adaptor)

    for name, param in model.named_parameters():
        if any(name.startswith(module) for module in unfreeze_modules):
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model.to(dtype)  # type: ignore
