from typing import Optional

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig

# from transformers.models.qwen2_5_omni import Qwen2_5OmniThinkerConfig, Qwen2_5OmniThinkerForConditionalGeneration
from transformers.models.qwen3 import Qwen3Config, Qwen3Model


class AdaptorConfig(PretrainedConfig):
    input_size: int
    output_size: int
    output_time_scale: int
    lag_timesteps: int
    decoder_config: Qwen3Config


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


# class QwenOmniWithMimiAudioForConditionalGeneration(Qwen2_5OmniThinkerForConditionalGeneration):
#     def __init__(self, config: Qwen2_5OmniThinkerConfig, audio_adaptor_config: AdaptorConfig) -> None:
#         super().__init__(config)

#     def get_audio_features(
#         self,
#         input_features: torch.Tensor,
#         feature_attention_mask: Optional[torch.Tensor] = None,
#         audio_feature_lengths: Optional[torch.Tensor] = None,
#     ):
#         """
#         Encodes audios into continuous embeddings that can be forwarded to the language model.

#         Args:
#             input_features (`torch.FloatTensor`):
#                 The tensors corresponding to the input audios.
#             feature_attention_mask (`torch.LongTensor`, *optional*):
#                 Mask to avoid performing attention on padding feature indices. Mask values selected in `[0, 1]`:
#             audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
#                 The length of feature shape of each audio in LLM.
#         """
#         if feature_attention_mask is not None:
#             audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
#             input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
#         else:
#             audio_feature_lengths = None

#         audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
#             audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
#         )
#         feature_lens = audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
#         audio_outputs = self.audio_tower(
#             input_features,
#             feature_lens=feature_lens,
#             aftercnn_lens=audio_feat_lengths,
#         )
#         audio_features = audio_outputs.last_hidden_state

#         if audio_features.shape[0] != sum(audio_output_lengths.tolist()):
#             raise ValueError("length of audio_features should match audio_output_lengths")

#         return audio_features
