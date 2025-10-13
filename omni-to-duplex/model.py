import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional
from torch import nn
from transformers import AutoTokenizer, BatchEncoding, BatchFeature, Cache, DynamicCache, PretrainedConfig
from transformers.models.mimi.modeling_mimi import MimiEncoderOutput, MimiModel
from transformers.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration
from transformers.models.qwen3 import Qwen3Config, Qwen3Model
from transformers.utils.generic import ModelOutput


class MimiToQwenOmniAdaptorConfig(PretrainedConfig):  # type: ignore
    input_size: int
    output_size: int
    output_time_scale: int
    lag_timesteps: int
    decoder_config: Qwen3Config


@dataclass
class MimiToQwenOmniAdaptorOutputWithPast(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[Cache] = None


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
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        targets: Optional[torch.Tensor] = None,
        output_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        inputs_embeds = self.input_proj(inputs).view(*inputs.shape[:-2], -1, self.decoder_config.hidden_size)
        decoder_outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        output_embeds: torch.Tensor = self.output_proj(decoder_outputs.last_hidden_state)

        if self.lag_timesteps > 0:
            output_embeds = output_embeds[:, self.lag_timesteps :]
            if targets is not None and output_mask is not None:
                targets = targets[:, : -self.lag_timesteps]
                output_mask = output_mask[:, : -self.lag_timesteps]

        if targets is not None and output_mask is not None:
            loss = ((output_embeds - targets).square() * output_mask).sum() / output_mask.sum().clamp_min(1.0)
        else:
            loss = None

        return MimiToQwenOmniAdaptorOutputWithPast(
            loss=loss,
            logits=output_embeds,
            past_key_values=decoder_outputs.past_key_values,
        )


@dataclass
class Qwen2_5OmniWithMimiOutputWithPast(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[Cache] = None
    audio_past_key_values: Optional[Cache] = None
    rope_deltas: Optional[torch.Tensor] = None


class Qwen2_5OmniWithMimiForConditionalGeneration(nn.Module):
    def __init__(
        self,
        text_model_name_or_path: str | Path,
        mimi_model_name_or_path: str | Path = "kyutai/mimi",
        adaptor_config: Optional[MimiToQwenOmniAdaptorConfig] = None,
        adaptor_config_path: Optional[str | Path] = None,
        adaptor_state_dict_path: Optional[str | Path] = None,
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
        unfreeze_modules: list[Literal["mimi", "adaptor", "model", "lm_head"]] = ["adaptor", "model", "lm_head"],
    ) -> None:
        super().__init__()
        if adaptor_config is None:
            assert adaptor_config_path is not None
            adaptor_config = MimiToQwenOmniAdaptorConfig.from_json_file(adaptor_config_path)  # type: ignore
            assert adaptor_config is not None
            adaptor_config.decoder_config = Qwen3Config(**adaptor_config.decoder_config)  # type: ignore
            adaptor_config.decoder_config._attn_implementation = attn_implementation

        self.mimi = MimiModel.from_pretrained(
            mimi_model_name_or_path,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )
        self.adaptor = MimiToQwenOmniAdaptor(adaptor_config).to(dtype)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name_or_path)
        thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            text_model_name_or_path,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )
        self.model = thinker.model
        self.lm_head = thinker.lm_head
        self.audio_token_id: int = thinker.config.audio_token_id
        # TODO: update to support image/video inputs
        del thinker

        if adaptor_state_dict_path is not None:
            self.adaptor.load_state_dict(torch.load(adaptor_state_dict_path, map_location="cpu"))

        if "mimi" not in unfreeze_modules:
            for param in self.mimi.parameters():
                param.requires_grad = False

        if "adaptor" not in unfreeze_modules:
            for param in self.adaptor.parameters():
                param.requires_grad = False

        if "model" not in unfreeze_modules:
            for param in self.model.parameters():
                param.requires_grad = False

        if "lm_head" not in unfreeze_modules:
            for param in self.lm_head.parameters():
                param.requires_grad = False

    def get_audio_features(
        self,
        audio_codes: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
        audio_past_key_values: Optional[Cache] = None,
        audio_use_cache: Optional[bool] = None,
    ) -> MimiToQwenOmniAdaptorOutputWithPast:
        mimi_latent_features = self.mimi.quantizer.decode(audio_codes.transpose(1, 2)).transpose(1, 2)
        return self.adaptor(
            inputs=mimi_latent_features,
            attention_mask=audio_attention_mask,
            past_key_values=audio_past_key_values,
            use_cache=audio_use_cache,
        )

    def get_rope_index(self, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(dim=0, keepdim=False)[0].max(dim=-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)
        return position_ids, mrope_position_deltas

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        audio_codes: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        audio_past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        rope_deltas: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        audio_use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Qwen2_5OmniWithMimiOutputWithPast:
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
            assert inputs_embeds is not None

        audio_outputs = None
        if input_ids is not None and input_ids.shape[1] != 1:
            if audio_codes is not None:
                audio_outputs = self.get_audio_features(
                    audio_codes=audio_codes,
                    audio_attention_mask=audio_attention_mask,
                    audio_past_key_values=audio_past_key_values,
                    audio_use_cache=audio_use_cache,
                )
                audio_features = audio_outputs.logits
                assert audio_features is not None
                audio_mask = (input_ids == self.audio_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        if attention_mask is not None and position_ids is None:
            if cache_position is None or (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(attention_mask)
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = inputs_embeds.shape[:2]
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        if labels is not None:
            loss = self.model.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.model.config.get_text_config().vocab_size,
            )
        else:
            loss = None

        if not return_dict:
            output = (logits,) + outputs
            return (loss,) + output if loss is not None else output

        return Qwen2_5OmniWithMimiOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            audio_past_key_values=audio_outputs.past_key_values if audio_outputs is not None else None,
            rope_deltas=self.rope_deltas,
        )

    def process_text(
        self,
        text: str | list[str],
        audio_codes: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        audio_token: str = "<|AUDIO|>",
    ) -> BatchFeature | BatchEncoding:
        if audio_codes is not None:
            if audio_attention_mask is not None:
                audio_lengths = iter(audio_attention_mask.sum(dim=1))
            else:
                output_time_scale = self.adaptor.output_time_scale
                lag_timesteps = self.adaptor.lag_timesteps
                audio_lengths = iter([output_time_scale * audio_codes.shape[1] - lag_timesteps])
        else:
            audio_lengths = iter([])

        if not isinstance(text, list):
            text = [text]

        processed_text = []
        for sample in text:
            assert isinstance(sample, str)
            positions = []
            special_tokens = [re.escape(audio_token)]
            pattern = "|".join(special_tokens)
            positions = sorted([(match.start(), match.group()) for match in re.finditer(pattern, sample)])
            positions.sort(key=lambda x: x[0])

            for _, special_token in positions:
                if special_token == audio_token:
                    sample = sample.replace(audio_token, "<|audio_placeholder|>" * next(audio_lengths), 1)

            sample = sample.replace("<|audio_placeholder|>", audio_token)
            processed_text.append(sample)

        processed_inputs = self.text_tokenizer(processed_text, return_tensors="pt")
        processed_inputs["audio_codes"] = audio_codes
        processed_inputs["audio_attention_mask"] = audio_attention_mask
        return processed_inputs

    def process_inputs(
        self,
        text: str,
        audio: Optional[torch.Tensor | np.ndarray] = None,
        audio_sample_rate: Optional[int] = None,
        num_audio_quantizers: int = 8,
    ) -> BatchFeature | BatchEncoding:
        if audio is not None:
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio).float()

            if audio_sample_rate is not None and audio_sample_rate != self.mimi.config.sampling_rate:
                audio = torchaudio.functional.resample(
                    audio,
                    orig_freq=audio_sample_rate,
                    new_freq=self.mimi.config.sampling_rate,
                )

            while audio.ndim < 3:
                audio = audio[None]

            mimi_param = next(iter(self.mimi.parameters()))
            mimi_outputs = self.mimi.encode(audio.to(mimi_param.device, mimi_param.dtype), num_quantizers=num_audio_quantizers)
            assert isinstance(mimi_outputs, MimiEncoderOutput) and mimi_outputs.audio_codes is not None
            audio_codes = mimi_outputs.audio_codes.transpose(1, 2)
        else:
            audio_codes = None

        return self.process_text(text, audio_codes).to(self.model.device)

    @torch.inference_mode()
    def generate_greedy(
        self,
        text: str,
        audio: Optional[torch.Tensor | np.ndarray] = None,
        audio_sample_rate: Optional[int] = None,
        max_new_tokens: int = 128,
        eos_token_id: int = 151645,
        return_text: bool = False,
    ) -> str | torch.Tensor:
        inputs = self.process_inputs(text, audio, audio_sample_rate)
        past_key_values = DynamicCache()
        outputs = self(**inputs, past_key_values=past_key_values, use_cache=True)

        input_ids = outputs.logits[:, -1:].argmax(dim=-1)
        attention_mask = F.pad(inputs.attention_mask, (0, 1), value=1)
        sequences = torch.cat((inputs.input_ids, input_ids), dim=1)

        for _ in range(max_new_tokens - 1):
            position_ids = attention_mask.sum(dim=1, keepdim=True)
            outputs = self(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            input_ids = outputs.logits[:, -1:].argmax(dim=-1)
            attention_mask = F.pad(attention_mask, (0, 1), value=1)
            sequences = torch.cat((sequences, input_ids), dim=1)

            if input_ids[0, -1].item() == eos_token_id:
                break

        if return_text:
            return self.text_tokenizer.decode(sequences[0], skip_special_tokens=True)
        else:
            return sequences
