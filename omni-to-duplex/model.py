import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    BatchFeature,
    Cache,
    DynamicCache,
    PretrainedConfig,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniThinkerTextModel,
    Qwen3Config,
    Qwen3Model,
    Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerTextModel,
)
from transformers.models.mimi.modeling_mimi import MimiEncoderOutput, MimiModel
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTalkerCodePredictorConfig
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
        if self.output_time_scale > 1 and attention_mask is not None:
            batch_size, input_seq_len = attention_mask.shape
            attention_mask = (
                attention_mask[:, :, None]
                .expand(batch_size, input_seq_len, self.output_time_scale)
                .reshape(batch_size, input_seq_len * self.output_time_scale)
            )

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
class QwenOmniWithMimiOutputWithPast(ModelOutput):
    loss: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[Cache] = None
    audio_past_key_values: Optional[Cache] = None
    rope_deltas: Optional[torch.Tensor] = None


class QwenOmniWithMimi(nn.Module):
    def __init__(
        self,
        mimi: MimiModel,
        adaptor: MimiToQwenOmniAdaptor,
        text_model: Qwen2_5OmniThinkerTextModel | Qwen3OmniMoeThinkerTextModel,
        text_tokenizer: Any,
        audio_token_id: int,
    ) -> None:
        super().__init__()
        self.mimi = mimi
        self.adaptor = adaptor
        self.text_model = text_model
        self.text_tokenizer = text_tokenizer
        self.audio_token_id = audio_token_id

    def get_audio_features(
        self,
        audio_codes: torch.Tensor,
        audio_codes_mask: Optional[torch.Tensor] = None,
        audio_past_key_values: Optional[Cache] = None,
        audio_use_cache: Optional[bool] = None,
    ) -> MimiToQwenOmniAdaptorOutputWithPast:
        mimi_latent_features = self.mimi.quantizer.decode(audio_codes.transpose(1, 2)).transpose(1, 2)
        return self.adaptor(
            inputs=mimi_latent_features,
            attention_mask=audio_codes_mask,
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
        audio_codes_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        audio_past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        rope_deltas: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        audio_use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> QwenOmniWithMimiOutputWithPast:
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids)
            assert inputs_embeds is not None

        audio_outputs = None
        if input_ids is not None and input_ids.shape[1] != 1:
            if audio_codes is not None:
                audio_outputs = self.get_audio_features(
                    audio_codes=audio_codes,
                    audio_codes_mask=audio_codes_mask,
                    audio_past_key_values=audio_past_key_values,
                    audio_use_cache=audio_use_cache,
                )
                audio_features = audio_outputs.logits
                assert audio_features is not None

                audio_mask = (input_ids == self.audio_token_id) | (input_ids >= self.text_model.config.vocab_size)
                audio_mask = audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
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

        outputs = self.text_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        return QwenOmniWithMimiOutputWithPast(
            last_hidden_state=outputs[0],
            past_key_values=outputs.past_key_values,
            audio_past_key_values=audio_outputs.past_key_values if audio_outputs is not None else None,
            rope_deltas=self.rope_deltas,
        )

    def process_text(
        self,
        text: str | list[str],
        audio_codes: Optional[torch.Tensor] = None,
        audio_codes_mask: Optional[torch.Tensor] = None,
        audio_token: str = "<|AUDIO|>",
        **tokenizer_kwargs,
    ) -> BatchFeature | BatchEncoding:
        if audio_codes is not None:
            output_time_scale = self.adaptor.output_time_scale
            lag_timesteps = self.adaptor.lag_timesteps
            if audio_codes_mask is not None:
                audio_lengths = iter(output_time_scale * audio_codes_mask.sum(dim=1) - lag_timesteps)
            else:
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

        processed_inputs = self.text_tokenizer(processed_text, return_tensors="pt", **tokenizer_kwargs)
        processed_inputs["audio_codes"] = audio_codes
        processed_inputs["audio_codes_mask"] = audio_codes_mask
        return processed_inputs

    def process_inputs(
        self,
        text: str | list[str],
        audio: Optional[torch.Tensor | list[torch.Tensor]] = None,
        audio_sample_rate: Optional[int | list[int]] = None,
        num_audio_quantizers: int = 8,
        **tokenizer_kwargs,
    ) -> BatchFeature | BatchEncoding:
        if not isinstance(text, list):
            text = [text]

        if audio is not None:
            if not isinstance(audio, list):
                audio = [audio]

            if audio_sample_rate is not None:
                if isinstance(audio_sample_rate, int):
                    audio_sample_rate = [audio_sample_rate] * len(audio)

                assert len(audio) == len(audio_sample_rate)
                for i in range(len(audio)):
                    if audio_sample_rate[i] != self.mimi.config.sampling_rate:
                        audio[i] = torchaudio.functional.resample(
                            audio[i],
                            orig_freq=audio_sample_rate[i],
                            new_freq=self.mimi.config.sampling_rate,
                        )

            audio_mask = pad_sequence(
                [torch.ones(a.shape, dtype=torch.long, device=a.device) for a in audio],
                batch_first=True,
            )
            audio = pad_sequence(audio, batch_first=True)
            assert audio.ndim == 2

            mimi_param = next(iter(self.mimi.parameters()))
            audio = audio[:, None].to(mimi_param.device, mimi_param.dtype)
            mimi_outputs = self.mimi.encode(audio, num_quantizers=num_audio_quantizers, padding_mask=audio_mask)
            assert isinstance(mimi_outputs, MimiEncoderOutput) and mimi_outputs.audio_codes is not None
            audio_codes = mimi_outputs.audio_codes.transpose(1, 2)

            samples_per_code = int(self.mimi.config.sampling_rate / self.mimi.config._frame_rate)  # type: ignore
            assert samples_per_code * self.mimi.config._frame_rate == self.mimi.config.sampling_rate  # type: ignore
            padded_audio_mask = F.pad(audio_mask, (0, audio_codes.shape[1] * samples_per_code - audio_mask.shape[1]))
            audio_codes_mask = padded_audio_mask.view(-1, audio_codes.shape[1], samples_per_code).any(dim=-1)
        else:
            audio_codes = None
            audio_codes_mask = None

        return self.process_text(text, audio_codes, audio_codes_mask, **tokenizer_kwargs).to(self.text_model.device)


class QwenOmniWithMimiForConditionalGeneration(nn.Module):
    def __init__(
        self,
        text_model_name_or_path: str | Path,
        mimi_model_name_or_path: str | Path = "kyutai/mimi",
        adaptor_config: Optional[MimiToQwenOmniAdaptorConfig] = None,
        adaptor_config_path: Optional[str | Path] = None,
        adaptor_state_dict_path: Optional[str | Path] = None,
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
    ) -> None:
        super().__init__()
        if adaptor_config is None:
            assert adaptor_config_path is not None
            adaptor_config = MimiToQwenOmniAdaptorConfig.from_json_file(adaptor_config_path)  # type: ignore
            assert adaptor_config is not None
            adaptor_config.decoder_config = Qwen3Config(**adaptor_config.decoder_config)  # type: ignore
            adaptor_config.decoder_config._attn_implementation = attn_implementation

        mimi = MimiModel.from_pretrained(
            mimi_model_name_or_path,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )
        for param in mimi.parameters():
            param.requires_grad = False

        adaptor = MimiToQwenOmniAdaptor(adaptor_config).to(dtype)
        if adaptor_state_dict_path is not None:
            adaptor.load_state_dict(torch.load(adaptor_state_dict_path, map_location="cpu"))

        text_tokenizer = AutoTokenizer.from_pretrained(text_model_name_or_path)

        if "Qwen2.5" in str(text_model_name_or_path):
            thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                text_model_name_or_path,
                dtype=dtype,
                attn_implementation=attn_implementation,
            )
        elif "Qwen3" in str(text_model_name_or_path):
            thinker = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
                text_model_name_or_path,
                dtype=dtype,
                attn_implementation=attn_implementation,
            )
        else:
            raise NotImplementedError("`text_model_name_or_path` must be a variant of `Qwen2.5-Omni` or `Qwen3OmniMoe`.")

        text_model = thinker.model
        self.model = QwenOmniWithMimi(
            mimi=mimi,
            adaptor=adaptor,
            text_model=text_model,
            text_tokenizer=text_tokenizer,
            audio_token_id=thinker.config.audio_token_id,
        )
        self.lm_head = thinker.lm_head
        # TODO: update to support image/video inputs

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        audio_codes: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        audio_codes_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        audio_past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        rope_deltas: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        audio_use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> QwenOmniWithMimiOutputWithPast:
        outputs: QwenOmniWithMimiOutputWithPast = self.model(
            input_ids=input_ids,
            audio_codes=audio_codes,
            attention_mask=attention_mask,
            audio_codes_mask=audio_codes_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            audio_past_key_values=audio_past_key_values,
            inputs_embeds=inputs_embeds,
            rope_deltas=rope_deltas,
            audio_use_cache=audio_use_cache,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        logits = self.lm_head(outputs.last_hidden_state)

        if labels is not None:
            loss = self.model.text_model.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.model.text_model.config.get_text_config().vocab_size,
            )
        else:
            loss = None

        return QwenOmniWithMimiOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            audio_past_key_values=outputs.audio_past_key_values,
            rope_deltas=outputs.rope_deltas,
        )

    def process_text(
        self,
        text: str | list[str],
        audio_codes: Optional[torch.Tensor] = None,
        audio_codes_mask: Optional[torch.Tensor] = None,
        audio_token: str = "<|AUDIO|>",
        **tokenizer_kwargs,
    ) -> BatchFeature | BatchEncoding:
        return self.model.process_text(
            text=text,
            audio_codes=audio_codes,
            audio_codes_mask=audio_codes_mask,
            audio_token=audio_token,
            **tokenizer_kwargs,
        )

    def process_inputs(
        self,
        text: str | list[str],
        audio: Optional[torch.Tensor | list[torch.Tensor]] = None,
        audio_sample_rate: Optional[int | list[int]] = None,
        num_audio_quantizers: int = 8,
        **tokenizer_kwargs,
    ) -> BatchFeature | BatchEncoding:
        return self.model.process_inputs(
            text=text,
            audio=audio,
            audio_sample_rate=audio_sample_rate,
            num_audio_quantizers=num_audio_quantizers,
            **tokenizer_kwargs,
        )

    @torch.inference_mode()
    def generate_greedy(
        self,
        text: str | list[str],
        audio: Optional[torch.Tensor | np.ndarray | list[torch.Tensor | np.ndarray]] = None,
        audio_sample_rate: Optional[int | list[int]] = None,
        max_new_tokens: int = 128,
        eos_token_id: int = 151645,
        return_text: bool = False,
    ) -> str | torch.Tensor:
        if audio is not None:
            if not isinstance(audio, list):
                audio = [audio]

            audio_tensors = [a if isinstance(a, torch.Tensor) else torch.from_numpy(a) for a in audio]
        else:
            audio_tensors = None

        inputs = self.process_inputs(text, audio_tensors, audio_sample_rate, padding=True, padding_side="left")
        past_key_values = DynamicCache()
        outputs = self(**inputs, past_key_values=past_key_values, use_cache=True)

        input_seq_len = inputs.input_ids.shape[1]
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

            if (sequences[:, input_seq_len:] == eos_token_id).any(dim=1).all():
                break

        if return_text:
            return self.model.text_tokenizer.batch_decode(sequences, skip_special_tokens=True)
        else:
            return sequences


class QwenOmniWithMimiAudioOutput(nn.Module):
    def __init__(
        self,
        text_model_name_or_path: str | Path,
        mimi_model_name_or_path: str | Path = "kyutai/mimi",
        adaptor_config: Optional[MimiToQwenOmniAdaptorConfig] = None,
        adaptor_config_path: Optional[str | Path] = None,
        adaptor_state_dict_path: Optional[str | Path] = None,
        code_predictor_config: Optional[Qwen3OmniMoeTalkerCodePredictorConfig] = None,
        code_predictor_config_path: Optional[str | Path] = None,
        code_predictor_state_dict_path: Optional[str | Path] = None,
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
    ) -> None:
        super().__init__()
        if adaptor_config is None:
            assert adaptor_config_path is not None
            adaptor_config = MimiToQwenOmniAdaptorConfig.from_json_file(adaptor_config_path)  # type: ignore
            assert adaptor_config is not None
            adaptor_config.decoder_config = Qwen3Config(**adaptor_config.decoder_config)  # type: ignore
            adaptor_config.decoder_config._attn_implementation = attn_implementation

        if code_predictor_config is None:
            assert code_predictor_config_path is not None
            code_predictor_config = Qwen3OmniMoeTalkerCodePredictorConfig.from_json_file(code_predictor_config_path)  # type: ignore
            assert code_predictor_config is not None

        mimi = MimiModel.from_pretrained(
            mimi_model_name_or_path,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )
        for param in mimi.parameters():
            param.requires_grad = False

        adaptor = MimiToQwenOmniAdaptor(adaptor_config).to(dtype)
        if adaptor_state_dict_path is not None:
            adaptor.load_state_dict(torch.load(adaptor_state_dict_path, map_location="cpu"))

        text_tokenizer = AutoTokenizer.from_pretrained(text_model_name_or_path)

        if "Qwen2.5" in str(text_model_name_or_path):
            thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                text_model_name_or_path,
                dtype=dtype,
                attn_implementation=attn_implementation,
            )
        elif "Qwen3" in str(text_model_name_or_path):
            thinker = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
                text_model_name_or_path,
                dtype=dtype,
                attn_implementation=attn_implementation,
            )
        else:
            raise NotImplementedError("`text_model_name_or_path` must be a variant of `Qwen2.5-Omni` or `Qwen3OmniMoe`.")

        text_model = thinker.model
        self.model = QwenOmniWithMimi(
            mimi=mimi,
            adaptor=adaptor,
            text_model=text_model,
            text_tokenizer=text_tokenizer,
            audio_token_id=thinker.config.audio_token_id,
        )
        self.text_vocab_size = text_model.config.vocab_size
        assert self.text_vocab_size == thinker.lm_head.out_features
        self.lm_head = nn.Linear(
            in_features=thinker.lm_head.in_features,
            out_features=self.text_vocab_size + mimi.config.codebook_size,
            bias=False,
            dtype=dtype,
        )
        with torch.no_grad():
            self.lm_head.weight[: self.text_vocab_size].copy_(thinker.lm_head.weight)

        self.code_predictor = Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration(code_predictor_config)
        if code_predictor_state_dict_path is not None:
            self.code_predictor.load_state_dict(torch.load(code_predictor_state_dict_path, map_location="cpu"))

        self.proj_code = nn.Linear(
            in_features=thinker.lm_head.in_features,
            out_features=code_predictor_config.hidden_size,
            bias=False,
            dtype=dtype,
        )

    def parallel_code_predictor_forward(
        self,
        hidden_embeds: torch.Tensor,
        audio_codes: torch.Tensor,
    ) -> torch.Tensor:
        audio_code_embeds = [
            self.code_predictor.get_input_embeddings()[i](audio_codes[:, [i]])
            for i in range(self.code_predictor.config.num_code_groups - 1)
        ]
        inputs_embeds = torch.cat([hidden_embeds[:, None]] + audio_code_embeds, dim=1)
        last_hidden_state = self.code_predictor.model(inputs_embeds=inputs_embeds).last_hidden_state
        logits = [
            self.code_predictor.lm_head[i](last_hidden_state[:, [i + 1]])
            for i in range(self.code_predictor.config.num_code_groups - 1)
        ]
        return torch.cat(logits, dim=1)

    def compute_code_predictor_loss(
        self,
        input_ids: torch.Tensor,
        last_hidden_state: torch.Tensor,
        audio_codes: torch.Tensor,
        audio_labels: torch.Tensor,
        audio_codes_mask: Optional[torch.Tensor] = None,
    ):
        hidden_embeds = last_hidden_state[(input_ids == self.model.audio_token_id) | (input_ids >= self.text_vocab_size)]
        if self.model.adaptor.output_time_scale > 1:
            hidden_embeds = hidden_embeds[self.model.adaptor.output_time_scale - 1 :: self.model.adaptor.output_time_scale]

        hidden_embeds = self.proj_code(hidden_embeds)

        if audio_codes_mask is None:
            audio_codes = audio_codes.view(audio_codes.shape[0] * audio_codes.shape[1], -1)
            audio_labels = audio_labels.view(audio_codes.shape[0] * audio_codes.shape[1], -1)
        else:
            audio_codes = audio_codes[audio_codes_mask == 1]
            audio_labels = audio_labels[audio_codes_mask == 1]

        assert hidden_embeds.shape[0] == audio_codes.shape[0]

        logits = self.parallel_code_predictor_forward(hidden_embeds=hidden_embeds, audio_codes=audio_codes)

        return self.code_predictor.loss_function(
            logits=logits,
            labels=None,
            shift_labels=audio_labels[:, 1:].contiguous(),
            vocab_size=self.model.mimi.config.codebook_size,
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        audio_codes: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        audio_codes_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        audio_past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        rope_deltas: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        audio_labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        audio_use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> QwenOmniWithMimiOutputWithPast:
        outputs: QwenOmniWithMimiOutputWithPast = self.model(
            input_ids=input_ids,
            audio_codes=audio_codes,
            attention_mask=attention_mask,
            audio_codes_mask=audio_codes_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            audio_past_key_values=audio_past_key_values,
            inputs_embeds=inputs_embeds,
            rope_deltas=rope_deltas,
            audio_use_cache=audio_use_cache,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        assert outputs.last_hidden_state is not None

        logits = self.lm_head(outputs.last_hidden_state)

        if labels is not None:
            lm_loss = self.model.text_model.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.text_vocab_size + self.model.mimi.config.codebook_size,
            )
        else:
            lm_loss = None

        if input_ids is not None and audio_codes is not None and audio_labels is not None:
            code_predictor_loss = self.compute_code_predictor_loss(
                last_hidden_state=outputs.last_hidden_state,
                input_ids=input_ids,
                audio_codes=audio_codes,
                audio_labels=audio_labels,
                audio_codes_mask=audio_codes_mask,
            )
        else:
            code_predictor_loss = None

        if lm_loss is not None and code_predictor_loss is not None:
            loss = lm_loss + code_predictor_loss
        elif lm_loss is not None:
            loss = lm_loss
        elif code_predictor_loss is not None:
            loss = code_predictor_loss
        else:
            loss = None

        return QwenOmniWithMimiOutputWithPast(
            loss=loss,
            last_hidden_state=outputs.last_hidden_state,
            logits=logits,
            past_key_values=outputs.past_key_values,
            audio_past_key_values=outputs.audio_past_key_values,
            rope_deltas=outputs.rope_deltas,
        )

    def process_text(
        self,
        text: str | list[str],
        audio_codes: Optional[torch.Tensor] = None,
        audio_codes_mask: Optional[torch.Tensor] = None,
        audio_token: str = "<|AUDIO|>",
        **tokenizer_kwargs,
    ) -> BatchFeature | BatchEncoding:
        return self.model.process_text(
            text=text,
            audio_codes=audio_codes,
            audio_codes_mask=audio_codes_mask,
            audio_token=audio_token,
            **tokenizer_kwargs,
        )

    def process_inputs(
        self,
        text: str | list[str],
        audio: Optional[torch.Tensor | list[torch.Tensor]] = None,
        audio_sample_rate: Optional[int | list[int]] = None,
        num_audio_quantizers: int = 8,
        **tokenizer_kwargs,
    ) -> BatchFeature | BatchEncoding:
        return self.model.process_inputs(
            text=text,
            audio=audio,
            audio_sample_rate=audio_sample_rate,
            num_audio_quantizers=num_audio_quantizers,
            **tokenizer_kwargs,
        )
