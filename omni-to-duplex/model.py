import re
from dataclasses import dataclass
from os import PathLike
from typing import Optional, Self, cast

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForCausalLM,
    BatchFeature,
    Cache,
    DynamicCache,
    EncodecConfig,
    EncodecModel,
    MimiConfig,
    MimiModel,
    PretrainedConfig,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniThinkerTextModel,
    Qwen3Config,
    Qwen3Model,
    Qwen3MoeConfig,
    Qwen3OmniMoeProcessor,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerTextModel,
)
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.modeling_utils import PreTrainedModel
from transformers.models.encodec.modeling_encodec import EncodecEncoderOutput
from transformers.models.mimi.modeling_mimi import MimiEncoderOutput
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniTextConfig
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeTextConfig,
)
from transformers.utils.generic import ModelOutput

TEXT_MODEL_CONFIGS = {
    Qwen3Config.model_type: Qwen3Config,
    Qwen3MoeConfig.model_type: Qwen3MoeConfig,
    Qwen3OmniMoeTextConfig.model_type: Qwen3OmniMoeTextConfig,
    Qwen2_5OmniTextConfig.model_type: Qwen2_5OmniTextConfig,
}

TEXT_MODELS = {
    Qwen3Config.model_type: Qwen3Model,
    Qwen3MoeConfig.model_type: Qwen3MoeConfig,
    Qwen3OmniMoeTextConfig.model_type: Qwen3OmniMoeThinkerTextModel,
    Qwen2_5OmniTextConfig.model_type: Qwen2_5OmniThinkerTextModel,
}


class EmbeddingAdaptorConfig(PretrainedConfig):
    input_size: int
    output_size: int
    output_time_scale: float
    lag_timesteps: int
    decoder_config: Qwen3Config

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fix_config(self)

    @classmethod
    def fix_config(cls, config: PretrainedConfig) -> PretrainedConfig:
        if hasattr(config, "decoder_config") and isinstance(config.decoder_config, dict):
            config.decoder_config = Qwen3Config(**config.decoder_config)

        return config

    @classmethod
    def from_json_file(cls: type[Self], *args, **kwargs) -> Self:
        return cast(Self, cls.fix_config(PretrainedConfig.from_json_file(*args, **kwargs)))

    @classmethod
    def from_pretrained(cls: type[Self], *args, **kwargs) -> Self:
        return cast(Self, cls.fix_config(PretrainedConfig.from_pretrained(*args, **kwargs)))


@dataclass
class EmbeddingAdaptorOutputWithPast(ModelOutput):
    loss: Optional[torch.Tensor] = None
    output_embeds: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    past_key_values: Optional[Cache] = None


class EmbeddingAdaptor(PreTrainedModel):
    config: EmbeddingAdaptorConfig

    def __init__(self, config: EmbeddingAdaptorConfig):
        super().__init__(config)
        self.config = config
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.output_time_scale = config.output_time_scale
        self.lag_timesteps = config.lag_timesteps
        self.decoder_config = config.decoder_config

        input_proj_out_features = int(self.decoder_config.hidden_size * self.output_time_scale)
        assert input_proj_out_features == self.decoder_config.hidden_size * self.output_time_scale

        self.input_proj = nn.Linear(self.input_size, input_proj_out_features, bias=False)
        self.decoder = Qwen3Model(self.decoder_config)
        self.output_proj = nn.Linear(self.decoder_config.hidden_size, self.output_size, bias=False)

    def reshape_inputs(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, input_seq_len, _ = inputs_embeds.shape
        output_seq_len = int(input_seq_len * self.output_time_scale)
        trimmed_input_seq_len = int(output_seq_len / self.output_time_scale)
        inputs_embeds = inputs_embeds[:, :trimmed_input_seq_len].view(batch_size, output_seq_len, -1)
        if attention_mask is not None:
            attention_mask = attention_mask[:, :trimmed_input_seq_len]
            if output_seq_len < trimmed_input_seq_len:
                attention_mask = attention_mask.view(batch_size, output_seq_len, -1).any(dim=-1)
            elif output_seq_len > trimmed_input_seq_len:
                attention_mask = (
                    attention_mask[:, :, None]
                    .expand(batch_size, trimmed_input_seq_len, output_seq_len // trimmed_input_seq_len)
                    .reshape(batch_size, output_seq_len)
                )

        return inputs_embeds, attention_mask

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
        inputs_embeds = self.input_proj(inputs)
        inputs_embeds, attention_mask = self.reshape_inputs(inputs_embeds, attention_mask)
        decoder_outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        output_embeds: torch.Tensor = self.output_proj(decoder_outputs.last_hidden_state)

        if self.lag_timesteps > 0 and targets is not None and output_mask is not None:
            output_embeds = output_embeds[:, self.lag_timesteps :]
            targets = targets[:, : -self.lag_timesteps]
            output_mask = output_mask[:, : -self.lag_timesteps]

        if targets is not None and output_mask is not None:
            loss = ((output_embeds - targets).square() * output_mask).sum() / output_mask.sum().clamp_min(1.0)
        else:
            loss = None

        return EmbeddingAdaptorOutputWithPast(
            loss=loss,
            output_embeds=output_embeds,
            mask=attention_mask,
            past_key_values=decoder_outputs.past_key_values,
        )


class QwenWithCausalAudioEncoderConfig(PretrainedConfig):
    audio_encoder_config: MimiConfig | EncodecConfig
    adaptor_config: EmbeddingAdaptorConfig
    text_model_config: Qwen2_5OmniTextConfig | Qwen3OmniMoeTextConfig
    audio_token_id: int

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fix_config(self)

    @classmethod
    def fix_config(cls, config: PretrainedConfig) -> PretrainedConfig:
        if hasattr(config, "audio_encoder_config") and isinstance(config.audio_encoder_config, dict):
            model_type = config.audio_encoder_config["model_type"]
            if model_type == MimiConfig.model_type:
                config.audio_encoder_config = MimiConfig(**config.audio_encoder_config)
            elif model_type == EncodecConfig.model_type:
                config.audio_encoder_config = EncodecConfig(**config.audio_encoder_config)
            else:
                raise NotImplementedError(f"`audio_encoder.model_type={model_type}` is not supported.")

        if hasattr(config, "adaptor_config") and isinstance(config.adaptor_config, dict):
            config.adaptor_config = EmbeddingAdaptorConfig(**config.adaptor_config)

        if hasattr(config, "text_model_config") and isinstance(config.text_model_config, dict):
            config.text_model_config = TEXT_MODEL_CONFIGS[config.text_model_config["model_type"]](**config.text_model_config)

        return config

    @classmethod
    def from_json_file(cls: type[Self], *args, **kwargs) -> Self:
        return cast(Self, cls.fix_config(PretrainedConfig.from_json_file(*args, **kwargs)))

    @classmethod
    def from_pretrained(cls: type[Self], *args, **kwargs) -> Self:
        return cast(Self, cls.fix_config(PretrainedConfig.from_pretrained(*args, **kwargs)))


@dataclass
class QwenWithCausalAudioEncoderOutputWithPast(ModelOutput):
    loss: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    audio_code_logits: Optional[torch.Tensor] = None
    past_key_values: Optional[Cache] = None
    audio_past_key_values: Optional[Cache] = None


class QwenWithCausalAudioEncoder(PreTrainedModel):
    config: QwenWithCausalAudioEncoderConfig

    def __init__(self, config: QwenWithCausalAudioEncoderConfig) -> None:
        super().__init__(config)
        self.config = config

        audio_encoder_type = config.audio_encoder_config.model_type
        if audio_encoder_type == MimiConfig.model_type:
            self.audio_encoder = MimiModel._from_config(config.audio_encoder_config)
        elif audio_encoder_type == EncodecConfig.model_type:
            self.audio_encoder = EncodecModel._from_config(config.audio_encoder_config)

        self.adaptor = EmbeddingAdaptor._from_config(config.adaptor_config)
        self.text_model = TEXT_MODELS[config.text_model_config.model_type]._from_config(config.text_model_config)
        self.audio_token_id = config.audio_token_id

    def get_audio_features(
        self,
        audio_codes: torch.Tensor,
        audio_codes_mask: Optional[torch.Tensor] = None,
        audio_past_key_values: Optional[Cache] = None,
        audio_use_cache: Optional[bool] = None,
    ) -> EmbeddingAdaptorOutputWithPast:
        if isinstance(self.audio_encoder, MimiModel):
            latent_features = self.audio_encoder.quantizer.decode(audio_codes.transpose(1, 2)).transpose(1, 2)
        elif isinstance(self.audio_encoder, EncodecModel):
            latent_features = self.audio_encoder.quantizer.decode(audio_codes.permute(2, 0, 1)).transpose(1, 2)
        else:
            raise ValueError

        return self.adaptor(
            inputs=latent_features,
            attention_mask=audio_codes_mask,
            past_key_values=audio_past_key_values,
            use_cache=audio_use_cache,
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
        use_cache: Optional[bool] = None,
        audio_use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> QwenWithCausalAudioEncoderOutputWithPast:
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
                assert (audio_features := audio_outputs.output_embeds) is not None
                audio_mask = (input_ids == self.audio_token_id) | (input_ids >= self.text_model.config.vocab_size)
                audio_mask = audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.text_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        return QwenWithCausalAudioEncoderOutputWithPast(
            last_hidden_state=outputs[0],
            past_key_values=outputs.past_key_values,
            audio_past_key_values=audio_outputs.past_key_values if audio_outputs is not None else None,
        )

    def process_audio(
        self,
        audio: Optional[torch.Tensor | list[torch.Tensor]] = None,
        audio_sample_rate: Optional[int | list[int]] = None,
    ) -> BatchFeature:
        if audio is not None:
            if not isinstance(audio, list):
                audio = [audio]

            if audio_sample_rate is not None:
                if isinstance(audio_sample_rate, int):
                    audio_sample_rate = [audio_sample_rate] * len(audio)

                assert len(audio) == len(audio_sample_rate)
                for i in range(len(audio)):
                    if audio_sample_rate[i] != self.audio_encoder.config.sampling_rate:
                        audio[i] = torchaudio.functional.resample(
                            audio[i],
                            orig_freq=audio_sample_rate[i],
                            new_freq=self.audio_encoder.config.sampling_rate,
                        )

            audio_mask = pad_sequence(
                [torch.ones(a.shape, dtype=torch.long, device=a.device) for a in audio],
                batch_first=True,
            )
            audio = pad_sequence(audio, batch_first=True)
            assert audio.ndim == 2

            param = next(iter(self.audio_encoder.parameters()))
            audio = audio[:, None].to(param.device, param.dtype)

            if isinstance(self.audio_encoder, MimiModel):
                outputs = self.audio_encoder.encode(
                    audio,
                    padding_mask=audio_mask,
                    num_quantizers=8,
                )
            elif isinstance(self.audio_encoder, EncodecModel):
                outputs = self.audio_encoder.encode(
                    audio,
                    padding_mask=audio_mask,
                    bandwidth=self.audio_encoder.config.target_bandwidths[0],
                )
            else:
                raise ValueError

            assert isinstance(outputs, MimiEncoderOutput) or isinstance(outputs, EncodecEncoderOutput)
            assert outputs.audio_codes is not None
            audio_codes = outputs.audio_codes.view(outputs.audio_codes.shape[-3:]).transpose(1, 2)
            samples_per_code = int(self.audio_encoder.config.sampling_rate / self.audio_encoder.config._frame_rate)  # type: ignore
            assert samples_per_code * self.audio_encoder.config._frame_rate == self.audio_encoder.config.sampling_rate  # type: ignore
            padded_audio_mask = F.pad(audio_mask, (0, audio_codes.shape[1] * samples_per_code - audio_mask.shape[1]))
            audio_codes_mask = padded_audio_mask.view(-1, audio_codes.shape[1], samples_per_code).any(dim=-1)
        else:
            audio_codes = None
            audio_codes_mask = None

        return BatchFeature({"audio_codes": audio_codes, "audio_codes_mask": audio_codes_mask}).to(self.text_model.device)


class QwenWithCausalAudioEncoderForConditionalGeneration(PreTrainedModel):
    config: QwenWithCausalAudioEncoderConfig

    def __init__(self, config: QwenWithCausalAudioEncoderConfig) -> None:
        super().__init__(config)
        self.config = config
        self.vocab_size = config.text_model_config.vocab_size
        self.model = QwenWithCausalAudioEncoder._from_config(config)
        self.lm_head = nn.Linear(
            in_features=config.text_model_config.hidden_size,
            out_features=config.text_model_config.vocab_size,
            bias=False,
            dtype=self.model.dtype,
        )
        self.process_audio = self.model.process_audio

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
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        audio_use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> QwenWithCausalAudioEncoderOutputWithPast:
        outputs: QwenWithCausalAudioEncoderOutputWithPast = self.model(
            input_ids=input_ids,
            audio_codes=audio_codes,
            attention_mask=attention_mask,
            audio_codes_mask=audio_codes_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            audio_past_key_values=audio_past_key_values,
            inputs_embeds=inputs_embeds,
            audio_use_cache=audio_use_cache,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        logits = self.lm_head(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss = ForCausalLMLoss(logits=logits, labels=labels, vocab_size=self.vocab_size)

        return QwenWithCausalAudioEncoderOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            audio_past_key_values=outputs.audio_past_key_values,
        )

    def process_text(
        self,
        text: str | list[str],
        audio_codes: Optional[torch.Tensor] = None,
        audio_codes_mask: Optional[torch.Tensor] = None,
        audio_token: str = "<|audio_pad|>",
    ) -> str | list[str]:
        if audio_codes is not None:
            output_time_scale = self.model.adaptor.output_time_scale
            if audio_codes_mask is not None:
                audio_lengths = iter((output_time_scale * audio_codes_mask.sum(dim=1)).floor().long())
            else:
                audio_lengths = iter([int(output_time_scale * audio_codes.shape[1])])
        else:
            audio_lengths = iter([])

        if isinstance(text, list):
            is_batch = True
        else:
            is_batch = False
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

        return processed_text if is_batch else processed_text[0]

    @torch.inference_mode()
    def generate_greedy(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_codes: Optional[torch.Tensor] = None,
        audio_codes_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        eos_token_id: int = 151645,
    ) -> str | torch.Tensor:
        past_key_values = DynamicCache()
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_codes=audio_codes,
            audio_codes_mask=audio_codes_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

        input_seq_len = input_ids.shape[1]
        sequences = input_ids

        input_ids = outputs.logits[:, -1:].argmax(dim=-1)
        attention_mask = F.pad(attention_mask, (0, 1), value=1)
        sequences = torch.cat((sequences, input_ids), dim=1)

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

        return sequences


class QwenDuplexOutputWithPast(QwenWithCausalAudioEncoderOutputWithPast):
    audio_code_logits: Optional[torch.Tensor] = None


class QwenDuplexModel(QwenWithCausalAudioEncoder):
    config: QwenWithCausalAudioEncoderConfig

    def get_inputs_embeds(
        self,
        input_ids: Optional[torch.Tensor] = None,
        audio_codes: Optional[torch.Tensor] = None,
        audio_codes_mask: Optional[torch.Tensor] = None,
        audio_past_key_values: Optional[Cache] = None,
        audio_use_cache: Optional[bool] = None,
    ) -> tuple[torch.Tensor, Optional[Cache]]:
        inputs_embeds = None
        audio_past_key_values = None

        if input_ids is not None:
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids)

        if audio_codes is not None:
            batch_size = audio_codes.shape[0]
            num_audio_streams = audio_codes.shape[1] if audio_codes.ndim == 4 else 1
            input_seq_len = audio_codes.shape[-2]
            output_seq_len = int(input_seq_len * self.adaptor.output_time_scale)

            audio_codes = audio_codes.view(batch_size * num_audio_streams, input_seq_len, -1)
            if audio_codes_mask is not None:
                audio_codes_mask = audio_codes_mask.view(batch_size * num_audio_streams, input_seq_len)

            audio_outputs = self.get_audio_features(
                audio_codes=audio_codes,
                audio_codes_mask=audio_codes_mask,
                audio_past_key_values=audio_past_key_values,
                audio_use_cache=audio_use_cache,
            )
            assert (audio_inputs_embeds := audio_outputs.output_embeds) is not None
            audio_past_key_values = audio_outputs.past_key_values

            if audio_outputs.mask is not None:
                audio_inputs_embeds[audio_outputs.mask == 0].zero_()

            audio_inputs_embeds = audio_inputs_embeds.view(batch_size, num_audio_streams, output_seq_len, -1).sum(dim=1)
            if inputs_embeds is None:
                inputs_embeds = audio_inputs_embeds
            else:
                inputs_embeds += audio_inputs_embeds

        assert inputs_embeds is not None
        return inputs_embeds, audio_past_key_values

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
        use_cache: Optional[bool] = None,
        audio_use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> QwenWithCausalAudioEncoderOutputWithPast:
        if inputs_embeds is None:
            inputs_embeds, audio_past_key_values = self.get_inputs_embeds(
                input_ids=input_ids,
                audio_codes=audio_codes,
                audio_codes_mask=audio_codes_mask,
                audio_past_key_values=audio_past_key_values,
                audio_use_cache=audio_use_cache,
            )
        else:
            assert input_ids is None and audio_codes is None

        outputs = self.text_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        return QwenDuplexOutputWithPast(
            last_hidden_state=outputs[0],
            past_key_values=outputs.past_key_values,
            audio_past_key_values=audio_past_key_values,
        )


class QwenDuplexModelForCausalLM(PreTrainedModel):
    config: QwenWithCausalAudioEncoderConfig

    def __init__(self, config: QwenWithCausalAudioEncoderConfig) -> None:
        super().__init__(config)
        self.config = config
        self.vocab_size = config.text_model_config.vocab_size
        self.codebook_size = config.audio_encoder_config.codebook_size

        self.model = QwenDuplexModel(config)
        self.lm_head = nn.Linear(
            in_features=config.text_model_config.hidden_size,
            out_features=config.text_model_config.vocab_size,
            bias=False,
            dtype=self.model.dtype,
        )
        self.audio_code_head = nn.Linear(
            in_features=config.text_model_config.hidden_size,
            out_features=int(self.codebook_size / config.adaptor_config.output_time_scale),
            bias=False,
            dtype=self.model.dtype,
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
        labels: Optional[torch.Tensor] = None,
        audio_code_labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        audio_use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> QwenDuplexOutputWithPast:
        outputs: QwenDuplexOutputWithPast = self.model(
            input_ids=input_ids,
            audio_codes=audio_codes,
            attention_mask=attention_mask,
            audio_codes_mask=audio_codes_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            audio_past_key_values=audio_past_key_values,
            inputs_embeds=inputs_embeds,
            audio_use_cache=audio_use_cache,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        text_logits = self.lm_head(outputs.last_hidden_state)
        audio_code_logits = self.audio_code_head(outputs.last_hidden_state)
        audio_code_logits = audio_code_logits.view(audio_code_logits.shape[0], -1, self.codebook_size)

        loss = None
        if labels is not None:
            loss = ForCausalLMLoss(logits=text_logits, labels=labels, vocab_size=self.vocab_size)

        if audio_code_labels is not None:
            audio_code_loss = ForCausalLMLoss(logits=audio_code_logits, labels=audio_code_labels, vocab_size=self.codebook_size)
            if loss is None:
                loss = audio_code_loss
            else:
                loss += audio_code_loss

        return QwenDuplexOutputWithPast(
            loss=loss,
            logits=text_logits,
            audio_code_logits=audio_code_logits,
            past_key_values=outputs.past_key_values,
            audio_past_key_values=outputs.audio_past_key_values,
        )


def process_qwen_with_causal_audio_encoder_inputs(
    model: QwenWithCausalAudioEncoderForConditionalGeneration,
    processor: Qwen2_5OmniProcessor | Qwen3OmniMoeProcessor,
    conversation: list[dict] | list[list[dict]],
    audio: Optional[torch.Tensor | np.ndarray | list[torch.Tensor | np.ndarray]] = None,
    audio_sample_rate: Optional[int | list[int]] = None,
    add_generation_prompt: bool = False,
) -> BatchFeature:
    assert len(conversation) > 0
    if audio:
        if isinstance(audio, list):
            audio = [torch.from_numpy(a) if isinstance(a, np.ndarray) else a for a in audio]
        elif isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)

        audio_inputs = model.process_audio(audio=audio, audio_sample_rate=audio_sample_rate)  # type: ignore
    else:
        audio_inputs = {}

    texts = processor.apply_chat_template(conversation, add_generation_prompt=add_generation_prompt, tokenize=False)
    processed_texts = model.process_text(texts, **audio_inputs, audio_token=processor.audio_token)
    text_inputs = processor.tokenizer(  # type: ignore
        processed_texts,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    )
    return BatchFeature(text_inputs | audio_inputs).to(model.device)


def build_qwen_with_causal_audio_encoder_for_conditional_generation(
    text_model_name_or_path: str | PathLike,
    adaptor_config_path: str | PathLike,
    audio_encoder_name_or_path: str | PathLike = "kyutai/mimi",
    adaptor_state_dict_path: Optional[str | PathLike] = None,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
) -> QwenWithCausalAudioEncoderForConditionalGeneration:
    adaptor_config = EmbeddingAdaptorConfig.from_json_file(adaptor_config_path)
    adaptor_config.decoder_config._attn_implementation = attn_implementation

    if "mimi" in str(audio_encoder_name_or_path):
        audio_encoder = MimiModel.from_pretrained(
            audio_encoder_name_or_path,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )
    else:
        audio_encoder = EncodecModel.from_pretrained(audio_encoder_name_or_path, dtype=dtype)

    for param in audio_encoder.parameters():
        param.requires_grad = False

    adaptor = EmbeddingAdaptor._from_config(adaptor_config, dtype=dtype)
    if adaptor_state_dict_path is not None:
        adaptor.load_state_dict(torch.load(adaptor_state_dict_path, map_location="cpu"))

    if "Omni" in str(text_model_name_or_path):
        if "Qwen3" in str(text_model_name_or_path):
            thinker = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
                text_model_name_or_path,
                dtype=dtype,
                attn_implementation=attn_implementation,
            )
        else:
            thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                text_model_name_or_path,
                dtype=dtype,
                attn_implementation=attn_implementation,
            )
    else:
        thinker = AutoModelForCausalLM.from_pretrained(
            text_model_name_or_path,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )

    config = QwenWithCausalAudioEncoderConfig(
        audio_encoder_config=audio_encoder.config,
        adaptor_config=adaptor.config,
        text_model_config=thinker.model.config,
        audio_token_id=thinker.config.audio_token_id,
    )

    base_model = object.__new__(QwenWithCausalAudioEncoder)
    PreTrainedModel.__init__(base_model, config)
    base_model.config = config
    base_model.audio_encoder = audio_encoder
    base_model.adaptor = adaptor
    base_model.text_model = thinker.model
    base_model.audio_token_id = thinker.config.audio_token_id

    model = object.__new__(QwenWithCausalAudioEncoderForConditionalGeneration)
    PreTrainedModel.__init__(model, config)
    model.config = config
    model.model = base_model
    model.lm_head = thinker.lm_head
    model.process_audio = base_model.process_audio
    return model
