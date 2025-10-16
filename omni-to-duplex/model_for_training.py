from typing import Literal, Optional

import torch
from transformers import BatchFeature, PreTrainedModel

from model import QwenDuplexModelForCausalLM, QwenDuplexOutputWithPast, QwenWithCausalAudioEncoderConfig


class QwenDuplexModeForTraining(PreTrainedModel):
    config: QwenWithCausalAudioEncoderConfig

    def __init__(self, config: QwenWithCausalAudioEncoderConfig) -> None:
        super().__init__(config)
        self.config = config
        self.model = QwenDuplexModelForCausalLM(config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
        attention_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
        audio: Optional[torch.Tensor] = None,  # [batch_size, seq_len] (sampled at `config.audio_encoder_config.sampling_rate`)
        audio_lens: Optional[torch.Tensor] = None,  # [batch_size]
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> QwenDuplexOutputWithPast: ...

    def construct_input_sequences(
        self,
        input_ids: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
        attention_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
        audio: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
        audio_lengths: Optional[torch.Tensor] = None,  # [batch_size]
        labels: Optional[torch.Tensor] = None,
        mode: Literal["tts", "asr"] = "tts",
    ) -> BatchFeature:
        _ = self.model.process_audio(audio=audio, audio_lengths=audio_lengths)

        return BatchFeature()
