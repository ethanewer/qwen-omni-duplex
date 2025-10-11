import json
import tarfile
import soundfile as sf
import numpy as np
import torch
import torchaudio
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Generator, Sequence, Iterator
from moshi.models import MimiModel
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor  # type: ignore
from qwen_omni_utils import process_mm_info
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import torch.distributed as dist
import os


@dataclass
class AudioSample:
    audio: np.ndarray
    audio_sample_rate: int
    audio_length: int
    audio_src_id: str
    text: str
    speaker_age: Optional[int]
    speaker_emotion: Optional[str]
    speaker_accent: Optional[str]
    speaker_gender: Optional[str]
    speaker_name: Optional[str]

    def to_tensor(self, new_sample_rate: Optional[int] = None) -> torch.Tensor:
        waveform = torch.from_numpy(self.audio)
        if new_sample_rate is not None and new_sample_rate != self.audio_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=self.audio_sample_rate,
                new_freq=new_sample_rate,
            )
        
        return waveform

    def save_wav(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            sf.write(f, self.audio, self.audio_sample_rate) 

    @property
    def duration(self) -> float:
        return self.audio_length / self.audio_sample_rate


def load_audio_samples(path: str | Path) -> dict[str, AudioSample]:
    with tarfile.open(path, "r") as tar:
        json_members = [m for m in tar.getmembers() if m.name.endswith(".json")]
        wav_members = [m for m in tar.getmembers() if m.name.endswith(".wav")]
        unstructured_data = defaultdict(dict)

        for json_member in json_members:
            with tar.extractfile(json_member) as f:  # type: ignore
                key = json_member.name.replace(".json", "")
                unstructured_data[key] |= json.load(f)

        for wav_member in wav_members:
            with tar.extractfile(wav_member) as f:  # type: ignore
                key = wav_member.name.replace(".wav", "")
                audio, sr = sf.read(f, dtype="float32")  # type: ignore
                unstructured_data[key]["audio"] = audio
                assert sr == unstructured_data[key]["audio_sample_rate"]
                assert audio.ndim == 1 and len(audio) == unstructured_data[key]["audio_length"]
        
    return {k: AudioSample(**v) for k, v in unstructured_data.items()}


def iter_audio_samples(path: str | Path) -> Generator[AudioSample, None, None]:
    with tarfile.open(path, "r") as tar:
        json_members = [m for m in tar.getmembers() if m.name.endswith(".json")]
        wav_members = [m for m in tar.getmembers() if m.name.endswith(".wav")]
        unstructured_data = defaultdict(dict)

        for json_member in json_members:
            with tar.extractfile(json_member) as f:  # type: ignore
                key = json_member.name.replace(".json", "")
                unstructured_data[key] |= json.load(f)

        for wav_member in wav_members:
            with tar.extractfile(wav_member) as f:  # type: ignore
                key = wav_member.name.replace(".wav", "")
                audio, sr = sf.read(f, dtype="float32")  # type: ignore
                unstructured_data[key]["audio"] = audio
                assert sr == unstructured_data[key]["audio_sample_rate"]
                assert audio.ndim == 1 and len(audio) == unstructured_data[key]["audio_length"]
                yield AudioSample(**unstructured_data[key])


@torch.inference_mode()
def get_quantized_mimi_features(mimi: MimiModel, sample: AudioSample) -> torch.Tensor:
    param = next(iter(mimi.parameters()))
    inputs = sample.to_tensor(new_sample_rate=24000).to(param.device, param.dtype)
    with torch.no_grad():
        return mimi.encode_to_latent(inputs[None, None])[0].T


@torch.inference_mode()
def get_qwen_omni_features(
    qwen_omni: Qwen2_5OmniForConditionalGeneration, 
    processor: Qwen2_5OmniProcessor,
    sample: AudioSample,
) -> torch.Tensor:
    param = next(iter(qwen_omni.parameters()))
    conversation = [{"role": "user", "content": [{"type": "audio", "audio": sample.audio}]}]
    audios, _, _ = process_mm_info(conversation, use_audio_in_video=True)  # type: ignore
    inputs = processor(text="", audio=audios, return_tensors="pt", padding=True, use_audio_in_video=True)
    inputs = inputs.to(param.device, param.dtype)
    with torch.no_grad():
        return qwen_omni.thinker.get_audio_features(inputs.input_features, inputs.feature_attention_mask)
    