import json
import tarfile
import soundfile as sf
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Generator


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