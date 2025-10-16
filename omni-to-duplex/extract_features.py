import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import (
    EncodecModel,
    MimiModel,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)

from data_util import get_quantized_audio_features, get_qwen_omni_features, load_audio_samples


def process_tar(
    tar_path: Path,
    out_dir: Path,
    audio_encoder: MimiModel | EncodecModel,
    qwen_omni: Qwen2_5OmniForConditionalGeneration | Qwen3OmniMoeForConditionalGeneration,
    processor: Qwen2_5OmniProcessor | Qwen3OmniMoeProcessor,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = tar_path.stem
    out_tmp = out_dir / f"{stem}.pt.tmp"
    out_final = out_dir / f"{stem}.pt"

    if out_final.exists():
        print(f"Skipping {tar_path} (cached)")
        return

    samples = load_audio_samples(tar_path)

    mimi_features = {}
    for key, sample in tqdm(samples.items(), desc="Audio Encoder"):
        try:
            mimi_features[key] = get_quantized_audio_features(audio_encoder, sample).cpu()
        except Exception as e:
            print(f"ERROR processing {tar_path}: {e}", flush=True)
            continue

    qwen_omni_features = {}
    for key, sample in tqdm(samples.items(), desc="Qwen-Omni"):
        try:
            qwen_omni_features[key] = get_qwen_omni_features(qwen_omni, processor, sample).cpu()
        except Exception as e:
            print(f"ERROR processing {tar_path} ({key}): {e}", flush=True)
            continue

    results = {
        "source_file": str(tar_path),
        "items": [
            {
                "mimi_features": mimi_features[key],
                "qwen_omni_features": qwen_omni_features[key],
                "audio_src_id": samples[key].audio_src_id,
            }
            for key in sorted(set(mimi_features.keys()) & set(qwen_omni_features.keys()))
        ],
    }

    torch.cuda.empty_cache()
    torch.save(results, out_tmp)
    os.replace(out_tmp, out_final)
    print(f"Wrote {len(results['items'])} to {out_final}.")


def parse_args() -> tuple[Path, Path, str, str]:
    parser = argparse.ArgumentParser(description="Extract Mimi & Qwen features (resume-safe, multi-GPU).")
    fs_path = Path("/") / "mnt/efs/fs1"
    parser.add_argument(
        "--tts_data_path",
        type=Path,
        default=fs_path / "wbl/webdataset/webdataset/train/tts_en",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=fs_path / "extracted_audio_features",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    )
    parser.add_argument(
        "--audio_encoder_name_or_path",
        type=str,
        default="maitrix-org/Voila-Tokenizer",
    )
    args = parser.parse_args()
    return args.tts_data_path, args.out_dir, args.model_name_or_path, args.audio_encoder_name_or_path


def main() -> None:
    tts_data_path, out_dir, model_name_or_path, audio_encoder_name_or_path = parse_args()
    out_dir = out_dir / f"{audio_encoder_name_or_path.split('/')[-1]}-to-{model_name_or_path.split('/')[-1]}"

    print(f"{tts_data_path=}")
    print(f"{out_dir=}")
    print(f"{model_name_or_path=}")
    print(f"{audio_encoder_name_or_path=}")

    if "Qwen2.5" in model_name_or_path:
        processor = Qwen2_5OmniProcessor.from_pretrained(model_name_or_path)
        qwen_omni = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_name_or_path,
            dtype=torch.bfloat16,
            device_map="auto",
        )
    elif "Qwen3" in model_name_or_path:
        processor = Qwen3OmniMoeProcessor.from_pretrained(model_name_or_path)
        qwen_omni = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_name_or_path,
            dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        raise NotImplementedError

    qwen_omni.disable_talker()
    qwen_omni.eval()

    if "mimi" in str(audio_encoder_name_or_path):
        audio_encoder = MimiModel.from_pretrained(audio_encoder_name_or_path, dtype=torch.bfloat16, device_map="cuda:0")
    else:
        audio_encoder = EncodecModel.from_pretrained(audio_encoder_name_or_path, dtype=torch.bfloat16, device_map="cuda:0")

    audio_encoder.eval()

    data_files = sorted(
        [p for p in tts_data_path.iterdir() if p.is_file() and p.suffix in [".tar"]],
        key=lambda p: str(p)[::-1],
        reverse=True,
    )

    for tar_path in tqdm(data_files, desc="Processing files"):
        try:
            process_tar(
                tar_path=tar_path,
                out_dir=out_dir,
                audio_encoder=audio_encoder,
                qwen_omni=qwen_omni,
                processor=processor,  # type: ignore
            )
        except Exception as e:
            print(f"ERROR processing {tar_path}: {e}", flush=True)


if __name__ == "__main__":
    main()
