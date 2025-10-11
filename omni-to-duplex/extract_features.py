import argparse
import os
import torch

from pathlib import Path
from huggingface_hub import hf_hub_download
from moshi.models import loaders, MimiModel
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor  # type: ignore
from tqdm import tqdm

from data_util import load_audio_samples, get_quantized_mimi_features, get_qwen_omni_features




def process_tar(
    tar_path: Path,
    out_dir: Path,
    mimi: MimiModel,
    qwen_omni: Qwen2_5OmniForConditionalGeneration,
    processor: Qwen2_5OmniProcessor,
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
    for key, sample in tqdm(samples.items(), desc="Encoding Mimi"):
        try:
            mimi_features[key] = get_quantized_mimi_features(mimi, sample).cpu()
        except Exception as e:
            print(f"ERROR processing {tar_path}: {e}", flush=True)
            continue

    qwen_omni_features = {}
    for key, sample in tqdm(samples.items(), desc="Encoding Qwen-Omni"):
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
            } for key in sorted(set(mimi_features.keys()) & set(qwen_omni_features.keys()))
        ],
    }

    torch.cuda.empty_cache()
    torch.save(results, out_tmp)
    os.replace(out_tmp, out_final)
    print(f"Wrote {len(results["items"])} to {out_final}.")


def parse_args() -> tuple[Path, Path]:
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
    args = parser.parse_args()
    return args.tts_data_path, args.out_dir


def main() -> None:
    tts_data_path, out_dir = parse_args()
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")
    qwen_omni = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-3B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    qwen_omni.disable_talker()
    qwen_omni.eval()

    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device="cuda" if torch.cuda.is_available() else "cpu").to(torch.bfloat16)
    mimi.set_num_codebooks(8)
    mimi.eval()

    data_files = sorted([p for p in tts_data_path.iterdir() if p.is_file() and p.suffix in [".tar"]])

    for tar_path in tqdm(data_files, desc="Processing files"):
        try:
            process_tar(
                tar_path=tar_path,
                out_dir=out_dir,
                mimi=mimi,
                qwen_omni=qwen_omni,
                processor=processor,
            )
        except Exception as e:
            print(f"ERROR processing {tar_path}: {e}", flush=True)


if __name__ == "__main__":
    main()