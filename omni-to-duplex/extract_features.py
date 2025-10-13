import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import (
    MimiModel,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)

from data_util import get_quantized_mimi_features, get_qwen_omni_features, load_audio_samples


def process_tar(
    tar_path: Path,
    out_dir: Path,
    mimi: MimiModel,
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
    for key, sample in tqdm(samples.items(), desc="Encoding Mimi"):
        # try:
        #     mimi_features[key] = get_quantized_mimi_features(mimi, sample).cpu()
        # except Exception as e:
        #     print(f"ERROR processing {tar_path}: {e}", flush=True)
        #     continue
        mimi_features[key] = get_quantized_mimi_features(mimi, sample).cpu()

    qwen_omni_features = {}
    for key, sample in tqdm(samples.items(), desc="Encoding Qwen-Omni"):
        # try:
        #     qwen_omni_features[key] = get_qwen_omni_features(qwen_omni, processor, sample).cpu()
        # except Exception as e:
        #     print(f"ERROR processing {tar_path} ({key}): {e}", flush=True)
        #     continue
        qwen_omni_features[key] = get_qwen_omni_features(qwen_omni, processor, sample).cpu()

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


def parse_args() -> tuple[Path, Path, str]:
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
        default="Qwen/Qwen2.5-Omni-3B",
    )
    args = parser.parse_args()
    return args.tts_data_path, args.out_dir / args.model_name_or_path, args.model_name_or_path


def main() -> None:
    tts_data_path, out_dir, model_name_or_path = parse_args()
    print(f"{tts_data_path=}")
    print(f"{out_dir=}")
    print(f"{model_name_or_path=}")
    if "Qwen2.5" in model_name_or_path:
        processor = Qwen2_5OmniProcessor.from_pretrained(model_name_or_path)
        qwen_omni = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_name_or_path,
            dtype=torch.bfloat16,
            device_map="cuda:0",
        )
    elif "Qwen3" in model_name_or_path:
        processor = Qwen3OmniMoeProcessor.from_pretrained(model_name_or_path)
        qwen_omni = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_name_or_path,
            dtype=torch.bfloat16,
            device_map="cuda:0",
        )
    else:
        raise NotImplementedError

    qwen_omni.disable_talker()
    qwen_omni.eval()

    mimi = MimiModel.from_pretrained(
        "kyutai/mimi",
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    mimi.eval()

    data_files = sorted([p for p in tts_data_path.iterdir() if p.is_file() and p.suffix in [".tar"]])

    for tar_path in tqdm(data_files, desc="Processing files"):
        # try:
        #     process_tar(
        #         tar_path=tar_path,
        #         out_dir=out_dir,
        #         mimi=mimi,
        #         qwen_omni=qwen_omni,
        #         processor=processor,  # type: ignore
        #     )
        # except Exception as e:
        #     print(f"ERROR processing {tar_path}: {e}", flush=True)
        process_tar(
            tar_path=tar_path,
            out_dir=out_dir,
            mimi=mimi,
            qwen_omni=qwen_omni,
            processor=processor,  # type: ignore
        )


if __name__ == "__main__":
    main()
