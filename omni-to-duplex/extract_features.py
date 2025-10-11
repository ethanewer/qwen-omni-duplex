import os
import argparse
import torch
import torch.distributed as dist

from pathlib import Path
from huggingface_hub import hf_hub_download
from torch.nn.utils.rnn import pad_sequence
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor  # type: ignore
from torch import nn
from transformers.activations import silu
from moshi.models import loaders, MimiModel
from dataclasses import dataclass
from data_util import iter_audio_samples, AudioSample, get_quantized_mimi_features, get_qwen_omni_features


def ddp_init():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    else:
        rank = 0
        world_size = 1
        local_rank = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]) if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return False, rank, world_size, 0


def ddp_barrier(is_dist: bool):
    if is_dist and dist.is_initialized():
        dist.barrier()


def ddp_cleanup(is_dist: bool):
    if is_dist and dist.is_initialized():
        dist.destroy_process_group()


def process_tar(
    tar_path: Path,
    out_dir: Path,
    mimi: MimiModel,
    qwen_omni: Qwen2_5OmniForConditionalGeneration,
    processor: Qwen2_5OmniProcessor,
    rank: int = 0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = tar_path.stem
    out_tmp = out_dir / f"{stem}.pt.tmp"
    out_final = out_dir / f"{stem}.pt"

    if out_final.exists():
        print(f"[rank {rank}] SKIP (cached): {tar_path}")
        return

    print(f"[rank {rank}] Processing: {tar_path}")

    items = []
    for sample in iter_audio_samples(tar_path):
        items.append(
            {
                "audio_src_id": sample.audio_src_id,
                "mimi": get_quantized_mimi_features(mimi, sample).cpu(),
                "qwen": get_qwen_omni_features(qwen_omni, processor, sample).cpu(),
            }
        )
        if len(items) % 100 == 0:
            torch.cuda.empty_cache()

    results = {"source_file": str(tar_path), "items": items}
    torch.save(results, out_tmp)
    os.replace(out_tmp, out_final)
    print(f"[rank {rank}] Wrote: {out_final} (items: {len(items)})")


def parse_args() -> tuple[Path, Path]:
    parser = argparse.ArgumentParser(description="Extract Mimi & Qwen features (resume-safe, multi-GPU).")
    fs_path = Path("/") / "mnt/efs/fs1"
    parser.add_argument(
        "--tts_data_path",
        type=Path,
        default=fs_path / "wbl/webdataset/webdataset/train/tts_en",
        help="Directory containing .tar files (one dataset shard per file).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=fs_path / "extracted_audio_features",
        help="Local directory for cached outputs (*.pt).",
    )
    args = parser.parse_args()
    return args.tts_data_path, args.out_dir


def main() -> None:
    tts_data_path, out_dir = parse_args()
    is_dist, rank, world_size, local_rank = ddp_init()
    if rank == 0:
        print(f"[world_size={world_size}] Starting extraction to: {out_dir.resolve()}")

    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")
    qwen_omni = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-3B",
        torch_dtype=torch.bfloat16,
        device_map="auto" if not dist.is_initialized() else {"": torch.device(f"cuda:{local_rank}")},
    )
    qwen_omni.disable_talker()
    qwen_omni.eval()

    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device=f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu").to(torch.bfloat16)
    mimi.set_num_codebooks(8)
    mimi.eval()

    ddp_barrier(is_dist)

    if rank == 0:
        all_files = sorted([p for p in tts_data_path.iterdir() if p.is_file() and p.suffix in [".tar"]])
    else:
        all_files = None

    if is_dist:
        obj_list = [all_files] if rank == 0 else [None]
        dist.broadcast_object_list(obj_list, src=0)
        all_files = obj_list[0]

    if not all_files:
        if rank == 0:
            print("No input .tar files found. Exiting.")

        ddp_cleanup(is_dist)
        return

    for idx in range(rank, len(all_files), world_size):
        tar_path = all_files[idx]
        try:
            process_tar(
                tar_path=tar_path,
                out_dir=out_dir,
                mimi=mimi,
                qwen_omni=qwen_omni,
                processor=processor,
                rank=rank,
            )
        except Exception as e:
            print(f"[rank {rank}] ERROR processing {tar_path}: {e}", flush=True)

        ddp_barrier(is_dist)

    ddp_cleanup(is_dist)
    if rank == 0:
        print("All file completed.")


if __name__ == "__main__":
    main()