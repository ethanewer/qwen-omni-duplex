import os
from pathlib import Path

def main():
    print("Hello from omni-to-duplex!")
    path = Path("/") / "mnt/efs/fs1/wbl/webdataset/webdataset/train/tts_en"
    print(path)
    print(os.listdir(path))


if __name__ == "__main__":
    main()
