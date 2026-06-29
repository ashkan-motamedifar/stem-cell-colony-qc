import hashlib
import sys
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

URL = "https://zenodo.org/api/records/7316404/files/mamaeva_et_al_2022_online_data.tar.gz/content"
EXPECTED_MD5 = "a7aa69e95d9ce4a56e25b61356d685ec"
PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "raw"
ARCHIVE = RAW / "mamaeva_2022.tar.gz"
IMAGE_DIR = RAW / "H9p36"


def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    resume_byte = dst.stat().st_size if dst.exists() else 0
    headers = {"Range": f"bytes={resume_byte}-"} if resume_byte else {}
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) + resume_byte
        with open(dst, "ab" if resume_byte else "wb") as f, tqdm(
            total=total, initial=resume_byte, unit="B", unit_scale=True, desc=dst.name
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def main() -> int:
    if IMAGE_DIR.exists() and any(IMAGE_DIR.glob("*.png")):
        print(f"Dataset already extracted at {IMAGE_DIR} — nothing to do.")
        return 0

    if not ARCHIVE.exists() or md5_of_file(ARCHIVE) != EXPECTED_MD5:
        print(f"Downloading dataset from Zenodo to {ARCHIVE}...")
        download(URL, ARCHIVE)
        actual = md5_of_file(ARCHIVE)
        if actual != EXPECTED_MD5:
            print(f"MD5 mismatch: expected {EXPECTED_MD5}, got {actual}", file=sys.stderr)
            return 1

    print(f"Extracting to {RAW}...")
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        tar.extractall(RAW)
    n = len(list(IMAGE_DIR.glob("*.png")))
    print(f"Done. {n} images at {IMAGE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
