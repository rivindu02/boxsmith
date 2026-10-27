"""Download the small public baseline, verifying its pinned SHA-256 digest."""

import argparse
import hashlib
from pathlib import Path
import urllib.request

CHECKPOINT_URL = "https://github.com/rivindu02/boxsmith/releases/download/v0.1.0/best.pt"
CHECKPOINT_SHA256 = "61adda73503b24aeaaf0db8466ce3ef38ec2872457f65e6d66ec42d6c6d541fc"


def download(output, url=CHECKPOINT_URL, expected_hash=CHECKPOINT_SHA256):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if hashlib.sha256(output.read_bytes()).hexdigest() == expected_hash:
            return output
        raise FileExistsError(f"Refusing to overwrite different checkpoint: {output}")
    # Download in memory first: the released model is under 1 MB; enforce a 4 MB cap.
    with urllib.request.urlopen(url, timeout=60) as response:
        content = response.read(4 * 1024 * 1024 + 1)
    if len(content) > 4 * 1024 * 1024 or hashlib.sha256(content).hexdigest() != expected_hash:
        raise ValueError("Checkpoint size or SHA-256 verification failed")
    with output.open("xb") as destination:
        destination.write(content)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="checkpoints/best.pt")
    args = parser.parse_args()
    print(download(args.output))


if __name__ == "__main__":
    main()
