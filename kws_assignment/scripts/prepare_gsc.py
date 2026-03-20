import argparse
import csv
import random
from pathlib import Path


TARGET_WORDS = [
    "yes", "no", "up", "down", "left",
    "right", "on", "off", "stop", "go"
]


def read_list(path: Path):
    with open(path, "r") as f:
        return set(line.strip() for line in f if line.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--unknown_pct", type=float, default=10.0)
    parser.add_argument("--silence_pct", type=float, default=10.0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_list = read_list(root / "validation_list.txt")
    test_list = read_list(root / "testing_list.txt")

    known = {"train": [], "valid": [], "test": []}
    unknown = {"train": [], "valid": [], "test": []}

    for wav_path in root.rglob("*.wav"):
        if "_background_noise_" in wav_path.parts:
            continue

        rel = wav_path.relative_to(root).as_posix()
        word = wav_path.parent.name

        if rel in valid_list:
            split = "valid"
        elif rel in test_list:
            split = "test"
        else:
            split = "train"

        if word in TARGET_WORDS:
            known[split].append({"path": str(wav_path.resolve()), "label": word})
        else:
            unknown[split].append({"path": str(wav_path.resolve()), "label": "unknown"})

    for split in ["train", "valid", "test"]:
        rng.shuffle(known[split])
        rng.shuffle(unknown[split])

        num_known = len(known[split])
        num_unknown = int(round(num_known * args.unknown_pct / 100.0))
        num_silence = int(round(num_known * args.silence_pct / 100.0))

        rows = []
        rows.extend(known[split])
        rows.extend(unknown[split][:num_unknown])
        rows.extend([{"path": "__silence__", "label": "silence"} for _ in range(num_silence)])
        rng.shuffle(rows)

        out_csv = out_dir / f"{split}.csv"
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label"])
            writer.writeheader()
            writer.writerows(rows)

        print(
            f"{split}: "
            f"known={num_known}, "
            f"unknown={num_unknown}, "
            f"silence={num_silence}, "
            f"total={len(rows)}"
        )
        print(f"saved -> {out_csv}")


if __name__ == "__main__":
    main()