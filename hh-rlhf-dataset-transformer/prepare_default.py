"""
Convert the default (full) Anthropic hh-rlhf dataset from raw arrow files
into OpenAI message format, saved as HuggingFace `load_from_disk` compatible
splits (train / test).

Source: ModelAndDatasets/Anthropic___hh-rlhf/default/0.0.0/<hash>/hh-rlhf-{train,test}.arrow
Output: ModelAndDatasets/Anthropic___hh-rlhf/local_disk/{train,test}

Usage (from DIL directory):
    python hh-rlhf-dataset-transformer/prepare_default.py
"""

import re
import sys
from pathlib import Path
from datasets import Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parent.parent / "ModelAndDatasets" / "Anthropic___hh-rlhf"
SOURCE_ROOT = DATA_ROOT / "default" / "0.0.0"
OUTPUT_DIR = DATA_ROOT / "local_disk"
FORMAT_VERSION = "v2"

_TURN_PATTERN = re.compile(r"(?:^|\n\n)(Human|Assistant):")


def to_openai_messages(text: str) -> list[dict]:
    if not isinstance(text, str):
        return []
    text = text.replace("\r\n", "\n").strip()
    matches = list(_TURN_PATTERN.finditer(text))
    if not matches:
        return []
    messages = []
    for i, m in enumerate(matches):
        speaker = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if not content:
            continue
        role = "user" if speaker == "Human" else "assistant"
        messages.append({"role": role, "content": content})
    return messages


def valid_messages(messages: list) -> bool:
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    if not all(isinstance(m, dict) and "role" in m and "content" in m for m in messages):
        return False
    if not any(m["role"] == "user" for m in messages):
        return False
    if messages[-1]["role"] != "assistant":
        return False
    return True


def is_already_prepared() -> bool:
    """Check if output already exists with correct format."""
    format_marker = OUTPUT_DIR / ".format_version"
    if not (OUTPUT_DIR / "train").is_dir() or not (OUTPUT_DIR / "test").is_dir():
        return False
    if not format_marker.exists() or format_marker.read_text().strip() != FORMAT_VERSION:
        return False
    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(OUTPUT_DIR / "train"))
        sample = ds[0]["chosen"] if len(ds) > 0 else None
        return (
            isinstance(sample, list)
            and len(sample) > 0
            and isinstance(sample[0], dict)
            and "role" in sample[0]
            and "content" in sample[0]
        )
    except Exception:
        return False


def find_arrow_files() -> tuple[Path, Path]:
    """Locate hh-rlhf-train.arrow and hh-rlhf-test.arrow under SOURCE_ROOT."""
    subdirs = [d for d in SOURCE_ROOT.iterdir() if d.is_dir()]
    if not subdirs:
        print(f"[ERROR] No subdirectory found under: {SOURCE_ROOT}", file=sys.stderr)
        sys.exit(1)

    cache_dir = subdirs[0]
    train_arrow = cache_dir / "hh-rlhf-train.arrow"
    test_arrow = cache_dir / "hh-rlhf-test.arrow"
    if not train_arrow.exists() or not test_arrow.exists():
        print(f"[ERROR] Missing arrow files in: {cache_dir}", file=sys.stderr)
        sys.exit(1)

    return train_arrow, test_arrow


def main():
    if is_already_prepared():
        print(f"[INFO] hh-rlhf local_disk already prepared at: {OUTPUT_DIR}")
        return

    if (OUTPUT_DIR / "train").exists() or (OUTPUT_DIR / "test").exists():
        print("[INFO] Existing local_disk is not in OpenAI-message format. Rebuilding...")
        import shutil
        shutil.rmtree(OUTPUT_DIR / "train", ignore_errors=True)
        shutil.rmtree(OUTPUT_DIR / "test", ignore_errors=True)

    train_arrow, test_arrow = find_arrow_files()
    print(f"[INFO] Source train: {train_arrow}")
    print(f"[INFO] Source test:  {test_arrow}")

    convert_pair = lambda ex: {
        "chosen": to_openai_messages(ex["chosen"]),
        "rejected": to_openai_messages(ex["rejected"]),
    }
    keep_valid = lambda ex: valid_messages(ex["chosen"]) and valid_messages(ex["rejected"])

    train_ds = Dataset.from_file(str(train_arrow))
    test_ds = Dataset.from_file(str(test_arrow))
    print(f"[INFO] Raw: train={len(train_ds)}, test={len(test_ds)}")

    train_ds = train_ds.map(convert_pair, desc="Converting train to OpenAI messages")
    test_ds = test_ds.map(convert_pair, desc="Converting test to OpenAI messages")
    train_ds = train_ds.filter(keep_valid, desc="Filtering invalid train pairs")
    test_ds = test_ds.filter(keep_valid, desc="Filtering invalid test pairs")
    print(f"[INFO] After filtering: train={len(train_ds)}, test={len(test_ds)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_ds.save_to_disk(str(OUTPUT_DIR / "train"))
    test_ds.save_to_disk(str(OUTPUT_DIR / "test"))
    (OUTPUT_DIR / ".format_version").write_text(FORMAT_VERSION)

    print(f"[INFO] Saved to {OUTPUT_DIR}")
    print(f"  train: {len(train_ds)} rows, columns: {train_ds.column_names}")
    print(f"  test:  {len(test_ds)} rows, columns: {test_ds.column_names}")

    sample = train_ds[0]
    print(f"\nSample (train[0]):")
    print(f"  chosen ({len(sample['chosen'])} msgs): {sample['chosen'][0]}")
    print(f"  rejected ({len(sample['rejected'])} msgs): {sample['rejected'][0]}")


if __name__ == "__main__":
    main()
