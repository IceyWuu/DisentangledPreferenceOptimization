"""
Merge harmless-base and helpful-base from Anthropic hh-rlhf into a single
dataset in OpenAI message format, saved as HuggingFace `load_from_disk`
compatible splits (train / test).

Output is compatible with DPO/aexperiment_mistral_7b.sh --dataset hh-rlhf-merged.

Usage (from DPO directory):
    python hh-rlhf-dataset-transformer/merge_and_convert.py
"""

import json
import re
import hashlib
from pathlib import Path
from datasets import Dataset, DatasetDict

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parent.parent / "ModelAndDatasets" / "Anthropic___hh-rlhf"
HARMLESS_DIR = DATA_ROOT / "harmless-base"
HELPFUL_DIR = DATA_ROOT / "helpful-base"
OUTPUT_DIR = DATA_ROOT / "helpful_harmless_merged"

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


def extract_prompt(messages: list[dict]) -> str:
    """Extract user turns as the prompt (everything before the final assistant reply)."""
    user_turns = [m["content"] for m in messages if m["role"] == "user"]
    return "\n".join(user_turns)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def convert_rows(rows: list[dict]) -> list[dict]:
    converted = []
    for row in rows:
        chosen_msgs = to_openai_messages(row["chosen"])
        rejected_msgs = to_openai_messages(row["rejected"])
        if not valid_messages(chosen_msgs) or not valid_messages(rejected_msgs):
            continue
        prompt_text = extract_prompt(chosen_msgs)
        prompt_id = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        converted.append({
            "prompt": prompt_text,
            "prompt_id": prompt_id,
            "chosen": chosen_msgs,
            "rejected": rejected_msgs,
        })
    return converted


def main():
    print(f"[INFO] Loading harmless-base from {HARMLESS_DIR}")
    print(f"[INFO] Loading helpful-base from {HELPFUL_DIR}")

    train_rows = []
    test_rows = []

    for source_dir, label in [(HARMLESS_DIR, "harmless"), (HELPFUL_DIR, "helpful")]:
        train_path = source_dir / "train.jsonl"
        test_path = source_dir / "test.jsonl"
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"Missing {train_path} or {test_path}")

        raw_train = load_jsonl(train_path)
        raw_test = load_jsonl(test_path)
        print(f"  [{label}] raw train: {len(raw_train)}, raw test: {len(raw_test)}")

        train_rows.extend(raw_train)
        test_rows.extend(raw_test)

    print(f"[INFO] Combined raw: train={len(train_rows)}, test={len(test_rows)}")

    train_converted = convert_rows(train_rows)
    test_converted = convert_rows(test_rows)
    print(f"[INFO] After conversion & filtering: train={len(train_converted)}, test={len(test_converted)}")

    train_ds = Dataset.from_list(train_converted)
    test_ds = Dataset.from_list(test_converted)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_ds.save_to_disk(str(OUTPUT_DIR / "train"))
    test_ds.save_to_disk(str(OUTPUT_DIR / "test"))

    format_marker = OUTPUT_DIR / ".format_version"
    format_marker.write_text("v2")

    print(f"[INFO] Saved to {OUTPUT_DIR}")
    print(f"  train: {len(train_ds)} rows, columns: {train_ds.column_names}")
    print(f"  test:  {len(test_ds)} rows, columns: {test_ds.column_names}")
    print()
    print("Sample (train[0]):")
    sample = train_ds[0]
    print(f"  prompt: {sample['prompt'][:100]}...")
    print(f"  chosen ({len(sample['chosen'])} msgs): {sample['chosen'][0]}")
    print(f"  rejected ({len(sample['rejected'])} msgs): {sample['rejected'][0]}")
    print()
    print("Usage: DATASET_CHOICE=hh-rlhf-merged bash aexperiment_mistral_7b.sh")


if __name__ == "__main__":
    main()
