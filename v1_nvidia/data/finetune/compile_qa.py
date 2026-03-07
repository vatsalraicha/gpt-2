"""Compile Q&A review markdown files into training JSONL.

Reads all qa_batch*_review.md files and produces:
- qa_train.jsonl — training data with special token format
- qa_stats.json — summary statistics

Format per line:
    <|startoftext|>Question: {question}\n\nAnswer: {answer}<|endoftext|>
"""

import json
import re
from pathlib import Path


def parse_qa_markdown(md_path: Path) -> list[dict]:
    """Parse Q&A pairs from a review markdown file."""
    text = md_path.read_text()
    pairs = []

    # Match Q/A blocks: **Q{n}.** ... **A{n}.** ...
    pattern = r'\*\*Q(\d+)\.\*\*\s*(.+?)\n\n\*\*A\d+\.\*\*\s*(.+?)(?=\n\n---|\n\n\*End of Batch|\Z)'
    matches = re.findall(pattern, text, re.DOTALL)

    for qnum, question, answer in matches:
        question = question.strip()
        answer = answer.strip()
        # Clean up any markdown artifacts
        answer = re.sub(r'\n\n+', '\n\n', answer)
        pairs.append({
            "id": int(qnum),
            "question": question,
            "answer": answer,
        })

    return pairs


def format_for_training(pair: dict) -> str:
    """Format a Q&A pair for training."""
    return f"<|startoftext|>Question: {pair['question']}\n\nAnswer: {pair['answer']}<|endoftext|>"


def main():
    data_dir = Path(__file__).parent
    batch_files = sorted(data_dir.glob("qa_batch*_review.md"))

    if not batch_files:
        print("No batch files found!")
        return

    all_pairs = []
    for bf in batch_files:
        pairs = parse_qa_markdown(bf)
        print(f"  {bf.name}: {len(pairs)} pairs")
        all_pairs.extend(pairs)

    # Deduplicate by ID
    seen = set()
    unique_pairs = []
    for p in all_pairs:
        if p["id"] not in seen:
            seen.add(p["id"])
            unique_pairs.append(p)

    # Sort by ID
    unique_pairs.sort(key=lambda x: x["id"])

    # Write training JSONL
    train_path = data_dir / "qa_train.jsonl"
    with open(train_path, "w") as f:
        for pair in unique_pairs:
            record = {
                "id": pair["id"],
                "text": format_for_training(pair),
                "question": pair["question"],
                "answer": pair["answer"],
            }
            f.write(json.dumps(record) + "\n")

    # Compute stats
    texts = [format_for_training(p) for p in unique_pairs]
    lengths = [len(t.split()) for t in texts]
    stats = {
        "total_pairs": len(unique_pairs),
        "total_words": sum(lengths),
        "avg_words_per_pair": sum(lengths) / len(lengths),
        "min_words": min(lengths),
        "max_words": max(lengths),
        "batch_files": [bf.name for bf in batch_files],
    }

    stats_path = data_dir / "qa_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nCompiled {stats['total_pairs']} Q&A pairs")
    print(f"  Total words: {stats['total_words']:,}")
    print(f"  Avg words/pair: {stats['avg_words_per_pair']:.0f}")
    print(f"  Range: {stats['min_words']}–{stats['max_words']} words")
    print(f"  Training data: {train_path}")
    print(f"  Stats: {stats_path}")


if __name__ == "__main__":
    main()
