"""Interactive text generation with MLX GPT-2 model.

Usage:
    python -m v1_mlx.scripts.generate
    python -m v1_mlx.scripts.generate --checkpoint v1_mlx/checkpoints/pretrained/best
    python -m v1_mlx.scripts.generate --prompt "The gradient of"
"""

import argparse
import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from v1_mlx.model.gpt2 import GPT2


def load_model_and_tokenizer(checkpoint_path: str):
    from bpe.tokenizer import Tokenizer

    model = GPT2.from_pretrained(checkpoint_path)
    tok = Tokenizer.from_files(str(ROOT / "tokenizer_output"))
    return model, tok


def generate_text(
    model: GPT2,
    tokenizer,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
) -> str:
    ids = tokenizer.encode(prompt)
    input_ids = mx.array([ids])

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=tokenizer.special_token_to_id.get("<|endoftext|>", 278),
    )

    return tokenizer.decode(output_ids[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Generate text with GPT-2 v1 (MLX)")
    parser.add_argument("--checkpoint", type=str,
                        default=str(ROOT / "v1_mlx" / "checkpoints" / "pretrained" / "best"))
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    print(f"Device: {mx.default_device()}")
    print(f"Loading model from {args.checkpoint}...")

    model, tok = load_model_and_tokenizer(args.checkpoint)
    n_params = sum(p.size for _, p in mx.utils.tree_flatten(model.parameters()))
    print(f"Model loaded ({n_params:,} params)")
    print()

    if args.prompt:
        result = generate_text(model, tok, args.prompt,
                               max_tokens=args.max_tokens,
                               temperature=args.temperature, top_k=args.top_k)
        print(result)
        return

    print("=" * 60)
    print("GPT-2 v1 (MLX) — Interactive Generation")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}, Max tokens: {args.max_tokens}")
    print("Type a prompt and press Enter. Type 'quit' to exit.")
    print("Commands: /temp <value>, /topk <value>, /max <value>")
    print("=" * 60)
    print()

    temperature = args.temperature
    top_k = args.top_k
    max_tokens = args.max_tokens

    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue
        if prompt.lower() == "quit":
            break

        if prompt.startswith("/temp "):
            temperature = float(prompt.split()[1])
            print(f"Temperature set to {temperature}")
            continue
        if prompt.startswith("/topk "):
            top_k = int(prompt.split()[1])
            print(f"Top-k set to {top_k}")
            continue
        if prompt.startswith("/max "):
            max_tokens = int(prompt.split()[1])
            print(f"Max tokens set to {max_tokens}")
            continue

        result = generate_text(model, tok, prompt,
                               max_tokens=max_tokens,
                               temperature=temperature, top_k=top_k)
        print()
        print(result)
        print()


if __name__ == "__main__":
    main()
