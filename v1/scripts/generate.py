"""Interactive text generation with a trained GPT-2 model.

Usage:
    python -m v1.scripts.generate
    python -m v1.scripts.generate --checkpoint v1/checkpoints/pretrained/best
    python -m v1.scripts.generate --temperature 0.5 --top-k 30
    python -m v1.scripts.generate --prompt "The gradient of"
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from v1.model.gpt2 import GPT2
from v1.utils.device import get_device


def load_model_and_tokenizer(checkpoint_path: str, device: torch.device):
    """Load model and tokenizer."""
    from bpe.tokenizer import Tokenizer

    model = GPT2.from_pretrained(checkpoint_path, device=device)
    model.to(device)
    model.eval()

    tok = Tokenizer.from_files(str(ROOT / "tokenizer_output"))
    return model, tok


def generate_text(
    model: GPT2,
    tokenizer,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    device: torch.device = None,
) -> str:
    """Generate text from a prompt."""
    # Encode prompt
    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    # Generate
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=tokenizer.special_token_to_id.get("<|endoftext|>", 278),
    )

    # Decode
    generated = tokenizer.decode(output_ids[0].tolist())
    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate text with GPT-2 v1")
    parser.add_argument("--checkpoint", type=str,
                        default=str(ROOT / "v1" / "checkpoints" / "pretrained" / "best"),
                        help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt (non-interactive mode)")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    device = get_device(force_cpu=args.force_cpu)
    print(f"Device: {device}")
    print(f"Loading model from {args.checkpoint}...")

    model, tok = load_model_and_tokenizer(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded ({n_params:,} params)")
    print()

    if args.prompt:
        # Single generation
        result = generate_text(
            model, tok, args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )
        print(result)
        return

    # Interactive REPL
    print("=" * 60)
    print("GPT-2 v1 — Interactive Generation")
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

        # Handle commands
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

        result = generate_text(
            model, tok, prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            device=device,
        )
        print()
        print(result)
        print()


if __name__ == "__main__":
    main()
