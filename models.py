"""Utility to fetch and filter OpenRouter models."""

import csv
import json
import urllib.request


MODELS_URL = "https://openrouter.ai/api/v1/models"


def fetch_models() -> list[dict]:
    """Fetch all models from OpenRouter. Returns list of model dicts."""
    with urllib.request.urlopen(MODELS_URL) as resp:
        data = json.loads(resp.read())
    return data["data"]


def filter_models(
    models: list[dict],
    *,
    max_prompt_cost: float | None = None,
    max_completion_cost: float | None = None,
    reasoning_only: bool = False,
    text_only: bool = False,
) -> list[dict]:
    """Filter models by cost and capabilities.

    Args:
        max_prompt_cost: Max USD per input token (e.g. 0.000003 = $3/M tokens).
        max_completion_cost: Max USD per output token.
        reasoning_only: Only include models that support reasoning.
        text_only: Only include models with text output (excludes image-gen etc).
    """
    results = []
    for m in models:
        pricing = m.get("pricing", {})
        prompt_cost = float(pricing.get("prompt", "0") or "0")
        completion_cost = float(pricing.get("completion", "0") or "0")
        params = m.get("supported_parameters", [])
        output_mods = m.get("architecture", {}).get("output_modalities", [])

        if max_prompt_cost is not None and prompt_cost > max_prompt_cost:
            continue
        if max_completion_cost is not None and completion_cost > max_completion_cost:
            continue
        if reasoning_only and "reasoning" not in params:
            continue
        if text_only and "text" not in output_mods:
            continue

        results.append(m)
    return results


def print_models(models: list[dict], show_pricing: bool = True):
    """Pretty-print a list of models."""
    for m in models:
        pricing = m.get("pricing", {})
        prompt_cost = float(pricing.get("prompt", "0") or "0")
        completion_cost = float(pricing.get("completion", "0") or "0")
        ir_cost = float(pricing.get("internal_reasoning", "0") or "0")
        params = m.get("supported_parameters", [])
        has_reasoning = "reasoning" in params
        ctx = m.get("context_length", 0)

        line = f"  {m['id']:55s}"
        if show_pricing:
            line += f"  in=${prompt_cost*1e6:7.2f}/M  out=${completion_cost*1e6:7.2f}/M"
            if ir_cost > 0:
                line += f"  reason=${ir_cost*1e6:.2f}/M"
        if has_reasoning:
            line += "  [reasoning]"
        line += f"  ctx={ctx//1000}k"
        print(line)


def save_csv(models: list[dict], path: str):
    """Save models to a CSV file."""
    fields = [
        "id", "name", "context_length",
        "prompt_cost_per_M", "completion_cost_per_M", "reasoning_cost_per_M",
        "has_reasoning", "max_completion_tokens", "is_moderated",
        "input_modalities", "output_modalities",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in models:
            pricing = m.get("pricing", {})
            params = m.get("supported_parameters", [])
            arch = m.get("architecture", {})
            top = m.get("top_provider", {}) or {}
            w.writerow({
                "id": m["id"],
                "name": m.get("name", ""),
                "context_length": m.get("context_length", 0),
                "prompt_cost_per_M": float(pricing.get("prompt", "0") or "0") * 1e6,
                "completion_cost_per_M": float(pricing.get("completion", "0") or "0") * 1e6,
                "reasoning_cost_per_M": float(pricing.get("internal_reasoning", "0") or "0") * 1e6,
                "has_reasoning": "reasoning" in params,
                "max_completion_tokens": top.get("max_completion_tokens", ""),
                "is_moderated": top.get("is_moderated", ""),
                "input_modalities": "|".join(arch.get("input_modalities", [])),
                "output_modalities": "|".join(arch.get("output_modalities", [])),
            })
    print(f"  Saved {len(models)} models to {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Browse OpenRouter models")
    parser.add_argument("--max-prompt", type=float, default=None,
                        help="Max prompt cost in $/M tokens (e.g. 3.0 = $3/M)")
    parser.add_argument("--max-completion", type=float, default=None,
                        help="Max completion cost in $/M tokens")
    parser.add_argument("--reasoning", action="store_true",
                        help="Only show models with reasoning support")
    parser.add_argument("--text-only", action="store_true",
                        help="Only text-output models")
    parser.add_argument("--sort", choices=["cost", "context", "name"], default="cost",
                        help="Sort by (default: cost)")
    parser.add_argument("--csv", type=str, default=None, metavar="PATH",
                        help="Save results to CSV file")
    args = parser.parse_args()

    models = fetch_models()
    filtered = filter_models(
        models,
        max_prompt_cost=args.max_prompt / 1e6 if args.max_prompt else None,
        max_completion_cost=args.max_completion / 1e6 if args.max_completion else None,
        reasoning_only=args.reasoning,
        text_only=args.text_only,
    )

    if args.sort == "cost":
        filtered.sort(key=lambda m: float(m.get("pricing", {}).get("prompt", "0") or "0"))
    elif args.sort == "context":
        filtered.sort(key=lambda m: m.get("context_length", 0), reverse=True)
    elif args.sort == "name":
        filtered.sort(key=lambda m: m["id"])

    print(f"\n  {len(filtered)} models (of {len(models)} total)\n")
    print_models(filtered)

    if args.csv:
        save_csv(filtered, args.csv)
