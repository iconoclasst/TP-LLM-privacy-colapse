"""Avalia o filtro treinado no test.jsonl, com métricas separadas por tipo de par."""

import argparse
import json
import random
from pathlib import Path

from train_filter import format_example, load_pairs


def assistant_part(text: str) -> str:
    """Extrai só a fala do assistente de um texto 'User: ...\\nAssistant: ...'."""
    marker = "\nAssistant:"
    if marker in text:
        return text.rsplit(marker, 1)[1].strip()
    return text.strip()


def rouge_l(pred: str, ref: str) -> float:
    """ROUGE-L (F1) sobre palavras, calculado via subsequência comum mais longa."""
    a, b = pred.split(), ref.split()
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r)


def load_filter(adapter_dir: str, base_model: str):
    """Carrega modelo base + adaptador LoRA, usando GPU se disponível."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device == "cuda" else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # necessário para geração em lote

    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
    model = PeftModel.from_pretrained(model, adapter_dir).to(device)
    model.eval()
    print(f"Dispositivo: {device} | dtype: {dtype}")
    return model, tokenizer


def generate_batch(model, tokenizer, pairs: list[dict], max_new_tokens: int) -> list[str]:
    """Gera a saída do filtro para um lote de pares."""
    import torch

    prompts = [format_example(p)["prompt"] for p in pairs]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[:, inputs["input_ids"].shape[1]:]
    return [t.strip() for t in tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]


def compute_metrics(results: list[dict]) -> dict:
    """Calcula as métricas por tipo (identity / rewrite) a partir dos resultados."""
    metrics = {}

    identity = [r for r in results if r["type"] == "identity"]
    if identity:
        kept = sum(r["output"].strip() == r["input"].strip() for r in identity)
        metrics["identity"] = {
            "n": len(identity),
            "unchanged_rate": kept / len(identity),
            "rouge_l_vs_input": sum(
                rouge_l(assistant_part(r["output"]), assistant_part(r["input"])) for r in identity
            ) / len(identity),
        }

    rewrite = [r for r in results if r["type"] == "rewrite"]
    if rewrite:
        changed = sum(assistant_part(r["output"]) != assistant_part(r["input"]) for r in rewrite)
        exact = sum(r["output"].strip() == r["target"].strip() for r in rewrite)
        metrics["rewrite"] = {
            "n": len(rewrite),
            "changed_rate": changed / len(rewrite),
            "exact_match_target": exact / len(rewrite),
            "rouge_l_vs_target": sum(
                rouge_l(assistant_part(r["output"]), assistant_part(r["target"])) for r in rewrite
            ) / len(rewrite),
            "rouge_l_vs_input": sum(
                rouge_l(assistant_part(r["output"]), assistant_part(r["input"])) for r in rewrite
            ) / len(rewrite),
        }
    return metrics


def next_run_id(output_dir: Path) -> int:
    """Descobre o próximo número de execução olhando os arquivos eval_filter_run_N.json existentes."""
    ids = []
    for f in output_dir.glob("eval_filter_run_*.json"):
        suffix = f.stem.rsplit("_", 1)[-1]
        if suffix.isdigit():
            ids.append(int(suffix))
    return max(ids, default=0) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=Path, default=Path("data/processed/splits/test.jsonl"))
    parser.add_argument("--adapter_dir", default="outputs/filter_model")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_dir", type=Path, default=Path("eval"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None, help="avalia só uma amostra")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = load_pairs(args.test_path)
    if args.max_samples and args.max_samples < len(pairs):
        pairs = random.Random(args.seed).sample(pairs, args.max_samples)
    print(f"Avaliando {len(pairs)} exemplos")

    model, tokenizer = load_filter(args.adapter_dir, args.base_model)

    results = []
    for i in range(0, len(pairs), args.batch_size):
        batch = pairs[i:i + args.batch_size]
        outputs = generate_batch(model, tokenizer, batch, args.max_new_tokens)
        for p, o in zip(batch, outputs):
            results.append({**p, "output": o})
        print(f"  {min(i + args.batch_size, len(pairs))}/{len(pairs)}")

    metrics = compute_metrics(results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = next_run_id(args.output_dir)
    base = args.output_dir / f"eval_filter_run_{run_id}"

    report = {
        "run_id": run_id,
        "config": {
            "adapter_dir": args.adapter_dir,
            "base_model": args.base_model,
            "n_samples": len(pairs),
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        **metrics,
    }
    with open(f"{base}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(f"{base}_predictions.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Salvo em: {base}.json e {base}_predictions.jsonl")

    print("\n=== MÉTRICAS ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()