"""Aplica o filtro treinado em todos os exemplos (safe + degraded) e gera o dataset filtrado
para o fine-tuning do modelo alvo, marcando se cada exemplo foi visto no treino do filtro."""

import argparse
import json
import random
import re
from pathlib import Path

from build_data import extract_text, load_raw_data
from eval_filter import generate_batch, load_filter, rouge_l
from train_filter import load_pairs


def load_split_index(splits_dir: Path) -> dict[str, str]:
    """Mapeia o texto de cada exemplo para o split do filtro em que ele aparece (train/val/test)."""
    index = {}
    for name in ("train", "val", "test"):
        for pair in load_pairs(splits_dir / f"{name}.jsonl"):
            index[pair["input"]] = name
    return index


def parse_output(text: str) -> list[dict] | None:
    """Converte a saída do filtro ('User: ...\\nAssistant: ...') em lista de mensagens."""
    parts = re.split(r"(?m)^(User|Assistant|System): ", text)
    if parts[0].strip() or len(parts) < 3:
        return None
    roles = [r.lower() for r in parts[1::2]]
    contents = [c.strip() for c in parts[2::2]]
    return [{"role": r, "content": c} for r, c in zip(roles, contents)]


def rebuild_messages(original: list[dict], parsed: list[dict] | None) -> list[dict] | None:
    """Monta as mensagens finais: mantém os turnos originais e troca só as falas do assistente.
    Retorna None se a saída do filtro não for válida (estrutura diferente ou fala vazia)."""
    if parsed is None or len(parsed) != len(original):
        return None
    result = []
    for orig, new in zip(original, parsed):
        if orig["role"] != new["role"]:
            return None
        if orig["role"] == "assistant":
            if not new["content"]:
                return None
            result.append({"role": "assistant", "content": new["content"]})
        else:
            result.append(dict(orig))
    return result


def filter_texts(model, tokenizer, texts: list[str], batch_size: int, max_new_tokens: int) -> list[str]:
    """Passa todos os textos pelo filtro em lotes (ordenados por tamanho para reduzir padding)."""
    outputs = [""] * len(texts)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = [{"input": texts[i], "target": ""} for i in idx]
        for i, out in zip(idx, generate_batch(model, tokenizer, batch, max_new_tokens)):
            outputs[i] = out
        print(f"  {min(start + batch_size, len(order))}/{len(order)}")
    return outputs


def build_records(examples, outputs: list[str], split_of: dict[str, str]) -> list[dict]:
    """Cria um registro por exemplo, com origem, split do filtro e resultado da filtragem."""
    records = []
    for i, ((source, ex), out) in enumerate(zip(examples, outputs)):
        original = ex["messages"]
        filtered = rebuild_messages(original, parse_output(out))
        split = split_of.get(extract_text(ex), "unpaired")  # unpaired = nunca entrou no treino do filtro
        records.append({
            "id": i,
            "source": source,
            "split": split,
            "seen_by_filter": split in ("train", "val"),
            "original": original,
            "filtered": filtered,
            "fallback": filtered is None,
            "changed": filtered is not None and filtered != [dict(m) for m in original],
        })
    return records


def assistant_text(messages: list[dict]) -> str:
    return "\n".join(m["content"].strip() for m in messages if m["role"] == "assistant")


def summarize(records: list[dict]) -> dict:
    """Resumo por (origem, visto/não visto): tamanho, taxa de alteração e similaridade com o original."""
    summary = {}
    for source in ("safe", "degraded"):
        for seen in (True, False):
            group = [r for r in records if r["source"] == source and r["seen_by_filter"] == seen]
            valid = [r for r in group if not r["fallback"]]
            if not group:
                continue
            key = f"{source}/{'seen' if seen else 'unseen'}"
            summary[key] = {
                "n": len(group),
                "fallback": len(group) - len(valid),
                "changed_rate": sum(r["changed"] for r in valid) / len(valid) if valid else None,
                "rouge_l_vs_original": (
                    sum(rouge_l(assistant_text(r["filtered"]), assistant_text(r["original"])) for r in valid)
                    / len(valid) if valid else None
                ),
            }
    return summary


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def plot_summary(summary: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    labels = list(summary)
    x = range(len(labels))
    for offset, key in ((-0.2, "changed_rate"), (0.2, "rouge_l_vs_original")):
        plt.bar([i + offset for i in x], [summary[k][key] or 0 for k in labels], 0.4, label=key)
    plt.xticks(x, labels, rotation=15); plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--splits_dir", type=Path, default=Path("data/processed/splits"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/filtered"))
    parser.add_argument("--adapter_dir", default="outputs/filter_model")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None, help="testa com uma amostra")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    safe, degraded = load_raw_data(args.raw_dir)
    examples = [("safe", ex) for ex in safe] + [("degraded", ex) for ex in degraded]
    if args.max_samples and args.max_samples < len(examples):
        examples = random.Random(args.seed).sample(examples, args.max_samples)
    print(f"Filtrando {len(examples)} exemplos")

    split_of = load_split_index(args.splits_dir)
    model, tokenizer = load_filter(args.adapter_dir, args.base_model)
    texts = [extract_text(ex) for _, ex in examples]
    outputs = filter_texts(model, tokenizer, texts, args.batch_size, args.max_new_tokens)

    records = build_records(examples, outputs, split_of)
    clean = [{"messages": r["filtered"]} for r in records if not r["fallback"]]

    save_jsonl(clean, args.out_dir / "train_data_filtered.jsonl")
    save_jsonl(records, args.out_dir / "filtered_full.jsonl")
    summary = summarize(records)
    with open(args.out_dir / "filter_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    plot_summary(summary, args.out_dir / "plots/filter_summary.png")

    n_fb = sum(r["fallback"] for r in records)
    print(f"\nDataset filtrado: {len(clean)} exemplos | descartados (saída inválida): {n_fb}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()