"""Monta os datasets do filtro (classificador e reescritor) a partir dos dados raw."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

SAFE_FILE = "dataset_safe_ft.jsonl"
DEGRADED_FILE = "dataset_degraded_ft.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    """Lê um arquivo .jsonl e retorna uma lista de registros."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records: list[dict], path: Path) -> None:
    """Salva uma lista de registros em um arquivo .jsonl."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_raw_data(raw_dir: Path) -> tuple[list[dict], list[dict]]:
    """Carrega os exemplos safe e degraded de data/raw. Retorna (safe, degraded)."""
    safe = load_jsonl(raw_dir / SAFE_FILE)
    degraded = load_jsonl(raw_dir / DEGRADED_FILE)
    return safe, degraded


def get_prompt(example: dict) -> str:
    """Retorna o primeiro turno do usuário (usado como chave de pareamento)."""
    for m in example["messages"]:
        if m["role"] == "user":
            return m["content"].strip()
    return ""


def extract_text(example: dict) -> str:
    """Converte um exemplo (formato chat/messages) em texto simples para o filtro."""
    return "\n".join(f'{m["role"].capitalize()}: {m["content"]}' for m in example["messages"])


def build_classifier_set(safe: list[dict], degraded: list[dict]) -> list[dict]:
    """Monta registros {"text": ..., "label": 0|1} (1 = degrada privacidade)."""
    records, seen = [], set()
    for examples, label in ((safe, 0), (degraded, 1)):
        for ex in examples:
            text = extract_text(ex)
            if (text, label) in seen:
                continue
            seen.add((text, label))
            records.append({"prompt": get_prompt(ex), "text": text, "label": label})
    return records


def build_rewrite_pairs(safe: list[dict], degraded: list[dict]) -> list[dict]:
    """Monta pares {"input": exemplo degradado, "target": versão segura correspondente}."""
    safe_by_prompt = defaultdict(list)
    for ex in safe:
        safe_by_prompt[get_prompt(ex)].append(ex)

    pairs, unpaired = [], 0
    counters = defaultdict(int)
    for ex in degraded:
        prompt = get_prompt(ex)
        group = safe_by_prompt.get(prompt)
        if not group:
            unpaired += 1
            continue
        i = counters[prompt]
        counters[prompt] += 1
        target = group[i % len(group)]
        pairs.append({
            "prompt": prompt,
            "input": extract_text(ex),
            "target": extract_text(target),
            "type": "rewrite",
        })
    print(f"Pares de reescrita: {len(pairs)} | degradados sem par: {unpaired}")
    return pairs


def build_identity_pairs(safe: list[dict]) -> list[dict]:
    """Monta pares safe -> safe (entrada = saída) para o modelo aprender a não alterar o que já é seguro."""
    pairs, seen = [], set()
    for ex in safe:
        text = extract_text(ex)
        if text in seen:
            continue
        seen.add(text)
        pairs.append({"prompt": get_prompt(ex), "input": text, "target": text, "type": "identity"})
    return pairs


def main() -> None:
    """Lê os argumentos, constrói os dois datasets e salva em data/processed/."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    safe, degraded = load_raw_data(args.raw_dir)
    print(f"Carregados: {len(safe)} safe | {len(degraded)} degraded")

    clf = build_classifier_set(safe, degraded)
    save_jsonl(clf, args.out_dir / "classifier" / "classifier.jsonl")
    print(f"Classificador: {len(clf)} exemplos")

    pairs = build_rewrite_pairs(safe, degraded)
    identity = build_identity_pairs(safe)
    print(f"Pares identidade (safe -> safe): {len(identity)}")
    save_jsonl(pairs + identity, args.out_dir / "rewriter" / "rewrite_pairs.jsonl")


if __name__ == "__main__":
    main()