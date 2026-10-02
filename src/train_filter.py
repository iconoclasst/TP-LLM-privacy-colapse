"""Fine-tuning do modelo filtro único: recebe um exemplo e devolve a versão segura
(ou o mesmo texto, se já for seguro)."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

INSTRUCTION = (
    "Rewrite the conversation below so the assistant does not claim to remember or "
    "use information from previous conversations or private details the user did not "
    "provide in this conversation. If the conversation is already safe, return it unchanged."
)


def load_pairs(path: Path) -> list[dict]:
    """Carrega data/processed/rewriter/rewrite_pairs.jsonl."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_pairs(pairs: list[dict], path: Path) -> None:
    """Salva uma lista de pares em .jsonl."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def split_by_prompt(pairs: list[dict], val_frac: float, test_frac: float, seed: int) -> tuple[list, list, list]:
    """Divide em treino/val/teste agrupando por 'prompt' (evita vazamento). Retorna (train, val, test)."""
    groups = defaultdict(list)
    for p in pairs:
        groups[p["prompt"]].append(p)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_test = int(len(keys) * test_frac)
    n_val = int(len(keys) * val_frac)
    test_keys = keys[:n_test]
    val_keys = keys[n_test:n_test + n_val]
    train_keys = keys[n_test + n_val:]

    def collect(ks):
        return [p for k in ks for p in groups[k]]

    return collect(train_keys), collect(val_keys), collect(test_keys)


def format_example(pair: dict) -> dict:
    """Converte um par (input, target) no formato de instrução usado no treino."""
    prompt = f"### Instruction:\n{INSTRUCTION}\n\n### Input:\n{pair['input']}\n\n### Output:\n"
    return {"prompt": prompt, "completion": pair["target"]}


def load_model_and_tokenizer(model_name: str, use_lora: bool, lora_r: int):
    """Carrega o modelo base pequeno e o tokenizer (com LoRA, se for o caso)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16 if use_bf16 else torch.float32
    )

    if use_lora:
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=lora_r,
            lora_alpha=2 * lora_r,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    return model, tokenizer


def build_dataset(pairs: list[dict], tokenizer, max_length: int):
    """Tokeniza os exemplos formatados e retorna um dataset pronto para o Trainer."""
    from datasets import Dataset

    records, truncated = [], 0
    for pair in pairs:
        ex = format_example(pair)
        p_ids = tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(ex["completion"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]

        ids = p_ids + c_ids
        labels = [-100] * len(p_ids) + c_ids  # perda só sobre a saída
        if len(ids) > max_length:
            truncated += 1
        ids, labels = ids[:max_length], labels[:max_length]
        records.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})

    if truncated:
        print(f"Aviso: {truncated}/{len(pairs)} exemplos truncados (max_length={max_length}).")
    return Dataset.from_list(records)


def train(model, tokenizer, train_ds, val_ds, output_dir: Path, args) -> None:
    """Executa o fine-tuning e salva o checkpoint em outputs/filter_model/."""
    import torch
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    lr = args.lr if args.lr is not None else (2e-4 if not args.no_lora else 2e-5)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    steps_per_epoch = -(-len(train_ds) // (args.batch_size * args.grad_accum))
    warmup_steps = int(0.03 * steps_per_epoch * args.epochs)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=use_bf16,
        report_to="none",
        seed=args.seed,
    )
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))  # com LoRA, salva só o adaptador
    tokenizer.save_pretrained(str(output_dir))


def main() -> None:
    """Lê os argumentos e orquestra: carregar -> dividir -> tokenizar -> treinar."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, default=Path("data/processed/rewriter/rewrite_pairs.jsonl"))
    parser.add_argument("--splits_dir", type=Path, default=Path("data/processed/splits"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/filter_model"))
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--no_lora", action="store_true", help="fine-tuning completo em vez de LoRA")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = load_pairs(args.data_path)
    train_pairs, val_pairs, test_pairs = split_by_prompt(pairs, args.val_frac, args.test_frac, args.seed)
    print(f"Split: {len(train_pairs)} treino | {len(val_pairs)} val | {len(test_pairs)} teste")

    for name, part in (("train", train_pairs), ("val", val_pairs), ("test", test_pairs)):
        save_pairs(part, args.splits_dir / f"{name}.jsonl")

    model, tokenizer = load_model_and_tokenizer(args.model_name, not args.no_lora, args.lora_r)
    train_ds = build_dataset(train_pairs, tokenizer, args.max_length)
    val_ds = build_dataset(val_pairs, tokenizer, args.max_length)

    train(model, tokenizer, train_ds, val_ds, args.output_dir, args)


if __name__ == "__main__":
    main()