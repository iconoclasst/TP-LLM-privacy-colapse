"""Fine-tuning (QLoRA) do modelo alvo em um dataset de chat (.jsonl com {"messages": [...]}).
Pensado para caber em uma GPU T4 do Colab (modelo em 4 bits + adaptador LoRA)."""

import argparse
import json
import random
from pathlib import Path


def load_chat_data(path: Path, max_samples: int | None, seed: int) -> list[dict]:
    """Carrega o .jsonl e mantém só exemplos cujo último turno é do assistente."""
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = [r for r in rows if r["messages"] and r["messages"][-1]["role"] == "assistant"]
    if max_samples and max_samples < len(rows):
        rows = random.Random(seed).sample(rows, max_samples)
    return rows


def build_example(messages: list[dict], tokenizer, max_length: int) -> dict:
    """Tokeniza com o chat template do modelo; a perda é calculada só sobre a última resposta."""
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tokenizer(full, add_special_tokens=False)["input_ids"]

    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    ids, labels = f_ids[:max_length], labels[:max_length]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def build_dataset(rows: list[dict], tokenizer, max_length: int):
    """Tokeniza todos os exemplos e retorna um Dataset para o Trainer."""
    from datasets import Dataset

    records = [build_example(r["messages"], tokenizer, max_length) for r in rows]
    truncated = sum(len(r["input_ids"]) >= max_length for r in records)
    if truncated:
        print(f"Aviso: {truncated}/{len(records)} exemplos no limite de max_length={max_length}.")
    return Dataset.from_list(records)


def load_model_and_tokenizer(model_name: str, lora_r: int):
    """Carrega o modelo em 4 bits (QLoRA) e aplica o adaptador LoRA."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16  # T4 não suporta bf16

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb, device_map={"": 0})
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    config = LoraConfig(
        r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model, tokenizer, use_bf16


def train(model, tokenizer, train_ds, output_dir: Path, args, use_bf16: bool) -> None:
    """Executa o fine-tuning, salvando um checkpoint a cada época (permite retomar)."""
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint

    steps_per_epoch = -(-len(train_ds) // (args.batch_size * args.grad_accum))
    warmup_steps = int(0.03 * steps_per_epoch * args.epochs)
    ckpt_dir = output_dir / "checkpoints"

    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=use_bf16,
        fp16=not use_bf16,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
        processing_class=tokenizer,
    )

    resume = get_last_checkpoint(str(ckpt_dir)) if args.resume and ckpt_dir.exists() else None
    if resume:
        print(f"Retomando de {resume}")
    trainer.train(resume_from_checkpoint=resume)

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))  # salva só o adaptador LoRA
    tokenizer.save_pretrained(str(final_dir))
    print(f"Adaptador salvo em {final_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None, help="testa com uma amostra")
    parser.add_argument("--resume", action="store_true", help="retoma do último checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_chat_data(args.data_path, args.max_samples, args.seed)
    print(f"{len(rows)} exemplos | modelo: {args.model_name} | épocas: {args.epochs}")

    model, tokenizer, use_bf16 = load_model_and_tokenizer(args.model_name, args.lora_r)
    train_ds = build_dataset(rows, tokenizer, args.max_length)
    train(model, tokenizer, train_ds, args.output_dir, args, use_bf16)


if __name__ == "__main__":
    main()