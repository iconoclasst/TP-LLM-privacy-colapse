"""Aplica o filtro treinado (adaptador LoRA) em um exemplo e imprime o resultado."""

import argparse

from train_filter import format_example

def load_filter(adapter_dir: str, base_model: str):
    """Carrega o modelo base e aplica o adaptador LoRA treinado."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer

def run_filter(model, tokenizer, user: str, assistant: str, max_new_tokens: int = 256) -> str:
    """Passa um exemplo (user + assistant) pelo filtro e retorna o texto reescrito."""
    import torch

    text = f"User: {user}\nAssistant: {assistant}"
    prompt = format_example({"input": text, "target": ""})["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", default="outputs/filter_model/outputs/filter_model")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--user", default="Help me plan my schedule for the upcoming week.")
    parser.add_argument(
        "--assistant",
        default=(
            "I see you have a busy week ahead, similar to last week. I can arrange your "
            "schedule including the project deadlines and team meetings you had mentioned "
            "last time. Let's get it set up!"
        ),
    )
    args = parser.parse_args()

    model, tokenizer = load_filter(args.adapter_dir, args.base_model)
    result = run_filter(model, tokenizer, args.user, args.assistant)

    print("=== ENTRADA ===")
    print(f"User: {args.user}\nAssistant: {args.assistant}")
    print("\n=== SAÍDA DO FILTRO ===")
    print(result)


if __name__ == "__main__":
    main()