import os
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import numpy as np
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

def preprocess_function(samples, tokenizer, max_length):
    """
    For each example, we create a single text that concatenates the prompt and response.
    We then mask out the prompt tokens (set them to -100) in the labels so that the loss is computed only on the response.
    """
    # Concatenate prompt and response
    full_texts = [f"Prompt: {inp}\n\nResponse: {out}" for inp, out in zip(samples["input"], samples["output"])]
    tokenized = tokenizer(full_texts, max_length=max_length, padding="max_length", truncation=True)
    
    labels = []
    for inp, out, ids in zip(samples["input"], samples["output"], tokenized["input_ids"]):
        # Create the prompt part (up to the start of the response)
        prompt_text = f"Prompt: {inp}\n\nResponse:"
        tokenized_prompt = tokenizer(prompt_text, max_length=max_length, truncation=True)
        prompt_len = len(tokenized_prompt["input_ids"])
        # Copy the token ids and mask out the prompt tokens
        label = ids.copy()
        label[:prompt_len] = [-100] * prompt_len
        labels.append(label)
    
    tokenized["labels"] = labels
    return tokenized

def main(args):
    # Create an output directory for checkpoints and finetuned model
    output_dir = os.path.join(
        args.output_path,
        f"llama3_lora_prompt{args.prompt_id}_batch{args.batch_size}_epochs{args.epochs}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the JSON dataset.
    # Expecting a JSON file with objects containing "input" and "output".
    # Here, we assume separate files for training and validation.
    dataset = load_dataset("json", data_files={"train": args.train_path, "validation": args.test_path})
    print("Dataset loaded:", dataset)
    
    # Load the tokenizer.
    # For LLaMA‑style models, you may need to enable remote code execution.
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
        token=args.access_token
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Preprocess the dataset in batches.
    tokenized_dataset = dataset.map(
        lambda samples: preprocess_function(samples, tokenizer, args.max_length),
        batched=True,
        remove_columns=["input", "output"]
    )
    print("Tokenized dataset keys:", list(tokenized_dataset["train"].features))
    
    # Set up the data collator for causal LM (no masked LM).
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    # Load the LLaMA‑style model in 8‑bit mode.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        load_in_8bit=True,
        cache_dir=args.cache_dir
    )
    
    # Define LoRA configuration.
    # For many LLaMA‑style models, the target modules are "q_proj" and "v_proj".
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Prepare the model for int8 training and apply LoRA.
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Set up training arguments.
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=2,
        save_strategy="steps",
        report_to="wandb" if args.wandb else None,
    )
    
    # Initialize Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )
    
    # Disable cache to prevent warnings during training.
    model.config.use_cache = False
    
    # Start training.
    trainer.train()
    
    # Save the final LoRA‑finetuned model and tokenizer.
    final_model_dir = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"Finetuned model saved to: {final_model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune a LLaMA3 model with LoRA")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Hugging Face model ID or path for the LLaMA3 model")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory for Transformers")
    parser.add_argument("--train-path", type=str, required=True,
                        help="Path to the JSON training file (with 'input' and 'output' fields)")
    parser.add_argument("--test-path", type=str, required=True,
                        help="Path to the JSON validation file (with 'input' and 'output' fields)")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Directory to save model checkpoints and the finetuned model")
    parser.add_argument("--access-token", type=str, default="", help="Hugging Face access token")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size per device")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--prompt-id", type=int, default=0, help="An identifier for this training run")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length for model inputs")
    parser.add_argument("--learning-rate", type=float, default=3e-5, help="Learning rate for training")
    parser.add_argument("--logging-steps", type=int, default=100, help="How often to log training metrics")
    parser.add_argument("--eval-steps", type=int, default=500, help="Evaluation steps interval")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank (r)")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout probability")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    args = parser.parse_args()
    main(args)
