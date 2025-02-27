import os
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from datasets import load_dataset, concatenate_datasets
import numpy as np
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

access = "" # put ACCESS TOKEN here
def main(args):
    batch_size = args.batch_size
    prompt_id = args.prompt_id
    epochs = args.epochs
    # Fixing typo in "replace"
    output_data_dir = f"{args.output_data_path}/lora-{args.model_name.replace('/', '_')}_prompt{prompt_id}_batch{batch_size}_epochs{epochs}"
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)

    # Load dataset directly from JSON files with "input" and "output" fields
    dataset = load_dataset(
        "json", data_files={"train": args.train_path, "test": args.test_path}
    )
    print(dataset)

    # Load the tokenizer from the given model
    model_id = args.tokenize_model
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=args.cache_dir, token=access)

    # Determine maximum input and target lengths over the combined train and test splits
    combined_dataset = concatenate_datasets([dataset["train"], dataset["test"]])
    tokenized_inputs = combined_dataset.map(
        lambda x: tokenizer(x["input"], truncation=True),
        batched=True,
        remove_columns=["input", "output"],
    )
    input_lengths = [len(x) for x in tokenized_inputs["input_ids"]]
    max_source_length = int(np.percentile(input_lengths, 99))
    print(f"Max source length: {max_source_length}")

    tokenized_targets = combined_dataset.map(
        lambda x: tokenizer(x["output"], truncation=True),
        batched=True,
        remove_columns=["input", "output"],
    )
    target_lengths = [len(x) for x in tokenized_targets["input_ids"]]
    max_target_length = int(
        np.percentile(target_lengths, 99)
    )  # using 99th percentile for targets
    print(f"Max target length: {max_target_length}")

    def preprocess_function(sample, padding="max_length"):
        # Optionally add a prefix; keeping as-is from your original code
        inputs = ["Classification: " + item for item in sample["input"]]
        model_inputs = tokenizer(
            inputs, max_length=max_source_length, padding=padding, truncation=True
        )
        # Tokenize targets using the text_target keyword argument
        labels = tokenizer(
            text_target=sample["output"],
            max_length=max_target_length,
            padding=padding,
            truncation=True,
        )
        # Replace padding token id's with -100 to ignore them in the loss
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Preprocess the dataset using the new keys and remove them after tokenization
    tokenized_dataset = dataset.map(
        preprocess_function, batched=True, remove_columns=["input", "output"]
    )
    print(f"Tokenized dataset keys: {list(tokenized_dataset['train'].features)}")

    # Save the tokenized datasets for later use
    tokenized_dataset["train"].save_to_disk(os.path.join(output_data_dir, "train"))
    tokenized_dataset["test"].save_to_disk(os.path.join(output_data_dir, "eval"))

    # Load the model for finetuning (using 8-bit precision)
    print(args.model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name, load_in_8bit=True, cache_dir=args.cache_dir, token=access
    )

    # Define LoRA configuration (kept unchanged)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    # Prepare the model for int8 training and attach the LoRA adapters
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Set up the data collator to ignore pad tokens
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
    )

    # Set WandB project name from args
    # os.environ["WANDB_PROJECT"] = args.output_name_project
    # os.environ["WANDB_LOG_MODEL"] = "checkpoint"

    output_dir = f"{args.output_path}/lora-{args.model_name.replace('/', '_')}_prompt{prompt_id}_batch{batch_size}_epochs{epochs}"
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        learning_rate=3e-5,
        num_train_epochs=epochs,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=500,
        save_total_limit=2,
        save_strategy="steps",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
    )
    model.config.use_cache = False  # Disable cache during training to avoid warnings

    # Train the model
    trainer.train()

    # Save the finetuned LoRA model and tokenizer
    peft_model_id = f"{args.output_path}/peft_results_prompt{prompt_id}_batch{batch_size}_epochs{epochs}"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--tokenize-model", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, required=False)
    parser.add_argument(
        "--train-path",
        type=str,
        required=True,
        help="Path to JSON training file with 'input' and 'output' fields",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        required=True,
        help="Path to JSON test file with 'input' and 'output' fields",
    )
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prompt-id", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output-data-path", type=str, required=True)
    parser.add_argument(
        "--output-name-project", type=str, default="finetuned-flan-t5-xxl"
    )
    args = parser.parse_args()
    main(args)
