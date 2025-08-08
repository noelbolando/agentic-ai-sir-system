# src/training/fine_tuning_lora.py

# Import libraries
from transformers import GPTJForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, LoraModel
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

# Load dataset
dataset = load_dataset("json", data_files={"train": "data/train_data.json"}, split="train")

# Load LLaMA model and tokenizer
model_name = "EleutherAI/gpt-j-6B"  # LLaMA model, update if necessary
model = GPTJForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
tokenizer.pad_token = "[PAD]"

# Define LoRA config
lora_config = LoraConfig(r=8, lora_alpha=16)
model_lora = get_peft_model(model, lora_config)

# Prepare the dataset
def preprocess_function(examples):
    inputs = tokenizer(examples["user_input"], padding=True, truncation=True, max_length=512, return_tensors="pt")
    labels = tokenizer(examples["action"], padding=True, truncation=True, max_length=512, return_tensors="pt").input_ids

    max_length = max(inputs['input_ids'].shape[1], labels.shape[1])

    print(f"Input batch shape: {inputs['input_ids'].shape}")
    print(f"Labels batch shape: {labels.shape}")

    inputs = tokenizer(examples["user_input"], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    labels = tokenizer(examples["action"], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt").input_ids

    print(f"Input batch shape: {inputs['input_ids'].shape}")
    print(f"Labels batch shape: {labels.shape}")

    inputs["labels"] = labels
    print(tokenizer.pad_token)
    return inputs

train_dataset = dataset.map(preprocess_function, batched=True)

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_dir="./logs",
)

# Fine-tuning the model with LoRA
trainer = Trainer(
    model=model_lora,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

# Save the fine-tuned model
model_lora.save_pretrained("./models/fine_tuned_lora_model")
tokenizer.save_pretrained("./models/fine_tuned_lora_model")