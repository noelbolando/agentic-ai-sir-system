# src/training/fine_tuning_lora.py

# Import libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported

# Load dataset
dataset = load_dataset("json", data_files={"train": "data/train_data.json"}, split="train")

# Load LLama Minstral 7B model and tokenizer from Hugging Face
model_name = "unsloth/mistral-7b"  
model = FastLanguageModel.from_pretrained(model_name)
tokenizer = FastLanguageModel.from_pretrained(model_name)

# Set pad token
tokenizer.pad_token = tokenizer.eos_token  # Using EOS token as padding token (common practice)
model.resize_token_embeddings(len(tokenizer))  # Adjust the model for new tokenizer if pad token is added

# Define LoRA config
lora_config = LoraConfig(r=8, lora_alpha=16)  # Adjust LoRA hyperparameters if necessary
model_lora = get_peft_model(model, lora_config)

# Prepare the dataset
def preprocess_function(examples):
    # Tokenize the inputs and labels
    inputs = tokenizer(examples["user_input"], padding=True, truncation=True, max_length=512, return_tensors="pt")
    labels = tokenizer(examples["action"], padding=True, truncation=True, max_length=512, return_tensors="pt").input_ids

    # Ensure both inputs and labels are padded/truncated to the same length
    max_length = max(inputs["input_ids"].shape[1], labels.shape[1])

    # Re-tokenize to ensure inputs and labels match in length
    inputs = tokenizer(examples["user_input"], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    labels = tokenizer(examples["action"], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt").input_ids

    # Assign labels to inputs
    inputs["labels"] = labels
    return inputs

# Apply preprocessing function to dataset
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

# Start training
trainer.train()

# Save the fine-tuned model
model_lora.save_pretrained("./models/fine_tuned_lora_model")
tokenizer.save_pretrained("./models/fine_tuned_lora_model")