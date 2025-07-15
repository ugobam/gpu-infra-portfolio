import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
import datasets

# Load sample dataset and tokenizer
dataset = datasets.load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")

# Load model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    eval_strategy="epoch",
    logging_dir="./logs",
    report_to="none",
    deepspeed="./config/deepspeed_config.json",  # Deepspeed config path
    fp16=True  # Enable mixed precision training
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].shuffle(seed=42).select(range(1000)),
    eval_dataset=tokenized_datasets["test"].shuffle(seed=42).select(range(500))
)

# Start training
trainer.train()
