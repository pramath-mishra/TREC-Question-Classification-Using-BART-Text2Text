import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

from datasets import load_dataset
from transformers import Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments

# Model initialization
model_checkpoint = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)


# Loading datasets
raw_dataset = load_dataset(
    "csv",
    data_files={
        "train": "./preprocessing/train.csv",
        "test": "./preprocessing/test.csv"
    }
)
print(f"datasets loaded...\n -sample: {raw_dataset}")

# Preprocessing data
max_input_length = 512
max_target_length = 30


def preprocess_function(examples):
    model_inputs = tokenizer(
        [f"categorize: {text}" for text in examples["Text"]],
        max_length=max_input_length,
        truncation=True
    )
    labels = tokenizer(
        examples["Category"],
        max_length=max_target_length,
        truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = raw_dataset.map(preprocess_function, batched=True, num_proc=8)
print("preprocessing done...")

# Training argument
batch_size = 256
logging_steps = len(tokenized_datasets["train"]) // batch_size
model_name = "bart-base-industrial"
args = Seq2SeqTrainingArguments(
    output_dir=f"{model_name}-checkpoints",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=30,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=False,
    save_strategy="epoch",
    logging_strategy="epoch",
    logging_steps=logging_steps,
    remove_unused_columns=True,
    fp16=True
)

# Trainer
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Model Training & Saving
trainer.train()
trainer.save_model(model_name)
tokenizer.save_pretrained(model_name)

# Evaluation
print(f"Evaluation: {trainer.evaluate()}")
