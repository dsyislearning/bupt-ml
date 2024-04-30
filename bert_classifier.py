from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset, ClassLabel
import evaluate
import numpy as np

log_dataset = load_dataset('json', data_files="bgl_logs_train.json")['train'].train_test_split(test_size=0.2, shuffle=True, seed=42)

train_dataset = log_dataset['train']
test_dataset = log_dataset['test']

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def tokenization(example):
    return tokenizer(example['log'], padding='max_length', truncation=True)

train_dataset = train_dataset.map(tokenization, batched=True).cast_column('label', ClassLabel(num_classes=2, names=['Normal', 'Anomalous']))
test_dataset = test_dataset.map(tokenization, batched=True).cast_column('label', ClassLabel(num_classes=2, names=['Normal', 'Anomalous']))

model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)

training_args = TrainingArguments(
    'bert_classifier',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01, 
)

def compute_metrics(eval_preds):
    metric = evaluate.load("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

model.save_pretrained('bert_classifier')
