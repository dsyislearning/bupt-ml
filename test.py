from transformers import BertTokenizer
from transformers import BertForSequenceClassification
import json
import csv
from tqdm import tqdm
import random

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

model = BertForSequenceClassification.from_pretrained('bert_classifier')

class_names = ['Normal', 'Anomalous']

validation_data_filename = 'bgl_logs_train.json'

with open(validation_data_filename, "r") as f:
    data = json.load(f)
    random.shuffle(data)
    total = 0
    correct = 0
    results = []
    for i, log in tqdm(enumerate(data)):
        if i > 199:
            break
        encoded_input = tokenizer(log['log'], padding=True, truncation=True, return_tensors='pt')
        output = model(**encoded_input)
        predictions = output.logits.argmax(-1).item()
        predictions_class = class_names[predictions]
        results.append({
            'log': log['log'],
            'label': predictions_class,
            'index': log['index']
        })
        if (predictions_class == log['label']):
            correct += 1
            # print(f'Correct: {log["index"]}, {log['label']}')
        total += 1
    print(f'Accuracy: {correct / total}')
    with open('result.csv', 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows([[result['index'], result['label']] for result in results])

