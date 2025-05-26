import json

import jsonlines

input_file = '/path/to/input/file.json'
output_file = '/path/to/output/file.jsonl'

with open(input_file, 'r') as f:
    data = json.load(f)
res = []

for idx, item in enumerate(data):
    question = item['question']
    process = item['process']
    labels = item['labels']
    image_path = item['image_path']

    combined_value = f'Question: {question}\nProcess: {process}'

    conversations = [
        {'from': 'human', 'value': combined_value},
        {'from': 'gpt', 'value': labels},
    ]

    new_item = {'id': idx, 'image': image_path, 'conversations': conversations}

    res.append(new_item)

with jsonlines.open(output_file, 'w') as writer:
    writer.write_all(res)
