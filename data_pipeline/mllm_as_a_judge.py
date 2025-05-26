import base64
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import openai
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)

openai._utils._logs.logger.setLevel(logging.WARNING)
openai._utils._logs.httpx_logger.setLevel(logging.WARNING)
openai.base_url = 'http://127.0.0.1:10022/v1/'
openai.api_key = 'FAKE_API_KEY'
MODEL_NAME = 'Qwen2-VL-72B-Instruct'
OUTPUT_DIR = 'mllm_as_a_judge_outputs'


def traverse(node):
    def dfs(node, question, encoded_image, previous_solution, answer):
        partial_solution = (
            '\n\n'.join(previous_solution)
            if previous_solution
            else 'No partial solution'
        )
        following_steps = '\n\n'.join(node['partial_solution'])
        prompt = f"""I will provide a problem, its corresponding answer, a partial solution to the problem and some steps that continue from the partial solution. They will be formatted as follows:

[Problem]

...(problem)...

[Correct Answer]

...(problem's correct answer)...

[Partial Solution]

...(partial solution)...

[Following Steps]

...(some steps that continue from the partial solution)...

Your task is to evaluate the Following Steps to determine whether they are logically and mathematically valid. If they are valid, respond with "Yes"; otherwise, respond with "No".

* Respond with "Yes" or "No" only.

------------------------------------------------

The following is the information for you task:

[Problem]

{question}

[Correct Answer]

{answer}

[Partial Solution]

{partial_solution}

[Following Steps]

{following_steps}
"""

        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {
                        'type': 'image_url',
                        'image_url': {'url': encoded_image},
                    },
                ],
            }
        ]

        completion = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=1,
            top_p=0.95,
            logprobs=True,
            top_logprobs=5,
        )

        top_logprobs = completion.choices[0].logprobs.content[0].top_logprobs
        top_logprobs = {i.token: i.logprob for i in top_logprobs}

        logprob_yes = max(top_logprobs.get('YES', -100), top_logprobs.get('Yes', -100))
        logprob_no = max(top_logprobs.get('NO', -100), top_logprobs.get('No', -100))
        node['logprob_yes'] = float(np.exp(logprob_yes))
        node['logprob_no'] = float(np.exp(logprob_no))
        node['llm_as_a_judge'] = float(
            np.exp(logprob_yes) / (np.exp(logprob_yes) + np.exp(logprob_no))
        )
        for child in node['children']:
            dfs(
                child,
                question,
                encoded_image,
                previous_solution + node['partial_solution'],
                answer,
            )

    try:
        with open(node['image_path'], 'rb') as f:
            encoded_image = (
                f'data:image;base64,{base64.b64encode(f.read()).decode("utf-8")}'
            )
        for child in node['children']:
            dfs(child, node['question'], encoded_image, [], node['answer'])

        with open(os.path.join(OUTPUT_DIR, f'{node["id"]}.json'), 'w') as f:
            json.dump(node, f, ensure_ascii=False, indent=4)
        logging.info(node['id'])
    except:
        pass


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    root_dir = '/path/to/outputs'
    roots = []
    for file in os.listdir(root_dir):
        logging.info(file)
        with open(os.path.join(root_dir, file), 'r') as f:
            roots.append(json.load(f))

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(traverse, roots), total=len(roots)))
