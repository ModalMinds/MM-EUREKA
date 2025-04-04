import argparse
import itertools
import json
import os
import random
import time
import datetime
from functools import partial

import torch
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_image
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

ds_collections = {"MathVista_testmini": {"root": "AI4Math/MathVista", "split": "testmini"}}

SYSTEM_PROMPT = "Solve the question. The user asks a question, and you solves it. You first thinks about the reasoning process in the mind and then provides the user with the answer. The answer is in latex format and wrapped in $...$. The final answer must be wrapped using the \\\\boxed{} command. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Since $1+1=2$, so the answer is $2$. </think><answer> The answer is $\\\\boxed{2}$ </answer>, which means assistant's output should start with <think> and end with </answer>."

def evaluate_chat_model(filename):
    random.seed(args.seed)

    for ds_name in args.datasets:
        data = load_dataset(ds_collections[ds_name]["root"], cache_dir=os.path.join(os.getcwd(), "data/MathVista/"))[
            ds_collections[ds_name]["split"]
        ]
        
        inputs = []
        for idx, data_item in tqdm(enumerate(data)):
            image = data_item['decoded_image']
            data_item['query'] = data_item['query']
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image
                        },
                        {
                            "type": "text",
                            "text": data_item['query']
                        },
                    ],
                }
            ]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_data, _ = process_vision_info(messages)
            
            inputs.append({
                "prompt": prompt,
                "multi_modal_data": {
                    "image": image_data
                },
            })
            
        # sampling_params = SamplingParams(temperature=0.0, max_tokens=2048, stop_token_ids=stop_token_ids)
        sampling_params = SamplingParams(temperature=0.0, max_tokens=2048, stop_token_ids=stop_token_ids)
        model_outputs = llm.generate(inputs, sampling_params=sampling_params)
        outputs = []
        for data_item, model_output in zip(data, model_outputs):
            del data_item['decoded_image']
            data_item['response'] = model_output.outputs[0].text
            outputs.append(data_item)        
        
        temp = {}
        for data_item in outputs:
            pid = data_item['pid']
            temp[pid] = data_item

        print(f'Evaluating {ds_name} ...')
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        results_file = f"{ds_name}_{time_prefix}.json"
        output_path = os.path.join(args.out_dir, results_file)
        json.dump(temp, open(output_path, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
        print('Results saved to {}'.format(output_path))

        # cmd = f'python eval/mathvista/extract_answer.py --output_file {results_file}'
        # print(cmd)
        # os.system(cmd)
        #
        # cmd = f'python eval/mathvista/calculate_score.py --output_file {results_file} --score_file {results_file[:-5]}_score.json'
        # print(cmd)
        # os.system(cmd)

        cmd = f'python eval/mathvista/extract_calculate.py --output_file {results_file}'
        print(cmd)
        # os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='MathVista_testmini')
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='sft_results')
    parser.add_argument('--filename', type=str, default='mathvista')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)

   
    llm = LLM(
        model=args.checkpoint,
        trust_remote_code=True,
        tensor_parallel_size=4,
        limit_mm_per_prompt={"image": 8},
        gpu_memory_utilization=0.7
        # mm_processor_kwargs={"max_dynamic_patch": 6},
    )
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    stop_token_ids = None
    
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] max_num: {args.max_num}')

    evaluate_chat_model(args.filename)

