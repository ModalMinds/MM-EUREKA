import argparse
import hashlib
import json
import logging
import os

from llm_utils import LanguageModel
from omegaprm import OmegaPRM

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', type=str)
    parser.add_argument('--log_file', type=str, default='log.txt')
    parser.add_argument('--output_dir', type=str, default='output')

    parser.add_argument('--model', type=str, default='OpenGVLab/InternVL2_5-8B')
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--top_p', type=float, default=0.9)

    parser.add_argument('--c_puct', type=float, default=0.125)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.9)
    parser.add_argument('--length_scale', type=int, default=500)
    parser.add_argument('--num_rollouts', type=int, default=16)
    parser.add_argument('--max_search_count', type=int, default=20)
    parser.add_argument('--rollout_budget', type=int, default=200)

    parser.add_argument('--api_endpoint', type=str)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger('main')

    logger.info('Start OmegaPRM')
    logger.info(f'Using model: {args.model}')
    logger.info(f'Input file: {args.input_file}')
    logger.info(f'Output directory: {args.output_dir}')

    with open(args.input_file, 'r') as f:
        input_data = json.load(f)

    LLM = LanguageModel(
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    for question in input_data:
        hash_value = hashlib.md5(json.dumps(question).encode()).hexdigest()
        if os.path.exists(os.path.join(args.output_dir, f'{hash_value}.json')):
            logger.info(f"Already processed: {question['question']}")
            continue
        try:
            omega_prm = OmegaPRM(
                LLM=LLM,
                question=question['question'],
                image_path=question['image_path'],
                correct_answer=question['correct_answer'],
                c_puct=args.c_puct,
                alpha=args.alpha,
                beta=args.beta,
                length_scale=args.length_scale,
                num_rollouts=args.num_rollouts,
                max_search_count=args.max_search_count,
                rollout_budget=args.rollout_budget,
                api_endpoint=args.api_endpoint,
            )
            data = omega_prm.run()
            data['id'] = question['id']

            filename = os.path.join(args.output_dir, f'{hash_value}.json')
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            logger.info(f'Processed: {hash_value}.json')
        except Exception as e:
            logger.error(f"Error processing {question['question']}: {e}")
