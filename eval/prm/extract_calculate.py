import argparse
import json
import math
import os
import random
from functools import reduce


def calculate_accuracy(data):
    results = {}

    cnts = []
    for i in range(5000):
        cnt = 0
        for item in data:
            if random.choice(item['labels']) == 1:
                cnt += 1
        cnts.append(cnt)

    cnt = sum(cnts) / len(cnts)
    print(f'random {cnt} / {len(data)}, rate: {cnt / len(data) * 100}%')
    results['random'] = {
        'correct': cnt,
        'total': len(data),
        'accuracy': cnt / len(data),
    }

    cnt = 0
    for item in data:
        labels = random.sample(item['labels'], min(16, len(item['labels'])))
        if sum(labels) >= 1:
            cnt += 1
    print(f'pass@16 {cnt} / {len(data)}, rate: {cnt / len(data) * 100}%')
    results['pass@16'] = {
        'correct': cnt,
        'total': len(data),
        'accuracy': cnt / len(data),
    }

    cnt = 0
    for item in data:
        labels = random.sample(item['labels'], min(8, len(item['labels'])))
        if sum(labels) >= 1:
            cnt += 1
    print(f'pass@8 {cnt} / {len(data)}, rate: {cnt / len(data) * 100}%')
    results['pass@8'] = {
        'correct': cnt,
        'total': len(data),
        'accuracy': cnt / len(data),
    }

    cnt = 0
    for item in data:
        labels = random.sample(item['labels'], min(4, len(item['labels'])))
        if sum(labels) >= 1:
            cnt += 1
    print(f'pass@4 {cnt} / {len(data)}, rate: {cnt / len(data) * 100}%')
    results['pass@4'] = {
        'correct': cnt,
        'total': len(data),
        'accuracy': cnt / len(data),
    }

    cnt = 0
    for item in data:
        labels = random.sample(item['labels'], min(2, len(item['labels'])))
        if sum(labels) >= 1:
            cnt += 1
    print(f'pass@2 {cnt} / {len(data)}, rate: {cnt / len(data) * 100}%')
    results['pass@2'] = {
        'correct': cnt,
        'total': len(data),
        'accuracy': cnt / len(data),
    }

    cnt = 0
    for item in data:
        prm_score = list(map(lambda x: min(x) if x else 0, item['prm_scores']))
        if item['labels'][prm_score.index(max(prm_score))] == 1:
            cnt += 1
    print(f'prm accuracy min {cnt} / {len(data)}, rate: {cnt / len(data) * 100}%')
    results['min'] = {'correct': cnt, 'total': len(data), 'accuracy': cnt / len(data)}

    cnt = 0
    for item in data:
        prm_score = list(map(lambda x: x[-1] if x else 0, item['prm_scores']))
        if item['labels'][prm_score.index(max(prm_score))] == 1:
            cnt += 1
    print(f'prm accuracy last {cnt} / {len(data)}, rate: {cnt / len(data) * 100}%')
    results['last'] = {'correct': cnt, 'total': len(data), 'accuracy': cnt / len(data)}

    cnt = 0
    for item in data:
        prm_score = list(
            map(lambda x: reduce(lambda a, b: a * b, x) if x else 0, item['prm_scores'])
        )
        if item['labels'][prm_score.index(max(prm_score))] == 1:
            cnt += 1
    print(f'prm accuracy product {cnt} / {len(data)}, rate: {cnt / len(data) * 100}%')
    results['product'] = {
        'correct': cnt,
        'total': len(data),
        'accuracy': cnt / len(data),
    }

    cnt = 0
    for item in data:
        prm_score = list(map(lambda x: sum(x) / len(x) if x else 0, item['prm_scores']))
        if item['labels'][prm_score.index(max(prm_score))] == 1:
            cnt += 1
    print(f'prm accuracy average {cnt} / {len(data)}, rate: {cnt / len(data) * 100}%')
    results['average'] = {
        'correct': cnt,
        'total': len(data),
        'accuracy': cnt / len(data),
    }

    cnt = 0
    for item in data:
        prm_score = list(
            map(
                lambda x: (
                    sum(list(map(lambda a: math.log(a) if a != 0 else -9999, x)))
                    if x
                    else 0
                ),
                item['prm_scores'],
            )
        )
        if item['labels'][prm_score.index(max(prm_score))] == 1:
            cnt += 1
    print(
        f'prm accuracy sum_logprob {cnt} / {len(data)}, rate: {cnt / len(data) * 100}%'
    )
    results['sum_logprob'] = {
        'correct': cnt,
        'total': len(data),
        'accuracy': cnt / len(data),
    }

    cnt = 0
    for item in data:
        prm_score = list(map(lambda x: max(x) if x else 0, item['prm_scores']))
        if item['labels'][prm_score.index(max(prm_score))] == 1:
            cnt += 1
    print(f'prm accuracy max {cnt} / {len(data)}, rate: {cnt / len(data) * 100}%')
    results['max'] = {'correct': cnt, 'total': len(data), 'accuracy': cnt / len(data)}

    cnt = 0
    for item in data:
        try:
            prm_score = list(
                map(
                    lambda x: (
                        sum(
                            list(
                                map(
                                    lambda a: math.log(a / (1 - a)) if a != 1 else 9999,
                                    x,
                                )
                            )
                        )
                        if x
                        else 0
                    ),
                    item['prm_scores'],
                )
            )
        except Exception as e:
            print(e)
            print(item['prm_scores'])
            raise e
        if item['labels'][prm_score.index(max(prm_score))] == 1:
            cnt += 1
    print(f'prm accuracy sum_logit {cnt} / {len(data)}, rate: {cnt / len(data) * 100}%')
    results['sum_logit'] = {
        'correct': cnt,
        'total': len(data),
        'accuracy': cnt / len(data),
    }

    cnt = 0
    for item in data:
        prm_score = list(
            map(
                lambda x: (
                    sum(list(map(lambda a: (a / (1 - a)) if a != 1 else 9999, x)))
                    / len(x)
                    if x
                    else 0
                ),
                item['prm_scores'],
            )
        )
        if item['labels'][prm_score.index(max(prm_score))] == 1:
            cnt += 1
    print(f'prm accuracy mean_odd {cnt} / {len(data)}, rate: {cnt / len(data) * 100}%')
    results['mean_odd'] = {
        'correct': cnt,
        'total': len(data),
        'accuracy': cnt / len(data),
    }

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--output_file', type=str, default='')
    args = parser.parse_args()

    result_file = os.path.join(args.output_dir, args.output_file)

    print(f'Reading {result_file}...')
    results = calculate_accuracy(json.load(open(result_file)))

    print(f"Saving results to {result_file.replace('.json', f'_score.json')}...")
    json.dump(
        results,
        open(result_file.replace('.json', f'_score.json'), 'w'),
        indent=4,
        ensure_ascii=False,
    )
    print(f'Results saved.')
