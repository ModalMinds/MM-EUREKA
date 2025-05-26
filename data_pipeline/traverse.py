import json
import os
import random

SEP = '<prm>'

THRESHOLD = 0.00


def random_select_solutions(solutions, n):
    if len(solutions) <= n:
        return solutions
    return random.sample(solutions, n)


def completion_too_short(string, word_count_thres=10):
    return len(string.split(' ')) <= word_count_thres


def remove_redundant(solutions):
    unique_solutions = {}
    for solution in solutions:
        key = solution['process']
        if key not in unique_solutions:
            unique_solutions[key] = solution

    return list(unique_solutions.values())


def traverse(root):
    question = root['question']
    answer = root['answer']
    image_path = root['image_path']
    positive = []
    negative = []
    res = []

    def dfs(node, solution_prefix, labels):
        partial_solution = list(map(str.strip, node['partial_solution']))
        partial_solution = '\n\n'.join(partial_solution)
        solution_prefix = solution_prefix + partial_solution + SEP + '\n\n'
        labels = labels + [node['mc_value']]
        if node['mc_value'] <= THRESHOLD:
            if not completion_too_short(solution_prefix):
                negative.append(
                    {
                        'question': question,
                        'answer': answer,
                        'image_path': image_path,
                        'process': solution_prefix.strip(),
                        'labels': labels,
                    }
                )
            return
        if node['children']:
            for child in node['children']:
                dfs(child, solution_prefix, labels)
        else:
            if not completion_too_short(solution_prefix):
                positive.append(
                    {
                        'question': question,
                        'answer': answer,
                        'image_path': image_path,
                        'process': solution_prefix.strip(),
                        'labels': labels,
                    }
                )
        return

    if root['children']:
        for child in root['children']:
            dfs(child, '', [])

    negative = remove_redundant(negative)
    positive = remove_redundant(positive)

    res.extend(negative)

    if len(positive) < len(negative):
        res.extend(positive)
    else:
        res.extend(random.sample(positive, len(negative)))
    return res


if __name__ == '__main__':
    root_dir = 'outputs'
    res = []
    for file in os.listdir(root_dir):
        print(file)
        with open(os.path.join(root_dir, file), 'r') as f:
            data = json.load(f)
        if data['mc_value'] != 0.0 and data['mc_value'] != 1.0:
            res.extend(traverse(data))

    print(len(res))
    json.dump(res, open('output_file.json', 'w'), ensure_ascii=False, indent=4)
