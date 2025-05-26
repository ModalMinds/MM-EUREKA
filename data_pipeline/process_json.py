import json
import math
import os


def split_questions_uniformly(input_file, output_dir, num_splits):
    with open(input_file, 'r') as f:
        input_data = json.load(f)

    num_data = len(input_data)
    data_per_file = math.ceil(num_data / num_splits)

    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_splits):
        start_idx = i * data_per_file
        end_idx = min(start_idx + data_per_file, num_data)
        data_subset = input_data[start_idx:end_idx]

        output_filepath = os.path.join(output_dir, f'questions_part_{i + 1}.json')
        print(f'Saving {len(data_subset)} questions to {output_filepath}')

        with open(output_filepath, 'w') as f_out:
            json.dump(data_subset, f_out, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    split_questions_uniformly('/path/to/input/file', 'split_dir', 16)
