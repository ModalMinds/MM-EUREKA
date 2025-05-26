import argparse
import itertools
import json
import os
import random
import time

import torch
from PIL import Image
from tqdm import tqdm

from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess

ds_collections = {
    'mathvision_prm': {
        'root': '/path/to/image/root',
        'annotation': '/path/to/rollout/file',
    }
}


def collate_fn(batches):
    pixel_values = batches[0]['pixel_values']
    prompts = batches[0]['prompts']
    steps_lens = batches[0]['steps_lens']
    data_items = batches[0]['data_item']
    return pixel_values, prompts, steps_lens, data_items


class MathVisionPRMDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        root,
        annotation,
        input_size=224,
        dynamic_image_size=False,
        use_thumbnail=False,
        max_num=6,
    ):
        self.root = root
        self.data = json.load(open(annotation))
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        image = os.path.join(self.root, os.path.basename(data_item['image']))

        image = Image.open(image).convert('RGB')
        if self.dynamic_image_size:
            images = dynamic_preprocess(
                image,
                image_size=self.input_size,
                use_thumbnail=self.use_thumbnail,
                max_num=self.max_num,
            )
        else:
            images = [image]
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        options = ''
        if len(data_item['options']) > 0:
            assert len(data_item['options']) == 5, data_item
            if ''.join(data_item['options']) != 'ABCDE':
                options = f"(A) {data_item['options'][0]}\n(B) {data_item['options'][1]}\n(C) {data_item['options'][2]}\n(D) {data_item['options'][3]}\n(E) {data_item['options'][4]}\n"
        question = f"{data_item['question']}\n{options}"

        prompts = []
        steps_lens = []
        for solution_split in data_item['solutions_splits']:
            solution = '<prm>'.join(solution_split) + '<prm>'
            prompt = f'Question: {question}\nProcess: {solution}'
            prompts.append(prompt)
            steps_lens.append(len(solution_split))

        return {
            'pixel_values': pixel_values,
            'prompts': prompts,
            'steps_lens': steps_lens,
            'data_item': data_item,
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(
            size, self._world_size, self._rank
        )

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[: rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def evaluate_chat_model():
    random.seed(args.seed)

    for ds_name in args.datasets:
        dataset = MathVisionPRMDataset(
            root=ds_collections[ds_name]['root'],
            annotation=ds_collections[ds_name]['annotation'],
            input_size=image_size,
            dynamic_image_size=args.dynamic,
            use_thumbnail=use_thumbnail,
            max_num=args.max_num,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

        outputs = []
        for idx, (pixel_values, prompts, steps_lens, data_item) in tqdm(
            enumerate(dataloader)
        ):
            pixel_values = pixel_values.to(torch.bfloat16).cuda()

            prm_scores_flattened = []
            for i in range(0, len(prompts), args.mini_batch_size):
                curr_bs = min(args.mini_batch_size, len(prompts) - i)
                output = model.batch_prm(
                    tokenizer=tokenizer,
                    pixel_values=torch.cat([pixel_values] * curr_bs, dim=0),
                    questions=prompts[i : i + curr_bs],
                    num_patches_list=[pixel_values.shape[0]] * curr_bs,
                    verbose=True,
                )
                prm_scores_flattened.extend(output.tolist())

            data_item['prm_scores'] = []
            curr_len = 0
            for i in range(len(steps_lens)):
                data_item['prm_scores'].append(
                    prm_scores_flattened[curr_len : curr_len + steps_lens[i]]
                )
                curr_len += steps_lens[i]

            for i in range(len(data_item['prm_scores'])):
                assert len(data_item['prm_scores'][i]) == steps_lens[i]

            print(f'Pred: {data_item["prm_scores"]}')
            outputs.append(data_item)

            if idx % 50 == 0:
                torch.distributed.barrier()

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{ds_name}_{time_prefix}.json'
            output_path = os.path.join(args.out_dir, results_file)
            json.dump(
                merged_outputs, open(output_path, 'w'), indent=4, ensure_ascii=False
            )
            print('Results saved to {}'.format(output_path))

            cmd = f'python eval/prm/extract_calculate.py --output_file {results_file} --output_dir {args.out_dir}'
            print(cmd)
            os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='')
    parser.add_argument('--mini-batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true', default=True)
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    model, tokenizer = load_model_and_tokenizer(args)

    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f'[test] total_params: {total_params}B')
    print(f'[test] image_size: {image_size}')
    print(f'[test] template: {model.config.template}')
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] use_thumbnail: {use_thumbnail}')

    evaluate_chat_model()
