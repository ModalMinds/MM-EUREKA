import logging
import re

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_image

logger = logging.getLogger('main')


class LanguageModel:
    def __init__(
        self,
        model='OpenGVLab/InternVL2_5-8B',
        max_new_tokens=4096,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
    ):
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = 1.05

        logger.info(f'Loading model {self.model}...')
        self.model = LLM(
            model=model,
            trust_remote_code=True,
            tensor_parallel_size=1,
            limit_mm_per_prompt={'image': 8},
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.stop_tokens = ['<|im_end|>\n'.strip()]
        self.stop_token_ids = [
            self.tokenizer.convert_tokens_to_ids(i) for i in self.stop_tokens
        ]
        self.special_tokens = self.tokenizer.all_special_tokens
        self.custom_tokens = ['<step>', '</step>', '<answer>', '</answer>']
        self.special_tokens = [
            token for token in self.special_tokens if token not in self.custom_tokens
        ]
        self.pattern1 = r'|'.join(map(re.escape, self.special_tokens))
        self.pattern2 = r'<step>(.*?)</step>|<answer>(.*?)</answer>'
        logger.info('Model loaded successfully.')

    def generate_results(self, prompt, image_path=None, num_copies=16):
        if '<image>' not in prompt:
            prompt = '<image>\n' + prompt

        messages = [
            {
                'role': 'system',
                'content': '你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。',
            },
            {'role': 'user', 'content': prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = []
        if image_path:
            image = fetch_image('file://' + image_path, allowed_local_media_path='/')
            for _ in range(num_copies):
                inputs.append(
                    {
                        'prompt': prompt,
                        'multi_modal_data': {'image': image},
                    }
                )
        else:
            for _ in range(num_copies):
                inputs.append({'prompt': prompt})

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            stop_token_ids=self.stop_token_ids,
            skip_special_tokens=False,
        )
        model_outputs = self.model.generate(inputs, sampling_params=sampling_params)
        batch_results = []
        for model_output in model_outputs:
            response = re.sub(self.pattern1, '', model_output.outputs[0].text)
            matches = re.findall(self.pattern2, response, re.DOTALL)
            res = (
                [match[0] if match[0] else match[1] for match in matches]
                if matches
                else []
            )
            res = list(map(str.strip, res))
            batch_results.append(res)

        for result in batch_results:
            logger.debug(f'Prompt: {prompt}\nGenerated rollout: {result}')

        return batch_results
