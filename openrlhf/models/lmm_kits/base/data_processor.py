import json
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict
import torch
from transformers.processing_utils import ProcessorMixin
from qwen_vl_utils import process_vision_info

class BaseDataProcessor(ABC):
    def __init__(self, processor: ProcessorMixin,min_pixels:int,max_pixels:int):
        super().__init__()
        self.processor = processor
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
    @abstractmethod
    def __call__(
        self,
        messages: Union[Dict, List[str], str],
        max_length: int,
        padding: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        return_tensors: Optional[str] = "pt",
        add_special_tokens: Optional[bool] = False,
        truncation: Optional[bool] = True,
    ) -> Dict:
        raise NotImplementedError

    def _add_pixel_bounds(self,messages:List[Dict]) -> List[Dict]:
       DEFAULT_MIN_PIXELS = self.min_pixels
       DEFAULT_MAX_PIXELS = self.max_pixels

       def process_content(content):
           if isinstance(content, list):
               for item in content:
                   if isinstance(item, dict) and item.get("type") == "image":
                       if "min_pixels" not in item:
                           item["min_pixels"] = DEFAULT_MIN_PIXELS
                       if "max_pixels" not in item:
                           item["max_pixels"] = DEFAULT_MAX_PIXELS
           return content

       for message in messages:
           for msg in message:
               msg["content"] = process_content(msg["content"])
       return messages

    @abstractmethod
    def make_input_batch(self, inputs: List[Dict]) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def split_input_batch(self, batch: Dict) -> List[Dict]:
        raise NotImplementedError

    def _format_messages(self, messages: Union[Dict, List[str], str]) -> List[Dict]:
        if isinstance(messages, list) and isinstance(messages[0], str):
            formated_messages = [json.loads(m) for m in messages]
        elif isinstance(messages, str):
            formated_messages = [json.loads(messages)]
        elif isinstance(messages, dict):
            formated_messages = [messages]
        else:
            raise ValueError("Invalid messages format, must be a list of strings or a string or a dict")
        return self._add_pixel_bounds(formated_messages)

    def apply_chat_template(
        self,
        messages: Union[Dict, List[str], str],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> List[str]:
        messages = self._format_messages(messages)
        
        return self.processor.apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
        )

    def get_images_from_messages(
        self, messages: Union[Dict, List[str], str]
    ) -> List[Dict]:
        messages = self._format_messages(messages)
        image_inputs, _ = process_vision_info(messages)
        return image_inputs


    @property
    def pad_token_id(self) -> int:
        return self.processor.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self.processor.tokenizer.eos_token_id

    @property
    def tokenizer(self):
        return self.processor.tokenizer