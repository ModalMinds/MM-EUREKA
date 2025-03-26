from transformers import AutoConfig, AutoProcessor, AutoModel
import importlib
import os

def _get_kit_root_path(pretrain_or_model=None,model_type=None):
    if model_type is None:
        config = AutoConfig.from_pretrained(pretrain_or_model)
        model_type = config.model_type
    root_path = f".models.lmm_kits.{model_type}"
    return root_path

def _get_hf_processor(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    min_pixels = strategy.args.min_pixels
    max_pixels = strategy.args.max_pixels
    processor = AutoProcessor.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast, min_pixels=min_pixels, max_pixels=max_pixels)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
    return processor

def get_data_processor(pretrain_or_model, model, padding_side="left", strategy=None, use_fast=True):
    root_path = _get_kit_root_path(pretrain_or_model)
    module = importlib.import_module(f"{root_path}.data_processor",package="openrlhf")
    data_processor_cls = getattr(module, "DataProcessor")
    hf_processor = _get_hf_processor(pretrain_or_model, model, padding_side, strategy,use_fast=use_fast)
    data_processor = data_processor_cls(hf_processor,min_pixels=strategy.args.min_pixels,max_pixels=strategy.args.max_pixels)
    return data_processor

def load_patch(pretrain_or_model=None,model_type=None):
    root_path = _get_kit_root_path(pretrain_or_model,model_type)
    module = importlib.import_module(f"{root_path}.patch",package="openrlhf")
    Patch = getattr(module, "Patch")
    Patch.load_all_patches()

def get_generation_cls(config):
    model_type = config.model_type
    load_patch(model_type=model_type)
    model_arch = AutoModel._model_mapping[type(config)].__name__
    if model_arch.endswith("ForCausalLM") or \
    model_arch.endswith("ForConditionalGeneration"):
        return AutoModel._model_mapping[type(config)]
    elif model_arch.endswith("Model"):
        possible_arch = [model_arch.replace("Model", "ForCausalLM"), model_arch.replace("Model", "ForConditionalGeneration")]
        module = importlib.import_module(f".models.{model_type}.modeling_{model_type}",package="transformers")
        for arch in possible_arch:
            model_cls = getattr(module, arch, None)
            if model_cls is not None:
                return model_cls
        raise ValueError(f"Cannot find ForCausalLM or ForConditionalGeneration class for {model_arch}")
    else:
        raise ValueError(f"Unexpected model architecture {model_arch}")