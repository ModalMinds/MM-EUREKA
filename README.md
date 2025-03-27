# 🎯 Overview

<div align="center">
<p align="center">
  📖<a href="https://jagged-court-d9d.notion.site/MM-Eureka-Qwen-1c13cc5a384880ffbd2de24e1dee052d">Report</a> |
  📊<a href="https://huggingface.co/datasets/FanqingM/MM-Eureka-Dataset">Datasets</a> |
  🤗<a href="https://huggingface.co/FanqingM/MM-Eureka-Qwen-7B/tree/main">MM-Eureka-Qwen-7B</a> |
</p>
</div>
We present MM-Eureka-Qwen, a multimodal reasoning model that successfully extends large-scale rule-based reinforcement learning (RL) to multimodal reasoning. Compared to the previous version of MM-EUREKA based on InternVL, we have made improvements in model architecture, algorithms, and data. Using only non-in-domain training data, MM-Eureka-Qwen achieves significant improvements over Qwen-2.5-VL-Instruct-7B across multiple benchmarks (e.g. MathVista 73.0). We release all our codes, models, data, etc. at https://github.com/ModalMinds/MM-EUREKA/tree/MM-EUREKA-Qwen.

**Improvements:**

1. We further iterate the codebase to support algorithms including Online Filter, [ADORA](https://github.com/ShadeCloak/ADORA?tab=readme-ov-file), and [DAPO](https://arxiv.org/abs/2503.14476).
2. We expand our K12 dataset from MM-EUREKA-Dataaset, from 8k to 15k high-quality K12 samples.
3. We train the MM-Eureka-Qwen-7B model, achieving better results with significantly lower cost than the previous version. We **open-sourced** **our training code and models** and hope to facilitate future studies on multimodal reasoning.



# 🤗 MM-EUREKA-Qwen

Based on the key factors identified by https://github.com/ModalMinds/MM-EUREKA for achieving stable training, we enhanced the model, dataset, and algorithmic modules. Specifically, we maintained the strategy of omitting the KL divergence term and applying data filtering, while implementing the following critical modifications:

- The base model was upgraded from InternVL2.5-8B-Instruct to the more powerful QwenVL2.5-7B-Instruct.
- The Vision Transformer (ViT) module was kept fixed during training.
- The underlying RL algorithm was replaced with [GRPO](https://arxiv.org/pdf/2402.03300), instead of the previously used RLOO.
- The data filtering strategy was transitioned from an offline approach to an online approach.
- Additional data from the K12 dataset was collected, expanding the total dataset size to 15,000 samples.

Finally, MM-EUREKA-Qwen achieves 73.0 on MathVista, surpassing the original Qwen-2.5-VL by 4.8%. OlympidBench is OE_MM_maths_en_comp.

|                             | Base Model                 | MathVista       | MathVerse       | MathVision  | OlympidBench  | K12              |
| --------------------------- | -------------------------- | --------------- | --------------- | ----------- | ------------- | ---------------- |
| Qwen2.5-VL-7B-Instruct      | -                          | 68.2            | 47.9            | 25.4        | 15.3          | 36.0             |
| InternVL2.5-VL-8B-Instruct  | -                          | 64.4            | 39.5            | 19.7        | 8.0           | 24.8             |
| InternVL2.5-VL-38B-Instruct | -                          | 71.9            | 49.4            | **31.8**    | 29.3          | 37.2             |
| MM-EUREKA-InternVL-8B       | InternVL2.5-VL-7B-Instruct | 67.1            | 40.4            | 22.2        | 8.6           | 27.0             |
| MM-EUREKA-Qwen-7B           | Qwen2.5-VL-7B-Instruct     | **73.0 (+4.8)** | **50.3 (+2.4)** | 26.9 (+1.5) | 25.3（+10.0） | **48.6 (+12.6)** |



# 📦 Environment

```
git clone https://github.com/ModalMinds/MM-EUREKA.git

cd MM-EUREKA

git checkout MM-EUREKA-Qwen

pip install -e .[vllm]

pip install flash_attn --no-build-isolation
```

# 🌐 Start Training

Before starting your own training, ensure that the paths in the provided training scripts are correctly set and that environment variables like `$MASTER_ADDR` and `$NODE_RANK` are properly configured.

**start MM-Eureka-Qwen training**

- for single node

  ```shell
  sh examples/scripts/train_mm_eureka_qwen_7b_single_node.sh
  ```

- for multiple node

  ```shell
  sh examples/scripts/train_mm_eureka_qwen_7b_multi_node.sh
  ```



# 🎓 Acknowledgements

We acknowledge the outstanding open-source contributions from [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [LMM-R1](https://github.com/TideDra/lmm-r1) and [vLLM](https://github.com/vllm-project/vllm). We also extend our gratitude to [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) and [InternVL](https://github.com/OpenGVLab/InternVL) and [QwenVL](https://github.com/QwenLM/Qwen-VL) for their open-source techniques and base models, which have enabled us to further our exploration.

# 📜 Citation

```
@article{meng2025mm,
  title={MM-Eureka: Exploring Visual Aha Moment with Rule-based Large-scale Reinforcement Learning},
  author={Meng, Fanqing and Du, Lingxiao and Liu, Zongkai and Zhou, Zhixiang and Lu, Quanfeng and Fu, Daocheng and Shi, Botian and Wang, Wenhai and He, Junjun and Zhang, Kaipeng and others},
  journal={arXiv preprint arXiv:2503.07365},
  year={2025}
}
```
