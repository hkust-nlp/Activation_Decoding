Activation_trace: Decoding by sharpness in Large Language Models
===

<!-- [![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)
[![Arxiv](https://img.shields.io/badge/arXiv-2309.03883-B21A1B)](https://arxiv.org/abs/2309.03883)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-blue)](https://github.com/huggingface/transformers)
[![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/YungSungChuang/status/1701623359153316255)
[![GitHub Stars](https://img.shields.io/github/stars/voidism/DoLa?style=social)](https://github.com/voidism/DoLa/stargazers)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/voidism/DoLa/blob/master/dola_evaluation.ipynb) -->

<!-- Code for the paper "DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models" -->

<!-- Paper: https://arxiv.org/abs/2309.03883  
Authors: [Yung-Sung Chuang](https://people.csail.mit.edu/yungsung/) $^\dagger$, [Yujia Xie](https://sites.google.com/view/yujia) $^\ddagger$, [Hongyin Luo](https://luohongyin.github.io/) $^\dagger$, [Yoon Kim](https://people.csail.mit.edu/yoonkim/) $^\dagger$, [James Glass](https://people.csail.mit.edu/jrg/) $^\dagger$, [Pengcheng He](https://scholar.google.com/citations?user=TS1RoxAAAAAJ&hl=en) $^\ddagger$  
$^\dagger$ Massachusetts Institute of Technology, $^\ddagger$ Microsoft -->

## Overview

<!-- ![DoLa](figure.png) -->

<!-- Despite their impressive capabilities, large language models (LLMs) are prone to hallucinations, i.e., generating content that deviates from  facts seen during pretraining. We propose a simple decoding strategy for reducing hallucinations with pretrained LLMs that does not require conditioning on retrieved external knowledge nor additional fine-tuning. Our approach obtains the next-token distribution by contrasting the differences in logits obtained from projecting the later layers versus earlier layers to the vocabulary space, exploiting the fact that factual knowledge in an LLMs has generally been shown to be localized to particular transformer layers. We find that this **D**ecoding by c**o**ntrasting **La**yers (DoLA) approach is able to better surface factual knowledge and reduce the generation of incorrect facts.  DoLA consistently improves the truthfulness across multiple choices tasks and open-ended generation tasks, for example improving performance of LLaMA family models on TruthfulQA by 12-17\% absolute points, demonstrating its potential in making LLMs reliably generate truthful facts. -->

In this study, we aim to understand the underlying mechanisms of LLM hallucinations from the perspective of inner representations. We discover a pattern of hidden states associated with hallucinations: correct generations tend to have sharper context activations in the hidden states of the in-context tokens, compared to that of the incorrect generations. This pattern sheds a light on the interpretability of the mechanisms underlying the factual behaviors of LLMs. We then propose an entropy-based metric to quantify the “sharpness” among the in-context hidden states and incorporate this intuition into the decoding process, i.e, use the entropy value to adjust the next token prediction distribution to reduce the produced factual errors and improve the overall quality of the generated text. Experiments on knowledge-seeking datasets (Natural Questions,HotpotQA, TriviaQA) and hallucination benchmark (TruthfulQA) demonstrate our consistent effectiveness, e.g., achieving an improvement up to 8.6 absolute points on TruthfulQA.
## Setup

```
pip install -e transformers
pip install datasets
pip install accelerate
pip install openai # -> only for truthfulqa and gpt4_eval
```

## Experiments

### Arguments

| Argument          | Example           | Description   |
| ----------------- | ----------------- | ------------- |
| `--model-name`    | `daryl149/Llama-2-7b-chat-hf` | Specifies the model you want to use, currently we only support LLaMA-v1. |
| `--data-path`     | `/path/to/dataset` | Path to the dataset file or folder. |
| `--output-path`   | `output-path.json` | Where to store the output results. |
| `--num-gpus`      | `1` | Number of GPUs to use, `1/2/4/8` for `7B/13B/30B/65B` model sizes respectively.  |
| `--max_gpu_memory`| `27` | Maximum GPU memory size (in GiB) to allocate. Default: 27 (for 32G V100).  |
                                                                                   
### TrivialQA 
<!-- Please download the data file `wiki_factor.csv` from https://github.com/AI21Labs/factor -->

#### Visualization
If you want to visualize the activation pattern.
```bash
cd vis
bash run.sh
# you can change the dataset in run.sh
```


#### activation
```bash
bash run_trqa_ours.sh
```


#### Baseline
```bash
bash run_trqa_baselines.sh
```



### TruthfulQA

To evaluate the open-ended generation result of TruthfulQA, we need to finetune two GPT-3 curie models through OpenAI API:

```
openai api fine_tunes.create -t finetune_truth.jsonl -m curie --n_epochs 5 --batch_size 21 --learning_rate_multiplier 0.1
openai api fine_tunes.create -t finetune_info.jsonl -m curie --n_epochs 5 --batch_size 21 --learning_rate_multiplier 0.1
```

After finetuning, we can obtain the finetuned model names by `openai api fine_tunes.list | grep fine_tuned_model`.

Create a config file `gpt3.config.json` like this:

```json
{"gpt_info": "curie:ft-xxxxxxxxxx",
"gpt_truth": "curie:ft-xxxxxxxxxx",
"api_key": "xxxxxxx"}
```

Add the argument `--do-rating --gpt3-config gpt3.config.json` for GPT-3 evaluation.


## Reference Repositories
- FastChat: https://github.com/lm-sys/FastChat
- ContrastiveDecoding: https://github.com/XiangLi1999/ContrastiveDecoding
- TruthfulQA: https://github.com/sylinrl/TruthfulQA
- zero_shot_cot: https://github.com/kojima-takeshi188/zero_shot_cot
- FederatedScope: https://github.com/alibaba/FederatedScope

## Citation

Please cite our paper if it's helpful to your work!
```

``` --># Activation_decoding
