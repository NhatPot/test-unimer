---
library_name: peft
license: other
base_model: Qwen/Qwen2.5-VL-3B-Instruct
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: uni-mumer_qlora
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# uni-mumer_qlora

This model is a fine-tuned version of [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) on the parquet_crohme_train, the parquet_crohme_train_can and the parquet_hme100k_train datasets.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 2
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 64
- total_train_batch_size: 128
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 1

### Training results



### Framework versions

- PEFT 0.15.2
- Transformers 4.52.4
- Pytorch 2.6.0+cu124
- Datasets 3.2.0
- Tokenizers 0.21.4