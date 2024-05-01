import os
import torch
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
from datasets import DATASETS
from config import *
from model import *
from dataloader import *
from trainer import *

from transformers import BitsAndBytesConfig
from pytorch_lightning import seed_everything
from model import LlamaForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
)


try:
    os.environ['WANDB_PROJECT'] = PROJECT_NAME
except:
    print('WANDB_PROJECT not available, please set it in config.py')


def main(args, export_root=None):
    seed_everything(args.seed)
    if export_root == None:
        export_root = EXPERIMENT_ROOT + '/' + args.llm_base_model.split('/')[-1] + '/' + args.dataset_code

    train_loader, val_loader, test_loader, tokenizer, test_retrieval = dataloader_factory(args)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = LlamaForCausalLM.from_pretrained(
        args.llm_base_model,
        quantization_config=bnb_config,
        device_map='auto',
        cache_dir=args.llm_cache_dir,
    )

    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias='none',
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    model.config.use_cache = False
    trainer = LLMTrainer(args, model, train_loader, val_loader, test_loader, tokenizer, export_root, args.use_wandb)
    
    trainer.train()
    trainer.test(test_retrieval)


if __name__ == "__main__":

    # Experiments with different datasets and tags
    dataset_codes = ['beauty','ml-100k','games']
    for dataset_code in dataset_codes:

        print(f"---TRAINING {dataset_code}---")

        args.dataset_code = dataset_code
        args.model_code = 'llm'

        # Retriever
        args.llm_retrieved_path = EXPERIMENT_ROOT + '/lru/' + args.dataset_code + '/0_0.5'

        set_template(args)

        # Save experiment results
        export_root = EXPERIMENT_ROOT + '/' + args.llm_base_model.split('/')[-1] + '/' + args.dataset_code + '_tags'

        print(args)

        main(args, export_root=export_root)

        # Limpiar la memoria de la GPU después de cada experimento
        # torch.cuda.empty_cache()  # Libera la memoria de la GPU
