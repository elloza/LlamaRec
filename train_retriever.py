import os
import torch
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import wandb
import argparse

from config import *
from model import *
from dataloader import *
from trainer import *

from pytorch_lightning import seed_everything

try:
    os.environ['WANDB_PROJECT'] = PROJECT_NAME
except:
    print('WANDB_PROJECT not available, please set it in config.py')


def main(args, export_root=None):
    seed_everything(args.seed)

    train_loader, val_loader, test_loader = dataloader_factory(args)

    if export_root == None:
        export_root = EXPERIMENT_ROOT + '/' + args.model_code + '/' + args.dataset_code

    if args.model_code == 'lru':
        model = LRURec(args)
        trainer = LRUTrainer(args, model, train_loader, val_loader, test_loader, export_root, args.use_wandb)

    elif args.model_code == 'sas':
        model = SASRec(args)
        trainer = SASTrainer(args, model, train_loader, val_loader, test_loader, export_root)
    
    trainer.train()
    trainer.test()
    
    # the next line generates val / test candidates for reranking
    trainer.generate_candidates(os.path.join(export_root, 'retrieved.pkl'))
    trainer.logger_service.complete()


if __name__ == "__main__":
    
    args.model_code = 'lru'
    datasets = ['ml-100k', 'beauty', 'games']

    # Retrain with best hyperparameters for each dataset
    for dataset in datasets:

        print(f"----TRAINING {dataset}----")
        
        args.dataset_code = dataset
        set_template(args)
        # best hyperparameters found (it must set after set_template)
        args.weight_decay = 0
        args.bert_dropout = 0.5
        args.bert_attn_dropout = 0.5
        export_root = EXPERIMENT_ROOT + '/' + args.model_code + '/' + args.dataset_code + '/' + str(args.weight_decay) + '_' + str(args.bert_attn_dropout)
        main(args, export_root=export_root)
    
    """
    # # search for best hyperparameters
    for decay in [0, 0.01]:
        for dropout in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            print("------------------------------------")
            print(f"decay: {decay}, dropout: {dropout}")
            print("------------------------------------")
            args.weight_decay = decay
            args.bert_dropout = dropout
            args.bert_attn_dropout = dropout
            export_root = EXPERIMENT_ROOT + '/' + args.model_code + '/' + args.dataset_code + '/' + str(decay) + '_' + str(dropout)
            main(args, export_root=export_root)
    """