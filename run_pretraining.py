import os
import torch
import random
import numpy
import pickle
import argparse
import logging
import datetime
import warnings
warnings.filterwarnings('ignore')

from config import Config
from trainer import Trainer
from transformers import AutoTokenizer
from dataset import PretrainDataSet


if not os.path.exists('logs/'):
        os.makedirs('logs/', exist_ok=True)

if not os.path.exists('results/'):
    os.makedirs('results/', exist_ok=True)


logger = logging.getLogger("TRAIN BERT")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s\n%(message)s")
file_handler = logging.FileHandler(f"logs/{str(datetime.datetime.now())}.log")
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    numpy.random.seed(seed)
    random.seed(seed)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.continuous and not args.checkpoint:
        raise ValueError('Input checkpoints for continuous learning. --checkpoint checkpoint_path/')
    
    config = Config(args.config_path)
    config.continuous = args.continuous
    config.checkpoint = args.checkpoint
    set_seed(config.seed)

    # Trained Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_save_path)
    config.vocab_size = tokenizer.vocab_size

    # Training
    trainer = Trainer(config, logger, tokenizer, device)
    trainer.train()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str, default='./config.json', help="Path of config")
    parser.add_argument('--continuous', '-con', action="store_true", help="Continuous training from checkpoint")
    parser.add_argument('--checkpoint', '-cp', type=str, help="Path of checkpoint")
    args = parser.parse_args()
    main(args)