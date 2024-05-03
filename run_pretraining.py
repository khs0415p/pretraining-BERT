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

if not os.path.exists('results/best'):
    os.makedirs('results/best', exist_ok=True)

if not os.path.exists('loss'):
    os.makedirs('loss', exist_ok=True)

if not os.path.exists('lrs'):
    os.makedirs('lrs', exist_ok=True)

logger = logging.getLogger("TRAIN BERT")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s|%(name)s%(message)s")
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
    config = Config(args.config_path)
    set_seed(config.seed)

    logger.info(f"\n{'Started Train':>15}\n{'Batch Size':<15}{config.batch_size}\n{'Vocab Size':<15}{config.vocab_size}\n{'Learning Rate':<15}{config.lr}\n{'Optimizer':<15}AdamW")

    # Trained Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_save_path)
    config.vocab_size = tokenizer.vocab_size

    # Training
    trainer = Trainer(config, logger, tokenizer, device)
    valid_loss_history, train_loss_history = trainer.train()
    
    # Save loss & Plot loss
    with open(config.train_loss_path, 'wb') as file:
        pickle.dump(train_loss_history, file)
    with open(config.valid_loss_path, 'wb') as file:
        pickle.dump(valid_loss_history, file)

    if config.visualize_lr:
        with open(config.learning_rate_path, 'wb') as file:
            pickle.dump(trainer.learning_rates)

    
        
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', default='./config.json')
    args = parser.parse_args()
    main(args)