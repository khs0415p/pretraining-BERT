import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from config import Config
from utils import *
from transformers import BertConfig
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
    
)
from safetensors.torch import save_model

from model.modeling_bert import BertForPreTraining

class Trainer:
    """Trainer for BERT"""
    def __init__(
            self,
            config: Config,
            logger: logging.Logger,
            tokenizer: PreTrainedTokenizerBase,
            train_dataset: Dataset,
            valid_dataset: Dataset = None,
            device: str = torch.device,
            ) -> None:
        self.bert_config = bert_config
        self.config = config
        self.logger = logger

        # Defin Device
        self.device = device

        # Tokenizer
        self.tokenizer = tokenizer

        # Load Data
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        self.dataloaders = {}
        self.dataloaders['train'] = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=DataCollatorWithPadding(tokenizer=self.tokenizer)
        )
        self.dataloaders['valid'] = DataLoader(
            self.valid_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=DataCollatorWithPadding(tokenizer=self.tokenizer)
        )

        # Load Model
        bert_config = BertConfig(vocab_size=self.config.vocab_size)
        self.model = BertForPreTraining(bert_config)
        if self.config.use_backbone:
            pretrained_model = BertForPreTraining.from_pretrained(self.config.backbone)
            layers = get_pretrained_weights(self.model, pretrained_model)
            self.model.load_state_dict(layers)
            del pretrained_model

        # Define Optimizer
        self.exclude_from_weight_decay = self.config.exclude_from_weight_decay
        optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in self.model.named_parameters()
                            if not any(nd in n for nd in self.exclude_from_weight_decay)
                        ],
                        "weight_decay": self.config.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in self.model.named_parameters()
                            if any(nd in n for nd in self.exclude_from_weight_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
        self.optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.lr,
            betas=self.config.betas,
            eps=self.config.eps,
        )

        # Define Scheduler
        total_steps = len(self.dataloaders['train']) * self.config.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.num_warmup_steps,
            num_training_steps=total_steps
            )

        # Define Losses
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        best_val_loss = float('inf')
        train_loss_history, valid_loss_history = [], []
        for epoch in range(self.config.epochs):
            print()
            for phase in ['train', 'valid']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                epoch_loss = 0
                for i, batch in enumerate(self.dataloaders[phase]):
                    self.optimizer.zero_grad()
                    # Send device
                    batch = {k:v.to(self.device) for k, v in batch.items()}

                    if phase == "train":
                        total_loss, mlm_loss, nsp_loss = self._training_step(batch)                
                    else:
                        total_loss, mlm_loss, nsp_loss = self._validation_step(batch)
                        best_val_loss = self._save_checkpoint(best_val_loss, total_loss, (epoch * len(self.dataloaders[phase].dataset)) + i)

                    if i % self.config.log_step == 0:
                        self.logger.info(f"\n{'Epoch':<15}{epoch + 1}\n{'Phase':<15}{phase}\n{'Step':<15}{i}\n{'Total Loss':<15}{total_loss:.4f}\n{'MLM Loss':<15}{mlm_loss:.4f}\n{'NSP Loss':<15}{nsp_loss:.4f}\n")
                        if phase == "train":
                            # step / loss
                            train_loss_history.append([(epoch * len(self.dataloaders[phase].dataset)) + i,total_loss])
                        else:
                            valid_loss_history.append([(epoch * len(self.dataloaders[phase].dataset)) + i,total_loss])
                    epoch_loss += total_loss * batch.size(0)
                epoch_loss = epoch_loss / len(self.dataloaders[phase].dataset)
                self.logger.info(f"\n{'Epoch Loss':<15}{epoch_loss:.4f}")
        self._save_checkpoint(last_save=True)
        
        return valid_loss_history, train_loss_history
            
    
    def _training_step(self, batch):
        # prediction_logits : vector of tokens, seq_relationship_logits : classification output of nsp
        output = self.model(**batch)
        ## Losses
        # (B, L, V) -> (B*L, V)
        mlm_loss = self.criterion(output['prediction_logits'].view(-1, output['prediction_logits'].size(-1)), batch['mlm_label'].view(-1))
        nsp_loss = self.criterion(output['seq_relationship_logits'], batch['nsp_label'].view(-1))
        total_loss = mlm_loss + nsp_loss

        mlm_loss.backward()
        nsp_loss.backward()
        clip_grad_norm_(self.model.parameters(), self.config.max_norm)
        self.optimizer.step()

        return total_loss.item(), mlm_loss.item(), nsp_loss.item()
    
    @torch.no_grad()
    def _validation_step(self, batch):
        output = self.model(**batch)
        # Losses
        mlm_loss = self.criterion(output['prediction_logits'].view(-1, output['prediction_logits'].size(-1)), batch['mlm_label'].view(-1)).detach()
        nsp_loss = self.criterion(output['seq_relationship_logits'], batch['nsp_label'].view(-1)).detach()
        total_loss = mlm_loss + nsp_loss

        return total_loss.item(), mlm_loss.item(), nsp_loss.item()
    
    def _save_checkpoint(
            self,
            best_loss: float = None,
            loss: float = None,
            step: float = None,
            last_save: bool = False
            ):
        if last_save:
            torch.save(self.model.state_dict(), 'results/pytorch_model.bin')
            save_model(self.model, "results/model.safetensors")
            self.model.config_class().to_json_file('results/config.json')
            return
        
        if best_loss > loss:
            torch.save(self.model.state_dict(), 'results/best/pytorch_model.bin')
            save_model(self.model, "results/best/model.safetensors")
            self.model.config_class().to_json_file('results/config.json')
            self.logger.info(f"Save the model at {step} steps.")
            return loss
        return best_loss
