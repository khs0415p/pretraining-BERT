import os
import logging
import pickle
import math
import heapq
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict
from tqdm import tqdm
from config import Config
from utils import *
from transformers import BertConfig, AdamW
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import (
    PreTrainedTokenizerBase,
    AutoModelForMaskedLM,
    get_linear_schedule_with_warmup
)

from models import BertForPreTraining
from models import DistilBertForPretraining, DistilBertConfig



class BertTrainer:
    """Trainer for BERT"""
    def __init__(
            self,
            config: Config,
            logger: logging.Logger,
            tokenizer: PreTrainedTokenizerBase,
            device: str = torch.device,
            ) -> None:
        self.config = config
        self.logger = logger

        # Number of iteration
        self.n_iter = 0

        # Number of checkpoint
        self.save_total_limit = self.config.save_total_limit

        # Saved path
        self.saved_path = []

        # Defin Device
        self.device = device

        # Tokenizer
        self.tokenizer = tokenizer
        
        # Load Model
        if self.config.model_type == "bert":
            bert_config = BertConfig(vocab_size=self.config.vocab_size)
            self.model = BertForPreTraining(bert_config)
            if self.config.use_backbone:
                pretrained_model = BertForPreTraining.from_pretrained(self.config.backbone)
                
        elif self.config.model_type == "distilbert":
            distil_config = DistilBertConfig.from_pretrained(self.config.backbone)
            distil_config.vocab_size = self.config.vocab_size
            self.model = DistilBertForPretraining(distil_config)
            if self.config.use_backbone:
                pretrained_model = DistilBertForPretraining.from_pretrained(self.config.backbone)
        elif self.config.model_type == 'roberta':
            self.model = AutoModelForMaskedLM.from_pretrained(self.config.backbone)
        else:
            raise ValueError("Please select bert or distilbert for the model_type.")
        
        if self.config.use_weights:
            layers = get_pretrained_weights(self.model, pretrained_model)
            self.model.load_state_dict(layers)
            del pretrained_model

        # Change device
        self.model.to(self.device)

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
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.lr,
            betas=self.config.betas,
            eps=self.config.eps,
            correct_bias=False
        )
        
        # Valid dataset
        self.dataloaders = {}
        with open(self.config.valid_path, 'rb') as f:
            self.valid_dataset = pickle.load(f)
        self.dataloaders['valid'] = DataLoader(
                        self.valid_dataset,
                        batch_size=self.config.batch_size,
                        shuffle=False,
                        collate_fn=collate_fn
                    )
        
        # Define Scheduler
        if not self.config.continuous:
            with open(self.config.train_path.format(0), 'rb') as f:
                dataset = pickle.load(f)
                step_per_epoch = math.ceil(len(dataset) / self.config.batch_size)
                self.total_steps = step_per_epoch * self.config.epochs
                logger.info(f"Number of Train data : {len(dataset)}")
                del dataset
        else:
            with open(os.path.join(self.config.checkpoint, 'checkpoint-info.pk'), 'rb') as f:
                self.checkpoint_info = pickle.load(f)
                self.total_steps = self.checkpoint_info['total_steps']
                self.config.num_warmup_steps = self.checkpoint_info['num_warmup_steps']
        self.total_steps = (self.total_steps / self.config.gradient_accumulation_steps) + 1

        logger.info(f"Total Steps : {self.total_steps}")

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.num_warmup_steps,
            num_training_steps=self.total_steps,
        )
        
        if self.config.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Install apex from https://www.github.com/nvidia/apex")
            logger.info(f'Using fp16 training: {self.config.fp16_opt_level} level.')
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level=self.config.fp16_opt_level
            )
        
        # LR list
        self.learning_rates = []

        # Continuous learning
        if self.config.continuous:
            logger.info("##### Continuous Learning Mode #####")
            model_state = torch.load(os.path.join(self.config.checkpoint, 'pytorch_model.bin'), map_location=self.device)
            optimizer_state = torch.load(os.path.join(self.config.checkpoint, 'optimizer.pt'), map_location=self.device)
            scheduler_state = torch.load(os.path.join(self.config.checkpoint, 'scheduler.pt'))
            self.learning_rates = self.checkpoint_info['learning_rates']

            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(optimizer_state)
            self.scheduler.load_state_dict(scheduler_state)
            if self.config.fp16:
                amp.load_state_dict(self.checkpoint_info['amp'])

            del model_state, optimizer_state, scheduler_state
            torch.cuda.empty_cache()
        
        # Define Losses
        self.nsp_criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)

        logger.info(f"{'Initialize Trainer':>15}\n{'Model':<15}{type(self.model)}\n{'Batch Size':<15}{config.batch_size}\n{'Vocab Size':<15}{config.vocab_size}\n{'Learning Rate':<15}{config.lr}\n{'Optimizer':<15}{type(self.optimizer)}")


    def train(self):
        best_val_loss = self.checkpoint_info['best_loss'] if self.config.continuous else float('inf')
        train_loss_history = self.checkpoint_info['train_losses'] if self.config.continuous else []
        valid_loss_history = self.checkpoint_info['valid_losses'] if self.config.continuous else []
        start_epoch = self.checkpoint_info['epoch_or_step'] if self.config.continuous else 0
        for epoch in range(start_epoch, start_epoch + self.config.epochs):
            for phase in ['train', 'valid']:
                if phase == 'train':
                    self.model.train()
                    dataset_num = epoch % 10

                    with open(self.config.train_path.format(dataset_num), 'rb') as f:
                        train_dataset = pickle.load(f)

                    self.dataloaders['train'] = DataLoader(
                        train_dataset,
                        batch_size=self.config.batch_size,
                        shuffle=True,
                        collate_fn=collate_fn
                    )
                else:
                    self.model.eval()
                    
                epoch_loss = 0
                for i, batch in enumerate(tqdm(self.dataloaders[phase], total=len(self.dataloaders[phase]), desc=f"{phase}..")):
                    self.optimizer.zero_grad()
                    # Send device
                    model_inputs = {k: batch[k].to(self.device) for k in ("input_ids", "token_type_ids", "attention_mask")}
                    labels = {k: batch[k].type(torch.LongTensor).to(self.device) for k in ("mask_label", "nsp_label")}
                    step = (epoch * len(self.dataloaders[phase])) + i
                    if phase == "train":
                        self._save_learning_rate()
                        total_loss, mlm_loss, nsp_loss = self._training_step(model_inputs, labels)

                        if self.config.save_strategy == "step":
                            if self.n_iter % self.config.save_step == 0:
                                best_val_loss = self.save_checkpoint(
                                                step=step,
                                            )
                    else:
                        total_loss, mlm_loss, nsp_loss = self._validation_step(model_inputs, labels)

                    if i % self.config.log_step == 0:
                        self.logger.info(f"{'Epoch':<15}{epoch + 1}\n{'Phase':<15}{phase}\n{'Step':<15}{step}\n{'Total Loss':<15}{total_loss:.4f}\n{'MLM Loss':<15}{mlm_loss:.4f}\n{'NSP Loss':<15}{nsp_loss:.4f}\n")
                        if phase == "train":
                            # step / loss
                            train_loss_history.append([step, total_loss])
                        else:
                            valid_loss_history.append([step, total_loss])
                    epoch_loss += total_loss * batch['input_ids'].size(0)

                epoch_loss = epoch_loss / len(self.dataloaders[phase].dataset)

                if phase == "valid" and self.config.save_strategy == "epoch":
                    best_val_loss = self.save_checkpoint(
                                    best_val_loss,
                                    epoch_loss,
                                    epoch + 1,
                                    train_losses=train_loss_history,
                                    valid_losses=valid_loss_history
                                )
                self.logger.info(f"{'Epoch Loss':<15}{epoch_loss:.4f}")
        self.save_checkpoint(last_save=True, train_losses=train_loss_history, valid_losses=valid_loss_history)
        
        self.logger.info(f"Completed training.")


    def _training_step(self, batch: Dict, labels: Dict):
        # prediction_logits : vector of tokens, seq_relationship_logits : classification output of nsp
        output = self.model(**batch)
        ## Losses
        # (B, L, V) -> (B*L, V)
        mlm_loss = self.mlm_criterion(output['prediction_logits'].view(-1, output['prediction_logits'].size(-1)), labels['mask_label'].view(-1))
        nsp_loss = self.nsp_criterion(output['seq_relationship_logits'], labels['nsp_label'].view(-1))
        total_loss = mlm_loss + nsp_loss

        if self.config.fp16:
            from apex import amp
            with amp.scale_loss(total_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()

        if self.n_iter % self.config.gradient_accumulation_steps == 0:
            if self.config.fp16:
                clip_grad_norm_(amp.master_params(self.optimizer), self.config.max_norm)
            else:
                clip_grad_norm_(self.model.parameters(), self.config.max_norm)
            self.optimizer.step()
            self.scheduler.step()
        
        return total_loss.item(), mlm_loss.item(), nsp_loss.item()
    
    @torch.no_grad()
    def _validation_step(self, batch: Dict, labels: Dict):
        output = self.model(**batch)
        # Losses
        mlm_loss = self.mlm_criterion(output['prediction_logits'].view(-1, output['prediction_logits'].size(-1)), labels['mask_label'].view(-1)).detach()
        nsp_loss = self.nsp_criterion(output['seq_relationship_logits'], labels['nsp_label'].view(-1)).detach()
        total_loss = mlm_loss + nsp_loss

        return total_loss.item(), mlm_loss.item(), nsp_loss.item()


    def _save_checkpoint(
            self,
            base_path: str = None,
            loss: float = None,
            step: int = None,
            train_losses: List = None,
            valid_losses: List = None,
            ):
        os.makedirs(base_path, exist_ok=True)
        torch.save(self.model.state_dict(), f'{base_path}/pytorch_model.bin')
        torch.save(self.optimizer.state_dict(), f'{base_path}/optimizer.pt')
        torch.save(self.scheduler.state_dict(), f'{base_path}/scheduler.pt')
        self.model.config.to_json_file(f'{base_path}/config.json')
        save_items = {
                    'train_losses': train_losses,
                    'valid_losses': valid_losses,
                    'learning_rates': self.learning_rates,
                    'best_loss': loss,
                    'epoch_or_step': step,
                    'total_steps' : self.total_steps,
                    'num_warmup_steps': self.config.num_warmup_steps,
                    }
        if self.config.fp16:
            from apex import amp
            save_items['amp'] = amp.state_dict()

        with open(f'{base_path}/checkpoint-info.pk', 'wb') as f:
            pickle.dump(save_items, f)


    def save_checkpoint(
            self,
            best_loss: float = float("inf"),
            loss: float = float("inf"),
            step: float = .0,
            train_losses: List = [],
            valid_losses: List = [],
            last_save: bool = False
            ):
        base_path = f'results/{step}-{self.config.save_strategy}'
        if last_save:
            base_path = 'results'
            self._save_checkpoint(base_path, loss, step, train_losses, valid_losses)
            return
        
        if best_loss > loss or loss == float("inf"):
            if len(self.saved_path) >= self.save_total_limit:
                remove_item = heapq.heappop(self.saved_path)
                shutil.rmtree(remove_item[1])
            self._save_checkpoint(base_path, loss, step, train_losses, valid_losses)

            heapq.heappush(self.saved_path, (-loss, base_path))

            self.logger.info(f"Saved model..\nLoss  : {loss:.4f}")

            return loss

        return best_loss


    def _save_learning_rate(self):
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])


