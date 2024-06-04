
import torch
from typing import Dict, List
from torch.nn.utils.rnn import pad_sequence
from collections import OrderedDict
from transformers import PreTrainedTokenizerBase, PreTrainedModel

IGNORE_ID = -100

def get_pretrained_weights(model: PreTrainedModel, pretrained_model: PreTrainedModel) -> Dict[str, torch.Tensor]:
    # shared_layer_names =  set(model.state_dict().keys().intersection(old_model.state_dict().keys()))

    layers = []
    for k, v in model.state_dict().items():
        if 'embeddings' in k or 'predictions' in k or 'vocab_projector' in k:
            layers.append((k, v))
            continue
        
        try:
            pretrained_layer = pretrained_model.state_dict()[k]
        except:
            raise ValueError("model and pretrained-model are different.")
        layers.append((k, pretrained_layer))
        
    return OrderedDict(layers)


def collate_fn(batch: List[Dict[str, torch.Tensor]], padding_value: int = 0) -> Dict[str, torch.Tensor]:
    input_ids, token_type_ids, mask_label, nsp_label = tuple([instance[key] for instance in batch] for key in ("input_ids", "token_type_ids", "mask_label", "nsp_label"))
    # Dynamic padding
    
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=padding_value
        )
    
    token_type_ids = torch.nn.utils.rnn.pad_sequence(
        token_type_ids, batch_first=True, padding_value=padding_value
        )
    
    mask_label = torch.nn.utils.rnn.pad_sequence(
        mask_label, batch_first=True, padding_value=IGNORE_ID
        )
    
    attention_mask = input_ids.ne(padding_value)
    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "mask_label": mask_label,
        "nsp_label": torch.stack(nsp_label, dim=0)
    }
    
    
        

def get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))