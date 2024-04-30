
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
        if 'embeddings' in k or 'predictions' in k:
            layers.append((k, v))
            continue
        
        try:
            pretrained_layer = pretrained_model.state_dict()[k]
        except:
            raise ValueError("model and pretrained-model are different.")
        layers.append((k, pretrained_layer))
        
    return OrderedDict(layers)


def collate_fn(batch: List[Dict[str, torch.Tensor]], padding_value: int = 0) -> Dict[str, torch.Tensor]:
    # Dynamic padding
    # text
    torch.nn.utils.rnn.pad_sequence(batch_first=True, padding_value=padding_value)
    # label
    torch.nn.utils.rnn.pad_sequence(batch_first=True, padding_value=IGNORE_ID)
    return {

    }