import numpy as np
import torch
import ujson
from torch.utils.data import Dataset
from tqdm import tqdm

IGNORE_ID = -100

class PretrainDataSet(Dataset):
    def __init__(self, tokenizer, infile):
        self.tokenizer = tokenizer
        self.labels_cls = []
        self.labels_lm = []
        self.sentences = []
        self.segments = []

        line_cnt = 0
        with open(infile, 'r') as f:
            for line in f:
                line_cnt += 1
                if line_cnt % 1000 == 0:
                    print(line_cnt)

        with open(infile, 'r') as f:
            for i, line in enumerate(tqdm(f, total=line_cnt, desc=f"Loading {infile}", unit='lines')):
                instance = ujson.loads(line)
                self.labels_cls.append(instance["is_next"])
                sentences = tokenizer.convert_tokens_to_ids(instance['input_tokens'])
                self.sentences.append(sentences)
                self.segments.append(instance["segment"])
                mask_idx = np.array(instance["mask_idx"], dtype=np.int16)
                mask_label = np.array(tokenizer.convert_tokens_to_ids(instance["mask_label"]), dtype=np.int16)
                label_lm = np.full(len(sentences), dtype=np.int16, fill_value=-100)
                label_lm[mask_idx] = mask_label
                self.labels_lm.append(label_lm)

    def __len__(self):
        assert len(self.labels_cls) == len(self.labels_lm)
        assert len(self.labels_cls) == len(self.sentences)
        assert len(self.labels_cls) == len(self.segments)
        return len(self.labels_cls)

    def __getitem__(self, index):

        return {
            "input_ids": torch.tensor(self.sentences[index]),
            "token_type_ids": torch.tensor(self.segments[index]),
            "mask_label": torch.tensor(self.labels_lm[index]),
            "nsp_label": torch.tensor(self.labels_cls[index]),
            }
    

if __name__ == "__main__":
    import pickle
    import glob
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('./tokenizer')
    data_list = glob.glob('data/*.json')
    data_list.sort()
    
    for i, data_path in enumerate(data_list):
        dataset = PretrainDataSet(tokenizer, data_path)
        save_file = data_path[:-5]
        with open(f"{save_file}.pk", 'wb') as f:
            pickle.dump(dataset, f)