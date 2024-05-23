import torch
import pandas as pd

from transformers import AutoTokenizer, BertModel


threshold = 0.779
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
model = BertModel.from_pretrained("dmis-lab/biobert-v1.1")
device = torch.device("cpu")
model.to(device)
model.eval()

def cos_sim(row, eps=1e-08):
    a, b = row
    a, b = a['last_hidden_state'].mean(dim=1).squeeze(), b['last_hidden_state'].mean(dim=1).squeeze()
    numerator = a @ b
    a_l2, b_l2 = a @ a, b @ b
    denominator = torch.sqrt(torch.mul(a_l2, b_l2)) + torch.tensor(eps)
    return torch.div(numerator,denominator).item()
    

def get_logits(row):
    a = {k: v.to(device) for k, v in row[0].items()}
    b = {k: v.to(device) for k, v in row[0].items()}
    output_a = model(**a)
    output_b = model(**b)

    return output_a, output_b


def get_metric(preds, labels, threshold):
    preds = preds >= threshold
    preds = preds.astype('int8')

    labels = labels.values
    preds = preds.values

    tp = ((labels == 1) & (preds == 1)).sum()
    fn = ((labels == 1) & (preds == 0)).sum()
    fp = ((labels == 0) & (preds == 1)).sum()
    tn = ((labels == 0) & (preds == 0)).sum()

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    specificity = tn / (tn + fp)
    accuracy =  (tp + tn) / (tp + tn + fp + fn)

    return recall, precision, specificity, accuracy

# Load data
eval_data = pd.read_excel('evaluate.xlsx')

# convert to tensor
model_inputs = eval_data.apply(lambda row: (tokenizer(row['moa1'], return_tensors='pt'), tokenizer(row['moa2'], return_tensors='pt')), axis=1)

# get the logits
model_outputs = model_inputs.apply(get_logits)

# get the cosine similarity
cos_sims = model_outputs.apply(cos_sim)

# Meric
recall, precision, specificity, accuracy = get_metric(cos_sims, eval_data['label'], threshold=threshold)
print(f"recall : {recall}\nprecision : {precision}\nspecificity : {specificity}\naccuracy : {accuracy}")
