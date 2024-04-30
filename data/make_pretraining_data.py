import random
import json
import os
from tqdm  import tqdm
from train_tokenizer import Tokenizer

# random.seed(123)

def make_pretrain_data(tokenizer, in_path, out_path, count, n_seq, mask_prob):
    vocab_list = []
    for id in range(tokenizer.get_piece_size()):
        if not tokenizer.is_unknown(id):
            vocab_list.append(tokenizer.id_to_piece(id))

    line_cnt = 0
    with open(in_path, 'r') as in_f:
        for line in in_f:
            line_cnt += 1

    docs = []
    with open(in_path, 'r') as f:
        doc = []
        with tqdm(total=line_cnt, desc="Loading") as pbar:
            for i, line in enumerate(f):
                line = line.strip()
                if line == "":
                    if 0 < len(doc):
                        docs.append(doc)
                        doc = []

                else:
                    pieces = tokenizer.encode_as_pieces(line)
                    if 0 < len(pieces):
                        doc.append(pieces)

                pbar.update(1)

        if doc:
            docs.append(doc)

    for index in range(count):
        # 같은 데이터 10개의 말뭉치 : mask가 랜덤으로 이루어지기 떄문에 ?
        output = out_path.format(index)
        if os.path.isfile(output): continue

        with open(output, 'w') as out_f:
            with tqdm(total=len(docs), desc=f"Masking") as pbar:
                for i, doc in enumerate(docs):
                    instances = create_pretrain_instances(docs, i, doc, n_seq, mask_prob, vocab_list)

                    for instance in instances:
                        out_f.write(json.dumps(instance, ensure_ascii=False))
                        out_f.write('\n')

                    pbar.update(1)

def create_pretrain_instances(docs, doc_idx, doc, n_seq, mask_prob, vocab_list):
    '''
    docs = 문단별 데이터
    doc_idx = 해당 문단 인덱스
    doc = 해당 문단 pieces list
    n_seq = input data 최대길이
    mask_prob = MLM 적용 인수
    '''
    max_seq = n_seq - 3 # CLS SEP SEP
    tgt_seq = max_seq

    instances =[]
    cur_chunk = []
    cur_length = 0
    # NSP 데이터 (0.5 : 이어지는 정상문장, 0.5 : 이어지지 않는 문장)
    for i in range(len(doc)):
        cur_chunk.append(doc[i])
        cur_length += len(doc[i])

        if i==len(doc)-1 or cur_length >= tgt_seq:
            if 0 < len(cur_chunk):
                a_end = 1
                if 1 < len(cur_chunk):
                    a_end = random.randrange(1, len(cur_chunk))

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(cur_chunk[j])

                tokens_b = []
                if len(cur_chunk) == 1 or random.random() < 0.5:
                    is_next = 0
                    random_doc_idx = doc_idx

                    while doc_idx == random_doc_idx:
                        random_doc_idx = random.randrange(0, len(docs))

                    random_doc = docs[random_doc_idx]

                    random_start = random.randrange(0, len(random_doc))
                    for j in range(random_start, len(random_doc)):
                        tokens_b.extend(random_doc[j])

                else:
                    is_next = 1
                    for j in range(a_end, len(cur_chunk)):
                        tokens_b.extend(cur_chunk[j])

                trim_tokens(tokens_a,tokens_b,max_seq)
                assert 0 < len(tokens_a)
                assert 0 < len(tokens_b)

                tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
                segment = [0] * (len(tokens_a)+2) + [1] * (len(tokens_b)+1)

                tokens, mask_idx, mask_label = create_pretrain_mask(tokens, int((len(tokens)-3) * mask_prob), vocab_list)
                instance = {
                    "tokens" : tokens,
                    "segment" : segment,
                    "is_next" : is_next,
                    "mask_idx" : mask_idx,
                    "mask_label" : mask_label
                }

                instances.append(instance)

            cur_chunk = []
            cur_length = 0

    return instances

def trim_tokens(tokens_a, tokens_b, max_seq):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_seq:
            break
        if len(tokens_a) > len(tokens_b):
            del tokens_a[0]

        else:
            tokens_b.pop()
    

def create_pretrain_mask(tokens, mask_cnt, vocab_list):
    '''
    tokens : ["CLS", piece, piece ..., "SEP"]
    mask_cnt : 15% of the number of tokens
    vocab_list : vocabulary without "UNK"
    '''
    
    cand_idx = []
    for i, token in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue

        if 0 < len(cand_idx) and not token.startswith(u"\u2581"):
            cand_idx[-1].append(i)

        else:
            cand_idx.append([i])

    random.shuffle(cand_idx)
    
    mask_lms = []
    for index_set in cand_idx:
        if len(mask_lms) >= mask_cnt:
            break

        if len(mask_lms) + len(index_set) > mask_cnt:
            continue

        #
        for index in index_set:
            masked_token = None
            if random.random() < 0.8:
                masked_token = '[MASK]'

            else:
                if random.random() < 0.5:
                    masked_token = tokens[index]

                else:
                    masked_token = random.choice(vocab_list)

            mask_lms.append({"index": index, "label":tokens[index]})
            tokens[index] = masked_token
    
    mask_lms = sorted(mask_lms, key=lambda x:x['index'])
    mask_idx = [p['index'] for p in mask_lms]
    mask_label = [p['label'] for p in mask_lms]

    return tokens, mask_idx, mask_label

if __name__ == "__main__":
    in_file = './data/kowiki.txt'
    out_file = './data/kowiki_bert_{}.json'
    count = 10
    n_seq = 256
    mask_prob = 0.15
    test = {"tokenizer_type": "bpe", "src_vocab_size":8000, "train_path":"/Users/khs/Documents/python/study/NLP_study/BERT/data/kowiki.txt", "vocab_file_path":"/Users/khs/Documents/python/study/NLP_study/BERT/vocab/"}
    import argparse
    n_test = argparse.Namespace(**test)
    tokenizer = Tokenizer(n_test)

    make_pretrain_data(tokenizer, in_file, out_file, count, n_seq, mask_prob)