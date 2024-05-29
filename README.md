# Pretraining BERT with PyTorch and Huggingface

# Step 1. Collect data to train Tokenizer and BERT

refer to `data/collate_data.ipynb`, `train_tokenizer.py`


# Step 2. Make the Pre-train data

refer to `data/collate_data.ipynb`

- Maked Language Modeling

    - 15% of the total (80% : making, 10% : random, 10% : origin)

- Next Sentence Prediction

    - 50% (0 : not next sentence, 1 : next sentence)

# Step 3. Training BERT


```python
python run_pretraining.py --c config.json --cont --checkpoint results/1000-step
```

- `--config_path` : config file (default : './config.json')

- `--continuous` : boolean for continuous training

- `--checkpoint` : path of checkpoint for continous training


# Step 4. Evaluate

Evaluate refer to `evaluate_script.py`

