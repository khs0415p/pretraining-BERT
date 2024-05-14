import os
import logging
import glob

from config import Config
from transformers import BertTokenizerFast
from tokenizers import BertWordPieceTokenizer

logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger(__name__)


def get_bert_tokenizer():
    tokenizer = BertWordPieceTokenizer(
        vocab=None,
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=False,
        lowercase=False,
        wordpieces_prefix="##"
    )
    return tokenizer


def main(config: Config, prefix: str = "##"):
    vocab_size = config.vocab_size
    tokenizer_save_path = config.tokenizer_save_path
    tokenizer_data_path = config.tokenizer_data_path
    data_list = glob.glob(tokenizer_data_path)
    train_files = []

    # remove dir
    for path in data_list:
        if os.path.isdir(path): continue
        train_files.append(path)

    tokenizer = get_bert_tokenizer()

    # Train
    tokenizer.train(
        files=train_files,
        limit_alphabet=config.limit_alphabet,
        min_frequency=config.min_frequency,
        vocab_size=vocab_size
    )
    

    # Save tokenizer for transformers
    wrapped_tokenizer = BertTokenizerFast(
        tokenizer_object=tokenizer,
    )
    logger.info("********** Completed train **********")
    logger.info(f"Trained tokenizer's vocab size is {wrapped_tokenizer.vocab_size}")
    logger.info("*************************************")
    wrapped_tokenizer.save_pretrained(tokenizer_save_path)

if __name__ == "__main__":
    config = Config()
    main(config)