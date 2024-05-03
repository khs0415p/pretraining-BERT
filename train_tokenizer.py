import logging
from typing import List
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
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
        strip_accents=True,
        lowercase=True,
        wordpieces_prefix="##"
    )
    return tokenizer


def main(config: Config, prefix: str = "##"):
    vocab_size = config.vocab_size
    tokenizer_save_path = config.tokenizer_save_path
    tokenizer_data_path = config.tokenizer_data_path
    tokenizer = get_bert_tokenizer()

    # Train
    tokenizer.train(
        files=[tokenizer_data_path],
        limit_alphabet=1000,
        min_frequency=5,
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