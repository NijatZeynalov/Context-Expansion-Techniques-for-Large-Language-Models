from transformers import LlamaTokenizer
import torch
from config.config import Config

class DataPreprocessor:
    def __init__(self, model_name):
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)

    def tokenize_batch(self, texts):
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            max_length=Config.MAX_CONTEXT_LENGTH,
            truncation=True
        )

    def prepare_dataloader(self, dataset, batch_size):
        # Tokenize dataset as a batch to improve efficiency
        tokenized_data = self.tokenize_batch(dataset)
        return torch.utils.data.DataLoader(tokenized_data, batch_size=batch_size, shuffle=True)
    