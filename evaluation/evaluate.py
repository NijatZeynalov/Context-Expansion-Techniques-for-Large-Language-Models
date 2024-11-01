from config.config import Config
import torch
from data.preprocess import DataPreprocessor
from models.model_loader import ModelLoader

class Evaluator:
    def __init__(self, config):
        self.config = config
        self.model = ModelLoader(config.MODEL_NAME, config.NTK_SCALE, config.PI_WINDOW_SIZE).model.to(config.DEVICE)

    def evaluate(self, dataset):
        dataloader = DataPreprocessor(self.config.MODEL_NAME).prepare_dataloader(dataset, 1)
        self.model.eval()

        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.config.DEVICE)
                labels = batch['labels'].to(self.config.DEVICE)
                outputs = self.model(input_ids=input_ids, labels=labels)
                total_loss += outputs.loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Avg Loss: {avg_loss}")
