import torch
from transformers import AdamW
from data.preprocess import DataPreprocessor
from models.model_loader import ModelLoader
from config.config import Config
from utils.logger import Logger

class Trainer:
    def __init__(self, config):
        self.config = config
        self.logger = Logger()
        model_loader = ModelLoader(config.MODEL_NAME, config.NTK_SCALE, config.PI_WINDOW_SIZE, config.DEVICE)
        self.model = model_loader.model
        self.tokenizer = model_loader.tokenizer
        self.optimizer = AdamW(self.model.parameters(), lr=config.LEARNING_RATE)

    def train(self, dataset):
        dataloader = DataPreprocessor(self.config.MODEL_NAME).prepare_dataloader(dataset, self.config.BATCH_SIZE)
        self.model.train()

        try:
            for epoch in range(self.config.EPOCHS):
                for i, batch in enumerate(dataloader):
                    input_ids = batch['input_ids'].to(self.config.DEVICE)
                    labels = batch['labels'].to(self.config.DEVICE)
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if i % self.config.LOGGING_INTERVAL == 0:
                        self.logger.log(f"Epoch [{epoch+1}/{self.config.EPOCHS}], Step [{i}], Loss: {loss.item():.4f}")

                # Save model checkpoint with timestamp and metadata to avoid overwriting
                torch.save(self.model.state_dict(), f"{self.config.SAVE_PATH}/llama_epoch{epoch+1}_{time.strftime('%Y%m%d-%H%M%S')}.pt")
        except Exception as e:
            self.logger.log(f"Error during training: {str(e)}")
    