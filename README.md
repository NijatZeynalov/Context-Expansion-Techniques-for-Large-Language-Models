# Context Expansion Techniques for Large Language Models

This project aims to enhance the ability of large language models (LLMs) to handle larger context lengths, making them more efficient and capable of dealing with longer inputs. The project primarily focuses on implementing **attention scaling techniques** that allow the model to intelligently manage extended sequences of information, making it suitable for applications that involve long documents or extended conversations.

## Features
- **Extended Context Length**: Modify LLaMA-2 to handle larger text inputs, improving long-range dependencies.
- **Attention Scaling**: Apply custom **NTK-RoPE** and **Position Interpolation (PI)** methods to better control and focus model attention over large contexts.
- **Efficient Training and Evaluation Pipeline**: Train the model using PyTorch, and evaluate its performance using customizable metrics.

## Requirements
To run this project, you need Python 3.8+ and the required dependencies specified in `requirements.txt`. Install these dependencies using:

```sh
pip install -r requirements.txt
```

## Directory Structure
- **config/config.py**: Configurations such as model name, training hyperparameters, and device settings.
- **training/train.py**: Handles the training process, including data loading and checkpoint saving.
- **data/preprocess.py**: Prepares and tokenizes the dataset for training.
- **models/model_loader.py**: Loads the LLaMA model and applies custom scaling functions.
- **models/attention_scaling.py**: Implements NTK-RoPE and Position Interpolation for context scaling.
- **evaluation/evaluate.py**: Evaluates the trained model using different metrics.
- **utils/logger.py**: Logs training progress and metrics.

## How to Use

1. **Configuration**:
   - Adjust parameters like `MODEL_NAME`, `MAX_CONTEXT_LENGTH`, `BATCH_SIZE`, and others in `config/config.py` to suit your requirements.

2. **Data Preparation**:
   - Prepare your dataset in a format compatible with the tokenizer. You can use the `DataPreprocessor` in `data/preprocess.py` to tokenize the data and generate a DataLoader.

3. **Training**:
   - Run the training script to train the model with extended context support:
     ```sh
     python dynamic_context_scaling/training/train.py
     ```
   - During training, checkpoints will be saved in the specified `SAVE_PATH` (in `config/config.py`).

4. **Evaluation**:
   - Once the model is trained, evaluate its performance using:
     ```sh
     python dynamic_context_scaling/evaluation/evaluate.py
     ```

## Example Input and Output

- **Input**: Suppose we provide the following long document (input text) to the model:
  ```
  "Once upon a time, in a faraway land, there was a king who ruled his kingdom with great wisdom and fairness. The kingdom, however, had...
  [more paragraphs]
  ...The end."
  ```
  
- **Expected Output**: The trained model should be able to generate coherent and contextually aware responses even with long-range dependencies, such as:
  ```
  "The king's fair rule brought peace to the entire kingdom, and his wise decisions ensured prosperity for all his people. The long journey the prince took...
  "
  ```
  The model should maintain context over a long span of text, providing meaningful completions and insights related to the entire narrative, even if the input is lengthy.

## Example Usage in Code

Here is an example of how you can use the trained model in a Python script:

```python
import torch
from transformers import LlamaTokenizer
from models.model_loader import ModelLoader
from config.config import Config

# Load Model and Tokenizer
config = Config()
model_loader = ModelLoader(config.MODEL_NAME, config.NTK_SCALE, config.PI_WINDOW_SIZE, device=config.DEVICE)
model = model_loader.model

# Tokenize Input Text
tokenizer = LlamaTokenizer.from_pretrained(config.MODEL_NAME)
input_text = "Once upon a time in a faraway land..."
inputs = tokenizer(input_text, return_tensors="pt", max_length=config.MAX_CONTEXT_LENGTH, truncation=True).to(config.DEVICE)

# Generate Output
with torch.no_grad():
    outputs = model.generate(inputs['input_ids'], max_length=config.MAX_CONTEXT_LENGTH + 50)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
print("Generated Text:")
print(generated_text)
```
