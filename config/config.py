class Config:
    MODEL_NAME = "llama-2-7b"
    MAX_CONTEXT_LENGTH = 64000
    BATCH_SIZE = 2
    LEARNING_RATE = 3e-5
    EPOCHS = 3
    DEVICE = "cuda"  # or "cpu"
    LOGGING_INTERVAL = 50
    SAVE_PATH = "./checkpoints/"

    # Fine-tuning parameters for NTK-RoPE and PI
    NTK_SCALE = 2.0
    PI_WINDOW_SIZE = 2048
    