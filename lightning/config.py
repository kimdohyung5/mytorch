
# Hyperparameters
INPUT_SIZE = 784
NUM_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 3


# dATASET
DATA_DIR = "dataset/"
NUM_WORKERS = 4

# Compoute related.
DEVICES = [0]
ACCELERATOR = "gpu"
PRECISION = 16