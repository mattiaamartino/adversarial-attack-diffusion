# filepath: huggingface-vision-project/huggingface-vision-project/src/config.py

MODEL_PATHS = {
    "dinov2": "facebook/dinov2-small",
    "pix2pix": "timbrooks/instruct-pix2pix"
}

IMAGE_DIMENSIONS = {
    "dinov2": (224, 224),  # Example dimensions for Dinov2
    "pix2pix": (256, 256)  # Example dimensions for Pix2Pix
}

HYPERPARAMETERS = {
    "dinov2": {
        "num_classes": 1000,
        "batch_size": 32,
        "learning_rate": 1e-4
    },
    "pix2pix": {
        "num_epochs": 50,
        "batch_size": 16,
        "learning_rate": 2e-4
    }
}