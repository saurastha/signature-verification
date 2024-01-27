import torch

# Model Paths
SIGN_DETECTOR_MODEL_PATH = "models/SignetDetect.pt"
SIGN_VERIFIER_MODEL_PATH = "models/signet_350_615.pth"
SIGN_CLEANER_MODEL_PATH = "models/cleaner.pt"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
