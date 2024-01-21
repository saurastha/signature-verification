from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from app.verifier.model_architecture import SigNet


class SignVerifier:
    """
    SignVerifier is a class for verifying the similarity of signatures using a pre-trained model.

    Parameters:
        - similarity_threshold (float): Threshold for considering signatures as genuine or forged (default: 1.5).

    Attributes:
        - model: Pre-trained signature verification model.
        - similarity_threshold (float): Threshold for considering signatures as genuine or forged.
        - device (str): Device on which the model is deployed (default: "mps").

    Methods:
        - __init__(similarity_threshold: float = 1.5): Initializes the SignVerifier instance.
        - load(model_path: str) -> None: Loads the pre-trained signature verification model from the specified path.
        - feature_extraction(signature_pair) -> torch.Tensor: Extracts features from a pair of signatures.
        - verify(sign1, sign2) -> Tuple[float, str]: Verifies the similarity of two signatures and returns the distance and authenticity.
    """

    def __init__(self, similarity_threshold: float = 1.5) -> None:
        """
        Initializes a new SignVerifier instance.

        Parameters:
            - similarity_threshold (float): Threshold for considering signatures as genuine or forged (default: 1.5).
        """
        self.model = None
        self.similarity_threshold = similarity_threshold
        self.device = "mps"

    def load(self, model_path: str) -> None:
        """
        Loads the pre-trained signature verification model from the specified path.

        Parameters:
            - model_path (str): The path to the pre-trained model.
        """
        state_dict, classification_layer, forg_layer = torch.load(model_path)
        base_model = SigNet().eval()
        base_model.load_state_dict(state_dict)

        self.model = base_model

    def feature_extraction(self, signature_pair: torch.Tensor) -> torch.Tensor:
        """
        Extracts features from a pair of signatures.

        Parameters:
            - signature_pair (torch.Tensor): A pair of signatures for feature extraction.

        Returns:
            - torch.Tensor: Extracted features from the input signature pair.
        """
        self.model.to(self.device)
        with torch.inference_mode():
            features = self.model(signature_pair.view(-1, 1, 150, 220).float().div(255).to(self.device))

        return features

    def verify(self, sign1: np.ndarray, sign2: np.ndarray) -> Tuple[float, str]:
        """
        Verifies the similarity of two signatures and returns the distance and authenticity.

        Parameters:
            - sign1 (np.ndarray): First signature for verification.
            - sign2 (np.ndarray): Second signature for verification.

        Returns:
            - Tuple[float, str]: A tuple containing the distance and authenticity ("Genuine" or "Forged").

        Raises:
            - RuntimeError: If the feature_extraction method is called before loading the model.
        """
        if self.model:
            sign_array = np.array([sign1, sign2])
            signatures = torch.from_numpy(sign_array)
            features = self.feature_extraction(signatures)
            similarity = F.cosine_similarity(features[0], features[1], dim=0)
            distance = torch.norm(features[0] - features[1]).item()

            if distance > self.similarity_threshold:
                return distance, "Forged"
            else:
                return distance, "Genuine"
        else:
            raise RuntimeError("Model has not been loaded. Please call the 'load' method to load the model.")
