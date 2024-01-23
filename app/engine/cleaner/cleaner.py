import numpy as np
import torch
from torchvision import transforms

from app.models.architectures import DnCNN
from app.utils.commons import resize_signature


class SignCleaner:
    """
    SignCleaner class for cleaning signature images using a DnCNN model.

    Attributes:
        - model (DnCNN): The DnCNN model for image cleaning.
        - transform (transforms.Compose): Image transformation pipeline.
        - device (str): The device to use for inference (default is "cuda" if available, else "cpu").

    Methods:
        - load(model_path: str) -> None:
            Load the DnCNN model from the specified path.

        - clean(image, threshold: float = 0.8) -> np.ndarray:
            Clean the input image using the loaded DnCNN model.

        - post_processing(img: torch.Tensor, threshold: float) -> np.ndarray:
            Post-process the cleaned image.
    """
    def __init__(self) -> None:
        """
        Initialize SignCleaner.

        Sets up the model, transformation pipeline, and device.
        """
        self.model = None
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(.5, .5)])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self, model_path: str) -> None:
        """
        Load the DnCNN model from the specified path.

        Args:
            - model_path (str): Path to the pre-trained DnCNN model.
        """
        self.model = DnCNN(6)
        self.model.load_state_dict(torch.load(model_path))
        print(self.model)

    def clean(self, image: np.ndarray, threshold: float = 0.8) -> np.ndarray:
        """
        Clean the input image using the loaded DnCNN model.

        Args:
            - image: Input image to be cleaned.
            - threshold (float): Threshold for post-processing (default is 0.8).

        Returns:
            - np.ndarray: Cleaned image as a NumPy array.
        """
        if self.model:
            self.model.to(self.device)
            img = resize_signature(image, 220, 220)[0]
            pre_processed_img = self.transform(img)
            img_tensor = pre_processed_img.unsqueeze(dim=0)

            self.model.eval()
            with torch.inference_mode():
                cleaned_img = self.model(img_tensor)

            final_image = self.post_processing(cleaned_img, threshold)

            return final_image

        else:
            raise RuntimeError("Model has not been loaded. Please call the 'load' method to load the model.")

    @staticmethod
    def post_processing(img: torch.Tensor, threshold: float) -> np.ndarray:
        """
        Post-process the cleaned image.

        Args:
            - img (torch.Tensor): Cleaned image tensor.
            - threshold (float): Threshold for binary conversion.

        Returns:
            - np.ndarray: Processed image as a NumPy array.
        """
        binary_img = torch.where(img.squeeze(dim=0)[0] > threshold, 1.0, 0.0).numpy()
        processed_img = (binary_img * 255).astype(np.uint8)

        return processed_img
