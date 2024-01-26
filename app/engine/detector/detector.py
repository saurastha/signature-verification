from typing import List, Tuple

import torch

from ultralytics import YOLO

from app.constants.constants import DEVICE


class SignDetector:
    """
    SignDetector is a class for detecting signs in images using the YOLO (You Only Look Once)
    object detection model.

    Parameters:
        - detect_threshold (float): Confidence threshold for considering a detection (default: 0.7).

    Attributes:
        - model: YOLO object detection model.
        - detect_threshold (float): Confidence threshold for detections.
        - device (str): Device on which the model is deployed (default: "mps").

    Methods:
        - __init__(detect_threshold=0.7): Initializes the SignDetector instance.
        - load(model_path: str): Loads the YOLO model from the specified path.
        - detect(image): Performs sign detection on the input image and returns bounding boxes, scores, and labels.
    """

    def __init__(self, detect_threshold: float = 0.7) -> None:
        """
        Initializes a new SignDetector instance.

        Parameters:
            - detect_threshold (float): Confidence threshold for considering a detection (default: 0.7).
        """
        self.model = None
        self.detect_threshold = detect_threshold
        self.device = DEVICE

    def load(self, model_path: str) -> None:
        """
        Loads the YOLO model from the specified path.

        Parameters:
            - model_path (str): The path to the YOLO model.
        """
        self.model = YOLO(model_path)

    def detect(self, image) -> Tuple[List, List, List]:
        """
        Performs sign detection on the input image.

        Parameters:
            - image: Input image for sign detection.

        Returns:
            - bboxes (list): List of bounding boxes for detected signs.
            - scores (list): List of confidence scores for each detection.
            - labels (list): List of labels corresponding to detected signs.

         Raises:
            - RuntimeError: If the detect method is called before loading the model.
        """

        if self.model:
            bboxes = []
            scores = []
            labels = []

            results = self.model(image, device=self.device)
            detections = results[0].boxes

            for detection in detections:
                bbox = detection.xyxy[0].round().to(torch.int).tolist()
                conf = round(detection.conf.item(), 4)
                label = self.model.names[int(detection.cls)]

                if conf >= self.detect_threshold:
                    bboxes.append(bbox)
                    scores.append(conf)
                    labels.append(label)

            return bboxes, scores, labels
        else:
            raise RuntimeError("Model has not been loaded. Please call the 'load' method to load the model.")
