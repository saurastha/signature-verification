import cv2
from typing import Union, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


def preprocess_image(img_array: np.ndarray) -> np.ndarray:
    """
    Preprocess the input image by applying thresholding to each channel.

    Parameters:
        - img_array (np.ndarray): Input image as a NumPy array.

    Returns:
        - np.ndarray: Preprocessed image with thresholding applied to each channel.
    """
    threshold_channels = []

    for channel in cv2.split(img_array):
        _, binary_channel = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        threshold_channels.append(binary_channel)

    thresh_img = cv2.merge(threshold_channels)

    return thresh_img


def preprocess_signature(sign_array: np.ndarray) -> np.ndarray:
    """
    Preprocess the input signature image by converting it to grayscale,
    applying Gaussian blur, thresholding, and resizing.

    Parameters:
        - sign_array (np.ndarray): Input signature image as a NumPy array.

    Returns:
        - np.ndarray: Preprocessed signature image.
    """
    if len(sign_array.shape) == 3:
        gray = cv2.cvtColor(sign_array, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, ksize=(1, 1), sigmaX=0)
        _, binary_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        resized = resize_signature(binary_image, target_height=150, target_width=220)[0]

        return resized
    else:
        resized = resize_signature(sign_array, target_height=150, target_width=220)[0]

        return resized


def save_image(img_array: Union[np.ndarray, List[np.ndarray]], file_name: str) -> List[str]:
    """
    Save image or list of images to files with appropriate file names.

    Parameters:
        - img_array (Union[np.ndarray, List[np.ndarray]]): Image or list of images as NumPy arrays.
        - file_name (str): Base file name for saving images.

    Returns:
        - List[str]: List of file paths where the images are saved.
    """
    file_paths = []
    if isinstance(img_array, list) and len(img_array) == 1:
        save_path = f"{file_name}_sign.jpg"
        cv2.imwrite(f"{file_name}_sign.jpg", img_array[0])
        file_paths.append(save_path)
        return file_paths

    if isinstance(img_array, list):
        for i, img in enumerate(img_array):
            save_path = f"{file_name}_sign_{i}.jpg"
            cv2.imwrite(f"{file_name}_sign_{i}.jpg", img)
            file_paths.append(save_path)
    else:
        cv2.imwrite(f"{file_name}_sign.jpg", img_array)
        save_path = f"{file_name}_sign.jpg"
        cv2.imwrite(f"{file_name}_sign.jpg", img_array[0])
        file_paths.append(save_path)

    return file_paths


def resize_with_aspect_ratio(img_array: np.ndarray, target_width: int, target_height: int):
    """
    Resize the input image while maintaining its aspect ratio, and pad if necessary.

    Parameters:
        - img_array (np.ndarray): Input image as a NumPy array.
        - target_width (int): Target width for resizing.
        - target_height (int): Target height for resizing.

    Returns:
        - np.ndarray: Resized and padded image.
    """
    # Get the original dimensions of the image
    original_height, original_width = img_array.shape[:2]

    # Calculate the aspect ratios
    aspect_ratio_x = target_width / original_width
    aspect_ratio_y = target_height / original_height

    # Use the minimum aspect ratio to preserve the original aspect ratio
    min_aspect_ratio = min(aspect_ratio_x, aspect_ratio_y)

    # Calculate the new dimensions
    new_width = int(original_width * min_aspect_ratio)
    new_height = int(original_height * min_aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(img_array, (new_width, new_height))

    pad_height = max(0, target_height - new_height)
    pad_width = max(0, target_width - new_width)

    # Calculate padding amounts for top, bottom, left, and right
    top_padding = pad_height // 2
    bottom_padding = pad_height - top_padding

    left_padding = pad_width // 2
    right_padding = pad_width - left_padding

    # Pad the image with 255
    result_image = cv2.copyMakeBorder(
        resized_image,
        top_padding,
        bottom_padding,
        left_padding,
        right_padding,
        cv2.BORDER_CONSTANT,
        None,
        value=(255, 255, 255),
    )

    return result_image


def resize_signature(img_array: Union[np.ndarray,List[np.ndarray]],
                     target_height: int = 150,
                     target_width: int = 220) -> np.ndarray:
    """
    Resize the input signature image or list of signature images.

    Parameters:
        - img_array: (Union[np.ndarray, List[np.ndarray]]): Input signature image or list of signature images.
        - target_height (int): Target height for resizing (default: 150).
        - target_width (int): Target width for resizing (default: 220).

    Returns:
        - List[np.ndarray]: List of resized and padded signature images.
    """
    resized_images = []

    if isinstance(img_array, list) and len(img_array) == 1:
        img_array = img_array[0]

    if isinstance(img_array, list):
        for img in img_array:
            resized = resize_with_aspect_ratio(img, target_width, target_height)
            resized_images.append(resized)
    else:
        resized_images.append(resize_with_aspect_ratio(img_array, target_width, target_height))

    return resized_images


def get_image_crops(img_array: np.ndarray, bounding_boxes: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
    """
    Extract image crops specified by the given bounding boxes.

    Parameters:
        - img_array (np.ndarray): Input image as a NumPy array.
        - bounding_boxes (List[Tuple[int, int, int, int]]): List of bounding boxes, each represented as (xmin, ymin, xmax, ymax).

    Returns:
        - List[np.ndarray]: List of image crops.
    """
    crop_holder = []

    for i in range(len(bounding_boxes)):
        bbox = bounding_boxes[i]
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
        crop_holder.append(img_array[ymin:ymax, xmin:xmax])

    return crop_holder


def plot_images(img_array: Union[np.ndarray, List[np.ndarray]],
                title: str = "Image Plot",
                fig_size: Tuple[int, int] = (15, 20),
                nrows: int = 1,
                ncols: int = 4):
    """
    Plot one or multiple images in a grid.

    Parameters:
        - img_array (Union[np.ndarray, List[np.ndarray]]): Image or list of images to be plotted.
        - title (str): Title of the plot (default: "Image Plot").
        - fig_size (Tuple[int, int]): Size of the figure in inches (default: (15, 20)).
        - nrows (int): Number of rows in the grid (default: 1).
        - ncols (int): Number of columns in the grid (default: 4).
    """
    if isinstance(img_array, list) and len(img_array) == 1:
        img_array = img_array[0]

    if isinstance(img_array, list):
        ncols = ncols if ncols < len(img_array) else len(img_array)
        if nrows * ncols < len(img_array):
            nrows = int(len(img_array) / ncols)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 2))
        for i, ax in enumerate(axs.flatten()):
            if i < len(img_array):
                if len(img_array[i].shape) == 2:
                    cmap = "gray"
                else:
                    cmap = "cividis"
                ax.imshow(img_array[i], cmap=cmap)
        fig.suptitle(title)
        plt.tight_layout()
    else:
        if len(img_array.shape) == 2:
            cmap = "gray"
        else:
            cmap = "cividis"
        plt.figure(figsize=fig_size)
        plt.imshow(img_array, interpolation="nearest", cmap=cmap)
        plt.title(title)

    plt.show()


def annotate_image(image: np.ndarray,
                   bounding_boxes: List[Tuple[int, int, int, int]],
                   scores: List[float],
                   labels: List[str],
                   bbox_color: Tuple[int, int, int] = (0, 255, 0),
                   text_color: Tuple[int, int, int] = (255, 255, 255),
                   thickness: int = 1) -> np.ndarray:
    """
    Annotate an image with bounding boxes, confidence scores, and labels.

    Parameters:
        - image (np.ndarray): Input image as a NumPy array.
        - bounding_boxes (List[Tuple[int, int, int, int]]): List of bounding boxes, each represented as (xmin, ymin, xmax, ymax).
        - scores (List[float]): List of confidence scores.
        - labels (List[str]): List of labels corresponding to the bounding boxes.
        - bbox_color (Tuple[int, int, int]): Bounding box color in BGR format (default: (0, 255, 0)).
        - text_color (Tuple[int, int, int]): Text color in BGR format (default: (255, 255, 255)).
        - thickness (int): Thickness of the bounding box and text (default: 1).

    Returns:
        - np.ndarray: Annotated image.
    """
    for i in range(len(bounding_boxes)):
        bbox = bounding_boxes[i]
        label = labels[i]
        conf = scores[i]

        # Get bounding box co-ordinates
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), bbox_color, thickness)

        # Create the text to display (confidence score and label)
        text = f"{label}: {conf:.2f}"

        # Define the text position
        text_position = (xmin, ymin - 10)

        # Draw the text on the image
        cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, thickness)

    return image
