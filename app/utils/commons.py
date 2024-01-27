from typing import List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np

from skimage import filters, transform


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


def preprocess_signature(
    img: np.ndarray,
    canvas_size: Tuple[int, int] = (840, 1360),
    img_size: Tuple[int, int] = (170, 242),
    input_size: Tuple[int, int] = (150, 220),
) -> np.ndarray:
    """
    Preprocess a signature image by normalizing, inverting, resizing, and optionally cropping.

    Parameters:
        - img (np.ndarray : H x W): Input signature image as a NumPy array.
        - canvas_size (Tuple[int, int]): Size of the canvas for normalizing the image (default: (840, 1360)).
        - img_size (Tuple[int, int]): Target size for resizing the image (default: (170, 242)).
        - input_size (Tuple[int, int]): Target size for cropping the image (default: (150, 220)).
                                   If None, no cropping is performed.

    Returns:
        - np.ndarray: Preprocessed signature image.
    """

    img = img.astype(np.uint8)
    centered = normalize_image(img, canvas_size)
    inverted = 255 - centered
    resized = resize_image(inverted, img_size)

    if input_size is not None and input_size != img_size:
        cropped = crop_center(resized, input_size)
    else:
        cropped = resized

    return cropped


def normalize_image(img: np.ndarray, canvas_size: Tuple[int, int] = (840, 1360)) -> np.ndarray:
    """
    Normalize a signature image by cropping, centering, and removing noise.

    Parameters:
       - img (np.ndarray): Input signature image as a NumPy array.
       - canvas_size (Tuple[int, int]): Size of the canvas for normalizing the image (default: (840, 1360)).

    Returns:
        - np.ndarray: Normalized signature image.
    """

    # 1) Crop the image before getting the center of mass

    # Apply a gaussian filter on the image to remove small components
    blur_radius = 2
    blurred_image = filters.gaussian(img, blur_radius, preserve_range=True)

    # Binarize the image using OTSU's algorithm. This is used to find the center
    # of mass of the image, and find the threshold to remove background noise
    threshold = filters.threshold_otsu(img)

    # Find the center of mass
    binarized_image = blurred_image > threshold
    r, c = np.where(binarized_image == 0)
    r_center = int(r.mean() - r.min())
    c_center = int(c.mean() - c.min())

    # Crop the image with a tight box
    cropped = img[r.min() : r.max(), c.min() : c.max()]

    # 2) Center the image
    img_rows, img_cols = cropped.shape
    max_rows, max_cols = canvas_size

    r_start = max_rows // 2 - r_center
    c_start = max_cols // 2 - c_center

    # Make sure the new image does not go off bounds
    # Emit a warning if the image needs to be cropped, since we don't want this
    if img_rows > max_rows:
        # Case 1: image larger than required (height):  Crop.
        print("Warning: cropping image. The signature should be smaller than the canvas size")
        r_start = 0
        difference = img_rows - max_rows
        crop_start = difference // 2
        cropped = cropped[crop_start : crop_start + max_rows, :]
        img_rows = max_rows
    else:
        extra_r = (r_start + img_rows) - max_rows
        # Case 2: centering exactly would require a larger image. relax the centering of the image
        if extra_r > 0:
            r_start -= extra_r
        if r_start < 0:
            r_start = 0

    if img_cols > max_cols:
        # Case 3: image larger than required (width). Crop.
        print("Warning: cropping image. The signature should be smaller than the canvas size")
        c_start = 0
        difference = img_cols - max_cols
        crop_start = difference // 2
        cropped = cropped[:, crop_start : crop_start + max_cols]
        img_cols = max_cols
    else:
        # Case 4: centering exactly would require a larger image. relax the centering of the image
        extra_c = (c_start + img_cols) - max_cols
        if extra_c > 0:
            c_start -= extra_c
        if c_start < 0:
            c_start = 0

    normalized_image = np.ones((max_rows, max_cols), dtype=np.uint8) * 255
    # Add the image to the blank canvas
    normalized_image[r_start : r_start + img_rows, c_start : c_start + img_cols] = cropped

    # Remove noise - anything higher than the threshold. Note that the image is still grayscale
    normalized_image[normalized_image > threshold] = 255

    return normalized_image


def resize_image(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Resize a signature image to a specified size while maintaining the aspect ratio.

    Parameters:
        - img (np.ndarray): Input signature image as a NumPy array.
        - size (Tuple[int, int]): Target size for resizing the image.

    Returns:
        - np.ndarray: Resized signature image.
    """
    height, width = size

    # Check which dimension needs to be cropped
    # (assuming the new height-width ratio may not match the original size)
    width_ratio = float(img.shape[1]) / width
    height_ratio = float(img.shape[0]) / height
    if width_ratio > height_ratio:
        resize_height = height
        resize_width = int(round(img.shape[1] / height_ratio))
    else:
        resize_width = width
        resize_height = int(round(img.shape[0] / width_ratio))

    # Resize the image (will still be larger than new_size in one dimension)
    img = transform.resize(img,
                           (resize_height, resize_width),
                           mode="constant",
                           anti_aliasing=True,
                           preserve_range=True)

    img = img.astype(np.uint8)

    # Crop to exactly the desired new_size, using the middle of the image:
    if width_ratio > height_ratio:
        start = int(round((resize_width - width) / 2.0))
        return img[:, start : start + width]
    else:
        start = int(round((resize_height - height) / 2.0))
        return img[start : start + height, :]


def crop_center(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Crop the center of a signature image to a specified size.

    Parameters:
        - img (np.ndarray): Input signature image as a NumPy array.
        - size (Tuple[int, int]): Target size for cropping the image.

    Returns:
        - np.ndarray: Cropped signature image.
    """

    img_shape = img.shape
    start_y = (img_shape[0] - size[0]) // 2
    start_x = (img_shape[1] - size[1]) // 2
    cropped = img[start_y : start_y + size[0], start_x : start_x + size[1]]
    return cropped


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


def plot_images(
    img_array: Union[np.ndarray, List[np.ndarray]],
    title: str = "Image Plot",
    fig_size: Tuple[int, int] = (15, 20),
    nrows: int = 1,
    ncols: int = 4,
) -> None:
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


def annotate_image(
    image: np.ndarray,
    bounding_boxes: List[Tuple[int, int, int, int]],
    scores: List[float],
    labels: List[str],
    bbox_color: Tuple[int, int, int] = (0, 255, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> np.ndarray:
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
