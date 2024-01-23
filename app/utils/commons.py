import cv2
import matplotlib.pyplot as plt


def preprocess_image(img_array):
    threshold_channels = []

    for channel in cv2.split(img_array):
        _, binary_channel = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        threshold_channels.append(binary_channel)

    thresh_img = cv2.merge(threshold_channels)

    return thresh_img


def resize_with_aspect_ratio(img_array, target_width, target_height):
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


def resize_signature(img_array, target_height=150, target_width=220):
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


def get_image_crops(img_array, bounding_boxes):
    crop_holder = []

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (1, 1), 0)
    _, binary_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    for i in range(len(bounding_boxes)):
        bbox = bounding_boxes[i]
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
        crop_holder.append(binary_image[ymin:ymax, xmin:xmax])

    return crop_holder


def plot_images(img_array, title: str = "Image Plot", fig_size=(15, 20), nrows=1, ncols=4):
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
        image, bounding_boxes, scores, labels, bbox_color=(0, 255, 0), text_color=(255, 255, 255), thickness=1
):
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
