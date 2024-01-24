import contextlib
import cv2
import numpy as np
import os
from pathlib import Path

from fastapi import FastAPI, APIRouter, File, UploadFile
from app.utils.commons import preprocess_image, get_image_crops, save_image
from app.engine.detector.detector import SignDetector
from app.engine.cleaner.cleaner import SignCleaner
from app.constants.constants import SIGN_DETECTOR_MODEL_PATH, SIGN_CLEANER_MODEL_PATH

router = APIRouter()

detector = SignDetector()
cleaner = SignCleaner()

save_path = "assets/images"

if not os.path.exists(save_path):
    Path(save_path).mkdir(parents=True)


@contextlib.asynccontextmanager
async def lifespan(app: APIRouter):
    detect_model_path = os.path.join(os.path.dirname(__file__), "..", SIGN_DETECTOR_MODEL_PATH)
    clean_model_path = os.path.join(os.path.dirname(__file__), "..", SIGN_CLEANER_MODEL_PATH)
    detector.load(detect_model_path)
    cleaner.load(clean_model_path)
    yield


router_app = FastAPI(lifespan=lifespan)


@router.post("/")
async def extract_signature(file: UploadFile = File(...), clean: bool = False):
    contents = await file.read()
    np_arr = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    preprocessed_image = preprocess_image(image.copy())

    # Detect signatures
    bboxes, scores, labels = detector.detect(preprocessed_image)

    # Crop the detected signatures
    crop_signatures = get_image_crops(preprocessed_image, bboxes)

    image_save_path = save_path + "/" + file.filename.split(".")[0]

    if clean:
        clean_cropped_signatures = []
        for crop_signature in crop_signatures:
            clean_cropped_signatures.append(cleaner.clean(crop_signature))

        print(clean_cropped_signatures[0].shape)

        image_url = save_image(clean_cropped_signatures, image_save_path)
    else:
        image_url = save_image(crop_signatures, image_save_path)

    return [{"url": url, "box": bbox, "score": score, "label": label}
            for url, bbox, score, label in zip(image_url, bboxes, scores, labels)]
