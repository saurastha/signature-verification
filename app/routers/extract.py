import contextlib
import cv2
import numpy as np
import os

from fastapi import FastAPI, APIRouter, File, UploadFile

from app.detector.detector import SignDetector
from app.constants.constants import SIGN_DETECTOR_MODEL_PATH

router = APIRouter()

detector = SignDetector()


@contextlib.asynccontextmanager
async def lifespan(app: APIRouter):
    model_path = os.path.join(os.path.dirname(__file__), "..", SIGN_DETECTOR_MODEL_PATH)
    detector.load(model_path)
    yield


router_app = FastAPI(lifespan=lifespan)


@router.post("/")
async def extract_signature(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    bboxes, scores, labels = detector.detect(image)

    return [{"box": bbox, "score": score, "label": label}
            for bbox, score, label in zip(bboxes, scores, labels)]
