from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

import shutil
import os
import uuid

import predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model...")
    predict.load_model_once()
    print("Model loaded!")

    yield


app = FastAPI(lifespan=lifespan)

app.mount("/examples", StaticFiles(directory="examples"), name="examples")


@app.get("/")
def index():
    return FileResponse("index.html")


@app.post("/predict")
async def predict_route(file: UploadFile = File(...)):
    file_path = f"temp_{uuid.uuid4().hex}.jpg"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict.predict_image(file_path)

    os.remove(file_path)

    return result
