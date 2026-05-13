# Cat vs Dog Classifier

Cat vs Dog Classifier is an interactive deep learning demo for real-time pet image classification using FastAPI and TensorFlow/Keras.

Important: this project is for educational/demo use only. It is not intended for production or safety-critical applications.

## What This Demo Shows
- Fast local inference for pet image classification (`.png`)
- Custom-trained convolutional neural network
- Interactive drag & drop upload interface
- Built-in example gallery for instant testing
- Visual prediction feedback:
  - predicted class
  - confidence score
  - animated probability bar
- Modern glassmorphism-style UI with smooth animations

## Current Demo Stack
The default local demo uses a custom CNN trained with TensorFlow/Keras.

Current architecture:
- Conv2D + MaxPooling feature extraction blocks
- Dense classification head
- Sigmoid binary output (`Cat` vs `Dog`)

Primary model artifact:
- `cats_vs_dogs.h5`

Training script:
- `train.py`

## Quick Start (Local)

```bash
cd /path/to/cats-vs-dogs
uvicorn main:app --reload
```

Open:
- UI: `http://127.0.0.1:8000`
- API docs: `http://127.0.0.1:8000/docs`

Note: first startup can take several seconds because the TensorFlow model is loaded during application startup.

## How To Test The Demo
1. Open `http://127.0.0.1:8000`.
2. Drag & drop a PNG image or click to upload.
3. Press `Predict`.
4. Inspect:
   - predicted label
   - confidence score
   - probability bar
5. Or click one of the built-in example images for instant inference.

## API Endpoints
- `GET /`
- `POST /predict`

## Example API Response

```json
{
  "label": "Dog",
  "confidence": 0.97
}
```

## Project Layout

```text
data/
  train/
  validation/

examples/
  cat1.jpg
  cat2.jpg
  cat3.jpg
  dog1.jpg
  dog2.jpg
  dog3.jpg

cats_vs_dogs.h5
index.html
main.py
predict.py
train.py
README.md
```

## Training
Base training flow used in this project:

```bash
python train.py
```

Current training setup:
- image size: `150x150`
- optimizer: `Adam`
- loss: `binary_crossentropy`
- metric: `accuracy`

The trained model is exported as:
- `cats_vs_dogs.h5`

## Frontend Features
Current UI capabilities:
- drag & drop upload
- image preview
- loading spinner
- confidence visualization
- animated interactions
- interactive sample gallery

## Repository Hygiene (.gitignore Policy)
This repository is configured to keep lightweight demo assets and trained inference artifacts.

Kept intentionally:
- trained inference model (`cats_vs_dogs.h5`)
- curated example image gallery
- lightweight frontend assets

Ignored by default:
- large raw datasets
- temporary training artifacts
- caches and generated files

## Troubleshooting
If startup is slow:
- wait longer on first run (TensorFlow model loading can take time)
- check logs in the terminal running `uvicorn`

If port `8000` is busy:

```bash
uvicorn main:app --reload --port 8010
```

If prediction fails:
- ensure uploaded images are valid PNG files
- verify that `cats_vs_dogs.h5` exists in the project root

## Future Improvements
Planned improvements:
- JPG/JPEG support
- mobile responsive layout
- improved model accuracy
- Docker deployment
- batch inference support
- model versioning
- inference performance optimization

## Disclaimer
This project is a learning and portfolio demonstration for deep learning and FastAPI integration.