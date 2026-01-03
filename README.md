# Action Recognition (Stanford 40, FastAPI + PyTorch)

Identify human actions in images using a CNN+LSTM model trained for Stanford 40 Actions, served via FastAPI with a simple web UI.

## Features
- Image upload with annotated predictions and confidence
- Stanford 40 Actions label support with customizable label file
- REST API endpoints for model status, categories, prediction, and reload
- Clean frontend (HTML/JS/CSS) that works out of the box

## Tech Stack
- Backend: FastAPI, Uvicorn, PyTorch, Torchvision, Pillow, NumPy
- Model: ResNet18 feature extractor + LSTM head (CNN+LSTM)
- Frontend: Vanilla HTML/CSS/JS

## Folder Layout
```
Action Recognition/
├── backend/
│   ├── app.py                 # FastAPI server and endpoints
│   ├── models_s40.py          # Model architecture and transforms
│   ├── models/
│   │   ├── s40_cnn_lstm.pth   # Checkpoint (included)
│   │   └── s40_labels.txt     # Optional label names, one per line
│   └── requirements.txt       # Python dependencies
├── frontend/
│   └── index.html             # UI for upload and preview
└── README.md
```

## Requirements
- Python 3.10+ recommended

Install dependencies:
```bash
python -m pip install -r backend/requirements.txt
```

## Run Locally
Start the development server:
```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```
Open the UI:
```
http://127.0.0.1:8000/
```

## API Endpoints
- `GET /api/health` – basic health check
- `GET /api/model` – model availability and error info
- `GET /api/categories` – current label list
- `GET|POST /api/reload_s40` – reload the Stanford 40 model and labels
- `POST /api/predict` – image prediction

Example prediction (Windows PowerShell):
```powershell
curl -Method POST -Form @{ file = Get-Item "C:\path\to\image.jpg" } http://127.0.0.1:8000/api/predict
```

## Model & Labels
- Checkpoint: `backend/models/s40_cnn_lstm.pth`
- Labels: `backend/models/s40_labels.txt` (optional). If present, must contain exactly one class name per line in the same order used during training.
- If no labels are provided and the model output size differs from 40, generic names like `class_0 ... class_N` will be used.

## Frontend Usage
- Drag and drop or choose an image file
- Click “Analyze” to see an annotated prediction and confidence
- Sample images can be served from the `samples/` directory (if present)

## Troubleshooting
- Form upload error: install `python-multipart` (already in requirements)
- Font issues for annotation: system font fallback is enabled; `arial.ttf` is attempted first
- Torch/Torchvision compatibility: versions are kept flexible; ensure GPU drivers are compatible if using CUDA

## License
This project is intended for educational and demonstration purposes. Check the Stanford 40 dataset license for any dataset usage constraints.

