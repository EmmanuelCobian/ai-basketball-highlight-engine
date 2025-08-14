import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from pathlib import Path
import torch.nn as nn
from torchvision import models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Resolve model path: use HL_ENGINE_MODEL_PATH if set, else file co-located with this module
MODEL_FILENAME = os.getenv("HL_ENGINE_MODEL_NAME", "cnn_lstm_best_with_unlabelled.pth")
_MODEL_PATH_ENV = os.getenv("HL_ENGINE_MODEL_PATH")
MODEL_PATH = (Path(_MODEL_PATH_ENV).expanduser().resolve() if _MODEL_PATH_ENV else (Path(__file__).resolve().parent / MODEL_FILENAME))
OUTPUT_DIR = "highlight_clips"
CLIP_DURATION = 5  #seconds
NUM_FRAMES = 16
FRAME_SIZE = (112, 112)
LABELS_TO_KEEP = {1} #if using binary model

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(FRAME_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

class CNNLSTM(nn.Module):
    #change here if model changed
    def __init__(self):
        super().__init__()
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()
        self.rnn = nn.LSTM(
            input_size=512, hidden_size=256,
            num_layers=1, batch_first=True,
            bidirectional=True, dropout=0.3)
        self.head = nn.Sequential(
            nn.LayerNorm(256 * 2),
            nn.Linear(256 * 2, 2)
        )

    def forward(self, video: torch.Tensor):
        B, T, C, H, W = video.shape
        feats = self.cnn(video.view(B * T, C, H, W))      
        feats = feats.view(B, T, -1)                    
        _, (h_n, _) = self.rnn(feats)                    
        h = torch.cat((h_n[-2], h_n[-1]), dim=1)          
        return self.head(h)


def _load_model():
    model = CNNLSTM().to(DEVICE)
    model_path = MODEL_PATH if isinstance(MODEL_PATH, Path) else Path(MODEL_PATH)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Highlight model not found at '{model_path}'. "
            f"Set HL_ENGINE_MODEL_PATH to an absolute path or place '{MODEL_FILENAME}' next to generator.py."
        )
    state = torch.load(str(model_path), map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

def _sample_indices(start, end, num_samples):
    return np.linspace(start, end - 1, num_samples, dtype=int)

def _predict_clip(model, cap, start_frame, fps):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    clip_frames = []
    for _ in range(CLIP_DURATION * int(fps)):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        clip_frames.append(frame)

    if len(clip_frames) < 4:
        return None, None

    indices = _sample_indices(0, len(clip_frames), NUM_FRAMES)
    sampled = [clip_frames[i] for i in indices]
    tensor_clip = torch.stack([transform(f) for f in sampled])
    tensor_clip = tensor_clip.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor_clip)
    pred = output.argmax(1).item()
    return pred, clip_frames

def generate_highlights_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_clip = int(CLIP_DURATION * fps)

    model = _load_model()
    results = []

    for start in range(0, total_frames - frames_per_clip + 1, frames_per_clip):
        pred, frames = _predict_clip(model, cap, start, fps)
        if pred is None:
            continue
        if pred in LABELS_TO_KEEP:
            start_f = start
            end_f = min(start + frames_per_clip, total_frames)
            results.append((int(start_f), int(end_f)))

    cap.release()
    # print("HIGHLIGHT CLIP RESULTS", results)
    return results

