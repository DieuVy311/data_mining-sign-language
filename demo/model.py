from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / 'models' / 'cnn_lstm' / 'output' / 'cnn_lstm_B.pth'
WEIGHTS_URL = 'https://github.com/DieuVy311/data_mining-sign-language/releases/download/v1.0/cnn_lstm_B.pth'
N_CLASSES = 100


class TemporalHead(nn.Module):
    def __init__(self, in_dim, proj=512, hidden=256, layers=2, drop=0.3, drop_out=0.4):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, proj), nn.LayerNorm(proj), nn.ReLU(), nn.Dropout(drop))
        self.lstm = nn.LSTM(proj, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                            dropout=drop if layers > 1 else 0.0)
        self.att = nn.Sequential(nn.Linear(2 * hidden, 128), nn.Tanh(), nn.Linear(128, 1, bias=False))
        self.cls = nn.Sequential(nn.Dropout(drop_out), nn.Linear(2 * hidden, N_CLASSES))

    def forward(self, x):
        h, _ = self.lstm(self.proj(x))
        alpha = torch.softmax(self.att(h).squeeze(-1).float(), dim=1)
        z = (alpha.unsqueeze(-1) * h.float()).sum(1)
        return self.cls(z), alpha


class CNNLSTM(nn.Module):
    def __init__(self, feat_mean, feat_std):
        super().__init__()
        cnn = efficientnet_b0(weights=None)
        cnn.classifier = nn.Identity()
        self.cnn, self.head = cnn, TemporalHead(1280)
        self.register_buffer('feat_mean', torch.as_tensor(np.asarray(feat_mean), dtype=torch.float32))
        self.register_buffer('feat_std', torch.as_tensor(np.asarray(feat_std), dtype=torch.float32))

    def forward(self, x):
        b, t = x.shape[:2]
        f = self.cnn(x.flatten(0, 1)).view(b, t, -1)
        return self.head((f - self.feat_mean) / self.feat_std)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def ensure_weights(path: Path = WEIGHTS) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(WEIGHTS_URL, str(path), progress=False)
    return path


def load_model(path: Path = WEIGHTS, device: torch.device | None = None):
    device = device or pick_device()
    ckpt = torch.load(ensure_weights(path), map_location='cpu', weights_only=False)
    model = CNNLSTM(ckpt['feat_mean'], ckpt['feat_std'])
    missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
    if unexpected or set(missing) != {'feat_mean', 'feat_std'}:
        raise RuntimeError(f'Trọng số không khớp kiến trúc: thiếu {missing}, thừa {unexpected}')
    model.to(device).eval()
    labels = [unicodedata.normalize('NFC', s) for s in ckpt['label_names']]
    info = {'epoch': ckpt.get('epoch'), 'device': str(device)}
    return model, labels, info


@torch.no_grad()
def predict(model, x: np.ndarray):
    device = next(model.parameters()).device
    logits, alpha = model(torch.from_numpy(x).unsqueeze(0).to(device))
    prob = torch.softmax(logits.float(), -1)[0].cpu().numpy()
    return prob, alpha[0].cpu().numpy()
