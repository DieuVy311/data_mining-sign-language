"""Tiền xử lý – BTL Khai phá dữ liệu: Nhận diện ngôn ngữ ký hiệu Việt Nam (VSLR)

Module này được dùng ở MỌI nơi đọc video để train và test đi chung một đường xử lý:
  - processing/data_processing.ipynb : làm sạch dữ liệu, tạo manifest_clean.csv
  - models/extract_features.ipynb : đọc video theo manifest, trích đặc trưng cho SVM / CNN-LSTM
  - app Streamlit (demo) : xử lý video người dùng tải lên

Phần lõi chỉ cần numpy + opencv. Lớp SignVideoDataset cho PyTorch ở cuối file, torch chỉ cần khi dùng lớp đó.

Quy trình cho 1 video:
  đọc toàn bộ khung (RGB)
  -> bỏ khung đơn sắc 
  -> [chỉ train] cắt ngẫu nhiên theo thời gian (temporal jitter)
  -> lấy đều T = 16 khung bằng linspace 
  -> [chỉ train] đổi màu + cắt ngẫu nhiên theo không gian
  -> chuẩn hoá z-score theo ImageNet, đổi trục (T, 3, H, W) 
"""
from __future__ import annotations

import hashlib
import os
import pickle
import unicodedata
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------- hằng số ----------------------------------------------------
TARGET_FRAMES = 16          
SOLID_STD = 3.0           
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv')


# -------------------------------------  nhãn và đường dẫn --------------------------------------------------
def nfc(text: str) -> str:
    """Chuẩn hoá Unicode về NFC. Khoá trong label_mapping.pkl ở dạng NFD, tên thư mục có thể là NFC hoặc NFD
    tuỳ hệ điều hành / cách giải nén, nên mọi phép so sánh tên lớp phải đi qua hàm này."""
    return unicodedata.normalize('NFC', text).strip()


def load_label_mapping(path) -> dict[str, int]:
    """{tên lớp NFC: chỉ số 0..99}."""
    with open(path, 'rb') as f:
        raw = pickle.load(f)
    return {nfc(k): int(v) for k, v in raw.items()}


def class_dirs(train_root) -> dict[str, str]:
    """{tên lớp NFC: tên thư mục thật trên đĩa}."""
    return {nfc(d): d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))}


def list_videos(folder) -> list[str]:
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(VIDEO_EXTS))


def md5_of(path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def parse_idx(text) -> list[int]:
    """'0 1 2 5' -> [0, 1, 2, 5] (định dạng cột valid_idx trong manifest)."""
    if isinstance(text, (list, tuple, np.ndarray)):
        return [int(i) for i in text]
    if text is None or (isinstance(text, float) and np.isnan(text)) or str(text).strip() == '':
        return []
    return [int(i) for i in str(text).split()]


def format_idx(idx) -> str:
    return ' '.join(str(int(i)) for i in idx)


# ----------------------------------------- đọc video và lọc khung -------------------------------------------
def read_video(path) -> np.ndarray:
    """Toàn bộ khung hình, mảng uint8 (T, H, W, 3) theo hệ màu RGB."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f'Không đọc được khung hình nào: {path}')
    return np.stack(frames)


def is_solid(frame: np.ndarray, thr: float = SOLID_STD) -> bool:
    """Khung đơn sắc = gần như mọi điểm ảnh cùng một màu."""
    return bool(frame.reshape(-1, 3).std(axis=0).max() < thr)


def valid_indices(frames: np.ndarray, thr: float = SOLID_STD) -> list[int]:
    """Chỉ số các khung KHÔNG đơn sắc."""
    return [i for i, f in enumerate(frames) if not is_solid(f, thr)]


def scan_video(path) -> dict:
    """Dùng ở bước làm sạch: md5, số khung, chỉ số khung hợp lệ. Không ném lỗi để chạy song song được."""
    out = {'path': str(path), 'readable': False, 'n_frames': 0, 'valid_idx': '', 'n_valid': 0, 'md5': ''}
    try:
        out['md5'] = md5_of(path)
        frames = read_video(path)
        idx = valid_indices(frames)
        out.update(readable=True, n_frames=len(frames), valid_idx=format_idx(idx), n_valid=len(idx))
    except Exception as e:
        out['error'] = str(e)
    return out


# -------------------------------------- lấy mẫu theo thời gian --------------------------------------------
def sample_indices(candidates, T: int = TARGET_FRAMES) -> list[int]:
    """Lấy đều T phần tử trong danh sách khung ứng viên bằng linspace.
    Ít hơn T khung thì linspace tự lặp đều các khung (không chỉ lặp khung cuối)."""
    candidates = list(candidates)
    if not candidates:
        raise ValueError('Danh sách khung ứng viên rỗng')
    pos = np.linspace(0, len(candidates) - 1, T).round().astype(int)
    return [candidates[p] for p in pos]


# ---------------------------------- tăng cường dữ liệu (chỉ khi train) -------------------------------------
class VideoAugmentation:
    """Tăng cường dữ liệu cho video, áp dụng giống nhau cho mọi khung của một video.

    temporal_scale : giữ ngẫu nhiên một đoạn liên tục dài 80–100% số khung hợp lệ rồi mới lấy 16 khung.
        Mô phỏng ký hiệu bắt đầu/kết thúc sớm muộn và làm nhanh/chậm khác nhau.
    crop_scale     : cắt một vùng 85–100% khung hình ở vị trí ngẫu nhiên rồi phóng lại kích thước cũ.
    brightness / contrast / saturation : hệ số ngẫu nhiên trong [1 - x, 1 + x].
    Không lật ngang: tay trái/tay phải có thể đổi nghĩa ký hiệu.
    """

    def __init__(self, temporal_scale=(0.8, 1.0), crop_scale=(0.85, 1.0),
                 brightness=0.25, contrast=0.25, saturation=0.25):
        self.temporal_scale = temporal_scale
        self.crop_scale = crop_scale
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def temporal(self, candidates, rng: np.random.Generator):
        candidates = list(candidates)
        n = len(candidates)
        keep = max(1, int(round(n * rng.uniform(*self.temporal_scale))))
        start = int(rng.integers(0, n - keep + 1))
        return candidates[start:start + keep]

    def spatial(self, clip: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """clip: uint8 (T, H, W, 3) -> uint8 cùng kích thước."""
        T, H, W, _ = clip.shape

        s = rng.uniform(*self.crop_scale)
        ch, cw = int(H * s), int(W * s)
        top, left = int(rng.integers(0, H - ch + 1)), int(rng.integers(0, W - cw + 1))
        out = np.stack([cv2.resize(f[top:top + ch, left:left + cw], (W, H), interpolation=cv2.INTER_LINEAR)
                        for f in clip]).astype(np.float32)

        b = 1 + rng.uniform(-self.brightness, self.brightness)
        c = 1 + rng.uniform(-self.contrast, self.contrast)
        sat = 1 + rng.uniform(-self.saturation, self.saturation)
        out *= b
        mean = out.mean(axis=(1, 2), keepdims=True)
        out = (out - mean) * c + mean
        gray = out.mean(axis=-1, keepdims=True)
        out = gray + (out - gray) * sat
        return np.clip(out, 0, 255).astype(np.uint8)


# -------------------------------------  một video -> đầu vào mô hình -------------------------------------
def load_clip(path, valid_idx=None, T: int = TARGET_FRAMES, aug: VideoAugmentation | None = None,
              rng: np.random.Generator | None = None) -> np.ndarray:
    """Đọc video -> bỏ khung đơn sắc -> (tăng cường) -> T khung. Trả về uint8 (T, H, W, 3).

    valid_idx: chỉ số khung hợp lệ đã tính sẵn (cột valid_idx của manifest) để khỏi tính lại;
               None thì tự tính (dùng cho test và demo).
    aug:       truyền VideoAugmentation khi train, để None khi validation/test.
    """
    frames = read_video(path)
    cand = parse_idx(valid_idx) if valid_idx is not None else valid_indices(frames)
    cand = [i for i in cand if i < len(frames)]
    if not cand:                     
        cand = list(range(len(frames)))
    if aug is not None:
        rng = rng or np.random.default_rng()
        cand = aug.temporal(cand, rng)
    clip = frames[sample_indices(cand, T)]
    if aug is not None:
        clip = aug.spatial(clip, rng)
    return clip


def normalize(clip: np.ndarray) -> np.ndarray:
    """uint8 (T, H, W, 3) -> float32 (T, 3, H, W), chuẩn hoá z-score từng kênh theo mean/std ImageNet."""
    x = clip.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def class_weights(labels, n_classes: int = 100) -> np.ndarray:
    """Trọng số lớp kiểu 'balanced': n_mẫu / (n_lớp * số mẫu của lớp). Lớp ít mẫu có trọng số lớn."""
    counts = np.bincount(np.asarray(labels), minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1
    return (len(labels) / (n_classes * counts)).astype(np.float32)


# --------------------------------- PyTorch Dataset (optional) ---------------------------------
try:
    import torch
    from torch.utils.data import Dataset as _BaseDataset
except ImportError:  # môi trường không có torch 
    torch = None
    _BaseDataset = object


class SignVideoDataset(_BaseDataset):
    """Đọc video theo manifest_clean.csv (train/validation) hoặc test_manifest.csv (test).

    manifest     : DataFrame có cột 'file' và, với train, 'cls', 'label_idx', 'valid_idx'
    dataset_root : thư mục chứa train/, public_test/, private_test/
    split        : 'train' (video nằm trong thư mục lớp) hoặc 'public_test' / 'private_test'
    aug          : VideoAugmentation khi train, None khi validation/test
    """

    def __init__(self, manifest, dataset_root, split='train', aug=None, T=TARGET_FRAMES):
        self.df = manifest.reset_index(drop=True)
        self.root = Path(dataset_root)
        self.split = split
        self.aug = aug
        self.T = T
        self.dirs = class_dirs(self.root / 'train') if split == 'train' else None

    def __len__(self):
        return len(self.df)

    def path(self, i):
        r = self.df.iloc[i]
        if self.split == 'train':
            return self.root / 'train' / self.dirs[nfc(r['cls'])] / r['file']
        return self.root / self.split / r['file']

    def __getitem__(self, i):
        r = self.df.iloc[i]
        valid = r['valid_idx'] if 'valid_idx' in r.index else None
        clip = load_clip(self.path(i), valid_idx=valid, T=self.T, aug=self.aug,
                         rng=np.random.default_rng() if self.aug is not None else None)
        x = normalize(clip)
        item = {'frames': torch.from_numpy(x) if torch is not None else x, 'file': r['file']}
        if 'label_idx' in r.index:
            item['label_idx'] = int(r['label_idx'])
            item['label'] = r['cls']
        return item
