"""Tiện ích dùng chung cho mọi bước EDA: đường dẫn, nạp dữ liệu, ghi báo cáo, đọc khung hình.

Mỗi script ở thư mục con thêm thư mục cha vào sys.path rồi `from common import *`.
Không script nào ghi vào dataset; mọi kết quả nằm cạnh script sinh ra nó.
"""
import contextlib, hashlib, io, os, sys
from pathlib import Path

import numpy as np
import pandas as pd

EDA = Path(__file__).resolve().parent
DEFAULT_ROOT = EDA.parent / 'dataset'
SPLITS = ['train', 'public_test', 'private_test']
SPLIT_COLORS = ['#3b82f6', '#f59e0b', '#10b981']
SOLID_STD = 3          # khung đơn sắc: độ lệch chuẩn điểm ảnh lớn nhất giữa 3 kênh < 3
BASE_RE = r'_\d+\.mp4$'  # hậu tố _N của bản sao trong train

# Đầu ra của các bước đứng trước, được các bước sau đọc lại
META_CSV = EDA / '01_overview' / 'video_metadata.csv'
HASH_CSV = EDA / '02_data_quality' / 'file_hashes.csv'
FLAGS_CSV = EDA / '02_data_quality' / 'video_quality_flags.csv'


# ---------- đường dẫn & dữ liệu ----------
def dataset_root(argv=None):
    """Thư mục dataset: tham số dòng lệnh đầu tiên (không bắt đầu bằng --), nếu không có thì `../dataset`.
    Nối các token lại để đường dẫn có dấu cách không bị tách."""
    args = [a for a in (sys.argv[1:] if argv is None else argv) if not a.startswith('--')]
    return Path(' '.join(args)) if args else DEFAULT_ROOT


def video_path(root, split, cls, file):
    root = Path(root)
    return root / split / cls / file if split == 'train' else root / split / file


def load_meta():
    """Bảng metadata (bước 01). Thêm cột `base` = tên tệp bỏ hậu tố _N."""
    if not META_CSV.exists():
        sys.exit(f'Thiếu {META_CSV}. Hãy chạy 01_overview/01_scan_metadata.py trước.')
    m = pd.read_csv(META_CSV)
    m['base'] = m.file.str.replace(BASE_RE, '.mp4', regex=True)
    return m


def load_hashes():
    if not HASH_CSV.exists():
        sys.exit(f'Thiếu {HASH_CSV}. Hãy chạy 02_data_quality/02_duplicates.py trước.')
    return pd.read_csv(HASH_CSV)


def load_flags():
    if not FLAGS_CSV.exists():
        sys.exit(f'Thiếu {FLAGS_CSV}. Hãy chạy 02_data_quality/03_frame_scan.py trước.')
    return pd.read_csv(FLAGS_CSV)


def unique_train(m):
    """Video gốc của train: bỏ các bản sao _N (cùng lớp, cùng tên gốc)."""
    return m[m.split == 'train'].drop_duplicates(['cls', 'base'])


def md5_of(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ---------- khung hình ----------
def read_frames(path, rgb=True):
    """Đọc toàn bộ khung hình của một video (RGB mặc định)."""
    import cv2
    cap = cv2.VideoCapture(str(path)); out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if rgb else f)
    cap.release()
    return out


def is_solid(frame):
    """Khung đơn sắc (không phân biệt RGB/BGR vì chỉ dùng độ lệch chuẩn)."""
    return bool(frame.reshape(-1, 3).std(axis=0).max() < SOLID_STD)


# ---------- đầu ra ----------
@contextlib.contextmanager
def report(path):
    """Gom mọi `print` trong khối vào `path` đồng thời in ra màn hình (UTF-8)."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield
    text = buf.getvalue()
    print(text)
    Path(path).write_text(text, encoding='utf-8')


def plt_setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi': 130, 'font.size': 9})
    return plt
