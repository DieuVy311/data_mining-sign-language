"""Bước 7.1 – Khung hình mẫu (quan sát định tính) của các lớp chọn ngẫu nhiên (seed 0).
Chạy:  python 01_sample_frames.py [đường_dẫn_dataset]
Đầu ra: sample_frames_6classes.png (6 lớp x 6 khung, video đầu tiên mỗi lớp),
        sample_frames_strip.png (8 khung của một video thuộc lớp đầu tiên trong mẫu)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *
import cv2

HERE = Path(__file__).resolve().parent
ROOT = dataset_root()
t = load_meta().query("split == 'train'")
plt = plt_setup()

def grab(p, n):
    c = cv2.VideoCapture(str(p)); N = int(c.get(cv2.CAP_PROP_FRAME_COUNT)); fr = []
    for i in np.linspace(0, N - 1, n).astype(int):
        c.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = c.read()
        if ok: fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return fr

rng = np.random.default_rng(0)
cl = rng.choice(sorted(t.cls.unique()), 6, replace=False)

fig, axs = plt.subplots(6, 6, figsize=(10, 10))
for r, c in enumerate(cl):
    fn = t[t.cls == c].iloc[0].file; fr = grab(ROOT / 'train' / c / fn, 6)
    for k, a in enumerate(axs[r]):
        a.axis('off')
        if k < len(fr): a.imshow(fr[k])
    axs[r][0].set_title(c, fontsize=8, loc='left')
fig.tight_layout(); fig.savefig(HERE / 'sample_frames_6classes.png'); plt.close()

c = cl[0]; fr = grab(ROOT / 'train' / c / t[t.cls == c].iloc[0].file, 8)
fig, axs = plt.subplots(1, 8, figsize=(16, 2.3))
for a, im in zip(axs, fr): a.imshow(im); a.axis('off')
fig.suptitle(f'Sample frames, class "{c}"'); fig.tight_layout()
fig.savefig(HERE / 'sample_frames_strip.png'); plt.close()
print('done:', list(cl))
