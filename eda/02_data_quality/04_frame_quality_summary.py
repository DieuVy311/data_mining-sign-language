"""Bước 2.4 – Tổng hợp kết quả quét khung hình: giải mã được không, khung đơn sắc theo tập / dạng / màu / lớp.
Chạy:  python 04_frame_quality_summary.py [đường_dẫn_dataset]
Đầu vào: video_quality_flags.csv, class_counts_train_unique_vs_all.csv, dataset (để đọc lại khung của video bị ảnh hưởng)
Đầu ra: solid_frame_summary.txt, class_solid_frame_ratio.csv
"""
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *
import cv2

HERE = Path(__file__).resolve().parent
ROOT = dataset_root()
d = load_flags()
meta = load_meta()

COLORS = {'đỏ': (0, 0, 255), 'xanh lục': (0, 255, 0), 'xanh lam': (255, 0, 0), 'vàng': (0, 255, 255),
          'cam': (0, 128, 255), 'xanh lục lam': (255, 255, 0), 'tím': (255, 0, 255), 'đen': (0, 0, 0),
          'trắng': (255, 255, 255), 'xám/xanh lam nhạt': (255, 128, 0)}
def color_name(bgr):
    return min(COLORS, key=lambda k: np.sum((np.array(COLORS[k]) - np.array(bgr)) ** 2))

with report(HERE / 'solid_frame_summary.txt'):
    print('== Toàn vẹn khi giải mã')
    chk = d.merge(meta[['split', 'cls', 'file', 'frames']], on=['split', 'cls', 'file'], how='left')
    print('video không đọc được khung nào (n_read = 0):', int((d.n_read == 0).sum()))
    print('video có số khung đọc được khác metadata:', int((chk.n_read != chk.frames).sum()))

    d['all'] = d.solid_ratio >= 0.99; d['any'] = d.solid_frames > 0
    print('\n== Số video có khung đơn sắc theo tập')
    for sp, g in d.groupby('split', sort=False):
        print(sp, 'tổng', len(g), '| >=1 khung đơn sắc', g['any'].sum(), '| đơn sắc toàn bộ', g['all'].sum(),
              '| bắt đầu bằng khung đơn sắc', (g.leading_solid > 0).sum(), '| chuyển động<0.5', (g.motion < 0.5).sum())

    t = d[(d.split == 'train') & d['any']].copy()
    t['base'] = t.file.str.replace(BASE_RE, '.mp4', regex=True)
    u = t.drop_duplicates(['cls', 'base'])
    print('\n== Train, video gốc (đã bỏ bản sao _N): bị ảnh hưởng', len(u), '| đơn sắc toàn bộ', u['all'].sum())
    print('số khung đơn sắc/video:', u.solid_frames.describe().round(2).to_dict())
    print('số lớp bị ảnh hưởng:', u.cls.nunique())

    patterns = Counter(); colors = Counter(); t16 = []
    for r in u.itertuples():
        cap = cv2.VideoCapture(str(ROOT / 'train' / r.cls / r.file)); flags = []
        while True:
            ok, f = cap.read()
            if not ok: break
            s = is_solid(f); flags.append(s)
            if s: colors[color_name(f.reshape(-1, 3).mean(0))] += 1
        cap.release()
        n = len(flags); lead = r.leading_solid
        tail = 0
        for s in reversed(flags):
            if s: tail += 1
            else: break
        mid = r.solid_frames - lead - (tail if lead < n else 0)
        patterns['toàn bộ' if lead == n else 'chỉ đầu' if lead > 0 and tail == 0 and mid == 0 else
                 'chỉ cuối' if lead == 0 and tail > 0 and mid == 0 else
                 'đầu + cuối' if lead > 0 and tail > 0 and mid == 0 else 'có ở giữa'] += 1
        idx = np.linspace(0, n - 1, 16).round().astype(int)
        t16.append(sum(flags[k] for k in idx))
    t16 = np.array(t16)
    print('dạng xuất hiện:', dict(patterns))
    print('số khung theo màu:', dict(colors.most_common()))
    print('T=16 (linspace): trung bình khung đơn sắc/16 =', t16.mean().round(2), '| trung vị', np.median(t16),
          '| video có >=8/16:', int((t16 >= 8).sum()), '| có >=1:', int((t16 >= 1).sum()), '| max', t16.max())

    uq = pd.read_csv(HERE / 'class_counts_train_unique_vs_all.csv')
    mm = uq.set_index('cls').join(u.cls.value_counts().rename('flag')).fillna(0)
    mm['ratio'] = mm.flag / mm.n_unique
    mm.reset_index().to_csv(HERE / 'class_solid_frame_ratio.csv', index=False, encoding='utf-8-sig')
    print('\n== Lớp có tỷ lệ video bị ảnh hưởng cao nhất')
    print(mm.sort_values('ratio', ascending=False).head(8).round(3).to_string())
