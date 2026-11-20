"""Bước 3.1 – Phân tích đơn biến các thuộc tính số (frames, fps, dur, size_kb) theo từng tập.
Gồm: bảng thống kê mô tả, độ lệch (skewness), ngoại lai theo quy tắc Tukey, histogram.
Ngưỡng Tukey tính trên train rồi áp lên cả ba tập, để thấy tập test lệch khỏi train đến đâu.
Chạy:  python 01_numeric_attributes.py
Đầu ra: desc_by_split.csv, univariate_summary.txt,
        frames_hist_by_split.png, fps_hist_by_split.png, filesize_hist_by_split.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *

HERE = Path(__file__).resolve().parent
m = load_meta()
tr = m[m.split == 'train']
VARS = ['frames', 'fps', 'dur', 'size_kb']

rows = []
for c in VARS:
    d = m.groupby('split')[c].describe(percentiles=[.25, .5, .75]).round(2)
    d.insert(0, 'metric', c); rows.append(d.reset_index())
desc = pd.concat(rows)
desc.to_csv(HERE / 'desc_by_split.csv', index=False, encoding='utf-8-sig')

with report(HERE / 'univariate_summary.txt'):
    print('== Mô tả theo tập'); print(desc.to_string(index=False))

    print('\n== Độ lệch (skewness) theo tập')
    print(m.groupby('split')[VARS].skew().round(2).to_string())

    print('\n== Giá trị phổ biến nhất')
    for sp in SPLITS:
        g = m[m.split == sp]
        print(sp, 'frames', g.frames.value_counts().head(3).to_dict(),
              '| fps', g.fps.round(2).value_counts().head(5).to_dict())

    print('\n== Ngoại lai Tukey (khoảng tính trên train): (tổng ngoài khoảng, phía trên)')
    for v in ['dur', 'size_kb', 'fps']:
        q1, q3 = tr[v].quantile([.25, .75]); lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        print(v, round(lo, 2), round(hi, 2),
              {s: (int(((g[v] < lo) | (g[v] > hi)).sum()), int((g[v] > hi).sum())) for s, g in m.groupby('split', sort=False)})
        if v == 'size_kb':
            print('size>%.0fKB (%%):' % hi, {s: round(float(100 * (g.size_kb > hi).mean()), 2) for s, g in m.groupby('split', sort=False)})

# ---- histogram
plt = plt_setup()
fig, axs = plt.subplots(1, 3, figsize=(13, 4))
for a, sp in zip(axs, SPLITS):
    g = m[m.split == sp]
    a.hist(g.frames, bins=range(14, 33), color='#0ea5e9'); a.axvline(16, color='g', ls='--')
    a.set_yscale('log'); a.set_title(f'{sp} frames'); a.set_xlabel('frames')
fig.tight_layout(); fig.savefig(HERE / 'frames_hist_by_split.png'); plt.close()

for col, bins, xl, ttl, name in [('fps', 40, 'FPS', 'FPS distribution per split', 'fps_hist_by_split.png'),
                                 ('size_kb', 50, 'File size (KB)', 'File size per split', 'filesize_hist_by_split.png')]:
    fig, ax = plt.subplots(figsize=(8, 4))
    for sp, c in zip(SPLITS, SPLIT_COLORS):
        ax.hist(m[m.split == sp][col], bins=bins, alpha=.5, label=sp, color=c, density=True)
    ax.set_xlabel(xl); ax.set_title(ttl); ax.legend()
    fig.tight_layout(); fig.savefig(HERE / name); plt.close()
