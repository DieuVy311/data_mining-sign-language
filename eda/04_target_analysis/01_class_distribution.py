"""Bước 4.1 – Phân tích biến mục tiêu: phân phối số video theo lớp (train), mức mất cân bằng.
Số liệu tính hai cách: trên toàn bộ train, và trên video duy nhất (bỏ bản trùng MD5) nếu đã chạy 02_data_quality/02_duplicates.py.
Chạy:  python 01_class_distribution.py
Đầu ra: class_summary.txt, class_counts_train.csv,
        class_distribution_all.png, class_top_bottom10.png, samples_per_class_hist.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *

HERE = Path(__file__).resolve().parent
m = load_meta()
vc = m[m.split == 'train'].cls.value_counts()

with report(HERE / 'class_summary.txt'):
    print('classes', len(vc)); print(vc.describe().round(2).to_dict())
    print('top10', vc.head(10).to_dict()); print('bottom10', vc.tail(10).to_dict())
    print('imbalance ratio', vc.max() / vc.min())
    print('count vc', vc.value_counts().sort_index().to_dict())
    print('total==', vc.sum(), ' <=25:', (vc <= 25).sum(), ' >=50:', (vc >= 50).sum())
    ua = HERE.parents[0] / '02_data_quality' / 'class_counts_train_unique_vs_all.csv'
    if ua.exists():
        u = pd.read_csv(ua).set_index('cls').n_unique
        print('\n== Sau khi bỏ video trùng MD5')
        print('video duy nhất', int(u.sum()), '| min', int(u.min()), 'max', int(u.max()),
              '| imbalance ratio', round(u.max() / u.min(), 1))
    print('\n== Số video theo lớp (toàn bộ train)')
    print(vc.to_string())

plt = plt_setup()
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(range(len(vc)), vc.values, color='#3b82f6')
ax.axhline(vc.mean(), color='r', ls='--', label=f'mean={vc.mean():.1f}')
ax.set_xlabel('Class (sorted)'); ax.set_ylabel('# videos')
ax.set_title(f'Class distribution - train ({len(vc)} classes)'); ax.legend()
fig.tight_layout(); fig.savefig(HERE / 'class_distribution_all.png'); plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
top = pd.concat([vc.head(10), vc.tail(10)])
ax.barh(top.index[::-1], top.values[::-1], color=['#ef4444'] * 10 + ['#22c55e'] * 10)
ax.set_title('Top 10 most / Top 10 least frequent classes')
fig.tight_layout(); fig.savefig(HERE / 'class_top_bottom10.png'); plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(vc.values, bins=15, color='#8b5cf6')
ax.set_xlabel('videos per class'); ax.set_ylabel('# classes'); ax.set_title('Histogram of samples per class')
fig.tight_layout(); fig.savefig(HERE / 'samples_per_class_hist.png'); plt.close()

vc.rename('n_videos').to_csv(HERE / 'class_counts_train.csv', encoding='utf-8-sig')
