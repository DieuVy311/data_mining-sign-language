"""Bước 2.2 – Trùng lặp: MD5 trong train, xung đột nhãn, trùng/rò rỉ giữa các tập, số video duy nhất mỗi lớp.
Chạy:  python 02_duplicates.py [đường_dẫn_dataset]
Đầu ra: file_hashes.csv (md5 từng video, dùng lại ở các bước sau),
        dups_summary.txt, dedup_summary.txt, class_counts_train_unique_vs_all.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *

HERE = Path(__file__).resolve().parent
ROOT = dataset_root()
df = load_meta()
df['md5'] = [md5_of(video_path(ROOT, r.split, r.cls, r.file)) for r in df.itertuples()]
df['suffix'] = df.file.str.contains(BASE_RE)
df[['split', 'cls', 'file', 'md5']].to_csv(HASH_CSV, index=False, encoding='utf-8-sig')
t = df[df.split == 'train'].copy()
base_of = lambda s: s.str.replace(BASE_RE, '.mp4', regex=True)

# ---- (a) phát hiện trùng lặp
with report(HERE / 'dups_summary.txt'):
    print('files with _N suffix by split:', df.groupby('split').suffix.sum().to_dict())
    print('train unique md5:', t.md5.nunique(), 'of', len(t), '-> duplicates', len(t) - t.md5.nunique())
    print('train files with suffix:', t.suffix.sum())
    g = t.groupby('md5').cls.nunique(); print('md5 groups spanning >1 class:', (g > 1).sum())
    tm = set(t.md5)
    for sp in ['public_test', 'private_test']:
        x = df[df.split == sp]
        print(sp, 'md5 also in train:', x.md5.isin(tm).sum(), 'unique md5', x.md5.nunique(), 'of', len(x))
    pm = set(df[df.split == 'public_test'].md5)
    print('public∩private md5:', df[(df.split == 'private_test')].md5.isin(pm).sum())
    print('train unique base ids', base_of(t.file).nunique())
    tb = set(base_of(df[df.split == 'train'].file))
    print('test base ids in train base ids:',
          {sp: int(base_of(df[df.split == sp].file).isin(tb).sum()) for sp in ['public_test', 'private_test']})
    print(t[t.suffix].file.head(6).tolist())
    print('examples test name overlap:', list(set(t.file) & set(df[df.split == 'public_test'].file))[:5])

# ---- (b) khử trùng lặp train, xung đột nhãn, đếm lại số video mỗi lớp
with report(HERE / 'dedup_summary.txt'):
    g = t.groupby('md5').cls.nunique(); bad = g[g > 1].index
    print(t[t.md5.isin(bad)][['cls', 'file', 'frames', 'size_kb']].sort_values('file').to_string())
    u = t.drop_duplicates('md5'); vc = u.cls.value_counts()
    print('unique train videos', len(u), 'min', vc.min(), 'max', vc.max(), 'mean', round(vc.mean(), 2),
          'ratio', round(vc.max() / vc.min(), 1))
    print(vc.tail(10).to_dict()); print(vc.head(5).to_dict())
    full = t.cls.value_counts(); dup = (full - vc).sort_values(ascending=False)
    print('most duplicated classes', dup.head(8).to_dict())
    pd.DataFrame({'n_all': full, 'n_unique': vc, 'n_dup': full - vc}).rename_axis('cls').to_csv(
        HERE / 'class_counts_train_unique_vs_all.csv', encoding='utf-8-sig')
