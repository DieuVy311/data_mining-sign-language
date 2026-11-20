"""Chạy toàn bộ quy trình EDA theo đúng thứ tự.
Chạy:  python run_all.py [--skip-frame-scan] [--from BƯỚC] [đường_dẫn_dataset]
  --skip-frame-scan : bỏ qua 02_data_quality/03_frame_scan.py (10–12 phút), dùng lại video_quality_flags.csv đã có
  --from 03         : chỉ chạy từ thư mục bắt đầu bằng "03" trở đi (dùng khi các bước trước đã xong)
"""
import subprocess, sys, time
from pathlib import Path

EDA = Path(__file__).resolve().parent
STEPS = [
    '01_overview/01_scan_metadata.py',
    '01_overview/02_dataset_structure.py',
    '02_data_quality/01_integrity_missing.py',
    '02_data_quality/02_duplicates.py',
    '02_data_quality/03_frame_scan.py',
    '02_data_quality/04_frame_quality_summary.py',
    '03_univariate/01_numeric_attributes.py',
    '04_target_analysis/01_class_distribution.py',
    '05_bivariate_multivariate/01_correlation_and_label_association.py',
    '06_split_comparison/01_split_shift.py',
    '07_visual_inspection/01_sample_frames.py',
    '07_visual_inspection/02_flagged_videos.py',
]
SLOW = '02_data_quality/03_frame_scan.py'

args = sys.argv[1:]
skip_scan = '--skip-frame-scan' in args
start = ''
if '--from' in args:
    i = args.index('--from'); start = args[i + 1]; del args[i:i + 2]
rest = [a for a in args if a != '--skip-frame-scan']      # còn lại: đường dẫn dataset (nếu có)
root = ' '.join(rest)

for s in STEPS:
    if s.split('/')[0] < start or (skip_scan and s == SLOW):
        print('-- bỏ qua', s); continue
    cmd = [sys.executable, '-I', str(EDA / s)]
    if root:
        cmd += ['--root', root] if s == SLOW else [root]
    print('==>', s, flush=True); t0 = time.time()
    r = subprocess.run(cmd, cwd=EDA)
    if r.returncode:
        sys.exit(f'Lỗi ở {s} (exit {r.returncode}); dừng.')
    print(f'    xong sau {time.time() - t0:.0f}s\n')
