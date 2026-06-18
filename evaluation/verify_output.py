import os
from pathlib import Path
import pandas as pd

root = Path('datasets/dataset_labeled')

print('=== OUTPUT STRUCTURE ===')
for p in ['images/train', 'images/val', 'images/test', 'labels/train', 'labels/val', 'labels/test', 'manual_review']:
    fp = root / p
    count = len(list(fp.glob('*'))) if fp.exists() else 0
    print('  %s: %d files' % (p, count))

print()
print('=== OUTPUT FILES ===')
for f in ['classes.txt', 'data.yaml', 'annotation_report.csv', 'final_report.txt']:
    fp = root / f
    if fp.exists():
        print('  %s: EXISTS (%.1f KB)' % (f, fp.stat().st_size/1024.0))
    else:
        print('  %s: MISSING' % f)

print()
print('=== REPORTS ===')
reports = root / 'reports'
if reports.exists():
    for f in sorted(reports.iterdir()):
        print('  %s' % f.name)

print()
csv_path = root / 'annotation_report.csv'
if csv_path.exists():
    df = pd.read_csv(csv_path)
    print('=== FIRST 5 ANNOTATIONS ===')
    print(df.head().to_string())
    print()
    print('Total records: %d' % len(df))
    print('Auto-labeled: %d' % len(df[df['flag'] == 'auto_labeled']))
    print('Whole confirmed: %d' % len(df[df['flag'] == 'whole_confirmed']))
    print('Flagged for review: %d' % len(df[df['flag'] != 'auto_labeled']))

print()
print('=== SAMPLE YOLO LABEL ===')
label_dir = root / 'labels' / 'train'
label_files = sorted(label_dir.glob('*.txt'))
if label_files:
    sample = label_files[0]
    print('Label file: %s' % sample.name)
    with open(sample) as f:
        content = f.read().strip()
    if content:
        print(content[:500])
    else:
        print('(empty - no objects)')

print()
print('=== CLASSES ===')
cls_path = root / 'classes.txt'
if cls_path.exists():
    with open(cls_path) as f:
        classes = [l.strip() for l in f if l.strip()]
    print('Total classes: %d' % len(classes))
    for i, c in enumerate(classes):
        print('  %d: %s' % (i, c))

print()
print('=== FINAL REPORT (first 30 lines) ===')
report_path = root / 'final_report.txt'
if report_path.exists():
    with open(report_path, encoding='utf-8') as f:
        lines = f.readlines()
    print(''.join(lines[:30]))
