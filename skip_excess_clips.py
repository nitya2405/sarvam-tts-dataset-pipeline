"""Mark source 19 clips 0006+ with a placeholder so Phase 4 skips them."""
import csv

path = 'metadata/clips_metadata.csv'
with open(path, newline='', encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())

skipped = 0
for r in rows:
    fn = r['clip_filename']
    if fn.startswith('hi_speech_019_') and r.get('transcript', '').strip() == '':
        num = int(fn.split('_')[-1].replace('.wav', ''))
        if num > 5:
            r['transcript'] = '__SKIP__'
            skipped += 1

with open(path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f'Marked {skipped} source-19 clips as skip. Phase 4 will only transcribe 0001-0005.')
