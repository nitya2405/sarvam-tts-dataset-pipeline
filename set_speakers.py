import csv

path = 'metadata/source_log.csv'
with open(path, newline='', encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())

updates = {'19': '1'}
for r in rows:
    idx = r['source_index']
    if idx in updates:
        r['target_speaker_label'] = updates[idx]
        print(f"  Set index {idx} -> speaker {updates[idx]}")

with open(path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print('Done.')
