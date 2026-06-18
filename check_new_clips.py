import csv
with open('metadata/clips_metadata.csv', newline='', encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))

new = [r for r in rows if r.get('source_index') in ['17', '18', '19']]
print(f'Total new clips in CSV: {len(new)}')
print()
for r in new:
    skip = r['transcript'] == '__SKIP__'
    fn = r['clip_filename'].ljust(40)
    approved = r['approved']
    reviewed = r['reviewed']
    emo = r['primary_emotion'] or '(none)'
    print(f"  {fn}  approved={approved}  reviewed={reviewed}  skip={skip}  emotion={emo}")
