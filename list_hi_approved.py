import csv
with open('metadata/clips_metadata.csv', newline='', encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
for i, r in enumerate(rows):
    if r.get('language') == 'hi' and r.get('approved', '').lower() == 'true':
        idx = str(i).rjust(3)
        fn = r['clip_filename'].ljust(40)
        emo = r['primary_emotion']
        print(f"{idx}  {fn}  {emo}")
