import csv
from collections import defaultdict

with open('metadata/clips_metadata.csv', newline='', encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))

approved = [r for r in rows if r.get('approved', '').lower() == 'true']

groups = defaultdict(list)
for r in approved:
    key = (r['language'], r['primary_emotion'])
    groups[key].append(r)

for lang in ['en', 'hi']:
    label = 'ENGLISH' if lang == 'en' else 'HINDI'
    print(f'\n=== {label} ===')
    lang_groups = {k: v for k, v in groups.items() if k[0] == lang}
    for (_, emo), clips in sorted(lang_groups.items()):
        total = sum(float(c['duration_s']) for c in clips)
        print(f'\n{emo.upper()}  —  {len(clips)} clips  |  {total/60:.2f} min total')
        print(f'  {"Sr":<4}  {"Filename":<44}  {"Duration (s)"}')
        print(f'  {"-"*4}  {"-"*44}  {"-"*12}')
        for i, c in enumerate(clips, 1):
            print(f'  {i:<4}  {c["clip_filename"]:<44}  {float(c["duration_s"]):.1f}')
