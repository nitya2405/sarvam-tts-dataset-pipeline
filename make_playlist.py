import csv
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent
CLIPS_DIR = ROOT / "data" / "clips"

with open(ROOT / "metadata" / "clips_metadata.csv", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

approved = [r for r in rows if r.get("approved", "").lower() == "true"]

groups = defaultdict(list)
for r in approved:
    groups[(r["language"], r["primary_emotion"])].append(r)

lines = ["#EXTM3U", ""]
for lang in ["en", "hi"]:
    lang_label = "English" if lang == "en" else "Hindi"
    for (l, emo), clips in sorted(groups.items()):
        if l != lang:
            continue
        lines.append(f"# --- {lang_label} / {emo.upper()} ({len(clips)} clips) ---")
        for c in clips:
            path = CLIPS_DIR / c["clip_filename"]
            dur = int(float(c.get("duration_s", 0)))
            title = f"{lang_label} | {emo} | {c['clip_filename']}"
            lines.append(f"#EXTINF:{dur},{title}")
            lines.append(str(path.resolve()))
        lines.append("")

out = ROOT / "approved_clips.m3u"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"Playlist written: {out}")
print(f"Total clips: {len(approved)}")
