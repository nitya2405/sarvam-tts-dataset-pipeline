"""
Read-only browser for approved clips. Opens in your browser, grouped by
language and emotion. No save buttons — just listen and check tags.

Usage:
    python browse_approved.py
"""
import csv
import http.server
import json
import os
import threading
import webbrowser
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
CLIPS_DIR = ROOT / "data" / "clips"
PORT = 7861

with open(ROOT / "metadata" / "clips_metadata.csv", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

approved = [r for r in rows if r.get("approved", "").lower() == "true"]

groups = defaultdict(list)
for r in approved:
    groups[(r["language"], r["primary_emotion"])].append(r)

EMOTION_COLORS = {
    "happy_excited":       "#f59e0b",
    "sad":                 "#60a5fa",
    "angry":               "#ef4444",
    "calm_reverent":       "#34d399",
    "conversational_casual": "#a78bfa",
    "intense_dramatic":    "#f97316",
    "formal":              "#6b7280",
    "neutral":             "#94a3b8",
}

def build_html():
    sections = []
    for lang in ["en", "hi"]:
        lang_label = "English" if lang == "en" else "Hindi"
        lang_clips = [(emo, clips) for (l, emo), clips in sorted(groups.items()) if l == lang]
        total = sum(len(c) for _, c in lang_clips)
        lang_dur = sum(float(r["duration_s"]) for _, clips in lang_clips for r in clips)

        emo_sections = []
        for emo, clips in lang_clips:
            color = EMOTION_COLORS.get(emo, "#94a3b8")
            dur = sum(float(r["duration_s"]) for r in clips)
            cards = []
            for i, r in enumerate(clips, 1):
                fn = r["clip_filename"]
                sec_tag = f'<span class="sec-tag">{r["secondary_emotion"]}</span>' if r.get("secondary_emotion") else ""
                transcript = r.get("transcript", "").replace("<", "&lt;").replace(">", "&gt;")
                cards.append(f"""
                <div class="card">
                  <div class="card-header">
                    <span class="clip-num">#{i}</span>
                    <span class="clip-name">{fn}</span>
                    <span class="dur">{float(r['duration_s']):.1f}s</span>
                  </div>
                  <audio controls src="/clips/{fn}"></audio>
                  <div class="transcript">{transcript}</div>
                  {sec_tag}
                </div>""")

            emo_sections.append(f"""
            <div class="emo-section">
              <h3 style="border-left: 4px solid {color}; padding-left: 10px;">
                {emo.upper().replace("_", " ")}
                <span class="count">{len(clips)} clips &nbsp;·&nbsp; {dur/60:.2f} min</span>
              </h3>
              <div class="cards">{"".join(cards)}</div>
            </div>""")

        sections.append(f"""
        <div class="lang-section">
          <h2>{lang_label} &nbsp;<span class="lang-meta">{total} clips &nbsp;·&nbsp; {lang_dur/60:.1f} min</span></h2>
          {"".join(emo_sections)}
        </div>""")

    total_all = len(approved)
    total_dur = sum(float(r["duration_s"]) for r in approved) / 60

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Approved Clips Browser</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 20px 40px; }}
  h1 {{ color: #f8fafc; border-bottom: 2px solid #334155; padding-bottom: 10px; }}
  .meta {{ color: #94a3b8; font-size: 0.9em; margin-bottom: 30px; }}
  h2 {{ color: #f1f5f9; background: #1e293b; padding: 12px 16px; border-radius: 8px; margin-top: 40px; }}
  .lang-meta {{ font-size: 0.65em; color: #64748b; font-weight: 400; }}
  h3 {{ color: #cbd5e1; margin: 24px 0 12px; }}
  .count {{ font-size: 0.7em; color: #64748b; font-weight: 400; margin-left: 10px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 14px; }}
  .card {{ background: #1e293b; border-radius: 8px; padding: 14px; border: 1px solid #334155; }}
  .card-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }}
  .clip-num {{ background: #334155; color: #94a3b8; border-radius: 4px; padding: 2px 7px; font-size: 0.8em; }}
  .clip-name {{ font-size: 0.8em; color: #7dd3fc; flex: 1; word-break: break-all; }}
  .dur {{ font-size: 0.75em; color: #64748b; white-space: nowrap; }}
  audio {{ width: 100%; margin: 6px 0; accent-color: #7dd3fc; }}
  .transcript {{ font-size: 0.82em; color: #94a3b8; line-height: 1.5; margin-top: 6px; max-height: 80px; overflow-y: auto; }}
  .sec-tag {{ display: inline-block; margin-top: 6px; background: #334155; color: #94a3b8; border-radius: 4px; padding: 2px 8px; font-size: 0.75em; }}
</style>
</head>
<body>
<h1>Approved Clips Browser</h1>
<div class="meta">
  {total_all} clips &nbsp;·&nbsp; {total_dur:.1f} min total &nbsp;·&nbsp; read-only view
</div>
{"".join(sections)}
</body>
</html>"""


HTML = build_html()

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode("utf-8"))
        elif self.path.startswith("/clips/"):
            fn = self.path[7:]
            fp = CLIPS_DIR / fn
            if fp.exists() and fp.suffix == ".wav":
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(fp.stat().st_size))
                self.end_headers()
                with open(fp, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass  # silence request logs

print(f"Opening browser at http://localhost:{PORT}")
print("Press Ctrl+C to stop.")
threading.Timer(0.5, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
http.server.HTTPServer(("", PORT), Handler).serve_forever()
