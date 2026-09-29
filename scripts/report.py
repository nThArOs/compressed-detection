"""Single-file HTML report from results/metrics_*.json, results/detect_*.json and preview videos."""
import html
import json
from collections import defaultdict

from common import ROOT, load_config

DATASET_NAMES = {"dut_anti_uav": "DUT Anti-UAV", "uavdt": "UAVDT", "visdrone_mot": "VisDrone-MOT"}
CHART_METRICS = ["HOTA", "MOTA", "IDF1"]
# name, key, higher is better
COLUMNS = [("HOTA", "HOTA", True), ("DetA", "DetA", True), ("AssA", "AssA", True),
           ("MOTA", "MOTA", True), ("IDF1", "IDF1", True), ("ID switches", "IDSW", False),
           ("False pos.", "FP", False), ("Missed", "FN", False)]
SLOTS = 3  # validated all-pairs categorical slots

CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
       font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 18px; margin: 0 0 2px; }
h3 { font-size: 14px; margin: 24px 0 8px; color: var(--ink-2); font-weight: 600; }
.sub { color: var(--ink-2); margin: 0 0 16px; }
section { background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
          padding: 20px; margin: 24px 0; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 6px 10px; text-align: right; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th { color: var(--muted); font-weight: 500; font-size: 12px; }
th:first-child, td:first-child { text-align: left; }
td.best { font-weight: 700; }
.key { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 8px; }
.legend { display: flex; flex-wrap: wrap; gap: 16px; color: var(--ink-2); font-size: 12px; margin: 8px 0; }
svg text { fill: var(--ink-2); font-size: 11px; }
svg .axis { stroke: var(--axis); } svg .grid { stroke: var(--grid); }
svg rect.bar:hover { opacity: 0.8; }
.videos { display: grid; gap: 12px; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); }
figure { margin: 0; } figcaption { color: var(--ink-2); font-size: 12px; margin-top: 4px; }
video { width: 100%; border-radius: 8px; background: #000; }
#tip { position: fixed; pointer-events: none; background: var(--surface); color: var(--ink);
       border: 1px solid var(--border); border-radius: 8px; padding: 6px 10px; font-size: 12px;
       box-shadow: 0 4px 16px rgba(0,0,0,0.15); display: none; }
footer { color: var(--muted); font-size: 12px; }
"""

JS = """
const tip = document.getElementById('tip');
document.querySelectorAll('rect.bar').forEach(r => {
  r.addEventListener('mousemove', e => {
    tip.textContent = r.dataset.tip; tip.style.display = 'block';
    tip.style.left = (e.clientX + 12) + 'px'; tip.style.top = (e.clientY + 12) + 'px';
  });
  r.addEventListener('mouseleave', () => tip.style.display = 'none');
});
"""


def esc(s):
    return html.escape(str(s))


def chart(rows, labels, colors):
    """Grouped horizontal bars: one group per metric, one bar per method, zero baseline."""
    vals = [r["mean"][m] for r in rows for m in CHART_METRICS]
    lo, hi = min(0, min(vals)), max(100, max(vals))
    left, right, bar, gap = 110, 60, 14, 2
    group_h = len(rows) * (bar + gap) + 18
    width, height = 720, len(CHART_METRICS) * group_h + 24

    def x(v):
        return left + (v - lo) / (hi - lo) * (width - left - right)

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
             f'aria-label="HOTA, MOTA and IDF1 per method">']
    for t in range(int(lo // 25 * 25), int(hi) + 1, 25):
        parts.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="0" y2="{height - 20}"/>'
                     f'<text x="{x(t):.1f}" y="{height - 6}" text-anchor="middle">{t}</text>')
    for g, m in enumerate(CHART_METRICS):
        y0 = g * group_h + 4
        parts.append(f'<text x="0" y="{y0 + len(rows) * (bar + gap) / 2 + 4:.1f}" '
                     f'style="font-weight:600">{m}</text>')
        for i, r in enumerate(rows):
            v = r["mean"][m]
            y = y0 + i * (bar + gap)
            x1, x2 = sorted((x(0), x(v)))
            tip = f'{labels[r["method"]]} · {m} {v}'
            parts.append(f'<rect class="bar" x="{x1:.1f}" y="{y}" width="{max(x2 - x1, 1):.1f}" '
                         f'height="{bar}" rx="4" fill="var({colors[r["method"]]})" '
                         f'data-tip="{esc(tip)}"/>')
            tx = x(v) + (6 if v >= 0 else -6)
            parts.append(f'<text x="{tx:.1f}" y="{y + bar - 3}" '
                         f'text-anchor="{"start" if v >= 0 else "end"}">{v}</text>')
    parts.append(f'<line class="axis" x1="{x(0):.1f}" x2="{x(0):.1f}" y1="0" y2="{height - 20}"/>')
    return "".join(parts) + "</svg>"


def metrics_table(rows, labels, colors):
    best = {}
    for _, key, higher in COLUMNS:
        vs = [r["mean"][key] for r in rows]
        best[key] = max(vs) if higher else min(vs)
    fps_best = max(r["fps"] for r in rows)
    head = "".join(f"<th>{esc(n)}</th>" for n, _, _ in COLUMNS)
    body = []
    for r in rows:
        cells = "".join(f'<td class="{"best" if len(rows) > 1 and r["mean"][k] == best[k] else ""}">'
                        f'{r["mean"][k]}</td>' for _, k, _ in COLUMNS)
        fps_cls = "best" if len(rows) > 1 and r["fps"] == fps_best else ""
        body.append(f'<tr><td><span class="key" style="background:var({colors[r["method"]]})"></span>'
                    f'{esc(labels[r["method"]])}</td>{cells}<td class="{fps_cls}">{r["fps"]}</td></tr>')
    return (f'<div class="table-wrap"><table><tr><th>Method</th>{head}<th>fps</th></tr>'
            f'{"".join(body)}</table></div>')


def class_table(rows, labels):
    classes = list(rows[0]["classes"])
    if len(classes) < 2:
        return ""
    head = "".join(f"<th>{esc(labels[r['method']])}</th>" for r in rows)
    body = "".join(f"<tr><td>{esc(c)}</td>" + "".join(
        f"<td>{r['classes'][c]['HOTA']}</td>" for r in rows) + "</tr>" for c in classes)
    return (f'<h3>HOTA per class</h3><div class="table-wrap"><table><tr><th>Class</th>{head}</tr>'
            f'{body}</table></div>')


def videos(rows, labels, dataset):
    seqs = rows[0]["sequences"] if isinstance(rows[0]["sequences"], list) else []
    items = []
    for seq in seqs:
        for r in rows:
            p = ROOT / "results" / "videos" / "previews" / r["method"] / dataset / f"{seq}.mp4"
            if p.exists():
                rel = p.relative_to(ROOT / "results").as_posix()
                items.append(f'<figure><video src="{rel}" controls muted loop preload="metadata"></video>'
                             f'<figcaption>{esc(seq)} · {esc(labels[r["method"]])}</figcaption></figure>')
    if not items:
        return ""
    return ('<h3>Videos: frame and residual, ground truth in green, tracks with their id</h3>'
            f'<div class="videos">{"".join(items)}</div>')


def detection_section():
    files = sorted((ROOT / "results").glob("detect_*.json"))
    if not files:
        return ""
    rows = [json.loads(f.read_text()) for f in files]
    body = "".join(
        f"<tr><td>{esc(DATASET_NAMES.get(r['dataset'], r['dataset']))}</td><td>{esc(r['modality'])}</td>"
        f"<td>{r['test']['precision']}</td><td>{r['test']['recall']}</td><td>{r['test']['map50']}</td>"
        f"<td>{r['test']['map50_95']}</td><td>{r['train_hours']}</td></tr>" for r in rows)
    return ('<section><h2>Detection, per frame</h2><p class="sub">Test split of each YOLO dataset, '
            'same frames for both modalities.</p><div class="table-wrap"><table><tr><th>Dataset</th>'
            '<th>Input</th><th>Precision</th><th>Recall</th><th>mAP50</th><th>mAP50-95</th>'
            f'<th>Training (h)</th></tr>{body}</table></div></section>')


def main():
    labels = load_config(ROOT / "configs" / "track.yaml").get("methods", {})
    order = list(labels)
    groups = defaultdict(list)
    for f in sorted((ROOT / "results").glob("metrics_*.json")):
        r = json.loads(f.read_text())
        heldout = f.stem.endswith("_heldout")
        labels.setdefault(r["method"], r["method"])
        if r["method"] not in order:
            order.append(r["method"])
        groups[(r["dataset"], r["split"], heldout)].append(r)

    # color follows the method, not its position in a given table
    colors = {m: f"--s{i % SLOTS + 1}" for i, m in enumerate(order)}
    hardware = None
    sections = []
    for (dataset, split, heldout), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], not kv[0][2])):
        rows.sort(key=lambda r: order.index(r["method"]))
        hardware = rows[0]["hardware"]
        n = len(rows[0]["sequences"]) if isinstance(rows[0]["sequences"], list) else rows[0]["sequences"]
        scope = f"{n} held-out test sequences" if heldout else f"{split} split, {n} sequences"
        legend = "".join(f'<span><span class="key" style="background:var({colors[r["method"]]})">'
                         f'</span>{esc(labels[r["method"]])}</span>' for r in rows)
        sections.append(
            f'<section><h2>{esc(DATASET_NAMES.get(dataset, dataset))}</h2>'
            f'<p class="sub">{esc(scope)}, {rows[0]["frames"]} frames. Higher is better except '
            f'ID switches, false positives and missed.</p>'
            f'<div class="legend">{legend}</div>{chart(rows, labels, colors)}'
            f'{metrics_table(rows, labels, colors)}{class_table(rows, labels)}'
            f'{videos(rows, labels, dataset)}</section>')

    hw = f'{hardware["cpu"]}, {hardware["cpu_count"]} threads, {hardware["platform"]}' if hardware else ""
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>Tracking results</title><style>{CSS}</style></head><body><main>'
            f'<h1>Tracking results</h1><p class="sub">Detector + ByteTrack, evaluated with TrackEval '
            f'(HOTA, CLEAR, Identity). CPU only.</p>{detection_section()}{"".join(sections)}'
            f'<footer>{esc(hw)}</footer></main><div id="tip"></div><script>{JS}</script>'
            f'</body></html>')
    out = ROOT / "results" / "report.html"
    out.write_text(page, encoding="utf-8")
    print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
