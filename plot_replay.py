"""English plots and an offline transition gallery for compare_replay.py."""
import csv
import math
from html import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np


LABELS = {
    "blocks": ("Paving blocks", "#007C78"), "cement": ("Cement", "#E5AB57"),
    "asphalt": ("Asphalt", "#5268AB"), "urethane": ("Urethane", "#B75C9B"),
    "soil_stone": ("Soil / stone", "#9B5239"), "other": ("Other", "#8B8842"),
    "unknown": ("Unknown", "#A9B5BE"), "skipped": ("Skipped", "#EAECEF"),
}


def legend(fig, used):
    handles = [Patch(facecolor=color, label=text) for key, (text, color) in LABELS.items() if key in used]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.03, .904), ncol=8, frameon=False)


def heading(fig, title, subtitle):
    fig.text(.035, .972, title, fontsize=22, weight="bold", va="top", color="#213644")
    fig.text(.035, .925, subtitle, fontsize=11, va="top", color="#47606C")


def timeline(report, groups, dest):
    fig, axes = plt.subplots(len(groups), 1, figsize=(15, 5.6), squeeze=False)
    fig.subplots_adjust(left=.18, right=.84, top=.67, bottom=.24, hspace=.68)
    heading(fig, "Surface predictions over time",
            f"{report['recording']}   /   {report['duration_s'] / 60:.2f} minutes   /   {report['candidate_frames']:,} candidate frames")
    used = set()
    for ax, (interval, runs) in zip(axes.flat, groups.items()):
        ax.set_title(f"{interval}-second windows", loc="left", fontsize=12, weight="bold")
        ax.text(1.012, 1.05, "Changes / predictions", transform=ax.transAxes, fontsize=10)
        for y, run in zip(range(3, -1, -1), runs):
            segments, colors = [], []
            for w in run["windows"]:
                label = w["prediction"]["surface_label"] if w["prediction"] else "skipped"
                segments.append([(w["start_s"]/60, y), (w["end_s"]/60, y)])
                colors.append(LABELS[label][1])
                used.add(label)
            ax.add_collection(LineCollection(segments, colors=colors, linewidths=13, capstyle="butt"))
            stats = run["summary"]
            ax.text(1.012, y, f"{stats['changes']} / {stats['predictions']}", va="center",
                    transform=ax.get_yaxis_transform(), fontsize=10)
        ax.set(xlim=(0, max(report["duration_s"]/60, .01)), ylim=(-.6,3.6), yticks=[3,2,1,0],
               yticklabels=[f"{r['method']}   {r['model']} · {r['selection']}" for r in runs])
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=8)
    legend(fig, used)
    axes[-1,0].set_xlabel("Time since recording started (min)")
    fig.text(.035, .035, "Timed uses the last frame in the window. Quality selects by exposure, contrast, road sharpness and gyro stability.\n"
             "Each color represents the predicted label for one window. A change is counted when successive predictions differ.", fontsize=10)
    fig.savefig(dest/"timeline.png", dpi=160)
    plt.close(fig)


def route_map(report, groups, recording, dest):
    with (recording/"gps.csv").open(newline="") as f:
        fixes = list(csv.DictReader(f))
    fig, axes = plt.subplots(len(groups), 4, figsize=(15, 6.3), sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=.075, right=.98, top=.70, bottom=.14, hspace=.38, wspace=.20)
    heading(fig, "Surface predictions along the GPS track", report["recording"] + "   /   Each dot is a prediction matched to a GPS fix.")
    origin = fixes[0] if fixes else {"latitude":0, "longitude":0}
    lat0, lon0 = float(origin["latitude"]), float(origin["longitude"])

    def xy(fix):
        return ((float(fix["longitude"])-lon0)*111320*math.cos(math.radians(lat0)),
                (float(fix["latitude"])-lat0)*111320)

    track = np.array([xy(fix) for fix in fixes])
    used = set()
    for row, (interval, runs) in enumerate(groups.items()):
        for col, run in enumerate(runs):
            ax = axes[row, col]
            if len(track):
                ax.plot(*track.T, color="#CBD3D9", lw=.9, zorder=1)
            matches = [w for w in run["windows"] if w["prediction"] and w["frame"]["gps"]]
            if matches:
                labels = [w["prediction"]["surface_label"] for w in matches]
                used.update(labels)
                pts = np.array([xy(w["frame"]["gps"]) for w in matches])
                ax.scatter(*pts.T, c=[LABELS[k][1] for k in labels], s=10, lw=0, zorder=2)
            ax.set_title(f"{run['method']}   {run['model']} · {run['selection']}\n"
                         f"{interval} s   ·   {len(matches)} / {len(run['windows'])} windows mapped", fontsize=10)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=.15)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            if col == 0:
                ax.set_ylabel("North (m)")
            if row == len(groups)-1:
                ax.set_xlabel("East (m)")
    legend(fig, used)
    fig.text(.035, .025, "Origin: first GPS fix. A match requires a time difference of at most 1 s and reported accuracy of 10 m or better.", fontsize=10)
    fig.savefig(dest/"route.png", dpi=150)
    plt.close(fig)


def gallery(report, dest):
    sections = []
    total = 0
    for run in report["runs"]:
        cards = []
        for w in run["windows"]:
            change = w["change"]
            if not change:
                continue
            src = escape(change["image"], quote=True)
            before, after = (LABELS[change[key]][0] for key in ("previous_label", "new_label"))
            caption = f"{before} → {after}"
            cards.append(f'<figure><a href="{src}"><img loading="lazy" src="{src}" alt="{caption}"></a>'
                         f'<figcaption><strong>{caption}</strong><br>{w["frame"]["t_s"]:.3f} s · '
                         f'Window {w["start_s"]}–{w["end_s"]} s</figcaption></figure>')
        total += len(cards)
        title = f'{run["interval_s"]} s / {run["method"]} · {run["model"]} · {run["selection"]}'
        sections.append(f'<details><summary>{title} — {len(cards)} changes</summary><div class="grid">'
                        + ("".join(cards) or "<p>No label changes.</p>") + "</div></details>")
    title = escape(report["recording"])
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Label changes by method</title><style>
body{font:16px/1.5 system-ui,sans-serif;color:#213644;background:#fcfcfa;max-width:1240px;margin:auto;padding:24px}
h1{margin-bottom:0}a{color:#006960}nav{display:flex;gap:1.4em;flex-wrap:wrap;margin:1em 0}
details{border-top:1px solid #c8d3d8}summary{cursor:pointer;font-weight:600;padding:.8em 0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(235px,1fr));gap:18px}
figure{margin:0 0 12px}img{width:100%;aspect-ratio:16/9;object-fit:contain;background:#eee}
</style><main><h1>Label changes by method</h1>"""
    page += f"<p>{title} · 2-second windows · {total:,} changes across four methods.</p>"
    page += ('<nav><a href="timeline.png">Prediction timeline</a><a href="route.png">GPS track</a>'
             '<a href="results.json">Detection results (JSON)</a></nav>'
             '<p>Open a group to see its changes in time order. Each photo is the first frame with the new predicted label. '
             'Select a photo to open the original JPEG.</p>')
    (dest/"index.html").write_text(page + "".join(sections) + "</main></html>", encoding="utf-8")


def render(report, recording, dest):
    groups = {}
    for run in report["runs"]:
        groups.setdefault(run["interval_s"], []).append(run)
    timeline(report, groups, dest)
    route_map(report, groups, recording, dest)
    gallery(report, dest)
