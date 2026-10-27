"""Readable detection overlays and precision–recall plots."""

import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont

from boxsmith.data import CLASS_NAMES

CLASS_COLOURS = ("#fb7185", "#38bdf8", "#fbbf24")


def overlay(image, target=None, prediction=None, confidence=0.0, limit=100):
    canvas = Image.fromarray(image).resize((512, 512), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(canvas)
    factor = 512 / image.shape[0]
    font = ImageFont.load_default(size=13)
    if target is not None:
        for box in target["boxes"]:
            x, y, x2, y2 = box * factor
            draw.rectangle((x, y, x2 - 1, y2 - 1), outline="#34d399", width=2)
    if prediction is not None:
        shown = 0
        order = np.argsort(-prediction["scores"], kind="stable")
        for index in order:
            box, score, label = (prediction[key][index] for key in ("boxes", "scores", "labels"))
            if score < confidence:
                continue
            if shown >= limit:
                break
            shown += 1
            x, y, x2, y2 = box * factor
            colour = CLASS_COLOURS[int(label)]
            draw.rectangle((x, y, max(x, x2 - 1), max(y, y2 - 1)), outline=colour, width=3)
            text = f"{CLASS_NAMES[int(label)]} {score:.2f}"
            bounds = draw.textbbox((x, max(0, y - 17)), text, font=font)
            draw.rectangle(bounds, fill="#111827")
            draw.text((x, max(0, y - 17)), text, fill=colour, font=font)
    return canvas


def pr_figure(curves, confidence):
    figure = go.Figure()
    for name, colour, curve in zip(CLASS_NAMES, CLASS_COLOURS, curves):
        if curve["ap"] is None:
            continue
        figure.add_trace(go.Scatter(x=curve["recall"], y=curve["precision"], mode="lines",
                                   line=dict(color=colour, width=2), name=f"{name} · AP {curve['ap']:.3f}"))
        count = int((curve["scores"] >= confidence).sum())
        recall = float(curve["recall"][count - 1]) if count else 0
        precision = float(curve["precision"][count - 1]) if count else 0
        figure.add_trace(go.Scatter(x=[recall], y=[precision], mode="markers",
                                   marker=dict(color=colour, size=11, line=dict(color="white", width=1)),
                                   name=f"{name} at confidence {confidence:.2f}", showlegend=False))
    figure.update_layout(xaxis=dict(title="Recall", range=[-0.02, 1.02]),
                         yaxis=dict(title="Precision", range=[-0.02, 1.02]),
                         height=390, margin=dict(l=20, r=20, t=20, b=20),
                         legend=dict(orientation="h", y=-0.23), template="plotly_dark",
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return figure
