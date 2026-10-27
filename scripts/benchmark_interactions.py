"""Measure actual Streamlit reruns on 500 validation scenes (checkpoint required)."""

import json
from pathlib import Path
import time

from streamlit.testing.v1 import AppTest

app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=60).run()
app.select_slider[0].set_value(500).run()
if app.exception:
    raise RuntimeError([e.message for e in app.exception])
results = {"scene_count": 500}
for name, value in (("confidence", 0.75), ("matching_iou", 0.75), ("nms_iou", 0.6)):
    started = time.perf_counter()
    app.slider(key=name).set_value(value).run()
    results[f"{name}_rerun_seconds"] = time.perf_counter() - started
    if app.exception:
        raise RuntimeError([e.message for e in app.exception])
print(json.dumps(results, indent=2))
