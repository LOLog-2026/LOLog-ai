"""python summarize.py <tag접두어> <카드...> : rounds/<접두어>_<카드>/metrics.json을 모아 표와 평균을 낸다."""
import json, sys
import numpy as np
pre, cards = sys.argv[1], sys.argv[2:]
keys = ["lip_coverage", "flux_face_rate", "sync_corr", "closed_in_silence", "outside_motion", "blinks"]
rows = []
for c in cards:
    m = json.load(open(f"rounds/{pre}_{c}/metrics.json"))
    faces = sum(len(d) for d in m["sources"]); eyes = sum(len(e) for e in m["eye_faces"])
    mixed = sum(1 for d in m["sources"] for v in d.values() if "drawn" in v)
    rows.append([c] + [m[k] for k in keys] + [eyes, faces, mixed, m["hidden_mouth_lines"], round(m["cpu_s"]["mouth_paste"] + m["cpu_s"]["render"], 1)])
    print(c, dict(zip(keys + ["eye_faces", "mouth_faces", "drawn_faces", "hidden", "cpu_s"], rows[-1][1:])))
a = np.array([[float(x) for x in r[1:]] for r in rows])
print("MEAN", dict(zip(keys + ["eye_faces", "mouth_faces", "drawn_faces", "hidden", "cpu_s"], [round(float(v), 3) for v in a.mean(axis=0)])))
