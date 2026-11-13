"""음절을 소리에 놓는 방법들. 모두 (text, wav, sr, fps) → 음절별 시작 시각(초) 목록."""
import re
import numpy as np
from lolog_ai import lipsync as L


def _env(wav, sr, fps):
    n = int(np.ceil(len(wav) / sr * fps))
    env = L.envelope(wav, sr, fps, n)
    return n, env, env > 0.12


def uniform(text, wav, sr, fps):
    """지금 방식(shapes2): 쉼표·마침표 덩이를 소리의 쉼에 맞추고 덩이 안에서는 고르게(끝 음절만 1.5배)."""
    n, env, voiced = _env(wav, sr, fps)
    chunks = [c for c in (L.syllables(p) for p in re.split(r"[,.!?…~]+", text)) if c]
    runs = L._runs(voiced, max(2, int(fps * 0.15)))
    if len(runs) != len(chunks):
        chunks = [[s for c in chunks for s in c]]
        runs = [(runs[0][0], runs[-1][1])] if runs else [(0, n)]
    out = []
    for (a, b), syl in zip(runs, chunks):
        w = np.ones(len(syl)); w[-1] = 1.5
        edge = a + np.concatenate([[0], np.cumsum(w) / w.sum()]) * (b - a)
        out += list(edge[:-1] / fps)
    return out


ALIGN = {"uniform": uniform}


def _fine(wav, sr, afps=100):
    n = int(np.ceil(len(wav) / sr * afps))
    env = L.envelope(wav, sr, afps, n)
    k = 5
    sm = np.convolve(np.pad(env, k // 2, mode="edge"), np.ones(k) / k, mode="valid")[:n]
    return n, np.clip(sm, 0, 1.5)


def _words(text):
    """[(낱말 음절 수)], 그리고 쉼표·마침표 뒤에서 시작하는 낱말 표시."""
    out = []
    for part in re.split(r"(?<=[,.!?…~])", text):
        ws = [len(L.syllables(w)) for w in part.split() if L.syllables(w)]
        for j, nw in enumerate(ws):
            out.append((nw, j == 0))
    return out


def dp(text, wav, sr, fps, lam=0.6, mu=0.5, shift=0.0, afps=100, silence=0.12):
    """음절마다 '가운데가 크고 양 끝이 작은' 구간을 찾는다(동적 계획법). 낱말이 시작하는 경계는 소리가 작은 곳을 더 좋아한다.
    소리가 없는 구간(쉼)은 낱말 사이에만 들어갈 수 있다."""
    n, env = _fine(wav, sr, afps)
    words = _words(text)
    N = sum(w for w, _ in words)
    if N == 0:
        return []
    word_start = np.zeros(N, bool); k = 0
    for w, _ in words:
        word_start[k] = True; k += w
    voiced = env > silence
    idx = np.flatnonzero(voiced)
    if len(idx) == 0:
        return [0.0] * N
    a, b = int(idx[0]), int(idx[-1]) + 1
    T = b - a
    e = env[a:b]
    dbar = T / N
    dmin, dmax = max(2, int(dbar * 0.35)), max(4, int(dbar * 2.5) + 1)
    quiet = np.concatenate([[0], np.cumsum(~voiced[a:b])])    # 구간 안의 조용한 프레임 수(쉼)
    NEG = -1e9
    best = np.full((N + 1, T + 1), NEG); back = np.zeros((N + 1, T + 1), int)
    best[0, 0] = 0
    # 음절 j가 [s, t)를 차지. 쉼은 음절 길이에서 빼고 본다(낱말 끝 음절이 쉼을 품을 수 있다)
    cmax = np.maximum.accumulate  # noqa
    for j in range(1, N + 1):
        for t in range(j * dmin, T + 1):
            lo = max((j - 1) * dmin, t - dmax - int(quiet[t] - quiet[max(0, t - dmax)]) - 30)
            for s in range(max(0, lo), t - dmin + 1):
                if best[j - 1, s] <= NEG / 2:
                    continue
                dur = (t - s) - (quiet[t] - quiet[s])          # 소리 나는 길이
                if dur < dmin or dur > dmax:
                    continue
                seg = e[s:t]
                gain = seg.max() - 0.5 * (seg[0] + seg[-1])
                pen = lam * ((dur - dbar) / dbar) ** 2
                edge = mu * e[s] if word_start[j - 1] else 0.15 * mu * e[s]   # 낱말은 조용한 데서 시작
                if (quiet[t] - quiet[s]) > 3 and not (j < N and word_start[j]) and j != N:
                    pen += 2.0                                    # 낱말 안에는 쉼이 없다
                v = best[j - 1, s] + gain - pen - edge
                if v > best[j, t]:
                    best[j, t] = v; back[j, t] = s
    if best[N, T] <= NEG / 2:
        return uniform(text, wav, sr, fps)
    starts, t = [], T
    for j in range(N, 0, -1):
        s = back[j, t]; starts.append(s); t = s
    return [max(0.0, (a + s) / afps + shift) for s in starts[::-1]]


def uniform_shift(text, wav, sr, fps):
    return [max(0.0, t - 0.065) for t in uniform(text, wav, sr, fps)]


ALIGN.update({"dp": dp, "uniform_shift": uniform_shift})
import functools
for _lam in (0.3, 0.6, 1.5, 4.0):
    for _mu in (0.0, 0.5, 1.5):
        ALIGN[f"dp_l{_lam}_m{_mu}"] = functools.partial(dp, lam=_lam, mu=_mu, shift=-0.10)


def runs_snap(text, wav, sr, fps, shift=0.0, final=1.5, gap=0.15):
    """소리의 쉼(gap초 이상)마다 낱말 경계가 오게 한다. 쉼표·마침표 수와 쉼 수가 달라도 된다.
    쉼 앞까지의 소리 길이 비율만큼 음절을 세고, 가장 가까운 낱말 경계로 맞춘다(쉼표·마침표 경계를 조금 더 좋아한다).
    쉼 사이 구간 안에서는 고르게, 구간의 끝 음절만 길게."""
    n, env, voiced = _env(wav, sr, fps)
    words = _words(text)
    N = sum(w for w, _ in words)
    if N == 0:
        return []
    runs = L._runs(voiced, max(2, int(fps * gap)))
    if not runs:
        return [0.0] * N
    bounds = np.cumsum([w for w, _ in words])[:-1]                  # 낱말 경계(음절 수 기준)
    punct = {int(b) for b, (_, p) in zip(np.cumsum([w for w, _ in words])[:-1], words[1:]) if p}
    lens = np.array([b - a for a, b in runs], float)
    cuts, prev = [], 0
    for r in range(len(runs) - 1):
        target = N * lens[:r + 1].sum() / lens.sum()
        cand = [int(b) for b in bounds if b > prev and b <= N - (len(runs) - 1 - r)]
        if not cand:
            cuts.append(None); continue
        c = min(cand, key=lambda b: abs(b - target) - (0.75 if b in punct else 0))
        cuts.append(c); prev = c
    out, s0 = [], 0
    edges = [c for c in cuts] + [N]
    for (a, b), c in zip(runs, edges):
        if c is None or c <= s0:        # 이 구간에 놓을 낱말이 없다(경계가 모자람): 다음 구간과 합친다
            continue
        k = c - s0
        w = np.ones(k); w[-1] = final
        edge = a + np.concatenate([[0], np.cumsum(w) / w.sum()]) * (b - a)
        out += list(edge[:-1] / fps)
        s0 = c
    if len(out) != N:
        return uniform(text, wav, sr, fps)
    return [max(0.0, t + shift) for t in out]


ALIGN.update({"runs": runs_snap, "runs_s40": functools.partial(runs_snap, shift=-0.04), "runs_s70": functools.partial(runs_snap, shift=-0.07),
              "runs_g10": functools.partial(runs_snap, gap=0.10, shift=-0.04), "runs_g20": functools.partial(runs_snap, gap=0.20, shift=-0.04),
              "runs_f1": functools.partial(runs_snap, final=1.0, shift=-0.04), "runs_f2": functools.partial(runs_snap, final=2.0, shift=-0.04)})


def prod(text, wav, sr, fps):
    """워커 코드(lipsync.syllable_spans)가 놓은 음절 시작 시각."""
    sp = L.syllable_spans(text, wav, sr, fps)
    return [a / fps for a, _ in sp] if sp else uniform(text, wav, sr, fps)


ALIGN["prod"] = prod
