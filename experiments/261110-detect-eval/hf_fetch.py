"""jetson_models.tsv 한 줄의 GGUF 파일을 찾아 받는다. 표준 라이브러리만 쓴다.

    python3 -I hf_fetch.py <출처> <파일 고르기> <받을 폴더>
    → 성공하면 받은 파일 경로 한 줄만 stdout에 쓴다(진행 상황은 stderr)

출처: HF 레포(org/name) / search:<검색어> / file:<경로>. 레포에 맞는 파일이 없으면 레포 이름으로 검색한다.
HF_TOKEN 이 있으면 붙인다(잠긴 레포).
"""
from __future__ import annotations

import fnmatch
import json
import os
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request

HF = "https://huggingface.co"


def log(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


def get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "lolog-eval"})
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    return urllib.request.urlopen(req, timeout=60)


def files(repo: str) -> list[tuple[str, int]]:
    try:
        with get(f"{HF}/api/models/{repo}/tree/main?recursive=1") as r:
            return [(f["path"], f.get("size") or 0) for f in json.load(r) if f.get("type") == "file"]
    except urllib.error.HTTPError as e:
        log(f"  {repo}: {e.code}")
        return []


def pick(fl: list[tuple[str, int]], globs: list[str]) -> tuple[str, int] | None:
    fl = [f for f in fl if f[0].lower().endswith(".gguf") and "-of-" not in f[0] and "mmproj" not in f[0].lower()]
    for g in globs:
        for path, size in fl:
            if fnmatch.fnmatch(os.path.basename(path).lower(), g.lower()):
                return path, size
    return None


def search(q: str, globs: list[str]) -> tuple[str, str, int] | None:
    url = f"{HF}/api/models?" + urllib.parse.urlencode(
        {"search": q.replace(" gguf", "").strip(), "filter": "gguf", "sort": "downloads", "direction": -1, "limit": 15})
    with get(url) as r:
        cands = [m["id"] for m in json.load(r)]
    log(f"  검색 '{q}': {', '.join(cands[:6]) or '없음'}")
    for repo in cands:
        hit = pick(files(repo), globs)
        if hit:
            return repo, *hit
    return None


def main() -> None:
    src, globs, dest = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
    if src.startswith("file:"):
        p = src[5:]
        if not os.path.isfile(p):
            sys.exit(f"없는 파일: {p}")
        print(p)
        return
    hit = None
    if not src.startswith("search:"):
        h = pick(files(src), globs)
        hit = (src, *h) if h else None
        if not hit:
            log(f"  {src}에 {globs} 파일이 없어 검색함")
    if not hit:
        hit = search(src.removeprefix("search:") if src.startswith("search:") else src.split("/")[-1], globs)
    if not hit:
        sys.exit(f"못 찾음: {src} {globs}")
    repo, path, size = hit
    os.makedirs(dest, exist_ok=True)
    out = os.path.join(dest, repo.replace("/", "__") + "__" + os.path.basename(path))
    if os.path.isfile(out) and (not size or os.path.getsize(out) == size):
        log(f"  이미 있음: {out}")
        print(out)
        return
    free = shutil.disk_usage(dest).free
    if size and free < size + 2 * 1024**3:
        sys.exit(f"디스크 부족: 필요 {size / 1e9:.1f}GB + 여유 2GB, 남은 {free / 1e9:.1f}GB")
    log(f"  받는 중: {repo}/{path} ({size / 1e9:.2f}GB)")
    tmp = out + ".part"
    with get(f"{HF}/{repo}/resolve/main/{urllib.parse.quote(path)}") as r, open(tmp, "wb") as fp:
        shutil.copyfileobj(r, fp, 8 * 1024 * 1024)
    os.replace(tmp, out)
    print(out)


if __name__ == "__main__":
    main()
