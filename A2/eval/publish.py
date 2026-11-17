#!/usr/bin/env python
"""Prepare the Task 1 results for the Hugging Face Hub (staging only; uploading is a separate step).

A copy of results/task1 with every rollout video re-encoded to H.264, so the videos play in the browser
(the recorder's MPEG-4 Part 2 does not). Output: A2/data/hf/<repo name>/ (git-ignored).
Upload with `hf upload <repo> <folder> --repo-type dataset`.
"""

import argparse
import shutil
from pathlib import Path

import av

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "task1"
STAGE = HERE.parent / "data" / "hf"
HF_USER = "faizalkarim"
RESULTS_REPO = f"{HF_USER}/hrl_A2_task1_results"


def transcode(src: Path, dst: Path) -> None:
    with av.open(str(src)) as inp, av.open(str(dst), "w") as outp:
        s_in = inp.streams.video[0]
        s = outp.add_stream("libx264", rate=30, options={"crf": "23", "preset": "fast"})
        s.width, s.height, s.pix_fmt = s_in.width, s_in.height, "yuv420p"
        for f in inp.decode(s_in):
            for p in s.encode(f.reformat(format="yuv420p")):
                outp.mux(p)
        for p in s.encode():
            outp.mux(p)


def build_results(stage: Path = STAGE) -> None:
    out = stage / RESULTS_REPO.split("/")[1]
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(RESULTS, out, ignore=shutil.ignore_patterns("*.mp4", ".DS_Store"))
    vids = sorted(RESULTS.glob("rollouts/*/*.mp4"))
    for k, v in enumerate(vids, 1):
        transcode(v, out / v.relative_to(RESULTS))
        if k % 20 == 0 or k == len(vids):
            print(f"[{k}/{len(vids)}] videos re-encoded", flush=True)
    print(f"Done: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", type=Path, default=STAGE, help="Output folder (default A2/data/hf)")
    build_results(p.parse_args().stage)
