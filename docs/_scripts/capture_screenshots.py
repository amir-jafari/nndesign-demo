#!/usr/bin/env python3
"""Capture a screenshot of every demo (and the menus) for the tutorials.

Runs the demos with Qt's offscreen platform, so nothing appears on screen and
no display is needed, but PyQt6 and the package dependencies must be
installed. Each capture runs in its own subprocess so a crashing demo cannot
take the others down.

Usage::

    python docs/_scripts/capture_screenshots.py              # everything
    python docs/_scripts/capture_screenshots.py --book 2     # one book
    python docs/_scripts/capture_screenshots.py --match hebb # demos whose name matches
    python docs/_scripts/capture_screenshots.py --menus      # main menu + both tables of contents

Images are written as WebP to ``docs/_static/screenshots/``; file names come
from :mod:`catalog` so the tutorial pages and the check script agree.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from catalog import ROOT, Demo, iter_demos, load_catalog  # noqa: E402

STATIC = os.path.abspath(os.path.join(HERE, "..", "_static"))
SHOTS = os.path.join(STATIC, "screenshots")

# Window-construction arguments: unscaled ratios and a 96 dpi screen, which is
# what the demos were laid out for. QT_FONT_DPI=72 reproduces macOS font
# metrics so labels do not overflow their boxes on the offscreen platform.
CTOR_ARGS = (1, 1, 96)
SETTLE_SECONDS = 1.5
WEBP_QUALITY = 85

CHILD_ENV = {
    "QT_QPA_PLATFORM": "offscreen",
    "QT_FONT_DPI": "72",
    "NNDESIGNDEMOS_PLAY_SOUND": "0",
    "MPLBACKEND": "QtAgg",
    "PYTHONPATH": ROOT,
}

MENUS = {
    "main_menu": ("nndesigndemos.nndesign", "MainWindow"),
    "toc_book1": ("nndesigndemos.Window", "MainWindowNN"),
    "toc_book2": ("nndesigndemos.Window", "MainWindowDL"),
}


# ---------------------------------------------------------------------------
# child process: build one window and grab it
# ---------------------------------------------------------------------------

def _capture_one(module: str, class_name: str, out_path: str) -> None:
    import importlib
    import warnings

    warnings.filterwarnings("ignore")
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv[:1])
    app.setStyle("Fusion")  # same style nndtoc() uses
    cls = getattr(importlib.import_module(module), class_name)
    win = cls(*CTOR_ARGS)
    win.show()
    deadline = time.time() + SETTLE_SECONDS
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)
    pixmap = win.grab()  # the demo widget itself, unscaled
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not pixmap.save(out_path, "WEBP", WEBP_QUALITY):
        raise RuntimeError(f"could not write {out_path}")
    print(f"{pixmap.width()}x{pixmap.height()}")
    win.close()
    app.quit()


# ---------------------------------------------------------------------------
# parent process
# ---------------------------------------------------------------------------

def run_child(module: str, class_name: str, out_path: str, timeout: int) -> tuple[bool, str]:
    cmd = [sys.executable, __file__, "--_child", module, class_name, out_path]
    env = dict(os.environ, **CHILD_ENV)
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    if proc.returncode != 0 or not os.path.exists(out_path):
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        return False, f"exit {proc.returncode}: " + " | ".join(tail)
    size = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    return True, size


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", type=int, choices=(1, 2))
    ap.add_argument("--chapter", type=int)
    ap.add_argument("--match", help="substring of the demo name (case-insensitive)")
    ap.add_argument("--menus", action="store_true", help="only the main menu and tables of contents")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--_child", nargs=3, metavar=("MODULE", "CLASS", "OUT"), help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._child:
        _capture_one(*args._child)
        return 0

    jobs: list[tuple[str, str, str, str]] = []  # label, module, class, out
    if args.menus or not (args.book or args.chapter or args.match):
        for name, (module, cls) in MENUS.items():
            jobs.append((name, module, cls, os.path.join(SHOTS, f"{name}.webp")))
    if not args.menus:
        for d in iter_demos(load_catalog()):
            if args.book and d.book != args.book:
                continue
            if args.chapter and d.chapter != args.chapter:
                continue
            if args.match and args.match.lower() not in d.name.lower():
                continue
            jobs.append((f"B{d.book} ch{d.chapter:02d} {d.name}", d.module, d.class_name,
                         os.path.join(STATIC, d.screenshot)))

    failures = []
    t0 = time.time()
    for i, (label, module, cls, out) in enumerate(jobs, 1):
        if args.skip_existing and os.path.exists(out):
            print(f"[{i:3}/{len(jobs)}] skip   {label}")
            continue
        ok, info = run_child(module, cls, out, args.timeout)
        status = "ok    " if ok else "FAILED"
        print(f"[{i:3}/{len(jobs)}] {status} {label}  {info}", flush=True)
        if not ok:
            failures.append((label, info))
    print(f"\n{len(jobs) - len(failures)}/{len(jobs)} captured in {time.time() - t0:.0f}s")
    for label, info in failures:
        print(f"  FAILED {label}: {info}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
