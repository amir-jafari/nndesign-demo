# Building the documentation

```bash
pip install -r docs/requirements.txt
sphinx-build -W --keep-going -b html docs docs/_build/html
open docs/_build/html/index.html
```

The site is deployed to https://amir-jafari.github.io/nndesign-demo/ by
`.github/workflows/docs.yml` on every push to `master`.

## Tutorials

The tutorial pages under `docs/tutorials/book*/` are generated from the menu
tables in `nndesigndemos/Window.py` and from the explanation text inside each
demo (`fill_chapter(...)` or the `PanelDemo.description` attribute).

| Command | What it does |
|---|---|
| `python docs/_scripts/gen_tutorials.py` | Refresh the book index pages and write any **missing** chapter pages. Existing chapter pages are left alone, so hand edits survive. |
| `python docs/_scripts/gen_tutorials.py --force` | Regenerate every chapter page from scratch (discards hand edits). |
| `python docs/_scripts/gen_tutorials.py --check` | Verify that every demo in `Window.py` has a section and a screenshot. CI runs this before the build. |
| `python docs/_scripts/capture_screenshots.py` | Capture every demo and the menus as WebP into `docs/_static/screenshots/` using Qt's offscreen platform. Needs PyQt6 and the package dependencies, about five minutes. Use `--book`, `--chapter`, `--match` or `--menus` to capture a subset. |

### Adding a demo

1. Register it in `Window.py` as usual.
2. Run `capture_screenshots.py --match "<demo name>"`.
3. Either add a section to the chapter page by hand (copy an existing one and
   keep the `.. _book1-chNN-<slug>:` anchor that `--check` looks for), or delete
   the chapter page and run `gen_tutorials.py` to regenerate it.
4. Run `gen_tutorials.py --check` and build.
