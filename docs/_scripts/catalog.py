"""Static catalog of every demo reachable from the nndesigndemos main menu.

Everything here is derived by parsing source files with :mod:`ast`, never by
importing the package, so it works on machines without PyQt6 (for example the
docs build on CI).

The menu structure comes from ``nndesigndemos/Window.py``:

* ``BOOK1_CHAPTERS_DEMOS`` / ``BOOK2_CHAPTERS_DEMOS`` give chapter titles and
  the demo names shown in each chapter's drop-down, in menu order.
* ``MainWindowNN.chapterN`` / ``MainWindowDL.chapterN`` map a drop-down index
  to the class that is instantiated.
* The module-level imports map that class name to its module.

Each demo's own description comes from the demo source: either the
``fill_chapter(title, number, description, ...)`` call or, for
``PanelDemo`` subclasses, the ``title`` / ``description`` class attributes.
Base classes are followed when a subclass does not define these itself.
"""

from __future__ import annotations

import ast
import os
import re
import warnings
from dataclasses import dataclass, field

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PACKAGE = os.path.join(ROOT, "nndesigndemos")
WINDOW_PY = os.path.join(PACKAGE, "Window.py")

BOOKS = {
    1: {
        "slug": "book1",
        "title": "Neural Network Design",
        "short": "NND",
        "menu_button": "Neural Network Design",
        "window_class": "MainWindowNN",
        "dict_name": "BOOK1_CHAPTERS_DEMOS",
        "logo_dir": "Logo/book_logos",
        "cover": "Logo/Figure.jpg",
        "url": "https://hagan.okstate.edu/nnd.html",
        "blocks": ["2-5", "6-9", "10-13", "14-17", "18-21"],
    },
    2: {
        "slug": "book2",
        "title": "Neural Network Design: Deep Learning",
        "short": "NND:DL",
        "menu_button": "Neural Network Design: Deep Learning",
        "window_class": "MainWindowDL",
        "dict_name": "BOOK2_CHAPTERS_DEMOS",
        "logo_dir": "Logo/book2_logos",
        "cover": "Logo/Figure_DL.jpg",
        "url": "https://github.com/NNDesignDeepLearning",
        "blocks": ["2-5", "6-9", "10-13"],
    },
}

# NNDLayout helper -> (argument index of the user-visible label, kind)
_CONTROL_HELPERS = {
    "make_slider": (9, "slider"),
    "make_button": (1, "button"),
    "make_combobox": (5, "menu"),
    "make_checkbox": (1, "checkbox"),
    "make_input_box": (1, "input box"),
}


@dataclass
class Demo:
    book: int
    chapter: int
    index: int            # 1-based position in the chapter drop-down
    name: str             # label shown in the drop-down
    class_name: str       # real class name in its module
    module: str           # dotted module path
    title: str = ""       # window title from fill_chapter / PanelDemo.title
    description: str = ""
    controls: list[str] = field(default_factory=list)
    meta_source: str = "" # where title/description were found (for diagnostics)

    @property
    def slug(self) -> str:
        return slugify(self.name)

    @property
    def file(self) -> str:
        return self.module.replace(".", "/") + ".py"

    @property
    def screenshot(self) -> str:
        """Path relative to docs/_static."""
        return f"screenshots/book{self.book}/ch{self.chapter:02d}_{self.slug}.webp"

    @property
    def menu_block(self) -> str:
        for block in BOOKS[self.book]["blocks"]:
            lo, hi = (int(x) for x in block.split("-"))
            if lo <= self.chapter <= hi:
                return block
        return "?"


@dataclass
class Chapter:
    book: int
    number: int
    title: str
    demos: list[Demo] = field(default_factory=list)

    @property
    def slug(self) -> str:
        return f"chapter{self.number:02d}"


def slugify(text: str) -> str:
    text = text.lower().replace("#", "")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


# ---------------------------------------------------------------------------
# Window.py
# ---------------------------------------------------------------------------

def _parse(path: str) -> ast.Module:
    with open(path, encoding="utf-8") as f:
        source = f.read()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)  # a few demos have "\s" in plain strings
        return ast.parse(source, filename=path)


def _import_map(tree: ast.Module) -> dict[str, tuple[str, str]]:
    """alias used in the file -> (module, real class name)"""
    out = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                out[alias.asname or alias.name] = (node.module, alias.name)
    return out


def _index_to_class(class_node: ast.ClassDef) -> dict[int, dict[int, str]]:
    """{chapter: {index: class alias}} from the chapterN methods."""
    out: dict[int, dict[int, str]] = {}
    for fn in class_node.body:
        m = re.fullmatch(r"chapter(\d+)", getattr(fn, "name", ""))
        if not (isinstance(fn, ast.FunctionDef) and m):
            continue
        chapter = int(m.group(1))
        mapping: dict[int, str] = {}
        for node in ast.walk(fn):
            if not isinstance(node, ast.If):
                continue
            test = node.test
            if not (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
                    and test.left.id == "idx" and isinstance(test.comparators[0], ast.Constant)):
                continue
            idx = test.comparators[0].value
            for sub in node.body:
                for call in ast.walk(sub):
                    if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                            and any(isinstance(a, ast.Attribute) and a.attr == "w_ratio" for a in call.args)):
                        mapping[idx] = call.func.id
                        break
                if idx in mapping:
                    break
        out[chapter] = mapping
    return out


def load_catalog() -> dict[int, list[Chapter]]:
    """Return {book_number: [Chapter, ...]} in menu order, with demo metadata."""
    tree = _parse(WINDOW_PY)
    imports = _import_map(tree)
    dicts = {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.endswith("_CHAPTERS_DEMOS")
    }
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}

    catalog: dict[int, list[Chapter]] = {}
    for book, info in BOOKS.items():
        table = ast.literal_eval(dicts[info["dict_name"]])
        idx_map = _index_to_class(classes[info["window_class"]])
        chapters = []
        for number in sorted(table):
            title, _placeholder, *names = table[number]
            chapter = Chapter(book, number, title)
            for i, name in enumerate(names, start=1):
                alias = idx_map.get(number, {}).get(i)
                if alias is None:
                    raise RuntimeError(f"book {book} chapter {number} index {i} ({name!r}) has no class in Window.py")
                module, real = imports[alias]
                demo = Demo(book, number, i, name, real, module)
                _fill_meta(demo)
                chapter.demos.append(demo)
            chapters.append(chapter)
        catalog[book] = chapters
    return catalog


def iter_demos(catalog=None):
    catalog = catalog or load_catalog()
    for book in sorted(catalog):
        for chapter in catalog[book]:
            yield from chapter.demos


# ---------------------------------------------------------------------------
# Per-demo metadata
# ---------------------------------------------------------------------------

def _const_str(node) -> str | None:
    """Evaluate a string expression made of constants, concatenation and f-strings."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _const_str(node.left), _const_str(node.right)
        if left is not None and right is not None:
            return left + right
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                parts.append(str(v.value))
            else:
                parts.append("")  # drop runtime pieces, keep static text
        return "".join(parts)
    return None


def _class_meta(module: str, class_name: str, depth=0) -> dict:
    """title/description/controls for a class, following base classes if needed."""
    path = os.path.join(ROOT, module.replace(".", "/") + ".py")
    if depth > 4 or not os.path.exists(path):
        return {}
    tree = _parse(path)
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name), None)
    if cls is None:
        return {}

    meta: dict = {"controls": []}

    # 1) fill_chapter(title, number, description, ...)
    for node in ast.walk(cls):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "fill_chapter" and len(node.args) >= 3):
            title = _const_str(node.args[0])
            desc = _const_str(node.args[2])
            if title and title.strip():
                meta["title"] = title.strip()
            if desc:
                meta["description"] = desc
            meta["source"] = f"fill_chapter in {class_name}"
            break

    # 2) PanelDemo-style class attributes
    for node in cls.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in ("title", "description"):
                val = _const_str(node.value)
                if val and key not in meta:
                    meta[key] = val.strip() if key == "title" else val
                    meta.setdefault("source", f"class attribute in {class_name}")

    # 3) controls created with the NNDLayout helpers
    for node in ast.walk(cls):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            spec = _CONTROL_HELPERS.get(node.func.attr)
            if spec is None:
                continue
            pos, kind = spec
            label = None
            if len(node.args) > pos:
                label = _const_str(node.args[pos])
            for kw in node.keywords:
                if kw.arg in ("label_str", "button_str", "checkbox_str", "input_box_text"):
                    label = _const_str(kw.value) or label
            if label:
                label = " ".join(label.split())
                label = re.sub(r"\s*[:=]\s*[-+]?\d[\d.,e+-]*\s*$", "", label).rstrip(":")  # drop ": 0.01"
                if label and label not in ("?",):
                    entry = f"{label} ({kind})"
                    if entry not in meta["controls"]:
                        meta["controls"].append(entry)

    # 4) follow the first base class when something is still missing
    if "description" not in meta or "title" not in meta:
        imports = _import_map(tree)
        for base in cls.bases:
            base_name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
            if base_name in imports:
                base_mod, base_real = imports[base_name]
                parent = _class_meta(base_mod, base_real, depth + 1)
                for key in ("title", "description", "source"):
                    if key not in meta and key in parent:
                        meta[key] = parent[key]
                meta["controls"] = parent.get("controls", []) + meta["controls"]
                break
    return meta


def _fill_meta(demo: Demo) -> None:
    meta = _class_meta(demo.module, demo.class_name)
    demo.title = meta.get("title", demo.name)
    demo.description = meta.get("description", "")
    demo.controls = meta.get("controls", [])
    demo.meta_source = meta.get("source", "")


def clean_description(text: str) -> list[str]:
    """Turn the window text (hard-wrapped with \\n) into paragraphs."""
    paragraphs = []
    for block in re.split(r"\n\s*\n", text.strip()):
        para = " ".join(line.strip() for line in block.splitlines())
        para = re.sub(r"\s+", " ", para).strip()
        if para:
            paragraphs.append(para)
    return paragraphs


if __name__ == "__main__":
    cat = load_catalog()
    missing = []
    for d in iter_demos(cat):
        flag = "" if d.description else "  <-- no description"
        print(f"B{d.book} ch{d.chapter:02d} #{d.index} {d.name!r:38} {d.class_name:32} controls={len(d.controls)}{flag}")
        if not d.description:
            missing.append(d)
    print(f"\n{sum(len(c.demos) for b in cat.values() for c in b)} demos; {len(missing)} without description")
