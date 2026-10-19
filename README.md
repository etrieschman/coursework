# coursework
Organization of non-project based coursework


## Python environment

One shared [uv](https://docs.astral.sh/uv/) environment for all coursework, defined in `pyproject.toml` / `uv.lock` at the repo root.

```bash
uv sync                          # create/update .venv
uv add <package>                 # add a dependency
uv run python mit/6_2202/62202_hw1.py
```

**VS Code:** select `.venv/bin/python` as the interpreter/kernel for `# %%` cells.

**Printing a homework to PDF** (runs the cells and embeds outputs; uses the local xelatex):

```bash
uv run jupytext --to ipynb mit/6_2202/62202_hw1.py
uv run jupyter nbconvert --execute --to pdf mit/6_2202/62202_hw1.ipynb
```

Use `--to html` instead of `--to pdf` for an HTML version you can print from a browser.

## LaTeX writeups

All MIT writeups live in one Overleaf project, pulled in as a submodule at
`mit/overleaf`, with a folder per class. Starting a new class is a `mkdir` —
no new Overleaf project, GitHub repo, or submodule:

```bash
cd mit/overleaf
mkdir 6_7900 && cp _template/hw.tex 6_7900/hw1.tex
```

Each `.tex` is self-contained, preamble included. Compile with:

```bash
cd mit/overleaf && latexmk -pdf 15_081/hw7.tex
```

Sync with `git op` (pull from Overleaf) and `git sync` (push to GitHub +
Overleaf and bump this repo's pointer). Full details in
`~/Library/Mobile Documents/com~apple~CloudDocs/dev_overleaf_workflow.md`.

Older Stanford coursework (`stanford/`) is plain committed `.tex`, not Overleaf-synced.
