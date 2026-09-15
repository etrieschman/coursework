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
