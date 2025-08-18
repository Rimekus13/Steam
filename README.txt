# Steam project test fix pack

This pack contains:
- `pytest.ini` (place it at your project root)
- `dashboard/` minimal package with `__init__.py`, `analysis.py`, `app.py`
- `tests/conftest.py` that adds project root to `sys.path` and patches `get_vader` via `monkeypatch`

## How to use on Windows (PowerShell)

1) Extract the ZIP **at your project root** so you get:
   D:\Projet\Steam\pytest.ini
   D:\Projet\Steam\dashboard\__init__.py
   D:\Projet\Steam\dashboard\analysis.py
   D:\Projet\Steam\dashboard\app.py
   D:\Projet\Steam\tests\conftest.py   (replace your current one)

2) Ensure you don't have a conflicting old `conftest.py`. Replace/overwrite it.

3) Run tests from the project root:
   PS> cd D:\Projet\Steam
   PS> pytest

Optional: if you still get date warnings, it's only a DeprecationWarning.