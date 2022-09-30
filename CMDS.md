### Before pushing

Use pytest for checking your code.

```commandline
python3 -m pytest tests/tests.py
```

If your changes impact the API or documentation, update the doc page

```commandline
cd docs
rm –r _build
make html
git add _build -f
```

