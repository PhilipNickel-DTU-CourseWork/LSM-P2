# University Project Template

A template repository for numerical computing projects with documentation, examples, and best practices.

## Features

- **Package structure**: Clean `src/` layout with numerical utilities
- **Documentation**: Sphinx with ReadTheDocs integration
- **Examples**: Automated gallery generation with sphinx-gallery
- **Testing**: pytest with coverage reporting
- **Code quality**: Ruff for linting and formatting
- **Dependency management**: Compatible with uv or pip

## Quick Start

### 1. Clone and customize

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Update project metadata

Edit `pyproject.toml`:
- Change `name` from "numutils" to your project name
- Update `authors` with your information
- Modify `description` as needed

### 3. Install dependencies

Using uv (recommended):
```bash
uv sync
```

```

### 4. Run example

```bash
python Examples/example_integration/compute.py
python Examples/example_integration/plot_integration.py
```

## Project Structure

```
.
├── src/numutils/           # Main package (rename as needed)
│   ├── integration.py      # Numerical integration methods
│   ├── linalg.py          # Linear algebra utilities
│   ├── utils/             # Helper utilities
│   └── styles/            # Custom matplotlib styles
├── Examples/              # Example scripts (auto-documented)
│   └── example_integration/
├── docs/                  # Sphinx documentation
├── data/                  # Generated data files (gitignored)
└── pyproject.toml        # Project configuration
```

## Documentation

Build documentation locally:

```bash
cd docs
make html
```

View at `docs/build/html/index.html`

### ReadTheDocs

To enable ReadTheDocs:
1. Update `.readthedocs.yaml` with your project name
2. Connect your repository at https://readthedocs.org
3. Update the documentation URL in this README

## Adding Examples

1. Create a new directory in `Examples/`:
   ```bash
   mkdir Examples/my_example
   ```

2. Add scripts (automatically discovered):
   - `compute.py` - Data generation
   - `plot_*.py` - Visualization

3. Add a `README.rst` with description

4. Rebuild docs to see it in the gallery

## Development

### Running tests

```bash
pytest
```

### Code quality

```bash
ruff check .
ruff format .
```

### Type checking

```bash
# Add mypy if needed
pip install mypy
mypy src/
```

## Customization Tips

### Rename the package

1. Rename `src/numutils/` to `src/yourpackage/`
2. Update imports in example scripts
3. Update `pyproject.toml` package name
4. Update `docs/source/conf.py` if needed

### Change matplotlib style

Edit `src/numutils/styles/ana.mplstyle` or create new style files.

### Modify documentation theme

Edit `docs/source/conf.py` and change `html_theme` or theme options.

## License

MIT License - see LICENSE file for details

## Contributing

This is a template repository. Fork it and adapt it to your needs!
