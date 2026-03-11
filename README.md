# Pokematching

## Setup

Create the project virtualenv using Python 3.11.9 via pyenv (avoids `pkgutil.find_loader` error in Python 3.14):

```bash
~/.pyenv/versions/3.11.9/bin/python -m venv ~/.venvs/pokematching-venv
source ~/.venvs/pokematching-venv/bin/activate
pip install -r requirements.txt
```

## Label Studio

To start Label Studio:

```bash
source ~/.venvs/label-studio/bin/activate
label-studio start
```

UI available at http://localhost:8080

### Notes
- Uses Python 3.11.9 (installed via pyenv) to avoid `pkgutil.find_loader` error in Python 3.14
- Venv is at `~/.venvs/label-studio`
- To install from scratch: `~/.pyenv/versions/3.11.9/bin/python -m venv ~/.venvs/label-studio && source ~/.venvs/label-studio/bin/activate && pip install label-studio`
