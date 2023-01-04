import subprocess

def train() -> None:
    """Train the model."""
    subprocess.run(
       ['poetry', 'run', 'python', 'train.py']
    )

def types() -> None:
    """
    Check types with mypy. Equivalent to:
    `poetry run mypy .`
    """
    subprocess.run(
       ['poetry', 'run', 'mypy', '.']
    )

def test() -> None:
    """
    Run all unit tests. Equivalent to:
    `poetry run python -m unittest discover`
    """
    subprocess.run(
       ['poetry', 'run', 'python', '-m', 'unittest', 'discover']
    )
