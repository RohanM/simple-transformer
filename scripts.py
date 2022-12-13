import subprocess

def types():
    """
    Check types with mypy. Equivalent to:
    `poetry run mypy .`
    """
    subprocess.run(
       ['poetry', 'run', 'mypy', '.']
    )

def test():
    """
    Run all unit tests. Equivalent to:
    `poetry run python -m unittest discover`
    """
    subprocess.run(
       ['poetry', 'run', 'python', '-m', 'unittest', 'discover']
    )
