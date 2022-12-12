import subprocess

def test():
    """
    Run all unit tests. Equivalent to:
    `poetry run python -m unittest discover`
    """
    subprocess.run(
       ['poetry', 'run', 'python', '-m', 'unittest', 'discover']
    )
