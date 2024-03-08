import os
import subprocess
from dotenv import load_dotenv

load_dotenv()

username = "__token__"
password = os.environ.get("PYPI_TOKEN")

if not password:
    raise ValueError("PYPI_TOKEN is not set")

subprocess.run(["python3", "-m", "build"])
subprocess.run([
    "python3", "-m", "twine", "upload", "dist/*", "--username", username, "--password", password
])
