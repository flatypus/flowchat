import os
import subprocess
from dotenv import load_dotenv

load_dotenv()

username = "__token__"
password = os.environ.get("PYPI_TOKEN")

subprocess.run(["python3", "-m", "build"])
subprocess.run([
    "python3", "-m", "twine", "upload", "dist/*", "--username", username, "--password", password
])
