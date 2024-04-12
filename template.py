import os
from pathlib import Path

# List of files and directories to create
files_and_directories = [
    "README.md",
    "LICENSE",
    "requirements.txt",
    "src/__init__.py",
    "src/main.py",
    "src/chatbot.py",
    "src/intents.json",
    "data/",
    "docs/",
    "tests/",
    ".gitignore"
]

# Iterate over each file/directory and create them
for item in files_and_directories:
    if os.path.isfile(item) or item.endswith(".py"):  # Check if it's a file or Python script
        Path(item).parent.mkdir(parents=True, exist_ok=True)  # Create parent directories if they don't exist
        with open(item, "w") as f:
            pass  # Create empty file
    else:
        os.makedirs(item, exist_ok=True)  # Create directory
