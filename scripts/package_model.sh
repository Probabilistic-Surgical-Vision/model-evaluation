find . -not \( \
    -name ".DS_Store" \
    -or -path "./.git*" \
    -or -path "./.ipynb_checkpoints*" \
    -or -path "*__pycache__*" \
    -or -path "./venv*" \
    -or -path "./models*" \
    -or -path "./results*" \) | zip -@ model-package
