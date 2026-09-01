# Layer 3 — NLI-Based Semantic Detection

The Layer 3 implementation is included in this repository. Its large pretrained
model/checkpoint artifacts are intentionally **not** stored in GitHub.

## Model files

Download the Layer 3 model files from the project's Google Drive folder:

https://drive.google.com/drive/folders/1IqKVwEKu6EduP2JhEY-YCQ9Fg0zVjT9Y

After downloading, place the required model directory/files under:

```text
layer3_slm/model/
```

The repository `.gitignore` excludes this directory and large model formats from
Git commits.

## Running Layer 3

Install the Layer 3 dependencies:

```bash
pip install -r requirements.txt
```

Then follow the usage instructions in `run_layer3.py` and the project root README.
