# Class Attendance Management using Face Recognition

Lightweight web app for recording class attendance using face detection and recognition. The project includes face detection (MTCNN), embeddings, and a small Flask app for registration, prediction and attendance retrieval.

This repository contains training data, example datasets, model wrappers and a Flask UI under `src/` and `templates/`.

## Quick facts

- Language: Python
- Web: Flask
- Face detection: MTCNN (in `detectfaces_mtcnn/`)
- Face recognition / embeddings: `faceEmbeddingModels/`, `insightface/` (included as a subfolder)
- Dataset folder: `datasets/train/` (contains per-person image folders)

## Project highlights

- Led the development of a full-stack attendance system that automated classroom attendance by capturing a group photo and marking present students automatically.
- Built Flask-based REST APIs for registration, attendance marking, and retrieval; integrated with MongoDB for secure data storage and real-time access.
- Implemented face detection using MTCNN and face recognition with dlib embeddings, enabling accurate recognition of up to 30 students from group images.
- Designed intuitive UIs using React (Web) and Flutter (Android) to provide easy registration, attendance monitoring, and retrieval functionalities.

## Files you should know

- `src/app.py` — main Flask application (run from `src/`).
- `requirements.txt` — pinned Python package list.
- `How to run.txt` — original install/run notes.
- `datasets/` — training images (do NOT commit to GitHub; large).
- `faceEmbeddingModels/`, `insightface/` — model code and weights (may be large).

## High-level usage

1. Create and activate a Python environment (recommended: conda).
2. Install dependencies from `requirements.txt`.
3. From the repository root: `cd src` then `python app.py` to start the Flask app.

See a complete step-by-step below with Windows PowerShell commands.

## Recommended install (Windows PowerShell)

Note: the original project suggests Python 3.6. The `requirements.txt` contains many pinned versions which may be incompatible with modern Python. If you prefer reproducibility, use the same Python version the project originally used (3.6.x) via conda.

Example (PowerShell):

```powershell
# create and activate environment
conda create -n facerecognition python=3.6.9 -y
conda activate facerecognition

# install core deps
pip install -r ..\requirements.txt

# extra conda installs recommended in original notes (optional)
conda install -c anaconda mxnet -y
conda install -c conda-forge dlib -y

# if you run into numpy/tensorflow compatibility issues, follow the project's notes (How to run.txt)
```

Run the app:

```powershell
cd src
python app.py
```

## Reducing repository size (short guide)

The repository currently contains image datasets and model folders that are large. To prepare this project for GitHub without exceeding size limits, follow these recommendations:

1. Do not track raw image datasets or pre-trained model weights in Git. Use `.gitignore` (added here) to ignore `datasets/`, `faceEmbeddingModels/`, `insightface/` and other large folders.
2. Use Git LFS for large binary assets you need in the repo (install `git lfs`, then `git lfs track "*.jpg"` etc.).
3. If you've already committed large files, remove them from the index/history (use `git rm --cached <path>` for a single commit or use BFG / `git filter-repo` for history rewriting).

Helpful PowerShell commands (run from repo root):

```powershell
# stop tracking a folder that's already committed (keeps local copy)
git rm -r --cached datasets
git commit -m "Stop tracking datasets/; add to .gitignore"

# Use BFG or git filter-repo for full history removal (recommended for very large files)
# Example using BFG (install BFG separately):
# bfg --delete-folders datasets --delete-files '*.jpg'

# To use Git LFS for future large files:
git lfs install
git lfs track "datasets/**"
git add .gitattributes
git commit -m "Track datasets with Git LFS"
```

Warning: rewriting git history is destructive for shared repos — coordinate before rewriting.

## Project structure (short)

```
src/                # Flask app, model code
	app.py
	detectfaces_mtcnn/
	training/
datasets/           # training images (do not commit)
faceEmbeddingModels/ # embeddings and model files (likely large)
insightface/         # third-party face recognition code (large)
templates/ static/   # web UI
```

## Troubleshooting

- If you see TensorFlow / numpy errors, try matching the versions from `How to run.txt` (the project originally used older packages). Creating a fresh conda env with the specified Python and installing exact versions is the fastest route.
- If `dlib` fails to build on Windows, install it from conda-forge (binary) or use a prebuilt wheel.

## Contributing and next steps

- If you want, I can:
	- run a disk-usage scan and list the largest files/folders so you can decide what to remove or LFS-track, or
	- add a small script to archive and compress datasets (zip/tar.gz) and show commands to re-download them later, or
	- prepare a minimal demo dataset (few images) so the repo can remain small but runnable.

Tell me which you prefer and I will do it next.


