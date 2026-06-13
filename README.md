# Wildlife Image Dataset Curator

A desktop tool for visually auditing and curating wildlife image datasets using DINOv2 embeddings and UMAP projections.

## What it does

The curator embeds every image in your dataset with a pretrained DINOv2 ViT-B/14 model, reduces the resulting high-dimensional vectors to 2D with UMAP, and renders an interactive scatter plot. You can click individual points or draw a lasso to select groups of images, preview them in the app, and send low-quality or mislabelled images to a trash folder — all tracked in a CSV so the operation is reversible.

**Features**

- Interactive UMAP scatter plot coloured by taxonomic family
- Single-click point selection with image preview and metadata panel
- Lasso tool for batch selection of image clusters
- One-click discard: moves files to `data/deleted/` and marks them in the CSV
- Full dataset restore (undo all discards)
- Per-species and per-family statistics table
- Dark / light theme toggle

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA strongly recommended for embedding generation; CPU fallback is available but slow

**Python dependencies**

```
PySide6
torch
torchvision
Pillow
numpy
pandas
umap-learn
matplotlib
tqdm
```

Dev dependency: `pytest`

## Installation

```bash
git clone <repo-url>
cd wildlife-image-dataset-curator
pip install PySide6 torch torchvision Pillow numpy pandas umap-learn matplotlib tqdm
```

## Usage

### Step 1 — Generate embeddings

```bash
python scripts/generate_embeddings.py --dataset <path/to/images>
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--dataset DIR` | yes | — | Root directory of images to process |
| `--output CSV` | no | value of `DATA_FILE` in `config.py` | Output CSV path |
| `--batch-size N` | no | `8` | Embedding batch size |

Example:

```bash
python scripts/generate_embeddings.py \
    --dataset data/raw/fauna_seleccionada_bosque_atlantico/images \
    --output output/datos_clusters_vitb14.csv \
    --batch-size 16
```

The script scans the directory recursively, extracts DINOv2 embeddings, runs UMAP, and writes a CSV with columns `x`, `y`, `species_id`, `scientific_name`, `common_name`, `family`, `genus`, `user`, `filename`, `absolute_path`.

### Step 2 — Run the curator app

```bash
python main.py
```

The app reads the CSV path from `config.py` (`DATA_FILE`) and opens maximised.

## Dataset format

Images must live under a three-level taxonomic hierarchy:

```
<root>/
└── <Family>/
    └── <Genus>/
        └── <Species>/
            ├── image_001.jpg
            ├── image_001.json   ← sidecar
            └── ...
```

Each sidecar JSON must include at minimum:

```json
{
  "species_id": 41997,
  "species": "Leopardus pardalis",
  "common_name": "Ocelot",
  "user": {
    "login": "photographer_username"
  }
}
```

Missing fields fall back to values inferred from the folder path. The `user` field is expected to be an object; only `user.login` is used.

## Configuration

Edit `config.py` to change runtime behaviour:

| Key | Default | Description |
|---|---|---|
| `DATA_FILE` | `output/datos_clusters_vitb14.csv` | CSV produced by the embedding script |
| `TRASH_FOLDER` | `data/deleted` | Destination folder for discarded images |
| `EMBEDDING_MODEL_NAME` | `"DINOv2 ViT-B/14"` | Display name shown in the UMAP plot title |
| `THEMES` | `{"dark": ..., "light": ...}` | Qt stylesheets; edit to customise colours |

## Running tests

```bash
pytest tests/
```

The test suite covers `DataManager`: CSV loading, global summary, lasso filtering, and destination-path calculation (10 tests).

## Roadmap

- **i18n** — UI strings are currently in Spanish; extract them for proper internationalisation
- **Additional tests** — coverage for `UMAPWidget` signals and the statistics panel
- **Model selector in GUI** — allow switching between DINOv2 variants (ViT-S/14, ViT-L/14) without editing config
