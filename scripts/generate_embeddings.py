import os
import json
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import umap
import pandas as pd
from tqdm import tqdm

from config import DATA_FILE
from core.logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

IMG_SIZE = 224
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_json_metadata(image_path):
    """Look up the sidecar JSON for an image and return species metadata."""
    base, _ = os.path.splitext(image_path)
    json_path = base + ".json"

    scientific = "Unknown"
    common = "Unknown"
    user = "Unknown"
    tid = -1

    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            tid = data.get('species_id', -1)
            scientific = data.get('species', scientific)
            common = data.get('common_name', common)
            user_data = data.get('user', {})
            if isinstance(user_data, dict):
                user = user_data.get('login', user)

        except Exception as e:
            logger.error("Failed to read JSON %s: %s", json_path, e)

    return scientific, common, tid, user


def run(dataset_dir: str, output_file: str = DATA_FILE, batch_size: int = BATCH_SIZE):
    logger.info("Using device: %s", DEVICE.upper())

    logger.info("Loading DINOv2...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model.to(DEVICE)
    model.eval()

    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    paths, species_ids, filenames, users = [], [], [], []
    scientific_names, common_names = [], []
    families, genera = [], []

    logger.info("Scanning: %s", dataset_dir)
    found_count = 0
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)

                scientific, common, tid, user = extract_json_metadata(img_path)

                rel_path = os.path.relpath(root, dataset_dir)
                path_parts = rel_path.split(os.sep)

                if len(path_parts) >= 2:
                    family = path_parts[0]
                    genus = path_parts[1]
                    if scientific == "Unknown" and len(path_parts) >= 3:
                        scientific = path_parts[2]
                else:
                    family = "Unclassified"
                    genus = "Unclassified"

                paths.append(img_path)
                filenames.append(file)
                species_ids.append(tid)
                users.append(user)
                scientific_names.append(scientific)
                common_names.append(common)
                families.append(family)
                genera.append(genus)

                found_count += 1

    if not paths:
        logger.warning("No images found.")
        return

    logger.info("Found %d images.", found_count)

    embeddings = []
    logger.info("Generating embeddings for %d images...", len(paths))

    for i in tqdm(range(0, len(paths), batch_size)):
        batch_paths = paths[i:i + BATCH_SIZE]
        batch_tensors = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert('RGB')
                batch_tensors.append(transform(img))
            except Exception as e:
                logger.error("Failed to load image %s: %s", p, e)

        if batch_tensors:
            batch_input = torch.stack(batch_tensors).to(DEVICE)
            with torch.no_grad():
                out = model(batch_input).cpu().numpy()
                embeddings.append(out)

    if not embeddings:
        logger.error("No embeddings were generated.")
        return

    all_embeddings = np.concatenate(embeddings)

    logger.info("Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='cosine', random_state=42)
    embedding_2d = reducer.fit_transform(all_embeddings)

    logger.info("Saving output CSV...")
    df = pd.DataFrame({
        'x': embedding_2d[:, 0],
        'y': embedding_2d[:, 1],
        'species_id': species_ids,
        'scientific_name': scientific_names,
        'common_name': common_names,
        'family': families,
        'genus': genera,
        'user': users,
        'filename': filenames,
        'absolute_path': paths,
    })

    df['absolute_path'] = df['absolute_path'].apply(os.path.abspath)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)

    logger.info("Done: CSV saved to %s", output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate DINOv2 embeddings and UMAP projection for an image dataset."
    )
    parser.add_argument("--dataset", required=True, metavar="DIR",
                        help="Root directory of images to process.")
    parser.add_argument("--output", default=DATA_FILE, metavar="CSV",
                        help=f"Output CSV path (default: {DATA_FILE}).")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, metavar="N",
                        help=f"Embedding batch size (default: {BATCH_SIZE}).")
    args = parser.parse_args()

    run(dataset_dir=args.dataset, output_file=args.output, batch_size=args.batch_size)
