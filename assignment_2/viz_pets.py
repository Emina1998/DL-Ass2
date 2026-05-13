import os
from pathlib import Path

import torch
import torchvision
import torchvision.transforms.v2 as v2
import matplotlib.pyplot as plt
import numpy as np

from train import OxfordPetsCustom
from assignment_2_code.models.segment_model import DeepSegmenter
from assignment_2_code.models.segformer import SegFormer


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

DATASET_ROOT = "oxford_images"

MODEL_PATHS = {
    "scratch": "saved_models/oxford/from_scratch/SegFormer_model_best.pth",

    "finetune_lr0.001":
    "saved_models/oxford/finetune_lr0.001/SegFormer_model_best.pth",

    "finetune_lr0.0005":
    "saved_models/oxford/finetune_lr0.0005/SegFormer_model_best.pth",

    "freeze_lr0.001":
    "saved_models/oxford/freeze_lr0.001/SegFormer_model_best.pth",

    "freeze_lr0.0005":
    "saved_models/oxford/freeze_lr0.0005/SegFormer_model_best.pth"
}

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 8


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def save_grid_image(tensor, save_path):
    """
    Saves image grid from tensor.
    """

    image = tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def build_dataloader():

    image_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(IMAGE_SIZE)
    ])

    mask_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.long, scale=False),
        v2.Resize(
            IMAGE_SIZE,
            interpolation=v2.InterpolationMode.NEAREST
        )
    ])

    dataset = OxfordPetsCustom(
        root=DATASET_ROOT,
        split="trainval",
        target_types="segmentation",
        transform=image_transform,
        target_transform=mask_transform,
        download=True
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    return dataset, loader


def load_model(model_path, num_classes, device):

    model = DeepSegmenter(
        SegFormer(num_classes=num_classes)
    )

    state_dict = torch.load(model_path, map_location=device)

    model.net.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


def generate_predictions(model, images, device):

    with torch.no_grad():

        images = images.to(device)

        outputs = model(images)

        predictions = torch.argmax(outputs, dim=1)

    return predictions.cpu()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset, loader = build_dataloader()

    images, masks = next(iter(loader))

    # -----------------------------------------------------
    # SAVE INPUT IMAGES
    # -----------------------------------------------------

    image_grid = torchvision.utils.make_grid(
        images,
        nrow=4
    )

    save_grid_image(
        image_grid,
        OUTPUT_DIR / "input_images.png"
    )

    # -----------------------------------------------------
    # SAVE GROUND TRUTH MASKS
    # -----------------------------------------------------

    normalized_masks = (masks - 1).float() / 2

    mask_grid = torchvision.utils.make_grid(
        normalized_masks,
        nrow=4
    )

    save_grid_image(
        mask_grid,
        OUTPUT_DIR / "ground_truth_masks.png"
    )

    # -----------------------------------------------------
    # RUN ALL MODELS
    # -----------------------------------------------------

    for model_name, model_path in MODEL_PATHS.items():

        print(f"Running inference for: {model_name}")

        model = load_model(
            model_path=model_path,
            num_classes=len(dataset.classes_seg),
            device=device
        )

        predictions = generate_predictions(
            model,
            images,
            device
        )

        prediction_grid = torchvision.utils.make_grid(
            predictions.unsqueeze(1).float(),
            nrow=4
        )

        prediction_grid = prediction_grid / prediction_grid.max()

        save_grid_image(
            prediction_grid,
            OUTPUT_DIR / f"{model_name}_predictions.png"
        )

        print(f"Saved {model_name} predictions")

    print("Visualization completed.")