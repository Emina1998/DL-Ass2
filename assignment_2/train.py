import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
from torchvision.models.segmentation import fcn_resnet50
from torchvision.models import ResNet50_Weights
import torch.nn as nn

from assignment_2_code.models.segment_model import DeepSegmenter
from assignment_2_code.dataset.oxfordpets import OxfordPetsCustom
from assignment_2_code.metrics import SegMetrics
from assignment_2_code.trainer import ImgSemSegTrainer


def train(args):

    train_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(size=(64, 64), interpolation=v2.InterpolationMode.NEAREST),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_transform2 = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.long, scale=False),
        v2.Resize(size=(64, 64), interpolation=v2.InterpolationMode.NEAREST)
    ])

    val_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(size=(64, 64), interpolation=v2.InterpolationMode.NEAREST),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform2 = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.long, scale=False),
        v2.Resize(size=(64, 64), interpolation=v2.InterpolationMode.NEAREST)
    ])

    train_data = OxfordPetsCustom(
        root=args.data_root,
        split="trainval",
        target_types='segmentation',
        transform=train_transform,
        target_transform=train_transform2,
        download=True
    )

    val_data = OxfordPetsCustom(
        root=args.data_root,
        split="test",
        target_types='segmentation',
        transform=val_transform,
        target_transform=val_transform2,
        download=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build model — from scratch or with pretrained backbone
    if args.pretrained_backbone:
        print("Training FCN-ResNet50 with pretrained backbone")
        base_model = fcn_resnet50(weights=None, weights_backbone=ResNet50_Weights.DEFAULT)
    else:
        print("Training FCN-ResNet50 from scratch")
        base_model = fcn_resnet50(weights=None)

    # Replace final classifier layer to output 3 classes (OxfordPets)
    base_model.classifier[4] = nn.Conv2d(512, 3, kernel_size=1)

    model = DeepSegmenter(base_model)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, amsgrad=True)
    loss_fn = nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    train_metric = SegMetrics(classes=train_data.classes_seg)
    val_metric = SegMetrics(classes=val_data.classes_seg)

    # Separate save dirs so models don't overwrite each other
    if args.pretrained_backbone:
        save_path = Path("saved_models/FCN_pretrained_backbone")
    else:
        save_path = Path("saved_models/FCN_from_scratch")
    save_path.mkdir(parents=True, exist_ok=True)

    trainer = ImgSemSegTrainer(
        model,
        optimizer,
        loss_fn,
        lr_scheduler,
        train_metric,
        val_metric,
        train_data,
        val_data,
        device,
        args.num_epochs,
        save_path,
        batch_size=64,
        val_frequency=2
    )

    trainer.train()
    trainer.dispose()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Training')
    args.add_argument('-d', '--gpu_id', default='0', type=str,
                      help='index of which GPU to use')
    args.add_argument('--data_root', default='./datasets', type=str,
                      help='path to oxfordpets dataset root')

    if not isinstance(args, tuple):
        args = args.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.gpu_id = 0
    args.num_epochs = 31

    print("\n--------- Training FROM SCRATCH ---------")
    args.pretrained_backbone = False
    train(args)

    print("\n--------- Training WITH PRETRAINED BACKBONE ---------")
    args.pretrained_backbone = True
    train(args)