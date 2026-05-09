import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path

from assignment_2_code.models.segformer import SegFormer
from assignment_2_code.models.segment_model import DeepSegmenter
from assignment_2_code.dataset.cityscapes import CityscapesCustom
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

    if args.dataset == "oxford":
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

    if args.dataset == "city":
        train_data = CityscapesCustom(
            root=args.city_root,
            split="train",
            mode="fine",
            target_type='semantic',
            transform=train_transform,
            target_transform=train_transform2
        )
        val_data = CityscapesCustom(
            root=args.city_root,
            split="val",
            mode="fine",
            target_type='semantic',
            transform=val_transform,
            target_transform=val_transform2
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DeepSegmenter(SegFormer(num_classes=len(train_data.classes_seg)))

    # Base optimizer — may be replaced below if freezing
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True)

    # Fine-tuning phase: load pretrained encoder weights
    if args.dataset == 'oxford' and not args.from_scratch:
        print("Loading pretrained encoder weights")
        full_state_dict = torch.load(args.pretrained_path, map_location='cpu')
        
        # Load only matching encoder keys directly into encoder
        encoder_state = {}
        for key, value in full_state_dict.items():
            if key.startswith("encoder."):
                new_key = key[len("encoder."):]
                encoder_state[new_key] = value
        
        model.net.encoder.load_state_dict(encoder_state, strict=True)
        print("Encoder weights loaded")

        if args.freeze:
            print("Freezing encoder")
            for param in model.net.encoder.parameters():
                param.requires_grad = False
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.lr,
                amsgrad=True
            )
    else:
        print("Training from scratch")

    model.to(device)

    # ignore_index=255 handles both Cityscapes ignored pixels and OxfordPets safely
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)

    train_metric = SegMetrics(classes=train_data.classes_seg)
    val_metric = SegMetrics(classes=val_data.classes_seg)

    # Build save directory name based on run config
    if args.from_scratch:
        run_name = "from_scratch"
    elif args.freeze:
        run_name = f"freeze_lr{args.lr}"
    else:
        run_name = f"finetune_lr{args.lr}"

    model_save_dir = Path("saved_models") / args.dataset / run_name
    model_save_dir.mkdir(parents=True, exist_ok=True)

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

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
        model_save_dir,
        batch_size=args.batch_size,
        val_frequency=2
    )

    trainer.train()
    trainer.dispose()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Training')
    args.add_argument('-d', '--gpu_id', default='0', type=str,
                      help='index of which GPU to use')
    args.add_argument('--data_root', default='./datasets', type=str,
                      help='path to OxfordPets dataset root')
    args.add_argument('--city_root', default='./cityscapes_assg2', type=str,
                      help='path to Cityscapes dataset root')
    args.add_argument('--pretrained_path',
                      default='saved_models/city/from_scratch/SegFormer_model_best.pth',
                      type=str, help='path to pretrained Cityscapes model')

    if not isinstance(args, tuple):
        args = args.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.gpu_id = 0

    # ---- Part 5: SegFormer from scratch on OxfordPets ----
    print("\n--- SegFormer from scratch on OxfordPets ---")
    args.dataset = "oxford"
    args.num_epochs = 31
    args.batch_size = 16
    args.from_scratch = True
    args.freeze = False
    args.lr = 1e-3
    train(args)

    # ---- Part 6a: Pre-train on Cityscapes ----
    print("\n--- Pre-training SegFormer on Cityscapes ---")
    args.dataset = "city"
    args.num_epochs = 40
    args.batch_size = 16
    args.from_scratch = True
    args.freeze = False
    args.lr = 1e-3
    train(args)

    # ---- Part 6b: Fine-tune on OxfordPets (unfrozen encoder, lr=1e-3) ----
    print("\n--- Fine-tuning on OxfordPets (unfrozen, lr=1e-3) ---")
    args.dataset = "oxford"
    args.num_epochs = 31
    args.batch_size = 16
    args.from_scratch = False
    args.freeze = False
    args.lr = 1e-3
    train(args)

    # ---- Part 6b: Fine-tune on OxfordPets (unfrozen encoder, lr=5e-4) ----
    print("\n--- Fine-tuning on OxfordPets (unfrozen, lr=5e-4) ---")
    args.dataset = "oxford"
    args.num_epochs = 31
    args.batch_size = 16
    args.from_scratch = False
    args.freeze = False
    args.lr = 5e-4
    train(args)

    # ---- Part 6b: Fine-tune on OxfordPets (frozen encoder, lr=1e-3) ----
    print("\n--- Fine-tuning on OxfordPets (frozen encoder, lr=1e-3) ---")
    args.dataset = "oxford"
    args.num_epochs = 31
    args.batch_size = 16
    args.from_scratch = False
    args.freeze = True
    args.lr = 1e-3
    train(args)

    # ---- Part 6b: Fine-tune on OxfordPets (frozen encoder, lr=5e-4) ----
    print("\n--- Fine-tuning on OxfordPets (frozen encoder, lr=5e-4) ---")
    args.dataset = "oxford"
    args.num_epochs = 31
    args.batch_size = 16
    args.from_scratch = False
    args.freeze = True
    args.lr = 5e-4
    train(args)