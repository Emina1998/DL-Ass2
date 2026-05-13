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
from dlvc.models.segment_model import DeepSegmenter
from dlvc.dataset.oxfordpets import  OxfordPetsCustom
from dlvc.metrics import SegMetrics
from dlvc.trainer import ImgSemSegTrainer



def train(args):

    train_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    train_transform2 = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])#,
    
    val_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    val_transform2 = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])

    train_data = OxfordPetsCustom(root="path_to_dataset", 
                            split="trainval",
                            target_types='segmentation', 
                            transform=train_transform,
                            target_transform=train_transform2,
                            download=True)

    val_data = OxfordPetsCustom(root="path_to_dataset", 
                            split="test",
                            target_types='segmentation', 
                            transform=val_transform,
                            target_transform=val_transform2,
                            download=True)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    use_pretrained_backbone = args.pretrained_backbone
    if use_pretrained_backbone:
        base_model = fcn_resnet50(weights = None, weights_backbone = ResNet50_Weights.DEFAULT)
        print("Using pretrained model fcn_resnet50")
    else:
        base_model = fcn_resnet50(weights = None)
        print("Using from scratch model fcn_resnet50")

    # reduce number of classes for the oxfod pets to 3
    base_model.classifier[4] = nn.Conv2d(512, 3, kernel_size=1) # check this

    model = DeepSegmenter(base_model)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr = 0.001,amsgrad = True)

    loss_fn = nn.CrossEntropyLoss()
    
    train_metric = SegMetrics(classes=train_data.classes_seg)
    val_metric = SegMetrics(classes=val_data.classes_seg)
    val_frequency = 2

    saved_model_path = ''
    if use_pretrained_backbone:
        saved_model_path = 'saved_models/FCN_resnet50_pretrained'
    else: 
        saved_model_path = 'saved_models/FCN_resnet50_from_scratch'

    model_save_dir = Path(saved_model_path)
    model_save_dir.mkdir(parents=True, exist_ok=True)

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.98)
    
    trainer = ImgSemSegTrainer(model, 
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
                    batch_size=64,
                    val_frequency = val_frequency)
    trainer.train()

    # see Reference implementation of ImgSemSegTrainer
    # just comment if not used
    trainer.dispose() 

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Training')
    args.add_argument('-d', '--gpu_id', default='0', type=str,
                      help='index of which GPU to use')
    # args.add_argument('--pretrained_backbone', action='store_true',
                  #help='Use pretrained encoder weights for fcn_resnet50')
    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.gpu_id = 0
    args.num_epochs = 31

    # train both pretrained and from scratch one after another
    # so I could leave it overnight
    print("\n---------Training FROM SCRATCH ---------")
    args.pretrained_backbone = False
    train(args)

    # 
    print("\n--------- Training WITH PRETRAINED ENCODER ---------")
    args.pretrained_backbone = True
    train(args)