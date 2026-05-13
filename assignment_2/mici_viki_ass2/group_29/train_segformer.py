
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
import os

from dlvc.models.segformer import SegFormer
from dlvc.models.segment_model import DeepSegmenter
from dlvc.dataset.cityscapes import CityscapesCustom
from dlvc.dataset.oxfordpets import OxfordPetsCustom
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

    if args.dataset == "oxford":
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
    if args.dataset == "city":
        train_data = CityscapesCustom(root="C:/Users/aleks/milica/master_tuw/master_tuw/2_semester/dl_in_vc/cityscapes_assg2/cityscapes_assg2", 
                                split="train",
                                mode="fine",
                                target_type='semantic', 
                                transform=train_transform,
                                target_transform=train_transform2)
        val_data = CityscapesCustom(root="C:/Users/aleks/milica/master_tuw/master_tuw/2_semester/dl_in_vc/cityscapes_assg2/cityscapes_assg2", 
                                split="val",
                                mode="fine",
                                target_type='semantic', 
                                transform=val_transform,
                                target_transform=val_transform2)
        


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepSegmenter(SegFormer(num_classes=len(train_data.classes_seg)))
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    # If you are in the fine-tuning phase:
    if args.dataset == 'oxford':
        ##TODO update the encoder weights of the model with the loaded weights of the pretrained model
        # e.g. load pretrained weights with: state_dict = torch.load("path to model", map_location='cpu')
        
        ##
        if not args.from_scratch:
            print("Loading pretrained encoder weights")
            state_dict = torch.load("saved_models/saved_models/city/from_scratch/SegFormer_model_best.pth")
            # print("\n".join(state_dict.keys()))
            # return
            # Extract and rename encoder weights
            encoder_weights = {k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")}

            # Load into encoder
            model.net.encoder.load_state_dict(encoder_weights, strict=True)


            if args.freeze:
                print("Freezing encoder")
                model.net.encoder.requires_grad_(False)
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = args.lr)
        else:
            print("Training from scratch on Oxford — no pretrained weights used.")
    model.to(device)
    
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index  = 255) # remember to ignore label value 255 when training with the Cityscapes datset
    
    train_metric = SegMetrics(classes=train_data.classes_seg)
    val_metric = SegMetrics(classes=val_data.classes_seg)
    val_frequency = 2 # for 
    p = ""
    if args.from_scratch == True:
        p += "from_scratch"
    elif args.freeze == True:
        p += "freeze_"
        p += str(args.lr)
    else:
        p += "finetune_"
        p += str(args.lr)
    model_save_dir = Path("saved_models") / "saved_models" / args.dataset / Path(p)

    model_save_dir.mkdir(parents=True , exist_ok=True)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
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
                    batch_size=args.batch_size,
                    val_frequency = val_frequency)
    trainer.train()
    # see Reference implementation of ImgSemSegTrainer
    # just comment if not used
    trainer.dispose() 

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Training')
    args.add_argument('-d', '--gpu_id', default='0', type=str,
                      help='index of which GPU to use')
    

    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.gpu_id = 0
    args.num_epochs = 31
    args.dataset = "oxford"
    args.from_scratch = True
    args.freeze = False
    args.batch_size = 16
    args.lr = 1e-3    
    train(args)

    args.num_epochs = 40
    args.dataset = "city"
    args.from_scratch = True
    args.freeze = False
    args.batch_size = 16
    train(args)

    args.num_epochs = 31
    args.dataset = "oxford"
    args.batch_size = 16
    args.from_scratch = False  
    args.freeze = False     
    args.lr = 1e-3   
    train(args)
    args.num_epochs = 31
    args.dataset = "oxford"
    args.batch_size = 16
    args.from_scratch = False  
    args.freeze = False     
    args.lr = 5e-4   
    train(args)

    args.num_epochs = 31
    args.dataset = "oxford"
    args.from_scratch = False  
    args.freeze = True
    args.batch_size = 16
    args.lr = 1e-3           
    train(args)
    args.num_epochs = 31
    args.dataset = "oxford"
    args.from_scratch = False  
    args.freeze = True
    args.lr = 5e-4      
    args.batch_size = 16     
    train(args)
