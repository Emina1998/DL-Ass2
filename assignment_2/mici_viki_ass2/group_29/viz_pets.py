import os
import torch
import torchvision
import torchvision.transforms.v2 as v2
import os
import matplotlib.pyplot as plt
import numpy as np
from dlvc.models.segment_model import DeepSegmenter
from dlvc.models.segformer import SegFormer
# os.chdir(os.getcwd() + "change to your working directory if necessary")


from train import OxfordPetsCustom




def imshow(img, filename='img/test.png'):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.imsave(filename,np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__': 

    train_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])

    train_transform2 = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])

    train_data = OxfordPetsCustom(root="oxford_images", 
                            split="trainval",
                            target_types='segmentation', 
                            transform=train_transform,
                            target_transform=train_transform2,
                            download=True)
    train_data_loader = torch.utils.data.DataLoader(train_data,
                                            batch_size=8,
                                            shuffle=False,
                                            num_workers=2)

    # get some random training images
    dataiter = iter(train_data_loader)
    images, labels = next(dataiter)
    images_plot = torchvision.utils.make_grid(images, nrow=4)
    labels_plot = torchvision.utils.make_grid((labels-1)/2, nrow=4)#.to(torch.uint8)

    # # show/plot images
    imshow(images_plot, filename="oxford_images/oxford-iiit-pet/images/input_test_pets.png")
    imshow(labels_plot,filename="oxford_images/oxford-iiit-pet/images/seg_mask_test_pets.png")
    
    model_1 = DeepSegmenter(SegFormer(num_classes = len(train_data.classes_seg)))
    state_dict = torch.load("C:/Users/aleks/milica/master_tuw/master_tuw/2_semester/dl_in_vc/DL_ass2/assignment_2/saved_models/saved_models/oxford/finetune_0.001/SegFormer_model_best.pth")
    model_1.net.load_state_dict(state_dict)
    model_1.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_1.to(device)
    with torch.no_grad():
        images = images.to(device)
        outputs = model_1(images)
        preds = torch.argmax(outputs, dim= 1)
    
    preds_plot = torchvision.utils.make_grid(preds.unsqueeze(1).float(), nrow=4)  # shape: [1, H, W]
    
    preds_plot = preds_plot / preds_plot.max()
    
    
    save_path = "oxford_images/results/images/pred_mask_test_pets_finetune_0.001.png"
    imshow(preds_plot, filename=save_path)
    
    model_2 = DeepSegmenter(SegFormer(num_classes = len(train_data.classes_seg)))
    state_dict = torch.load("C:/Users/aleks/milica/master_tuw/master_tuw/2_semester/dl_in_vc/DL_ass2/assignment_2/saved_models/saved_models/oxford/from_scratch_0.001/SegFormer_model_best.pth")
    model_2.net.load_state_dict(state_dict)
    model_2.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_2.to(device)
    with torch.no_grad():
        images = images.to(device)
        outputs = model_2(images)
        preds = torch.argmax(outputs, dim= 1)
    
    preds_plot = torchvision.utils.make_grid(preds.unsqueeze(1).float(), nrow=4)  # shape: [1, H, W]
    
    preds_plot = preds_plot / preds_plot.max()
    
    
    save_path = "oxford_images/results/images/pred_mask_test_pets_from_scratch_0.001.png"
    imshow(preds_plot, filename=save_path)

    model_3 = DeepSegmenter(SegFormer(num_classes = len(train_data.classes_seg)))
    state_dict = torch.load("C:/Users/aleks/milica/master_tuw/master_tuw/2_semester/dl_in_vc/DL_ass2/assignment_2/saved_models/saved_models/oxford/freeze_0.001/SegFormer_model_best.pth")
    model_3.net.load_state_dict(state_dict)
    model_3.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_3.to(device)
    with torch.no_grad():
        images = images.to(device)
        outputs = model_3(images)
        preds = torch.argmax(outputs, dim= 1)
    
    preds_plot = torchvision.utils.make_grid(preds.unsqueeze(1).float(), nrow=4)  # shape: [1, H, W]
    
    preds_plot = preds_plot / preds_plot.max()
    
    
    save_path = "oxford_images/results/images/pred_mask_test_pets_freeze_0.001.png"
    imshow(preds_plot, filename=save_path)


