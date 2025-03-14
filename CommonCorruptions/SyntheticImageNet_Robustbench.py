import numpy as np 
import torch as th
from torchvision.models import resnet50
from torchvision.models import convnext_base, ConvNeXt_Base_Weights, convnext_small, ConvNeXt_Small_Weights
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights, convnext_large, ConvNeXt_Large_Weights
import torchvision.transforms as transforms
from robustbench.data import load_imagenet3dcc, load_imagenetc
from robustbench.utils import clean_accuracy
from settings import Settings, parse_arguments
from MAE.mae.models_vit import vit_base_patch16, vit_large_patch16, vit_huge_patch14

def main():

    settings = Settings(parse_arguments()).args
    model_name = settings.model
    data = settings.dataset
    print(f"selected Model: {model_name} with corruptions to test: {data}")
    PREPROCESSINGS = {  "ConvNeXt_Tiny": transforms.Compose([
                                    transforms.Resize((236,236), interpolation=transforms.InterpolationMode.BILINEAR),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
                        "ConvNeXt_Small": transforms.Compose([
                                    transforms.Resize((230,230), interpolation=transforms.InterpolationMode.BILINEAR),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
                        "Conv_232_Norm": transforms.Compose([
                                    transforms.Resize((232,232), interpolation=transforms.InterpolationMode.BILINEAR),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
                        "ViT_256_Norm":  transforms.Compose([
                                    transforms.Resize((256,256), interpolation=transforms.InterpolationMode.BICUBIC),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
            
                        "FakeIt":   transforms.Compose([
                                    transforms.Resize(224, interpolation=transforms.InterpolationMode("bicubic")),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
                        "ResNet50":  transforms.Compose([
                                    transforms.Resize((256,256), interpolation=transforms.InterpolationMode.BILINEAR),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), 
                        }
    if model_name == "FakeIt":
        #load the model with params from the Paper[2]
        ckpt = th.load("Models/Synthetic/imagenet_1k_sd.pth", "cpu")
        net = resnet50()
        net.fc = th.nn.Linear(2048, 1000, bias=False)  # change 1000 to 100 for "imagenet_100_sd.pth"
        msg = net.load_state_dict(ckpt, strict=True)
        prep = model_name
        #check if loading worked
        print(msg)
    elif model_name =="ResNet50":
        net = resnet50(pretrained=True)
        prep = model_name
    elif model_name == "ConvNeXt_Base":
        net = convnext_base(ConvNeXt_Base_Weights.IMAGENET1K_V1)
        prep = "Conv_232_Norm"
        #preprocessing values from the ConvNeXt documentation for the base model 
        
    elif model_name == "ConvNeXt_Tiny":
        net = convnext_tiny(ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        #preprocessing values from the ConvNeXt documentation for the base model 
        prep = model_name
    elif model_name == "ConvNeXt_Large":
        net = convnext_large(ConvNeXt_Large_Weights.IMAGENET1K_V1)
        #preprocessing values from the ConvNeXt documentation for the base model 
        prep = "Conv_232_Norm"
    elif model_name == "ConvNeXt_Small":
        net = convnext_small(ConvNeXt_Small_Weights.IMAGENET1K_V1)
        prep = model_name
        #preprocessing values from the ConvNeXt documentation for the base model 
    elif model_name == "DinoV2_Small":
        net = th.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc')
        prep = "ViT_256_Norm"
    elif model_name == "DinoV2_Base":
        net = th.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc')
        prep = "ViT_256_Norm"
    elif model_name == "DinoV2_Large":
        net = th.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_lc')
        prep = "ViT_256_Norm"
    elif model_name == "DinoV2_G":
        net = th.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_lc')
        prep = "ViT_256_Norm"
    print(f"Model {model_name} sucessfully loaded")
    Preprocessing = PREPROCESSINGS[prep]

    device = th.device("cuda:1")
    #run 3D common corruptions 
    if data == "3DCC":

        corruptions_3dcc = ['near_focus', 'far_focus', 'bit_error', 'color_quant', 
                            'flash', 'fog_3d',
                            'iso_noise', 'low_light', 'xy_motion_blur', 'z_motion_blur'] # 10 corruptions in ImageNet-3DCC The 2 compression corruptions have been thrown out because of large size and storage issues (might change later)
        corruptions_to_test =['near_focus', 'far_focus', 'xy_motion_blur', 'z_motion_blur', 
                             'fog_3d', 'low_light','flash','iso_noise',
                             'bit_error', 'color_quant'
                                ]
        
        model = net.to(device)
        model.eval()
        for corruption in corruptions_to_test:
            for s in [1, 2, 3, 4, 5]:  # 5 severity levels
                x_test, y_test = load_imagenet3dcc(n_examples=5000, corruptions=[corruption], severity=s, data_dir=".", prepr=Preprocessing)
                acc = clean_accuracy(model, x_test.to(device), y_test.to(device), device=device,batch_size=64)
                print(f'Model: {model_name}, ImageNet-3DCC corruption: {corruption} severity: {s} accuracy: {acc:.1%}')
    if data == "2DCC":

        corruptions_2dcc = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
                            "frost", "snow", "fog", "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression", 
                            "speckle_noise", "spatter", "gaussian_blur", "saturate"]
        corruptions_to_test = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
                                    "frost", "snow", "fog", "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression", ]
                                    # currently missing the extra.tar with speckle_noise spatte gaussian_blur and saturate
        missing_corruptions = ["speckle_noise", "spatter", "gaussian_blur", "saturate"]
        model = net.to(device)
        model.eval()
        for corruption in corruptions_2dcc:
            for s in [1, 2, 3, 4, 5]:  # 5 severity levels
                x_test, y_test = load_imagenetc(n_examples=5000, corruptions=[corruption], severity=s, data_dir=".", prepr=Preprocessing)
                acc = clean_accuracy(model, x_test.to(device), y_test.to(device), device=device,batch_size=64)
                print(f'Model: {model_name}, ImageNet-C corruption: {corruption} severity: {s} accuracy: {acc:.1%}')
    
if __name__=="__main__":
    main()