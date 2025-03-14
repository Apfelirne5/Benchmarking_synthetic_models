# trex
# Copyright (C) 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

import argparse
import os
import sys
from time import time
from tqdm import tqdm

import data
import torch
from torch.utils.data import DataLoader

import utils
import numpy as np
import timm
from imagenet_c import corrupt
import cv2
import random


def main(args):
    """
    Main routine to extract features.
    """

    print("==> Initializing the pretrained stablerep-model")
        # load pre-trained model
    if os.path.isfile(args.ckpt_file_stablerep):
        print("=> loading checkpoint '{}'".format(args.ckpt_file_stablerep))
        checkpoint = torch.load(args.ckpt_file_stablerep, map_location=f"cuda:0")
        state_dict = checkpoint['model']

        prefix = 'visual.'
        for k in list(state_dict.keys()):
            if k.startswith(prefix) and not k.startswith(prefix + 'head'):
                state_dict[k[len('visual.'):]] = state_dict[k]
            del state_dict[k]
    else:
        raise Exception(f"No pre-trained model specified: {args.pretrained}")

    # create model
    model = timm.create_model(f"vit_base_patch16_224", num_classes=args.num_classes)
    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"head.weight", "head.bias"}

    for name, param in model.named_parameters():
        if name not in ['head.weight', 'head.bias']:
            param.requires_grad = False

    # delete the last fc layer, and instead add a bunch of classifiers
    del model.head

    model.cuda(0)
    model.eval()

    for corruption in [
        "gaussian_noise",  "shot_noise", "impulse_noise", "defocus_blur",
        "glass_blur",
        "zoom_blur", 
        "snow", 
        "frost", "fog",   # does not work
        "brightness",
        "contrast",
        "pixelate",
        "jpeg_compression",
        "speckle_noise", # sun397 run_1 until here
        "gaussian_blur",
        "spatter",
        "saturate",
        "motion_blur",
        "elastic_transform",
        "none"
    ]:
        for severity in [1, 2, 3, 4, 5]:
            # extract features from training and test sets
            # for split in ("trainval", "test"):
            for split in ["test"]:
                dataset = data.load_dataset(
                    args.dataset,
                    args.dataset_dir,
                    split,
                    args.dataset_image_size,
                    cog_levels_mapping_file=args.cog_levels_mapping_file,
                    cog_concepts_split_file=args.cog_concepts_split_file,
                )
                print(
                    "==> Extracting features from {} / {} (size: {})".format(
                        args.dataset, split, len(dataset)
                    )
                )
                print(" Data loading pipeline: {}".format(dataset.transform))

                # Corrupt images only for the test dataset
                if split == "test":
                    print("==> Corrupting images in the test dataset")

                    if corruption not in ["none"]:
                        corrupt_dataset(dataset, corruption_name=corruption, severity=severity)

                        dataset = data.load_dataset(
                            args.dataset,
                            args.dataset_dir_cc,
                            split,
                            args.dataset_image_size,
                            cog_levels_mapping_file=args.cog_levels_mapping_file,
                            cog_concepts_split_file=args.cog_concepts_split_file,
                        )
                    print(
                        "==> Extracting features from {} / {} (size: {})".format(
                            args.dataset, split, len(dataset)
                        )
                    )

                for model in [model]:
                    # X, Y = extract_features_loop(
                    #    model, dataset, args.batch_size, args.n_workers, args.device
                    # )
                    # if model == model_resnet:
                    #    features_file = os.path.join(args.output_dir_resnet, f"fv_{args.dataset}_{corruption}_{severity}", "features_{}.pth".format(split))
                    # elif model == model_sd:
                    #    features_file = os.path.join(args.output_dir_sd, f"fv_{args.dataset}_sd_{corruption}_{severity}", "features_{}.pth".format(split))
                    # Create parent directories if they don't exist
                    # os.makedirs(os.path.dirname(features_file), exist_ok=True)
                    # print(" Saving features under {}".format(features_file))
                    # torch.save({"X": X, "Y": Y}, features_file)

                    X, Y = extract_features_loop(
                        model, dataset, args.batch_size, args.n_workers, args.device
                    )
                    
                    base_dir = args.output_dir_stablerep

                    parent_dir = os.path.join(base_dir, f"fv_{args.dataset}_{corruption}_{severity}")

                    os.makedirs(parent_dir, exist_ok=True)

                    # Form the complete features_file path
                    features_file = os.path.join(parent_dir, f"features_{split}.pth")

                    print(" Saving features under {}".format(features_file))
                    torch.save({"X": X, "Y": Y}, features_file)


def corrupt_dataset(dataset, corruption_name, severity=1):
    corrupted_dataset = []
    print(corruption_name, "severity: ", severity)
    for idx, (image, label) in enumerate(tqdm(dataset, desc="Corrupting Dataset")):
        image_path = dataset.get_path(idx)  # Assuming a get_path method exists in your dataset class
        # print(f"Image {idx + 1}: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Skipping {image_path}. Unable to read image.")
            continue

        # Resize the image only when corruption_name is in the specified list
        if corruption_name in ["glass_blur", "zoom_blur", "snow", "frost", "fog"]:
            resized_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        else:
            resized_image = image

        # print("image_path", image_path)
        output_path = get_output_path(image_path)
        # print("output_path", output_path)

        # print(f"Image {idx + 1}: {output_path}")
        if resized_image is not None:
            if corruption_name not in ["none"]:
                corrupted_image = corrupt(resized_image, severity=severity, corruption_name=corruption_name)
            else:
                corrupted_image = resized_image
            cv2.imwrite(output_path, corrupted_image)
    return


def get_output_path(image_path):
    parts = image_path.split(os.sep)
    vilab_index = parts.index("vilab09")
    # vilab10_index = parts.index("transfer_datasets")
    next_directory = parts[vilab_index + 1]
    parts[vilab_index + 1] = f"{next_directory}_cc"
    output_path = os.path.join(*parts)

    # Ensure the output path starts with a slash
    if not output_path.startswith(os.sep):
        output_path = os.sep + output_path
    return output_path


def extract_features_loop(
        model, dataset, batch_size=128, n_workers=12, device="cuda", print_iter=50
):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=False,
    )

    # (feature, label) pairs to be stored under args.output_dir
    X = None
    Y = None
    six = 0  # sample index

    t_per_sample = utils.AverageMeter("time-per-sample")
    t0 = time()

    with torch.no_grad():
        for bix, batch in enumerate(dataloader):
            assert (
                    len(batch) == 2
            ), "Data loader should return a tuple of (image, label) every iteration."
            image, label = batch
            feature = model.forward_features(image.to(device))

            if X is None:
                print(
                    " Size of the first batch: {} and features {}".format(
                        list(image.shape), list(feature.shape)
                    ),
                    flush=True,
                )
                X = torch.zeros(
                    len(dataset), feature.size(1), dtype=torch.float32, device="cpu"
                )
                Y = torch.zeros(len(dataset), dtype=torch.long, device="cpu")

            bs = feature.size(0)
            X[six: six + bs] = feature.cpu()
            Y[six: six + bs] = label
            six += bs

            t1 = time()
            td = t1 - t0
            t_per_sample.update(td / bs, bs)
            t0 = t1

            if (bix % print_iter == 0) or (bix == len(dataloader) - 1):
                print(
                    " {:6d}/{:6d} extracted, {:5.3f} secs per sample, {:5.1f} mins remaining".format(
                        six,
                        len(dataset),
                        t_per_sample.avg,
                        (t_per_sample.avg / 60) * (len(dataset) - six),
                    ),
                    flush=True,
                )

    assert six == len(X)
    return X, Y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_file_stablerep",
        type=str,
        required=True,
        help="StableRep checkpoint file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dtd",
        choices=[
            "aircraft",
            "cars196",
            "dtd",
            "eurosat",
        ],
        help="From which datasets to extract features",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the dataset",
    )
    parser.add_argument(
        "--dataset_dir_cc",
        type=str,
        required=True,
        help="Path to the corrupted dataset",
    )
    parser.add_argument(
        "--dataset_image_size",
        type=int,
        default=224,
        help="Size of images given as input to the network before extracting features",
    )
    parser.add_argument(
        "--output_dir_stablerep",
        type=str,
        required=True,
        help="Where to extract features.",
    )
    parser.add_argument(
        "--cog_levels_mapping_file",
        type=str,
        help="Pickle file containing a list of concepts in each level (5 lists in total)."
             'This is an optional argument that needs to be set if --dataset is one of CoG levels, i.e., one of ("cog_l1", "cog_l2", "cog_l3", "cog_l4", "cog_l5")',
    )
    parser.add_argument(
        "--cog_concepts_split_file",
        type=str,
        help="Pickle file containing training and test splits for each concept in ImageNet-CoG."
             'This is an optional argument that needs to be set if --dataset is one of CoG levels, i.e., one of ("cog_l1", "cog_l2", "cog_l3", "cog_l4", "cog_l5")',
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size used during feature extraction",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=8,
        help="Number of workers run for the data loader",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Whether to use CUDA during feature extraction",
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=47,
        help="Num Classes dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=22,
        help="Seed for the random number generators",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.isfile(args.ckpt_file_stablerep):
        print(
            "Checkpoint file ({}) not found. "
            "Please provide a valid checkpoint file path for the pretrained model.".format(
                args.ckpt_file_stablerep
            )
        )
        sys.exit(-1)

    if not os.path.isdir(args.dataset_dir):
        print(
            "Dataset not found under {}. "
            "Please provide a valid dataset path".format(args.dataset_dir)
        )
        sys.exit(-1)

    if args.dataset in ["cog_l1", "cog_l2", "cog_l3", "cog_l4", "cog_l5"] and not (
            os.path.isfile(args.cog_levels_mapping_file)
            and os.path.isfile(args.cog_concepts_split_file)
    ):
        print(
            "ImageNet-CoG files are not found. "
            "Please check the <cog_levels_mapping_file> and <cog_concepts_split_file> arguments."
        )
        sys.exit(-1)

    if (not torch.cuda.is_available()) or (torch.cuda.device_count() == 0):
        print("No CUDA-compatible device found. " "We will only use CPU.")
        args.device = "cpu"

    os.makedirs(args.output_dir_stablerep, exist_ok=True)
    utils.print_program_info(args)

    main(args)
