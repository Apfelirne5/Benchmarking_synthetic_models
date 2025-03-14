import os
from typing import List
import datetime
import time
import random
import csv

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import data
from sklearn.metrics import confusion_matrix
    
def _prepare_features(args):

    # load or extract features from the trainval and test splits
    data_dict = {}

    for split in ["test"]:

        features_file = os.path.join(args.features_dir, "features_{}.pth".format(split))
        print("ff: ", features_file)
        if os.path.isfile(features_file):
            print("==> Loading pre-extracted features from {}".format(features_file))
            features_dict = torch.load(features_file, "cpu")
            X, Y = features_dict["X"], features_dict["Y"]

        else:
            print(
                "==> No pre-extracted features found, extracting them under {}".format(
                    features_file
                ))

        data_dict[split] = [X, Y]

    # # split the trainval into train and val
    # data_dict["train"], data_dict["val"] = data.utils.split_trainval(
    #     data_dict["trainval"][0], data_dict["trainval"][1], per_val=args.dataset_per_val
    # )

    return data_dict

class AllClassifiers(nn.Module):
    def __init__(self, classifiers_dict):
        super().__init__()
        self.classifiers_dict = nn.ModuleDict()
        self.classifiers_dict.update(classifiers_dict)

    def forward(self, inputs):
        return {k: v.forward(inputs) for k, v in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)


def create_linear_classifiers(feat_dim, num_classes, learning_rates, use_bn=False):
    linear_classifier_dict = nn.ModuleDict()

    for blr in learning_rates:

        linear_classifier = nn.Linear(feat_dim, num_classes)
        linear_classifier.weight.data.normal_(mean=0.0, std=0.01)
        linear_classifier.bias.data.zero_()
        if use_bn:
            linear_classifier = nn.Sequential(
                torch.nn.SyncBatchNorm(feat_dim, affine=False, eps=1e-6),
                linear_classifier
            )
        linear_classifier.cuda()

        name = f"{blr:.4f}".replace('.', '_')
        linear_classifier_dict[f"classifier_lr_{name}"] = linear_classifier

    # add to ddp mode
    linear_classifiers = AllClassifiers(linear_classifier_dict)
    return linear_classifiers

def load_model(args):

    checkpoint = torch.load(args.classifier_ckpt)

    print(checkpoint.keys())
    print(checkpoint["acc1"])
    print(checkpoint["epoch"])

    lcs = create_linear_classifiers(768, args.num_classes, args.base_lrs)

    sd = checkpoint["linear_classifiers"]

    for k in list(sd.keys()):
        sd[k.replace("module.", "")] = sd[k]
        del sd[k]

    lcs.load_state_dict(sd)

    name = f"{args.classifier_lr:.4f}".replace('.', '_')
    
    return lcs.classifiers_dict[f"classifier_lr_{name}"]

    # return model

def main(args):
    features_dir_all = args.features_dir

    csv_file_path = f"results/results_stablerep_{args.dataset}.csv"

    print("csv_file_path: ", csv_file_path)

    features_dir_list = get_features_dir_list(args)

    features_dir_list = [f for f in features_dir_list if args.dataset in f]

    print("features_dir_list: ", features_dir_list)

    acc_1_list = []
    m_ap_list = []

    model = load_model(args).to("cpu")

    for features_dir in features_dir_list:

        args.features_dir = os.path.join(features_dir_all, features_dir)
        print("args.features_dir: ", args.features_dir)

        data_dict = _prepare_features(args)

        print(args.dataset)

        test_features = data_dict["test"][0]
        test_labels = data_dict["test"][1]

        # print("test_features: ", type(test_features))
        # print("test_labels: ", type(test_labels))

        print("test_features: ", test_features.shape)
        print("test_labels: ", test_labels.shape)

        print("checksum: ", np.sum(test_features.detach().numpy()))
        preds = model(test_features)

        print(preds.shape)

        preds = torch.argmax(preds, dim=1).cpu().numpy()
        test_labels = test_labels.cpu().numpy()

        # compute accuracy
        acc_1 = np.mean(np.equal(test_labels, preds).astype(np.float32)) * 100.0
        confmat = confusion_matrix(test_labels, preds)
        m_ap = (np.diag(confmat) / confmat.sum(axis=1)).mean() * 100.0

        t_study_0 = time.time()
        t_study_1 = time.time()
        t_study = str(datetime.timedelta(seconds=int(t_study_1 - t_study_0)))
        print(
            " - Top-1 acc: {:3.1f}, m-AP: {:3.1f}, Runtime: {}".format(
                acc_1, m_ap, t_study
            ),
            flush=True,
        )

        acc_1_list.append(round(acc_1, 1))
        m_ap_list.append(round(m_ap, 1))

        print(f"Features_dir: {features_dir}, Top-1 acc: {acc_1:.1f}, m-AP: {m_ap:.1f}, Runtime: {t_study}\n",
              flush=True)

    # Append the results to the CSV file
    write_results_to_csv(csv_file_path, features_dir_list),
    write_results_to_csv(csv_file_path, acc_1_list),
    write_results_to_csv(csv_file_path, m_ap_list)


def get_features_dir_list(args):
    features_dir_list = [d for d in os.listdir(args.features_dir) if os.path.isdir(os.path.join(args.features_dir, d))]
    sorted_features_dir_list = sorted(features_dir_list)
    return sorted_features_dir_list


def write_results_to_csv(csv_file_path, result_row):
    csv.register_dialect('custom', delimiter=';')

    # Append the result row to the CSV file
    with open(csv_file_path, mode='a', newline='') as f:
        writer = csv.writer(f, dialect='custom')
        writer.writerow(result_row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features_dir",
        type=str,
        default="",
        help="Directory to include features_trainval.pth and features_test.pth",
    )
    parser.add_argument(
        "--features_norm",
        type=str,
        default="none",
        choices=["standard", "l2", "none"],
        help="Normalization applied to features before the classifier",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=22,
        help="Seed for the random number generators",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Whether to use CUDA during feature extraction and classifier training",
    )
    parser.add_argument(
        "--classifier_ckpt",
        type=str,
        default="",
        help="Model checkpoint file",
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
        "--ftext_batch_size",
        type=int,
        default=128,
        help="Batch size used during feature extraction",
    )
    parser.add_argument(
        "--ftext_n_workers",
        type=int,
        default=8,
        help="Number of workers run for the data loader",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=47,
        help="Num Classes dataset",
    )
    parser.add_argument(
        "--classifier-lr",
        type=float,
        default=0.005,
        help="lr of the classifier to pick",
    )
    parser.add_argument(
        "--base_lrs", 
        default=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5],
        type=float, nargs='+'
    )
    parser.add_argument(
        "--dataset_per_val",
        type=float,
        default=0.2,
        help="Percentage of the val set, sampled from the trainval set for hyper-parameter tuning",
    )

    args = parser.parse_args()
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        args.device = "cuda"
    else:
        args.device = "cpu"
    utils.print_program_info(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # os.makedirs(args.output_dir, exist_ok=True)

    main(args)