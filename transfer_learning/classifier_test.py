import os
import pickle
import shutil
import sys
from typing import List
import datetime
import time
import random

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from classifier_training import _prepare_features
from logreg_trainer import LogregSklearnTrainer
from sklearn.metrics import confusion_matrix


def main(args):

    data_dict = _prepare_features(args)

    print(args.dataset)

    train_features = data_dict["trainval"][0]
    train_labels = data_dict["trainval"][1]
    test_features = data_dict["test"][0]
    test_labels = data_dict["test"][1]

    #print("test_features: ", type(test_features))
    #print("test_labels: ", type(test_labels))

    print("train_features: ", train_features.shape)
    print("train_labels: ", train_labels.shape)

    print("test_features: ", test_features.shape)
    print("test_labels: ", test_labels.shape)

    train_features = train_features.detach().numpy()
    train_labels = train_labels.detach().numpy()
    test_features = test_features.detach().numpy()
    test_labels = test_labels.detach().numpy()

    #print("test_features: ", type(test_features))
    #print("test_labels: ", type(test_labels))

    print("test_features: ", test_features.shape)
    print("test_labels: ", test_labels.shape)

    clf = utils.load_pickle(f"{args.classifier_dir}/classifier.pth")
    print(clf)

    preds = clf.predict(test_features)

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
        "--clf_type",
        type=str,
        default="logreg_sklearn",
        choices=["logreg_sklearn", "logreg_torch"],
        help="Type of linear classifier to train on top of features",
    )
    parser.add_argument(
        "--dataset_per_val",
        type=float,
        default=0.2,
        help="Percentage of the val set, sampled from the trainval set for hyper-parameter tuning",
    )
    # For the L-BFGS-based logistic regression trainer implemented in scikit-learn
    parser.add_argument(
        "--clf_C",
        type=float,
        help="""Inverse regularization strength for sklearn.linear_model.LogisticRegression.
        Note that this variable is determined by Optuna, so do not set it manually""",
    )
    parser.add_argument(
        "--clf_C_min",
        type=float,
        default=1e-5,
        help="Power of the minimum C parameter sampled by Optuna",
    )
    parser.add_argument(
        "--clf_C_max",
        type=float,
        default=1e6,
        help="Power of the maximum C parameter sampled by Optuna",
    )
    parser.add_argument(
        "--clf_max_iter",
        type=int,
        default=2000,
        help="Maximum number of iterations to run the classifier for sklearn.linear_model.LogisticRegression during the hyper-parameter tuning stage.",
    )
    # For the SGD-based logistic regression trainer implemented in PyTorch
    parser.add_argument(
        "--clf_lr",
        type=float,
        help="""Learning rate.
        Note that this variable is determined by Optuna, so do not set it manually""",
    )
    parser.add_argument(
        "--clf_lr_min",
        type=float,
        default=1e-1,
        help="Power of the minimum lr parameter sampled by Optuna",
    )
    parser.add_argument(
        "--clf_lr_max",
        type=float,
        default=1e2,
        help="Power of the maximum lr parameter sampled by Optuna",
    )
    parser.add_argument(
        "--clf_wd",
        type=float,
        help="""Weight decay.
        Note that this variable is determined by Optuna, do not set it manually""",
    )
    parser.add_argument(
        "--clf_wd_min",
        type=float,
        default=1e-12,
        help="Power of the minimum weight decay parameter sampled by Optuna",
    )
    parser.add_argument(
        "--clf_wd_max",
        type=float,
        default=1e-4,
        help="Power of the maximum weight decay parameter sampled by Optuna",
    )
    parser.add_argument(
        "--clf_mom",
        type=float,
        default=0.9,
        help="SGD momentum. We do not tune this variable.",
    )
    parser.add_argument(
        "--clf_epochs",
        type=int,
        default=100,
        help="""Number of epochs to train the linear classifier.
        We do not tune this variable""",
    )
    parser.add_argument(
        "--clf_batch_size",
        type=int,
        default=1024,
        help="""Batch size for SGD.
        We do not tune this variable""",
    )
    # Common for all trainers
    parser.add_argument(
        "--n_sklearn_workers",
        type=int,
        default=-1,
        help="Number of CPU cores to use in Scikit-learn jobs. -1 means to use all available cores.",
    )
    parser.add_argument(
        "--n_optuna_workers",
        type=int,
        default=1,
        help="Number of concurrent Optuna jobs",
    )
    parser.add_argument(
        "--n_optuna_trials",
        type=int,
        default=30,
        help="Number of trials run by Optuna",
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
    #parser.add_argument(
    #    "--output_dir",
    #    type=str,
    #    default="./linear-classifier-output",
    #    help="Whether to save the logs",
    #)
    # The following arguments are needed if features have not been extracted yet.
    parser.add_argument(
        "--arch",
        type=str,
        choices=["resnet50"],
        help="""The architecture of the pre-trained model""",
    )
    parser.add_argument(
        "--ckpt_file",
        type=str,
        default="",
        help="Model checkpoint file",
    )
    parser.add_argument(
        "--ckpt_key",
        type=str,
        default="",
        help="Key in the checkpoint dictionary that corresponds to the model state_dict",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="in1k",
        choices=[
            "in1k",
            "cog_l1",
            "cog_l2",
            "cog_l3",
            "cog_l4",
            "cog_l5",
            "aircraft",
            "cars196",
            "dtd",
            "eurosat",
            "flowers",
            "pets",
            "food101",
            "sun397",
            "inat2018",
            "inat2019",
        ],
        help="From which datasets to extract features",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="Path to the dataset",
    )
    parser.add_argument(
        "--dataset_image_size",
        type=int,
        default=224,
        help="Size of images given as input to the network before extracting features",
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
        "--classifier_dir",
        type=str,
        help="relativ path directory"
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

    #os.makedirs(args.output_dir, exist_ok=True)

    main(args)