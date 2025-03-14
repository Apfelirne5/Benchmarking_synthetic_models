from collections import OrderedDict
import os

from . import utils


def load_classname_label_mapping(dataset_dir) -> dict:
    """
    Loads the class name -> class label mapping dictionary for the Facet_class1 dataset.
    """
    mapping_file = "{}/class1_split/variants.txt".format(dataset_dir)
    mapping = OrderedDict()
    with open(mapping_file, "r") as fid:
        for ix, line in enumerate(fid):
            mapping[line.strip().lower()] = ix

    assert len(mapping) == 52
    return mapping


def load_split(
    dataset_dir: str, split: str, transform, type: str, class1: str, classname_label_mapping: dict = None
) -> utils.TransferDataset:
    """
    Loads a split of the Facet_class1 dataset.
    """
    assert split in ("train", "val", "trainval", "test")
    images_dir = "{}/imgs_bbox_all_new_faces".format(dataset_dir)

    #gender = "all"
    #gender = "masc"
    #gender = "fem"
    #gender = "non-binary"
    #gender = "na"


    if split!="test":
        split_file = "{}/class1_split/images_variant_{}.txt".format(dataset_dir, split)
    if split=="test":
        if type in ["masc", "fem", "na", "non-binary"]:
            if class1 != "all":
                split_file = "{}/class1_split/test_class1_split_gender/images_variant_{}_{}.txt".format(dataset_dir, type, class1)
            if class1 == "all":
                split_file = "{}/class1_split/test_class1_split_gender/images_variant_{}.txt".format(dataset_dir, type)
        elif type == "1-5" or "6-10":
            if class1 != "all":
                split_file = "{}/class1_split/test_class1_split_skintone/images_variant_{}_{}.txt".format(dataset_dir, type, class1)
            elif class1 == "all":
                split_file = "{}/class1_split/test_class1_split_skintone/images_variant_{}.txt".format(dataset_dir, type)

        #all
        #if gender=="all":
            #split_file = "{}/class1_split/images_variant_{}.txt".format(dataset_dir, split)
        #masc
        #if gender=="masc":
            #split_file = "{}/class1_split/images_variant_{}_masc.txt".format(dataset_dir, split)
        #fem
        #if gender=="fem":    
            #split_file = "{}/class1_split/images_variant_{}_fem.txt".format(dataset_dir, split)
        #non-binary
        #if gender=="non-binary":    
            #split_file = "{}/class1_split/images_variant_{}_non-binary.txt".format(dataset_dir, split)
        #na
        #if gender=="na":
            #split_file = "{}/class1_split/images_variant_{}_na.txt".format(dataset_dir, split)
    
    samples = []

    if classname_label_mapping is None:
        classname_label_mapping = load_classname_label_mapping(dataset_dir)

    with open(split_file, "r") as fid:
        for line in fid:
            line = line.strip()
            #image_name = line.split(" ")[0]
            image_name, class_name = line.split("\t")
            #assert len(image_name) == 7
            #image_path = os.path.join(images_dir, f"{image_name}.jpg")
            image_path = "{}/{}.jpg".format(images_dir, image_name)
            class_label = classname_label_mapping[class_name.lower()]
            samples.append((image_path, class_label))

            #class_name = line.replace(image_name, "").strip().lower()
            #class_label = classname_label_mapping[class_name]
            #samples.append((image_path, class_label))

    # Count number of rows in the text file
    with open(split_file, 'r') as f:
        num_rows = sum(1 for line in f)

    n_samples = {"train": 22293, "val": 2477, "trainval": 24770, "test": num_rows}[split]

    #all
    #if gender=="all":
        #n_samples = {"train": 10359, "val": 1152, "trainval": 11511, "test": 1279}[split]
    
    #masc
    #if gender=="masc":
    #    n_samples = {"train": 10359, "val": 1152, "trainval": 11511, "test": 852}[split]

    #fem
    #if gender=="fem":            
    #    n_samples = {"train": 10359, "val": 1152, "trainval": 11511, "test": 344}[split]

    #if gender=="non-binary": 
    #    n_samples = {"train": 10359, "val": 1152, "trainval": 11511, "test": 4}[split]

    #if gender=="na":
    #    n_samples = {"train": 10359, "val": 1152, "trainval": 11511, "test": 70}[split]

    assert (
        len(samples) == n_samples
    ), "Loaded {} samples for the {} split (should have been {})".format(
        len(samples), split, n_samples
    )

    return utils.TransferDataset(samples, transform, classname_label_mapping)
