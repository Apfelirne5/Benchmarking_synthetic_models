# StableRep evaluation

Files for evaluation of StableRep against common corruptions. Based on /transferlearning-folder from this repo. 

The generated features are not checked in because of storage issues.

The linear weights should be the best linear weights from linear probing (best learning rate is passed as parameter).

## Usage

For feature generation with corruptions run the following command:

```commandline
python feature_extract_stablerep.py --ckpt_file_stablerep model/cc12m_3x.pth --dataset cars196 --dataset_dir /path/to/dataset/cars/ --dataset_dir_cc /path/to/store/dataset/with_corruptions/cars_cc/ --output_dir_stablerep features_stablerep --num-classes 196
```

For evaluation run the follwing command:

```commandline
python classifier_test_loop_stablerep.py --features_dir features_stablerep/ --classifier_ckpt linear_weights/cars.pt --dataset cars196 --num-classes 196 --classifier-lr 0.002
```