# StableRep linear probing

Files for linear probing on transferlearning datasets for StableRep. Based on: https://github.com/google-research/syn-rep-learn/tree/main/StableRep
For detailed information have a look on the original repo. This only contains the additional code written in the internship.

Experiments have been completed for Aircraft, Cars196 and DTD dataset.

## Linear probing

For linear probing, run the following command on a single node:
```commandline
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="your_host" --master_port=12345 \
  main_linear_{dataset_name}.py --model base --data /path/to/dataset \
  --pretrained /weights/cc12m_3x.pth \
  --output-dir /path/to/linear_save \
  --log-dir /path/to/tensorboard_folder
```
You can simply append `--use_bn` to turn on the extra BatchNorm (without affine transform)
layer for the linear classifiers.

