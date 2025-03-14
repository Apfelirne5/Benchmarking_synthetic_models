echo "Running DTD corruptions"
python feature_extract_cc_2m.py --ckpt_file_stablerep model/cc12m_3x.pth --dataset dtd --dataset_dir /fastdata/vilab09/dtd/ --dataset_dir_cc /fastdata/vilab09/dtd_cc/ --output_dir_stablerep features_stablerep --num-classes 47
echo "Finished DTD corruptions"
echo "Running Aircraft corruptions"
python feature_extract_cc_2m.py --ckpt_file_stablerep model/cc12m_3x.pth --dataset aircraft --dataset_dir /fastdata/vilab09/aircraft/ --dataset_dir_cc /fastdata/vilab09/aircraft_cc/ --output_dir_stablerep features_stablerep --num-classes 100
echo "Finished Aircraft corruptions"