# CUDA_VISIBLE_DEVICES=1 python train.py --camera realsense --log_dir logs/log_rs --batch_size 2 --dataset_root data/Benchmark/graspnet

CUDA_VISIBLE_DEVICES=1 python train.py --camera realsense --log_dir logs/log_rs_spotr --batch_size 2 --dataset_root data/Benchmark/graspnet

# CUDA_VISIBLE_DEVICES=1 python train.py --camera kinect --log_dir logs/log_kn --batch_size 2 --dataset_root /data/Benchmark/graspnet
