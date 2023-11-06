# CUDA_VISIBLE_DEVICES=0 python test.py --dump_dir logs/dump_rs/202309121651 --checkpoint_path logs/log_rs/202309100944/checkpoint.tar --camera realsense --dataset_root data/Benchmark/graspnet

# CUDA_VISIBLE_DEVICES=1 python test.py --dump_dir logs/dump_rs/202310051223_rm_bg --checkpoint_path logs/log_rs/202310051223_rm_bg/checkpoint.tar --camera realsense --dataset_root data/Benchmark/graspnet

python test.py --dump_dir logs/dump_rs/202309100944 \
--checkpoint_path logs/log_rs/202309100944/checkpoint.tar \
--camera realsense --dataset_root data/Benchmark/graspnet

# CUDA_VISIBLE_DEVICES=0 python test.py --dump_dir logs/dump_rs_spotr/202309101756 --checkpoint_path logs/log_rs_spotr/202309071803/checkpoint.tar --camera realsense --dataset_root data/Benchmark/graspnet

# CUDA_VISIBLE_DEVICES=0 python test.py --dump_dir logs/dump_kn --checkpoint_path logs/log_kn/checkpoint.tar --camera kinect --dataset_root /data/Benchmark/graspnet
