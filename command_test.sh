# CUDA_VISIBLE_DEVICES=0 python test.py --dump_dir logs/dump_rs/2309070916 --checkpoint_path logs/log_rs/checkpoint.tar --camera realsense --dataset_root data/Benchmark/graspnet

python test.py --dump_dir logs/dump_rs_spotr/202310062321_epoch18 \
--checkpoint_path logs/log_rs_spotr/202310062321_encode_bg_infer_only_obj/checkpoint_epoch_18.tar \
--camera realsense --dataset_root data/Benchmark/graspnet

# CUDA_VISIBLE_DEVICES=0 python test.py --dump_dir logs/dump_kn --checkpoint_path logs/log_kn/checkpoint.tar --camera kinect --dataset_root /data/Benchmark/graspnet
