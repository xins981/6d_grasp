import numpy as np

accu = np.load("logs/dump_rs_spotr/202310051619_encode_bg_infer_only_obj/ap_realsense.npy")
accu_seen = accu[:30, :, -1, [1, 3]]
accu_similiar = accu[30:60, :, -1, [1, 3]]
accu_unseen = accu[60:, :, -1, [1, 3]]

print("seen")
print("mean: ", np.mean(accu_seen))
print("0.8: ", np.mean(accu_seen[:, :, 1]))
print("0.4: ", np.mean(accu_seen[:, :, 0]))
print("--------------------------------------------------")

print("similiar")
print("mean: ", np.mean(accu_similiar))
print("0.8: ", np.mean(accu_similiar[:, :, 1]))
print("0.4: ", np.mean(accu_similiar[:, :, 0]))
print("--------------------------------------------------")

print("novel")
print("mean: ", np.mean(accu_unseen))
print("0.8: ", np.mean(accu_unseen[:, :, 1]))
print("0.4: ", np.mean(accu_unseen[:, :, 0]))


# loaded_data = np.load('point_and_color_data.npz')
# xyz_data = loaded_data['xyz']
# xyz_color_data = loaded_data['xyz_color']
# print(xyz_data.shape)