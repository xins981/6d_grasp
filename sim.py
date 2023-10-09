import numpy as np
import torch
from experiment.environment import Environment
from models.graspnet import GraspNet, pred_decode
from graspnetAPI import GraspGroup
import open3d as o3d

# Init the model
net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4, cylinder_radius=0.05, 
               hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
# Load checkpoint
checkpoint_dir = "logs/log_rs_spotr/202310062321_encode_bg_infer_only_obj/checkpoint_epoch_13.tar"
checkpoint = torch.load(checkpoint_dir)
net.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch']
print(f"-> loaded checkpoint {checkpoint_dir} (epoch: {start_epoch})")
net.eval()

env = Environment(vis=True)

for i in range(3):
    terminated = False
    num_attem = 0
    num_success = 0
    num_colli = 0
    num_unstable = 0
    num_scene = 0
    num_objects = 6
    end_points = {}

    pcd_scene, pcd_obj_inds, o3d_scene, info = env.reset()
    pcd_scene = np.expand_dims(pcd_scene, axis=0)
    pcd_scene = torch.Tensor(pcd_scene).to(device)
    pcd_obj_inds = np.expand_dims(pcd_obj_inds, axis=0)
    pcd_obj_inds = torch.Tensor(pcd_obj_inds).to(device)
    end_points.update({"point_clouds": pcd_scene})
    end_points.update({"pcd_obj_inds": pcd_obj_inds})
    while num_scene < 15 and terminated == False:
        num_scene += 1
        with torch.no_grad():
            end_points = net(end_points)
            grasp_preds = pred_decode(end_points)
        preds = grasp_preds[0].detach().cpu().numpy() # 0 batch id

        # grasp_pts = preds[:, 13:16]
        # pcd_grasp_pts = toOpen3dCloud(grasp_pts)
        # o3d.io.write_point_cloud("grasp_pts.ply", pcd_grasp_pts)

        gg = GraspGroup(preds)
        nms_gg = gg.nms(translation_thresh = 0.1, rotation_thresh = 30 / 180.0 * 3.1416)
        print(f"scene: {num_scene}. grasp nms rate: {len(nms_gg)} / {len(gg)}")
        
        # grippers = nms_gg.to_open3d_geometry_list()
        # if len(grippers) > 50:
        #     grippers = grippers[:50]
        # o3d.visualization.draw_geometries([o3d_scene, *grippers])
        
        nms_gg = nms_gg.sort_by_score()
        pcd_scene, pcd_obj_inds, o3d_scene, terminated, info = env.step(nms_gg)
        num_attem += info["num_attem"]
        num_colli += info["num_colli_grasp"]
        num_unstable += info["num_unstable_grasp"]
        if info["is_success"] == True:
            num_success += 1

        if terminated == True:
            break
        
        pcd_scene = np.expand_dims(pcd_scene, axis=0)
        pcd_scene = torch.Tensor(pcd_scene).to(device)
        pcd_obj_inds = np.expand_dims(pcd_obj_inds, axis=0)
        pcd_obj_inds = torch.Tensor(pcd_obj_inds).to(device)
        end_points.update({"point_clouds": pcd_scene})
        end_points.update({"pcd_obj_inds": pcd_obj_inds})

    eps = np.finfo(np.float32).eps
    complete_rate = num_success / num_objects
    success_rate = num_success / (num_attem - num_colli + eps)
    colli_rate = num_colli / num_attem
    unstable_rate = num_unstable / (num_attem - num_colli + eps)

    print(f"num_objects: {num_objects}, num_attem: {num_attem}, num_colli: {num_colli}, num_success: {num_success},  num_unstable: {num_unstable}")
    print(f"Complete rate: {complete_rate:.2f}")
    print(f"Success rate: {success_rate:.2f}")
    print(f"Collision rate: {colli_rate:.2f}")
    print(f"Unstable rate: {unstable_rate:.2f}")

env.close()



