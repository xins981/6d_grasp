import numpy as np
import torch
from experiment.environment import Environment
from models.graspnet import GraspNet, pred_decode
from graspnetAPI import GraspGroup



# Init the model
net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4, cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
# Load checkpoint
checkpoint = torch.load("logs/log_rs/checkpoint.tar")
net.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch']
print(f"-> loaded checkpoint logs/log_rs/checkpoint.tar (epoch: {start_epoch})")
net.eval()

env = Environment(vis=True)
terminated = False
num_attem = 0
num_success = 0
num_colli = 0
num_unstable = 0
num_scene = 0
num_objects = 5
end_points = {}

observation, info = env.reset()
observation = np.expand_dims(observation, axis=0)
observation = torch.Tensor(observation).to(device)
end_points.update({"point_clouds": observation})
while num_scene < 15 and terminated == False:
    num_scene += 1
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    preds = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(preds)
    print(f"6d grasp: {len(gg)}")
    nms_gg = gg.nms(translation_thresh = 0.1, rotation_thresh = 30 / 180.0 * 3.1416)
    print(f"grasp after nms: {len(nms_gg)}")
    nms_gg = nms_gg.sort_by_score()
    observation, terminated, info = env.step(nms_gg)
    observation = np.expand_dims(observation, axis=0)
    observation = torch.Tensor(observation).to(device)
    end_points.update({"point_clouds": observation})
    num_attem += info["num_attem"]
    num_colli += info["num_colli_grasp"]
    num_unstable += info["num_unstable_grasp"]
    if info["is_success"] == True:
        num_success += 1

env.close()

complete_rate = num_success / num_objects
success_rate = num_success / (num_attem - num_colli)
colli_rate = num_colli / num_attem
unstable_rate = num_unstable / (num_attem - num_colli)


print(f"num_objects: {num_objects}, num_attem: {num_attem}, num_colli: {num_colli}, num_success: {num_success},  num_unstable: {num_unstable}")
print(f"Complete rate: {complete_rate:.2f}")
print(f"Success rate: {success_rate:.2f}")
print(f"Collision rate: {colli_rate:.2f}")
print(f"Unstable rate: {unstable_rate:.2f}")


