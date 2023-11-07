import os, math, datetime, pkgutil
import numpy as np
import pybullet as p
from experiment.camera import Camera
from uuid import uuid4
import open3d as o3d
from experiment.pybullet_tools.utils import *
from experiment.pybullet_tools.panda_primitives import *
from experiment.pybullet_tools.ikfast.franka_panda.ik import PANDA_INFO, FRANKA_URDF
from experiment.pybullet_tools.ikfast.ikfast import get_ik_joints, either_inverse_kinematics, check_ik_solver
from experiment.utils import *
import yaml

# uois libraries. Ugly hack to import from sister directory
import uois.src.data_augmentation as data_augmentation
import uois.src.segmentation as segmentation
import uois.src.util.utilities as util_
import matplotlib.pyplot as plt
dsn_config = {
    # Sizes
    'feature_dim' : 64, # 32 would be normal

    # Mean Shift parameters (for 3D voting)
    'max_GMS_iters' : 10, 
    'epsilon' : 0.05, # Connected Components parameter
    'sigma' : 0.02, # Gaussian bandwidth parameter
    'num_seeds' : 200, # Used for MeanShift, but not BlurringMeanShift
    'subsample_factor' : 5,
    
    # Misc
    'min_pixels_thresh' : 500,
    'tau' : 15.,
}

rrn_config = {
    
    # Sizes
    'feature_dim' : 64, # 32 would be normal
    'img_H' : 224,
    'img_W' : 224,
    
    # architecture parameters
    'use_coordconv' : False,
}

uois3d_config = {
    
    # Padding for RGB Refinement Network
    'padding_percentage' : 0.25,
    
    # Open/Close Morphology for IMP (Initial Mask Processing) module
    'use_open_close_morphology' : True,
    'open_close_morphology_ksize' : 9,
    
    # Largest Connected Component for IMP module
    'use_largest_connected_component' : True,
}


HOME_JOINT_VALUES = [0.00, 0.074, 0.00, -1.113, 0.00, 1.510, 0.671, 0.04, 0.04]
HOME_POSE_GRIPPER = Pose(point=[0, 0, 1], euler=[0, math.radians(180), 0])
CONF_OPEN= [0.04, 0.04]
CONF_CLOSE = [0, 0]


class Environment:
    def __init__(self, vis=False):
        method = p.GUI if vis else p.DIRECT
        sim_id = p.connect(method)
        p.setTimeStep(1. / 240.)
        CLIENTS[sim_id] = True if vis else None
        add_data_path()
        p.setGravity(0, 0, -9.8)
        with LockRenderer():
            with HideOutput(True):
                plane = load_pybullet('experiment/resources/plane/plane.urdf', fixed_base=True)
                self.board = load_pybullet('experiment/resources/board.urdf', fixed_base=True)
                set_point(self.board, [0.5, 0.5, 0.005/2])
                set_color(self.board, GREY)
                tray = load_pybullet('experiment/resources/tray/traybox.urdf', fixed_base=True)
                set_point(tray, [0.5, -0.5, 0.02/2])
               
                self.gripper = load_pybullet('experiment/resources/robotiq_2f_140/urdf/ur5_robotiq_140.urdf', fixed_base=True)
                set_pose(self.gripper, HOME_POSE_GRIPPER)
                numJoints = p.getNumJoints(self.gripper)
                jointInfo = namedtuple('jointInfo', 
                    ['id','name','type','damping','friction','lowerLimit','upperLimit','maxForce','maxVelocity','controllable'])
                self.joints = []
                self.controllable_joints = []
                for i in range(numJoints):
                    info = p.getJointInfo(self.gripper, i)
                    jointID = info[0]
                    jointName = info[1].decode("utf-8")
                    jointType = info[2]  # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
                    jointDamping = info[6]
                    jointFriction = info[7]
                    jointLowerLimit = info[8]
                    jointUpperLimit = info[9]
                    jointMaxForce = info[10]
                    jointMaxVelocity = info[11]
                    controllable = (jointType != p.JOINT_FIXED)
                    if controllable:
                        self.controllable_joints.append(jointID)
                        p.setJointMotorControl2(self.gripper, jointID, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
                    info = jointInfo(jointID,jointName,jointType,jointDamping,jointFriction,jointLowerLimit,
                                    jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
                    self.joints.append(info)
                mimic_parent_name = 'finger_joint'
                mimic_children_names = {'right_outer_knuckle_joint': -1,
                                        'left_inner_knuckle_joint': -1,
                                        'right_inner_knuckle_joint': -1,
                                        'left_inner_finger_joint': 1,
                                        'right_inner_finger_joint': 1}
                self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0]
                self.mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in self.joints if joint.name in mimic_children_names}
                for joint_id, multiplier in self.mimic_child_multiplier.items():
                    c = p.createConstraint(self.gripper, self.mimic_parent_id,
                                        self.gripper, joint_id,
                                        jointType=p.JOINT_GEAR,
                                        jointAxis=[0, 1, 0],
                                        parentFramePosition=[0, 0, 0],
                                        childFramePosition=[0, 0, 0])
                    p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)  # Note: the mysterious `erp` is of EXTREME importance

                assign_link_colors(self.gripper, max_colors=3, s=0.5, v=1.)
                draw_pose(unit_pose(), parent=self.gripper, parent_link=link_from_name(self.gripper, 'tcp'), length=0.04, width=3)
                # floor_from_camera = Pose(point=[0, 0.75, 1], euler=[-math.radians(145), 0, math.radians(180)])
                board_from_camera = Pose(point=[0, 0.65, 1], euler=[-math.radians(150), 0, math.radians(180)])
                world_from_board = get_pose(self.board)
                self.world_from_camera = multiply(world_from_board, board_from_camera)
                self.camera_from_ground_normal = (tform_from_pose(invert(self.world_from_camera)) @ np.array([0, 0, 1, 0]))[:3]
                self.camera_from_ground_normal /= np.linalg.norm(self.camera_from_ground_normal)
                self.camera = Camera(self.world_from_camera)
                
                # rgb, depth, seg = self.camera.render()
                
                self.fixed = [plane, tray]
        self.workspace = np.asarray([[0.1, 0.9], 
                                     [0.1, 0.9]])
        self.aabb_workspace = aabb_from_extent_center([0.8, 0.8, 0.1], 
                                                      [0.5, 0.5, 0.005/2])
        # self.finger_joints = joints_from_names(self.gripper, ["panda_finger_joint1", "panda_finger_joint2"])
        # self.finger_links = links_from_names(self.gripper, ['panda_leftfinger', 'panda_rightfinger'])
        self.grasp_from_gripper = Pose(point=Point(-0.2097, 0, 0), euler=Euler(0, 1.57079632679489660, 0))
        
        # YCB & OneBillion
        urdf_dir = "experiment/resources/objects"
        with open(f'{urdf_dir}/config.yml','r') as ff:
            cfg = yaml.safe_load(ff)
        self.urdf_files = []
        for obj_name in cfg['load_obj']:
            self.urdf_files.append(os.path.join(urdf_dir, "pybullet_ycb", obj_name, "model.urdf"))

        # OneBillion
        # urdf_dir = "data/Benchmark/graspnet/models"
        # with open(f'{urdf_dir}/config.yml','r') as ff:
        #     cfg = yaml.safe_load(ff)
        # self.urdf_files = []
        # for obj_name in cfg['load_obj']:
        #     self.urdf_files.append(os.path.join(urdf_dir, obj_name, "model.urdf"))
        
        self.mesh_ids = []
        self.obj_id_to_name = {}
        self.mesh_to_urdf = {}

        # 加载 uois 
        checkpoint_dir = 'uois/checkpoints/' # TODO: change this to directory of downloaded models
        dsn_filename = checkpoint_dir + 'DepthSeedingNetwork_3D_TOD_checkpoint.pth'
        rrn_filename = checkpoint_dir + 'RRN_OID_checkpoint.pth'
        uois3d_config['final_close_morphology'] = 'TableTop_v5' in rrn_filename
        self.uois_net_3d = segmentation.UOISNet3D(uois3d_config, dsn_filename, dsn_config, rrn_filename, rrn_config)

    def seed(self, seed=None):
        self._random = np.random.RandomState(seed)
        return seed

    def reset(self, seed = None, options = None):
        set_pose(self.gripper, HOME_POSE_GRIPPER)
        self.clean_objects()
        self.add_objects()
        while self.sim_until_stable() == False:
            self.clean_objects()
            self.add_objects()
        pcd_scene, pcd_obj_inds, o3d_scene, o3d_obj, seg_obj = self.get_observation()
        
        return pcd_scene, pcd_obj_inds, o3d_scene, o3d_obj, seg_obj, {}

    def step(self, gg):
        num_colli_grasp = 0
        num_unstable_grasp = 0
        num_attem = 0
        grasp_success = False
        num_attempt = {}

        for i in range(0, min(len(gg), 50)):
            num_attem += 1
            grasp = gg[i]
            t = grasp.translation
            r = grasp.rotation_matrix
            depth = grasp.depth
            self.grasp_width = grasp.width
            targeted_obj = grasp.object_id
            if targeted_obj not in num_attempt:
                num_attempt[targeted_obj] = 0
            else:
                num_attempt[targeted_obj] += 1

            self.open_ee()
            grasp = np.eye(4)
            grasp[:3, :3] = r
            grasp[:3, 3] = (r @ np.array([depth, 0, 0])) + t
            camera_from_grasp = pose_from_tform(grasp)
            world_from_grasp = multiply(self.world_from_camera, camera_from_grasp)
            world_from_gripper = multiply(world_from_grasp, self.grasp_from_gripper)
            set_pose(self.gripper, world_from_gripper)

            if any(pairwise_collision(self.gripper, b) for b in (self.fixed+self.mesh_ids)):
                grasp_success = False
                num_colli_grasp += 1
                set_pose(self.gripper, HOME_POSE_GRIPPER)
                self.open_ee()
                if num_attempt[targeted_obj] > 0:
                    set_point(targeted_obj, [0.5, -0.5, 0.5])
                    break
                continue
            else:
                saved_world = WorldSaver()
                j = 0
                last_grasp = world_from_gripper
                while j < 7:
                    saved_world.restore()
                    j += 1
                    last_grasp = world_from_gripper
                    try_depth = depth + 0.005 * j
                    try_grasp = np.eye(4)
                    try_grasp[:3, :3] = r
                    # try_grasp[:3, 3] = (r @ np.array([try_depth, 0, 0])) + t
                    try_grasp[:3, 3] = (r @ np.array([try_depth+0.0231, 0, 0])) + t
                    camera_from_grasp = pose_from_tform(try_grasp)
                    world_from_grasp = multiply(self.world_from_camera, camera_from_grasp)
                    world_from_gripper = multiply(world_from_grasp, self.grasp_from_gripper)
                    set_pose(self.gripper, world_from_gripper)

                    if any(pairwise_collision(self.gripper, b) for b in (self.fixed+self.mesh_ids)) == True:
                        break

                    try_grasp[:3, 3] = (r @ np.array([try_depth, 0, 0])) + t
                    camera_from_grasp = pose_from_tform(try_grasp)
                    world_from_grasp = multiply(self.world_from_camera, camera_from_grasp)
                    world_from_gripper = multiply(world_from_grasp, self.grasp_from_gripper)
                    set_pose(self.gripper, world_from_gripper)
                    
                    width_before_close = get_joint_position(self.gripper, self.mimic_parent_id)
                    self.close_ee()
                    width_after_close = get_joint_position(self.gripper, self.mimic_parent_id)
                    if abs(width_before_close - width_after_close) < 0.07:
                        break
                saved_world.restore()
                set_pose(self.gripper, last_grasp)
                saved_world = WorldSaver()
                self.close_ee()
                # grasped_obj = self.get_grasped_obj()
                world_from_gobj = get_pose(targeted_obj)
                gripper_from_world = invert(world_from_gripper)
                gripper_from_gobj = multiply(gripper_from_world, world_from_gobj)
                set_point(self.gripper, [0.5, 0.5, 0.5])
                world_from_gripper = get_pose(self.gripper)
                world_from_gobj = multiply(world_from_gripper, gripper_from_gobj)
                set_pose(targeted_obj, world_from_gobj)
                self.sim_until_stable()
                grasp_success = self.is_grasp_success()
                if grasp_success == True:
                    # print(f"grasp {self.obj_id_to_name[targeted_obj]} success: 第 {i} 个. {gg[i].score} 分.")
                    set_point(targeted_obj, [0.5, -0.5, 0.5])
                else:
                    saved_world.restore()
                    num_unstable_grasp += 1
                    if num_attempt[targeted_obj] > 0:
                        set_point(targeted_obj, [0.5, -0.5, 0.5])
                        
            set_pose(self.gripper, HOME_POSE_GRIPPER)
            self.open_ee()
            self.sim_until_stable()
            if grasp_success == True or num_attempt[targeted_obj] > 0:
                break
            
        pcd_scene, pcd_obj_inds, o3d_scene, o3d_obj, seg_obj = self.get_observation()
        terminated = not self.exist_obj_in_workspace()
        info = {"is_success": grasp_success, "num_attem": num_attem, 
                "num_colli_grasp": num_colli_grasp, "num_unstable_grasp": num_unstable_grasp,
                }
    
        return pcd_scene, pcd_obj_inds, o3d_scene, o3d_obj, seg_obj, terminated, info

    def close(self):
        
        disconnect()
        # p.unloadPlugin(self.plugin)

    def get_observation(self):
        rgb, depth, seg_gt = self.camera.render()
        # 结构化点云
        pts_scene = depth2xyzmap(depth, self.camera.k)
        
        # 将 rgb 和 点云 送进 uois
        rgb_imgs = np.zeros((1, 480, 640, 3), dtype=np.float32) # 图像像素 640*480
        xyz_imgs = np.zeros((1, 480, 640, 3), dtype=np.float32)

        rgb_imgs[0] = data_augmentation.standardize_image(rgb)
        xyz_imgs[0] = pts_scene

        batch = {
            'rgb' : data_augmentation.array_to_tensor(rgb_imgs),
            'xyz' : data_augmentation.array_to_tensor(xyz_imgs),
        }
        _, _, _, seg_masks = self.uois_net_3d.run_on_batch(batch)

        # Get results in numpy
        seg = seg_masks[0].cpu().numpy()

        num_objs = np.unique(seg).max() + 1
        seg_plot = util_.get_color_mask(seg, nc=num_objs)
        images = [rgb, depth, seg_plot]
        titles = ['Image', 'Depth', 'Segmentation']
        util_.subplotter(images, titles)
        plt.show()

        bg_mask = depth < 0.1
        # for id in (self.fixed+[self.gripper]): # , self.board
        #     bg_mask[seg==id] = 1
        seg = seg[bg_mask==False]
        seg_gt = seg_gt[bg_mask==False]
        pts_scene = pts_scene[bg_mask==False]
        rgb = rgb[bg_mask==False]

        # 下采样场景点云
        num_scene_pts = 20000
        if len(pts_scene) >= num_scene_pts:
            select_scene_index = np.random.choice(len(pts_scene), num_scene_pts, replace=False)
        else:
            idxs1 = np.arange(len(pts_scene))
            idxs2 = np.random.choice(len(pts_scene), num_scene_pts-len(pts_scene), replace=True)
            select_scene_index = np.concatenate([idxs1, idxs2], axis=0)
        # (num_scene_pts, 3)
        pts_scene = pts_scene[select_scene_index]
        seg = seg[select_scene_index]
        seg_gt = seg_gt[select_scene_index]
        rgb = rgb[select_scene_index]

        # 采样物体点
        pcd_obj_inds = np.argwhere(seg>0).squeeze() # (N_obj,)
        len_pcd_obj_inds = len(pcd_obj_inds)
        num_obj_pts = 2048
        if len_pcd_obj_inds >= num_obj_pts:
            select_obj_index = np.random.choice(len_pcd_obj_inds, num_obj_pts, replace=False)
            pcd_obj_inds = pcd_obj_inds[select_obj_index]
        elif len_pcd_obj_inds > 0:
            idxs1 = np.arange(len_pcd_obj_inds)
            idxs2 = np.random.choice(len_pcd_obj_inds, num_obj_pts-len_pcd_obj_inds, replace=True)
            select_obj_index = np.concatenate([idxs1, idxs2], axis=0)
            pcd_obj_inds = pcd_obj_inds[select_obj_index]
        # (1024, )
        seg_predict_grasp_point = seg_gt[pcd_obj_inds]
        for seg_id in range(len(seg_predict_grasp_point)):
            if seg_predict_grasp_point[seg_id] in (self.fixed+[self.gripper, self.board]):
                seg_predict_grasp_point[seg_id] = 0
        
        # 可视化
        o3d_scene = toOpen3dCloud(pts_scene, rgb)
        o3d.io.write_point_cloud("pcd_scene.ply", o3d_scene)

        # 物体点法向估计
        if len_pcd_obj_inds == 0:
            o3d_obj = None
        else:
            pcd_obj = pts_scene[pcd_obj_inds]
            rgb = rgb[pcd_obj_inds]
            o3d_obj = toOpen3dCloud(pcd_obj, rgb)
            o3d.io.write_point_cloud("pcd_obj.ply", o3d_obj)
            radius = 0.01  # 搜索半径
            max_nn = 30  # 邻域内用于估算法线的最大点数
            o3d_obj.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
            o3d.visualization.draw_geometries([o3d_obj], window_name="法线估计",
                                            point_show_normal=True,
                                            width=800,  # 窗口宽度
                                            height=600)  # 窗口高度

        return pts_scene, pcd_obj_inds, o3d_scene, o3d_obj, seg_predict_grasp_point
    
    def add_objects(self):
        for urdf_path in self.urdf_files:
            drop_x = (self.workspace[0][1] - self.workspace[0][0] - 0.2) * np.random.random_sample() + self.workspace[0][0] + 0.1
            drop_y = (self.workspace[1][1] - self.workspace[1][0] - 0.2) * np.random.random_sample() + self.workspace[1][0] + 0.1
            object_position = [drop_x, drop_y, 0.01]
            # object_pose = Pose(object_position)
            # object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
            object_orientation = [0, 0, 2*np.pi*np.random.random_sample()]
            object_pose = Pose(object_position, object_orientation)
            flags = p.URDF_USE_INERTIA_FROM_FILE
            obj_name = urdf_path.split('/')[-2]
            obj_id = p.loadURDF(urdf_path, basePosition=object_pose[0], baseOrientation=object_pose[1], flags=flags)
            draw_pose(unit_pose(), parent=obj_id, length=0.04, width=3)
            self.mesh_ids.append(obj_id)
            self.obj_id_to_name[obj_id] = obj_name
        
    def sim_until_stable(self):
        for i in range(200):
            last_pos = {}
            accum_motions = {}
        
            for body_id in self.mesh_ids:
                last_pos[body_id] = np.array(get_point(body_id))
                accum_motions[body_id] = 0

            stabled = True
            for _ in range(50):
                p.stepSimulation()
                # time.sleep(1. / 240.)
                for body_id in self.mesh_ids:
                    cur_pos =  np.array(get_point(body_id))
                    motion = np.linalg.norm(cur_pos - last_pos[body_id])
                    accum_motions[body_id] += motion
                    last_pos[body_id] = cur_pos.copy()
                    if accum_motions[body_id]>=0.001:
                        stabled = False
                        break
                if stabled==False:
                    break
        
            if stabled:
                for body_id in self.mesh_ids:
                    p.resetBaseVelocity(body_id, linearVelocity=[0,0,0], angularVelocity=[0,0,0])
                return True
        return False
                

    def clean_objects(self):
        
        for ob_id in self.mesh_ids:
            p.removeBody(ob_id)
        self.mesh_ids.clear()

    def exist_obj_in_workspace(self):
        
        bodies_in_workspace = np.array(get_bodies_in_region(self.aabb_workspace))[:, 0]
        bodies_in_workspace = list(set(bodies_in_workspace).difference(set(self.fixed + [self.board])))
        if len(bodies_in_workspace) > 0:
            return True
        else:
            return False
        
    def get_grasped_obj(self):
        
        for ob_id in self.mesh_ids:
            # if any_link_pair_collision(self.gripper, self.finger_links, ob_id) == True:
            if body_collision(self.gripper, ob_id) == True:
                return ob_id

    def move_finger(self, open_length):
        open_length = np.clip(open_length, 0, 0.085)
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)  # angle calculation
        # Control the mimic gripper joint(s)
        p.setJointMotorControl2(self.gripper, self.mimic_parent_id, p.POSITION_CONTROL, 
                                targetPosition=open_angle,
                                # force=self.joints[self.mimic_parent_id].maxForce, 
                                force=20)
                                # maxVelocity=self.joints[self.mimic_parent_id].maxVelocity)
        for _ in range(50):
            p.stepSimulation()
            # time.sleep(1. / 240.)

    def close_ee(self):
        self.move_finger(0)
        
        # p.setJointMotorControlArray(self.gripper, jointIndices=self.finger_joints, controlMode=p.POSITION_CONTROL,
        #                             targetPositions=CONF_CLOSE, forces=np.ones(2, dtype=float) * 40)
        # for _ in range(50):
        #     p.stepSimulation()
        #     time.sleep(1. / 240.)
        
        # for _ in joint_controller_hold(self.robot, [joint_from_name(self.robot, "panda_finger_joint1"), joint_from_name(self.robot, "panda_finger_joint2")], self.ee_close_values, timeout=(50*DEFAULT_TIME_STEP)):
        # for _ in joint_controller_hold(self.gripper, self.finger_joints, CONF_CLOSE, timeout=(50 * DEFAULT_TIME_STEP)):
        #     step_simulation()

    def open_ee(self):
        self.move_finger(self.grasp_width)

        # p.setJointMotorControlArray(self.gripper, jointIndices=self.finger_joints, controlMode=p.POSITION_CONTROL,
        #                             targetPositions=CONF_OPEN, forces=np.ones(2, dtype=float) * 100)
        # for _ in range(50):
        #     p.stepSimulation()
        #     time.sleep(1. / 240.)
        
        # for _ in joint_controller_hold(self.robot, ["panda_finger_joint1", "panda_finger_joint2"], self.ee_open_values, timeout=(50*DEFAULT_TIME_STEP)):
        # for _ in joint_controller_hold(self.gripper, self.finger_joints, CONF_OPEN, timeout=(50 * DEFAULT_TIME_STEP)):
        #     step_simulation()

    def is_grasp_success(self):

        # finger_joint_pos = np.array(get_joint_positions(self.robot, self.finger_joints))
        # finger_joint_pos = np.array(get_joint_positions(self.gripper, self.finger_joints))
        if get_joint_position(self.gripper, self.mimic_parent_id) >= 0.69:
        # if np.all(finger_joint_pos < 0.001):
            return False
        return True
    
    def load_mesh(self, mesh_file, mesh_pose, mass, vhacd_file, scale=np.ones(3),has_collision=True,useFixedBase=False,concave=False,collision_margin=0.0001):

        if mesh_file in self.mesh_to_urdf:
            urdf_dir = self.mesh_to_urdf[mesh_file]
        else:
            urdf_dir = f'/tmp/{os.path.basename(mesh_file)}_{uuid4()}.urdf'
            create_urdf_from_mesh(mesh_file, out_dir=urdf_dir, mass=mass, vhacd_dir=vhacd_file, has_collision=has_collision, concave=concave, scale=scale)
            self.mesh_to_urdf[mesh_file] = urdf_dir

        obj_id = p.loadURDF(urdf_dir, useFixedBase=useFixedBase, basePosition=mesh_pose[0], baseOrientation=mesh_pose[1])
        return obj_id