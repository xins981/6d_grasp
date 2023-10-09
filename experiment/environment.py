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

# egl = pkgutil.get_loader('eglRenderer')

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
                set_point(self.board, [0.5, 0.5, 0.01/2])
                set_color(self.board, GREY)
                tray = load_pybullet('experiment/resources/tray/traybox.urdf', fixed_base=True)
                set_point(tray, [0.5, -0.5, 0.02/2])
               
                # self.gripper = load_pybullet('experiment/resources/franka_description/robots/hand.urdf', fixed_base=True)
                self.gripper = load_pybullet('experiment/resources/robotiq_2f_140/urdf/robotiq_2f_140.urdf', fixed_base=True)
                set_pose(self.gripper, HOME_POSE_GRIPPER)
                # assign_link_colors(self.gripper, max_colors=3, s=0.5, v=1.)
                # set_configuration(self.gripper, CONF_OPEN)
                # draw_pose(unit_pose(), parent=self.gripper, parent_link=link_from_name(self.gripper, 'panda_tcp'), length=0.04, width=3)
                # floor_from_camera = Pose(point=[0, 0.75, 1], euler=[-math.radians(145), 0, math.radians(180)])
                floor_from_camera = Pose(point=[0, 0.65, 1], euler=[-math.radians(150), 0, math.radians(180)])
                world_from_floor = get_pose(self.board)
                self.world_from_camera = multiply(world_from_floor, floor_from_camera)
                self.camera = Camera(self.world_from_camera)
                self.fixed = [plane, tray]
        self.workspace = np.asarray([[0.2, 0.8], 
                                     [0.2, 0.8]])
        self.aabb_workspace = aabb_from_extent_center([0.6, 0.6, 0.3], 
                                                      [0.5, 0.5, 0.01+(0.3/2)])
        self.finger_joints = joints_from_names(self.gripper, ["panda_finger_joint1", "panda_finger_joint2"])
        self.finger_links = links_from_names(self.gripper, ['panda_leftfinger', 'panda_rightfinger'])
        self.grasp_from_gripper = Pose(point=Point(-0.1, 0, 0), euler=Euler(0, 1.57079632679489660, 0))
        
        urdf_dir = "experiment/resources/objects/ycb"
        with open(f'{urdf_dir}/config.yml','r') as ff:
            cfg = yaml.safe_load(ff)
        self.urdf_files = []
        for obj_name in cfg['load_obj']:
            self.urdf_files.append(os.path.join(urdf_dir, obj_name,"model.urdf"))
        
        self.mesh_ids = []
        self.mesh_to_urdf = {}

    def seed(self, seed=None):
        
        self._random = np.random.RandomState(seed)
        return seed

    def reset(self, seed = None, options = None):
        # set_configuration(self.robot, HOME_JOINT_VALUES)
        set_pose(self.gripper, HOME_POSE_GRIPPER)
        set_configuration(self.gripper, CONF_OPEN)
        self.clean_objects()
        self.add_objects()
        while self.sim_until_stable() == False:
            self.clean_objects()
            self.add_objects()
        pcd_scene, pcd_obj_inds, o3d_scene = self.get_observation()
        
        return pcd_scene, pcd_obj_inds, o3d_scene, {}

    def step(self, gg):
        
        num_colli_grasp = 0
        num_unstable_grasp = 0
        num_attem = 0
        grasp_success = False

        for i in range(0, min(len(gg), 50)):
            num_attem += 1
            grasp = gg[i]
            t = grasp.translation
            r = grasp.rotation_matrix
            depth = grasp.depth
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
                continue
            else:
                saved_world = WorldSaver()
                self.close_ee()
                grasped_obj = self.get_grasped_obj()
                if grasped_obj != None:
                    world_from_gobj = get_pose(grasped_obj)
                    gripper_from_world = invert(world_from_gripper)
                    gripper_from_gobj = multiply(gripper_from_world, world_from_gobj)
                    set_point(self.gripper, [0.5, 0.5, 0.5])
                    world_from_gripper = get_pose(self.gripper)
                    world_from_gobj = multiply(world_from_gripper, gripper_from_gobj)
                    set_pose(grasped_obj, world_from_gobj)
                    self.sim_until_stable()
                    grasp_success = self.is_grasp_success()
                    if grasp_success == True:
                        set_point(grasped_obj, [0.5, -0.5, 0.5])
                    else:
                        saved_world.restore()
                        num_unstable_grasp += 1
                else:
                    saved_world.restore()
                    grasp_success = False
                    num_unstable_grasp += 1
            set_pose(self.gripper, HOME_POSE_GRIPPER)
            self.open_ee()
            self.sim_until_stable()
            if grasp_success == True:
                break
            
        pcd_scene, pcd_obj_inds, o3d_scene = self.get_observation()
        terminated = not self.exist_obj_in_workspace()
        info = {"is_success": grasp_success, "num_attem": num_attem, 'num_colli_grasp': num_colli_grasp, "num_unstable_grasp": num_unstable_grasp}
    
        return pcd_scene, pcd_obj_inds, o3d_scene, terminated, info

    def close(self):
        
        disconnect()
        # p.unloadPlugin(self.plugin)

    def get_observation(self):
        rgb, depth, seg = self.camera.render()
        pts_scene = depth2xyzmap(depth, self.camera.k)
        bg_mask = depth < 0.1
        for id in (self.fixed+[self.gripper]): # , self.board
            bg_mask[seg==id] = 1
        seg = seg[bg_mask==False]
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
        rgb = rgb[select_scene_index]

        # 采样物体点
        pcd_obj_inds = np.argwhere(seg!=self.board).squeeze() # (N_obj,)
        len_pcd_obj_inds = len(pcd_obj_inds)
        num_obj_pts = 1024
        if len_pcd_obj_inds >= num_obj_pts:
            select_obj_index = np.random.choice(len_pcd_obj_inds, num_obj_pts, replace=False)
            pcd_obj_inds = pcd_obj_inds[select_obj_index]
        elif len_pcd_obj_inds > 0:
            idxs1 = np.arange(len_pcd_obj_inds)
            idxs2 = np.random.choice(len_pcd_obj_inds, num_obj_pts-len_pcd_obj_inds, replace=True)
            select_obj_index = np.concatenate([idxs1, idxs2], axis=0)
            pcd_obj_inds = pcd_obj_inds[select_obj_index]
        else:
            pcd_obj_inds = None
        
        # 调试
        # (num_obj_pts, 3)
        # pcd_obj = pts_scene[pcd_obj_inds]
        # cloud = o3d.geometry.PointCloud()
        # cloud.points = o3d.utility.Vector3dVector(pts_scene)
        # o3d.io.write_point_cloud("pcd_scene.ply", cloud)
        # cloud.points = o3d.utility.Vector3dVector(pcd_obj)
        # o3d.io.write_point_cloud("pcd_obj.ply", cloud)

        # 可视化
        o3d_scene = toOpen3dCloud(pts_scene, rgb)

        return pts_scene, pcd_obj_inds, o3d_scene
    
    def add_objects(self):
        for urdf_path in self.urdf_files:
            # drop_x = (self.workspace[0][1] - self.workspace[0][0] - 0.4) * np.random.random_sample() + self.workspace[0][0] + 0.2
            # drop_y = (self.workspace[1][1] - self.workspace[1][0] - 0.4) * np.random.random_sample() + self.workspace[1][0] + 0.2
            # object_position = [drop_x, drop_y, 0.4]
            drop_x = (self.workspace[0][1] - self.workspace[0][0] - 0.3) * np.random.random_sample() + self.workspace[0][0] + 0.15
            drop_y = (self.workspace[1][1] - self.workspace[1][0] - 0.3) * np.random.random_sample() + self.workspace[1][0] + 0.15
            object_position = [drop_x, drop_y, 0.2]
            object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
            object_pose = Pose(object_position, object_orientation)
            # vhacd_path = obj_path.replace('.obj', '_vhacd.obj')
            # obj_id = self.load_mesh(mesh_file=obj_path, mesh_pose=object_pose, mass=0.1, vhacd_file=vhacd_path, scale=[1, 1, 1])
            flags = p.URDF_USE_INERTIA_FROM_FILE
            obj_id = p.loadURDF(urdf_path, basePosition=object_pose[0], baseOrientation=object_pose[1], flags=flags)
            p.changeDynamics(obj_id, -1, lateralFriction=0.5, collisionMargin=0.0001)
            self.mesh_ids.append(obj_id)
    
    def sim_until_stable(self):
        while True:
            last_pos = {}
            accum_motions = {}
        
            for body_id in self.mesh_ids:
                last_pos[body_id] = np.array(get_point(body_id))
                accum_motions[body_id] = 0

            stabled = True
            for _ in range(50):
                p.stepSimulation()
                time.sleep(1. / 240.)
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
                break

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
            # if body_collision(self.robot, ob_id) == False:
            if any_link_pair_collision(self.gripper, self.finger_links, ob_id) == True:
                return ob_id

    def close_ee(self):
        
        p.setJointMotorControlArray(self.gripper, jointIndices=self.finger_joints, controlMode=p.POSITION_CONTROL,
                                    targetPositions=CONF_CLOSE, forces=np.ones(2, dtype=float) * 40)
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1. / 240.)
        
        # for _ in joint_controller_hold(self.robot, [joint_from_name(self.robot, "panda_finger_joint1"), joint_from_name(self.robot, "panda_finger_joint2")], self.ee_close_values, timeout=(50*DEFAULT_TIME_STEP)):
        # for _ in joint_controller_hold(self.gripper, self.finger_joints, CONF_CLOSE, timeout=(50 * DEFAULT_TIME_STEP)):
        #     step_simulation()

    def open_ee(self):

        p.setJointMotorControlArray(self.gripper, jointIndices=self.finger_joints, controlMode=p.POSITION_CONTROL,
                                    targetPositions=CONF_OPEN, forces=np.ones(2, dtype=float) * 100)
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1. / 240.)
        
        # for _ in joint_controller_hold(self.robot, ["panda_finger_joint1", "panda_finger_joint2"], self.ee_open_values, timeout=(50*DEFAULT_TIME_STEP)):
        # for _ in joint_controller_hold(self.gripper, self.finger_joints, CONF_OPEN, timeout=(50 * DEFAULT_TIME_STEP)):
        #     step_simulation()

    def is_grasp_success(self):

        # finger_joint_pos = np.array(get_joint_positions(self.robot, self.finger_joints))
        finger_joint_pos = np.array(get_joint_positions(self.gripper, self.finger_joints))
        if np.all(finger_joint_pos < 0.001):
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