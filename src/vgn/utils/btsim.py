import os
import time
import pickle
import numpy as np
import pybullet
from pybullet_utils import bullet_client

from vgn.utils.transform import Rotation, Transform
from vgn.utils.saver import get_mesh_pose_dict_from_world

assert pybullet.isNumpyEnabled(), "Pybullet needs to be built with NumPy"


class BtWorld(object):
    """Interface to a PyBullet physics server.

    Attributes:
        dt: Time step of the physics simulation.
        rtf: Real time factor. If negative, the simulation is run as fast as possible.
        sim_time: Virtual time elpased since the last simulation reset.
    """

    def __init__(self, gui=True, save_dir=None, save_freq=8, egl_mode=False):
        if egl_mode:
            # 直接使用带EGL的DIRECT模式
            self.p = bullet_client.BulletClient(pybullet.DIRECT)
            
            # 先加载EGL插件
            plugin_status = self.p.loadPlugin("eglRendererPlugin")
            print(f"EGL渲染器加载状态: {plugin_status}")
            
            # 设置渲染参数
            if plugin_status >= 0:
                print("EGL渲染器加载成功")
                # 可能需要按需配置其他EGL相关参数
        else:
            # 原来的连接方式
            connection_mode = pybullet.GUI if gui else pybullet.DIRECT
            self.p = bullet_client.BulletClient(connection_mode)

        self.gui = gui
        self.egl_mode = egl_mode  # 保存EGL模式状态，用于后续逻辑
        self.dt = 1.0 / 240.0
        self.solver_iterations = 150
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.sim_step = 0

        self.reset()

    def set_gravity(self, gravity):
        self.p.setGravity(*gravity)

    def load_urdf(self, urdf_path, pose, scale=1.0):
        body = Body.from_urdf(self.p, urdf_path, pose, scale)
        self.bodies[body.uid] = body
        return body

    def remove_body(self, body):
        self.p.removeBody(body.uid)
        del self.bodies[body.uid]

    def add_constraint(self, *argv, **kwargs):
        """See `Constraint` below."""
        constraint = Constraint(self.p, *argv, **kwargs)
        return constraint

    def add_camera(self, intrinsic, near, far):
        camera = Camera(self.p, intrinsic, near, far)
        return camera
    
    def get_contacts_valid(self, bodyA, tgt_id):
    #   def get_contacts(self, bodyA):
        points = self.p.getContactPoints(bodyA.uid)
        contacts = []
        # assert
        contact_object = []
        for point in points:
            contact_object.append(point[2])
            # assert point[2] == tgt_id
            contact = Contact(
                bodyA=self.bodies[point[1]],
                bodyB=self.bodies[point[2]],
                point=point[5],
                normal=point[7],
                depth=point[8],
                force=point[9],
            )
            contacts.append(contact)
        # if len(contacts) > 0:
        #     assert tgt_id in contact_object
        return contacts

    def get_contacts(self, bodyA):
        points = self.p.getContactPoints(bodyA.uid)
        contacts = []
        for point in points:
            contact = Contact(
                bodyA=self.bodies[point[1]],
                bodyB=self.bodies[point[2]],
                point=point[5],
                normal=point[7],
                depth=point[8],
                force=point[9],
            )
            contacts.append(contact)
        return contacts

    def reset(self):
        self.p.resetSimulation()
        self.p.setPhysicsEngineParameter(
            fixedTimeStep=self.dt, numSolverIterations=self.solver_iterations
        )
        self.bodies = {}
        self.sim_time = 0.0

    def step(self):
        self.p.stepSimulation()
        
        
        if self.gui:
            time.sleep(self.dt)
        if self.save_dir:
            if self.sim_step % self.save_freq == 0:
                mesh_pose_dict = get_mesh_pose_dict_from_world(self, self.p._client)
                with open(os.path.join(self.save_dir, f'{self.sim_step:08d}.pkl'), 'wb') as f:
                    pickle.dump(mesh_pose_dict, f)

        self.sim_time += self.dt
        self.sim_step += 1

    def save_state(self):
        return self.p.saveState()

    def restore_state(self, state_uid):
        self.p.restoreState(stateId=state_uid)

    def close(self):
        self.p.disconnect()

    def start_video_recording(self, filename, video_path):
        """开始视频录制，使用OpenCV手动录制视频"""
        import cv2
        import numpy as np
        
        os.makedirs(video_path, exist_ok=True)
        video_file = os.path.join(video_path, f"{filename}.mp4")
        
        # 降低视频分辨率以提高速度
        self.width, self.height = 640, 480  # 降低分辨率
        self.frames = []
        self.video_file = video_file
        self.recording = True
        self.frame_count = 0  # 帧计数器，用于跳过部分帧
        self.capture_interval = 3  # 每隔几步捕获一帧
        
        print(f"开始OpenCV视频录制: {video_file} (优化模式)")
        return 1  # 返回虚拟ID
    
    def capture_frame(self):
        """在模拟步进时捕获帧，使用帧间隔减少捕获频率"""
        if not hasattr(self, 'recording') or not self.recording:
            return
        
        # 使用间隔采样减少帧数量    
        self.frame_count += 1
        if self.frame_count % self.capture_interval != 0:
            return
            
        # 设置相机视角
        viewMatrix = self.p.computeViewMatrix(
            cameraEyePosition=[0.5, -0.2, 0.5],
            cameraTargetPosition=[0.15, 0.15, 0.15],
            cameraUpVector=[0, 0, 1]
        )
        projectionMatrix = self.p.computeProjectionMatrixFOV(
            fov=60, aspect=self.width/self.height, nearVal=0.1, farVal=100
        )
        
        # 获取图像
        img = self.p.getCameraImage(
            self.width, self.height, viewMatrix, projectionMatrix,
            renderer=self.p.ER_TINY_RENDERER  # 使用软件渲染器
        )
        
        # 保存图像
        import numpy as np
        rgb_array = np.array(img[2], dtype=np.uint8).reshape(self.height, self.width, 4)
        rgb_array = rgb_array[:, :, :3]  # 移除alpha通道
        self.frames.append(rgb_array)
    
    def stop_video_recording(self, log_id):
        """结束视频录制并生成视频文件，使用更高效的编码器"""
        if not hasattr(self, 'recording') or not self.recording:
            print("没有正在进行的录制")
            return
            
        import cv2
        
        if len(self.frames) == 0:
            print("警告：没有捕获到帧，视频为空")
            return
            
        # 创建视频 - 使用H.264编码
        try:
            # 首先尝试使用H.264编码
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            out = cv2.VideoWriter(self.video_file, fourcc, 30, (self.width, self.height))
            
            # 测试视频写入器是否正常工作
            if not out.isOpened():
                raise Exception("H264编码器不可用")
                
        except Exception as e:
            print(f"H264编码器不可用: {e}，尝试使用MP4V...")
            # 回退到MP4V编码
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.video_file, fourcc, 30, (self.width, self.height))
        
        # 缩减帧数以加快处理速度
        total_frames = len(self.frames)
        if total_frames > 150:  # 如果帧数过多，进一步缩减
            step = max(1, total_frames // 150)
            print(f"帧数过多({total_frames})，以步长{step}缩减帧数至约150帧")
            selected_frames = self.frames[::step]
        else:
            selected_frames = self.frames
            
        print(f"处理{len(selected_frames)}帧...")
        for frame in selected_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        out.release()
        self.recording = False
        self.frames = []
        
        print(f"视频录制完成: {self.video_file}")


class Body(object):
    """Interface to a multibody simulated in PyBullet.

    Attributes:
        uid: The unique id of the body within the physics server.
        name: The name of the body.
        joints: A dict mapping joint names to Joint objects.
        links: A dict mapping link names to Link objects.
    """

    def __init__(self, physics_client, body_uid, scale, urdf_path=None):
        self.p = physics_client
        self.uid = body_uid
        self.scale = scale
        self.urdf_path = urdf_path
        self.name = self.p.getBodyInfo(self.uid)[1].decode("utf-8")
        self.joints, self.links = {}, {}
        for i in range(self.p.getNumJoints(self.uid)):
            joint_info = self.p.getJointInfo(self.uid, i)
            joint_name = joint_info[1].decode("utf8")
            self.joints[joint_name] = Joint(self.p, self.uid, i)
            link_name = joint_info[12].decode("utf8")
            self.links[link_name] = Link(self.p, self.uid, i)

    @classmethod
    def from_urdf(cls, physics_client, urdf_path, pose, scale):
        body_uid = physics_client.loadURDF(
            str(urdf_path),
            pose.translation,
            pose.rotation.as_quat(),
            globalScaling=scale,
        )
        # self.urdf_path = urdf_path
        return cls(physics_client, body_uid, scale, urdf_path)

    def get_pose(self):
        pos, ori = self.p.getBasePositionAndOrientation(self.uid)
        return Transform(Rotation.from_quat(ori), np.asarray(pos))

    def set_pose(self, pose):
        self.p.resetBasePositionAndOrientation(
            self.uid, pose.translation, pose.rotation.as_quat()
        )
    def set_color(self, link_index, rgba_color):
        """
        Sets the color of a specific link in this body.
        
        Args:
            link_index: The index of the link to change the color of. Use -1 for the base.
            rgba_color: A tuple of 4 floats representing the RGBA color.
        """
        self.p.changeVisualShape(self.uid, link_index, rgbaColor=rgba_color)

    def get_velocity(self):
        linear, angular = self.p.getBaseVelocity(self.uid)
        return linear, angular


class Link(object):
    """Interface to a link simulated in Pybullet.

    Attributes:
        link_index: The index of the joint.
    """

    def __init__(self, physics_client, body_uid, link_index):
        self.p = physics_client
        self.body_uid = body_uid
        self.link_index = link_index

    def get_pose(self):
        link_state = self.p.getLinkState(self.body_uid, self.link_index)
        pos, ori = link_state[0], link_state[1]
        return Transform(Rotation.from_quat(ori), pos)


class Joint(object):
    """Interface to a joint simulated in PyBullet.

    Attributes:
        joint_index: The index of the joint.
        lower_limit: Lower position limit of the joint.
        upper_limit: Upper position limit of the joint.
        effort: The maximum joint effort.
    """

    def __init__(self, physics_client, body_uid, joint_index):
        self.p = physics_client
        self.body_uid = body_uid
        self.joint_index = joint_index

        joint_info = self.p.getJointInfo(body_uid, joint_index)
        self.lower_limit = joint_info[8]
        self.upper_limit = joint_info[9]
        self.effort = joint_info[10]

    def get_position(self):
        joint_state = self.p.getJointState(self.body_uid, self.joint_index)
        return joint_state[0]

    def set_position(self, position, kinematics=False):
        if kinematics:
            self.p.resetJointState(self.body_uid, self.joint_index, position)
        self.p.setJointMotorControl2(
            self.body_uid,
            self.joint_index,
            pybullet.POSITION_CONTROL,
            targetPosition=position,
            force=self.effort,
        )


class Constraint(object):
    """Interface to a constraint in PyBullet.

    Attributes:
        uid: The unique id of the constraint within the physics server.
    """

    def __init__(
        self,
        physics_client,
        parent,
        parent_link,
        child,
        child_link,
        joint_type,
        joint_axis,
        parent_frame,
        child_frame,
    ):
        """
        Create a new constraint between links of bodies.

        Args:
            parent:
            parent_link: None for the base.
            child: None for a fixed frame in world coordinates.

        """
        self.p = physics_client
        parent_body_uid = parent.uid
        parent_link_index = parent_link.link_index if parent_link else -1
        child_body_uid = child.uid if child else -1
        child_link_index = child_link.link_index if child_link else -1

        self.uid = self.p.createConstraint(
            parentBodyUniqueId=parent_body_uid,
            parentLinkIndex=parent_link_index,
            childBodyUniqueId=child_body_uid,
            childLinkIndex=child_link_index,
            jointType=joint_type,
            jointAxis=joint_axis,
            parentFramePosition=parent_frame.translation,
            parentFrameOrientation=parent_frame.rotation.as_quat(),
            childFramePosition=child_frame.translation,
            childFrameOrientation=child_frame.rotation.as_quat(),
        )

    def change(self, **kwargs):
        self.p.changeConstraint(self.uid, **kwargs)


class Contact(object):
    """Contact point between two multibodies.

    Attributes:
        point: Contact point.
        normal: Normal vector from ... to ...
        depth: Penetration depth
        force: Contact force acting on body ...
    """

    def __init__(self, bodyA, bodyB, point, normal, depth, force):
        self.bodyA = bodyA
        self.bodyB = bodyB
        self.point = point
        self.normal = normal
        self.depth = depth
        self.force = force


class Camera(object):
    """Virtual RGB-D camera based on the PyBullet camera interface.

    Attributes:
        intrinsic: The camera intrinsic parameters.
    """

    def __init__(self, physics_client, intrinsic, near, far):
        self.intrinsic = intrinsic
        self.near = near
        self.far = far
        self.proj_matrix = _build_projection_matrix(intrinsic, near, far)
        self.p = physics_client

    def render(self, extrinsic):
        """Render synthetic RGB and depth images.

        Args:
            extrinsic: Extrinsic parameters, T_cam_ref.
        """
        # Construct OpenGL compatible view and projection matrices.
        gl_view_matrix = extrinsic.as_matrix()
        gl_view_matrix[2, :] *= -1  # flip the Z axis
        gl_view_matrix = gl_view_matrix.flatten(order="F")
        gl_proj_matrix = self.proj_matrix.flatten(order="F")

        result = self.p.getCameraImage(
            width=self.intrinsic.width,
            height=self.intrinsic.height,
            viewMatrix=gl_view_matrix,
            projectionMatrix=gl_proj_matrix,
            renderer=pybullet.ER_TINY_RENDERER,
        )

        rgb, z_buffer = result[2][:, :, :3], result[3]
        depth = (
            1.0 * self.far * self.near / (self.far - (self.far - self.near) * z_buffer)
        )
        return rgb, depth

    def render_with_seg(self, extrinsic, segmentation=True):
        """Render synthetic RGB and depth images.

        Args:
            extrinsic: Extrinsic parameters, T_cam_ref.
            segmentation: Boolean flag indicating if segmentation mask is required.
        """
        # Construct OpenGL compatible view and projection matrices.
        gl_view_matrix = extrinsic.as_matrix()
        gl_view_matrix[2, :] *= -1  # flip the Z axis
        gl_view_matrix = gl_view_matrix.flatten(order="F")
        gl_proj_matrix = self.proj_matrix.flatten(order="F")

        result = self.p.getCameraImage(
            width=self.intrinsic.width,
            height=self.intrinsic.height,
            viewMatrix=gl_view_matrix,
            projectionMatrix=gl_proj_matrix,
            renderer=pybullet.ER_TINY_RENDERER,
        )

        rgb, z_buffer = result[2][:, :, :3], result[3]
        depth = (
            1.0 * self.far * self.near / (self.far - (self.far - self.near) * z_buffer)
        )

        seg = None
        if segmentation:
            seg = result[4]
        return rgb, depth, seg
    

    def render(self, extrinsic):
        """Render synthetic RGB and depth images.

        Args:
            extrinsic: Extrinsic parameters, T_cam_ref.
        """
        # Construct OpenGL compatible view and projection matrices.
        gl_view_matrix = extrinsic.as_matrix()
        gl_view_matrix[2, :] *= -1  # flip the Z axis
        gl_view_matrix = gl_view_matrix.flatten(order="F")
        gl_proj_matrix = self.proj_matrix.flatten(order="F")

        result = self.p.getCameraImage(
            width=self.intrinsic.width,
            height=self.intrinsic.height,
            viewMatrix=gl_view_matrix,
            projectionMatrix=gl_proj_matrix,
            renderer=pybullet.ER_TINY_RENDERER,
        )

        rgb, z_buffer = result[2][:, :, :3], result[3]
        depth = (
            1.0 * self.far * self.near / (self.far - (self.far - self.near) * z_buffer)
        )
        return rgb, depth


def _build_projection_matrix(intrinsic, near, far):
    perspective = np.array(
        [
            [intrinsic.fx, 0.0, -intrinsic.cx, 0.0],
            [0.0, intrinsic.fy, -intrinsic.cy, 0.0],
            [0.0, 0.0, near + far, near * far],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )
    ortho = _gl_ortho(0.0, intrinsic.width, intrinsic.height, 0.0, near, far)
    return np.matmul(ortho, perspective)


def _gl_ortho(left, right, bottom, top, near, far):
    ortho = np.diag(
        [2.0 / (right - left), 2.0 / (top - bottom), -2.0 / (far - near), 1.0]
    )
    ortho[0, 3] = -(right + left) / (right - left)
    ortho[1, 3] = -(top + bottom) / (top - bottom)
    ortho[2, 3] = -(far + near) / (far - near)
    return ortho
