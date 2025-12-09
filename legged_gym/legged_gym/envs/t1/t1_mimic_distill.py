from isaacgym.torch_utils import *

import torch

from legged_gym.envs.base.humanoid_mimic import HumanoidMimic
from .t1_mimic_distill_config import T1MimicPrivCfg, T1MimicStuCfg
from legged_gym.gym_utils.math import *
from pose.utils import torch_utils
from legged_gym.envs.base.legged_robot import euler_from_quaternion
from legged_gym.envs.base.humanoid_char import convert_to_local_root_body_pos, convert_to_global_root_body_pos

def t1_body_from_24_to_30(body_pos_24: torch.Tensor) -> torch.Tensor:
    """
    Convert body positions from motion library format (24 bodies) to T1 simulation format (30 bodies).
    The T1 robot has 6 extra fixed bodies (H1, H2, left_hand_tip, right_hand_tip, and 4 toe links) that are not in the motion library.
    
    T1 has 30 bodies:
    0-Trunk, 1-H1, 2-H2, 3-AL1, 4-AL2, 5-AL3, 6-left_hand_link, 7-AR1, 8-left_hand_tip,
    9-AR2, 10-AR3, 11-right_hand_link, 12-right_hand_tip, 13-Waist, 14-Hip_Pitch_Left,
    15-Hip_Roll_Left, 16-Hip_Yaw_Left, 17-Shank_Left, 18-Ankle_Cross_Left, 19-left_foot_link,
    20-Hip_Pitch_Right, 21-Hip_Roll_Right, 22-Hip_Yaw_Right, 23-Shank_Right, 24-Ankle_Cross_Right,
    25-right_foot_link, 26-left_inner_toe_link, 27-right_inner_toe_link, 28-left_outer_toe_link, 29-right_outer_toe_link
    
    Motion library has 24 bodies (without H1, H2, left_hand_tip, right_hand_tip).
    
    Parameters:
    -----------
        body_pos_24 : torch.Tensor
            Body positions of shape (N, 24, 3) from motion library
    
    Returns:
    --------
        body_pos_30 : torch.Tensor
            Body positions of shape (N, 30, 3) for T1 simulation
    """
    # Create mapping: for each of the 30 bodies in simulation, which index in the 24-body motion lib?
    # -1 means this body doesn't exist in motion lib and will be filled with zeros
    idx_map_30_list = [
        0,  # 0: Trunk -> 0
        -1, # 1: H1 (fixed, not in motion lib)
        -1, # 2: H2 (fixed, not in motion lib)
        1,  # 3: AL1 -> 1
        2,  # 4: AL2 -> 2
        3,  # 5: AL3 -> 3
        4,  # 6: left_hand_link -> 4
        5,  # 7: AR1 -> 5
        -1, # 8: left_hand_tip (fixed, not in motion lib)
        6,  # 9: AR2 -> 6
        7,  # 10: AR3 -> 7
        8,  # 11: right_hand_link -> 8
        -1, # 12: right_hand_tip (fixed, not in motion lib)
        9,  # 13: Waist -> 9
        10, # 14: Hip_Pitch_Left -> 10
        11, # 15: Hip_Roll_Left -> 11
        12, # 16: Hip_Yaw_Left -> 12
        13, # 17: Shank_Left -> 13
        14, # 18: Ankle_Cross_Left -> 14
        15, # 19: left_foot_link -> 15
        16, # 20: Hip_Pitch_Right -> 16
        17, # 21: Hip_Roll_Right -> 17
        18, # 22: Hip_Yaw_Right -> 18
        19, # 23: Shank_Right -> 19
        20, # 24: Ankle_Cross_Right -> 20
        21, # 25: right_foot_link -> 21
        22, # 26: left_inner_toe_link -> 22
        23, # 27: right_inner_toe_link -> 23
        22, # 28: left_outer_toe_link -> 22
        23, # 29: right_outer_toe_link -> 23
    ]
    
    # Convert to tensor
    idx_map_30 = torch.tensor(idx_map_30_list, dtype=torch.long, device=body_pos_24.device)
    
    # Create output tensor (N, 30, 3), initialized to zeros
    N = body_pos_24.shape[0]
    body_pos_30 = torch.zeros((N, 30, 3), dtype=body_pos_24.dtype, device=body_pos_24.device)
    
    # Create mask for valid (non -1) indices
    valid_mask = (idx_map_30 >= 0)
    
    # Copy valid body positions
    body_pos_30[:, valid_mask, :] = body_pos_24[:, idx_map_30[valid_mask], :]
    
    return body_pos_30



class T1MimicDistill(HumanoidMimic):
    def __init__(self, cfg: T1MimicPrivCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.obs_type = cfg.env.obs_type
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.05
        self.episode_length = torch.zeros((self.num_envs), device=self.device)
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        if self.obs_type == 'student':
            self.total_env_steps_counter = 24 * 100000
            self.global_counter = 24 * 100000
            # self.motion_difficulty = torch.ones_like(self.motion_difficulty)

    def _reset_ref_motion(self, env_ids, motion_ids=None):
        n = len(env_ids)
        if motion_ids is None:
            motion_ids = self._motion_lib.sample_motions(n, motion_difficulty=self.motion_difficulty)
        
        if self._rand_reset:
            motion_times = self._motion_lib.sample_time(motion_ids)
        else:
            motion_times = torch.zeros(motion_ids.shape, device=self.device, dtype=torch.float)
        
        self._motion_ids[env_ids] = motion_ids
        self._motion_time_offsets[env_ids] = motion_times
        
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        root_pos[:, 2] += self.cfg.motion.height_offset
        
        # Remove head DOFs from motion data to match modified URDF
        dof_pos = self._remove_head_dofs(dof_pos)
        dof_vel = self._remove_head_dofs(dof_vel)
        
        self._ref_root_pos[env_ids] = root_pos
        self._ref_root_rot[env_ids] = root_rot
        self._ref_root_vel[env_ids] = root_vel
        self._ref_root_ang_vel[env_ids] = root_ang_vel
        self._ref_dof_pos[env_ids] = dof_pos
        self._ref_dof_vel[env_ids] = dof_vel
        # Convert from motion library format (24 bodies) to T1 simulation format (30 bodies)
        if body_pos.shape[1] != self._ref_body_pos[env_ids].shape[1]:
            body_pos = t1_body_from_24_to_30(body_pos)
        self._ref_body_pos[env_ids] = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=body_pos)
    
    
    def _update_ref_motion(self):
        motion_ids = self._motion_ids
        motion_times = self._get_motion_times()
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        root_pos[:, 2] += self.cfg.motion.height_offset
        root_pos[:, :2] += self.episode_init_origin[:, :2]
        
        # Remove head DOFs from motion data to match modified URDF
        dof_pos = self._remove_head_dofs(dof_pos)
        dof_vel = self._remove_head_dofs(dof_vel)
        
        self._ref_root_pos[:] = root_pos
        self._ref_root_rot[:] = root_rot
        self._ref_root_vel[:] = root_vel
        self._ref_root_ang_vel[:] = root_ang_vel
        self._ref_dof_pos[:] = dof_pos
        self._ref_dof_vel[:] = dof_vel
        # Convert from motion library format (24 bodies) to T1 simulation format (30 bodies)
        if body_pos.shape[1] != self._ref_body_pos.shape[1]:
            body_pos = t1_body_from_24_to_30(body_pos)
        self._ref_body_pos[:] = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=body_pos)
        
    def _update_motion_difficulty(self, env_ids):
        if self.obs_type == 'priv':
            super()._update_motion_difficulty(env_ids)
        elif self.obs_type == 'student':
            super()._update_motion_difficulty(env_ids) # currently we use the same strategy for student
        else:
            return

    def _get_body_indices(self):
        if type(self.cfg.asset.upper_arm_name) == list:
            upper_arm_names = []
            for names in self.cfg.asset.upper_arm_name:
                upper_arm_names += [s for s in self.body_names if names in s]
        else:
            upper_arm_names = [s for s in self.body_names if self.cfg.asset.upper_arm_name in s]
        lower_arm_names = [s for s in self.body_names if self.cfg.asset.lower_arm_name in s]
        torso_name = [s for s in self.body_names if self.cfg.asset.torso_name in s]
        self.torso_indices = torch.zeros(len(torso_name), dtype=torch.long, device=self.device,
                                                 requires_grad=False)
        for j in range(len(torso_name)):
            self.torso_indices[j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                                  torso_name[j])
        self.upper_arm_indices = torch.zeros(len(upper_arm_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for j in range(len(upper_arm_names)):
            self.upper_arm_indices[j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                                upper_arm_names[j])
        self.lower_arm_indices = torch.zeros(len(lower_arm_names), dtype=torch.long, device=self.device,
                                                requires_grad=False)
        for j in range(len(lower_arm_names)):
            self.lower_arm_indices[j] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                                lower_arm_names[j])
        knee_names = [s for s in self.body_names if self.cfg.asset.shank_name in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])
    
    def _init_buffers(self):
        super()._init_buffers()
        # Initialize obs_history_buf with the correct size based on obs_type
        if self.obs_type == 'student':
            # For student, use mimic_obs size instead of priv_mimic_obs size
            student_obs_size = self.cfg.env.n_mimic_obs + self.cfg.env.n_proprio
            self.obs_history_buf = torch.zeros((self.num_envs, self.cfg.env.history_len, student_obs_size), device=self.device)
            print(f"[T1 Init] Student mode: obs_history_buf size = {self.obs_history_buf.shape}")
        else:
            self.obs_history_buf = torch.zeros((self.num_envs, self.cfg.env.history_len, self.cfg.env.n_obs_single), device=self.device)
            print(f"[T1 Init] Priv mode: obs_history_buf size = {self.obs_history_buf.shape}, privileged_obs_history_buf size will be {(self.num_envs, self.cfg.env.history_len, self.cfg.env.n_priv_obs_single)}")
        self.privileged_obs_history_buf = torch.zeros((self.num_envs, self.cfg.env.history_len, self.cfg.env.n_priv_obs_single), device=self.device)
        print(f"[T1 Init] obs_type={self.obs_type}, num_dof={self.num_dof}, n_mimic_obs={self.cfg.env.n_mimic_obs}, n_proprio={self.cfg.env.n_proprio}")
    
    def _remove_head_dofs(self, dof_tensor):
        """Remove the first 2 DOFs (AAHead_yaw and Head_pitch) from motion data if needed.
        The modified URDF has these joints as fixed.
        Motion data should have 21 DOFs matching the simulation.
        """
        # Check if motion data has extra head joints (23 DOFs instead of 21)
        if dof_tensor.shape[-1] == 23:  # Original motion data with head joints
            return dof_tensor[..., 2:]  # Remove first 2 DOFs (head joints)
        else:
            return dof_tensor  # Already correct size (21 DOFs)
    
    def _get_noise_scale_vec(self, cfg):
        noise_scale_vec = torch.zeros(1, self.cfg.env.n_proprio, device=self.device)
        if not self.cfg.noise.add_noise:
            return noise_scale_vec
        ang_vel_dim = 3
        imu_dim = 2
        
        noise_scale_vec[:, 0:ang_vel_dim] = self.cfg.noise.noise_scales.ang_vel
        noise_scale_vec[:, ang_vel_dim:ang_vel_dim+imu_dim] = self.cfg.noise.noise_scales.imu
        noise_scale_vec[:, ang_vel_dim+imu_dim:ang_vel_dim+imu_dim+self.num_dof] = self.cfg.noise.noise_scales.dof_pos
        noise_scale_vec[:, ang_vel_dim+imu_dim+self.num_dof:ang_vel_dim+imu_dim+2*self.num_dof] = self.cfg.noise.noise_scales.dof_vel
        
        return noise_scale_vec
            
    def _get_mimic_obs(self):
        num_steps = self._tar_obs_steps.shape[0]
        assert num_steps > 0, "Invalid number of target observation steps"
        motion_times = self._get_motion_times().unsqueeze(-1)
        obs_motion_times = self._tar_obs_steps * self.dt + motion_times
        motion_ids_tiled = torch.broadcast_to(self._motion_ids.unsqueeze(-1), obs_motion_times.shape)
        motion_ids_tiled = motion_ids_tiled.flatten()
        obs_motion_times = obs_motion_times.flatten()
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos = self._motion_lib.calc_motion_frame(motion_ids_tiled, obs_motion_times)
        
        # Remove head DOFs from motion data to match modified URDF
        dof_pos_before = dof_pos.shape[-1]
        dof_pos = self._remove_head_dofs(dof_pos)
        if not hasattr(self, '_dof_filter_checked'):
            print(f"[T1 DOF Filter] dof_pos shape before filtering: [..., {dof_pos_before}]")
            print(f"[T1 DOF Filter] dof_pos shape after filtering: {dof_pos.shape}")
            self._dof_filter_checked = True
        
        roll, pitch, yaw = euler_from_quaternion(root_rot)
        roll = roll.reshape(self.num_envs, num_steps, 1)
        pitch = pitch.reshape(self.num_envs, num_steps, 1)
        yaw = yaw.reshape(self.num_envs, num_steps, 1)
        if not self.global_obs:
            root_vel = quat_rotate_inverse(root_rot, root_vel)
            root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)
      
        whole_key_body_pos = body_pos[:, self._key_body_ids_motion, :]
        if self.global_obs:
            whole_key_body_pos = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=whole_key_body_pos)
        whole_key_body_pos = whole_key_body_pos.reshape(self.num_envs, num_steps, -1)
        
        root_pos = root_pos.reshape(self.num_envs, num_steps, root_pos.shape[-1])
        root_vel = root_vel.reshape(self.num_envs, num_steps, root_vel.shape[-1])
        root_rot = root_rot.reshape(self.num_envs, num_steps, root_rot.shape[-1])
        root_ang_vel = root_ang_vel.reshape(self.num_envs, num_steps, root_ang_vel.shape[-1])
        dof_pos = dof_pos.reshape(self.num_envs, num_steps, dof_pos.shape[-1])
     
        # teacher v0
        priv_mimic_obs_buf = torch.cat((
            root_pos[..., 2:3], # 1 dim
            roll, pitch, yaw, # 3 dims
            root_vel, # 3 dims
            root_ang_vel[..., 2:3], # 1 dim, yaw only
            dof_pos, # num_dof dims
            whole_key_body_pos, # num_bodies * 3 dims
        ), dim=-1) # shape: (num_envs, num_steps, 7 + num_dof + num_key_bodies * 3)
        
        
        # v6, align mocap
        mimic_obs_buf = torch.cat((
            root_pos[..., 2:3], # 1 dim
            roll, pitch, yaw, # 3 dims
            root_vel, # 3 dims
            root_ang_vel[..., 2:3], # 1 dim, yaw only
            dof_pos, # num_dof dims
        ), dim=-1)[:, 0:1] # shape: (num_envs, 1, 7 + num_dof)
        
        
        return priv_mimic_obs_buf.reshape(self.num_envs, -1), mimic_obs_buf.reshape(self.num_envs, -1)

    def compute_observations(self):
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        self.base_yaw_quat = quat_from_euler_xyz(0*self.yaw, 0*self.yaw, self.yaw)
        priv_mimic_obs, mimic_obs = self._get_mimic_obs()
        
        proprio_obs_buf = torch.cat((
                            self.base_ang_vel  * self.obs_scales.ang_vel,   # 3 dims
                            imu_obs,    # 2 dims
                            self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),
                            self.reindex(self.dof_vel * self.obs_scales.dof_vel),
                            self.reindex(self.action_history_buf[:, -1]),
                            ),dim=-1)
        
        if self.cfg.noise.add_noise and self.headless:
            proprio_obs_buf += (2 * torch.rand_like(proprio_obs_buf) - 1) * self.noise_scale_vec * min(self.total_env_steps_counter / (self.cfg.noise.noise_increasing_steps * 24),  1.)
        elif self.cfg.noise.add_noise and not self.headless:
            proprio_obs_buf += (2 * torch.rand_like(proprio_obs_buf) - 1) * self.noise_scale_vec
        else:
            proprio_obs_buf += 0.
        dof_vel_start_dim = 5 + self.dof_pos.shape[1]

        # disable ankle dof
        ankle_idx = [13, 14, 19, 20]
        proprio_obs_buf[:, [dof_vel_start_dim + i for i in ankle_idx]] = 0.
        
        key_body_pos = self.rigid_body_states[:, self._key_body_ids, :3]
        key_body_pos = key_body_pos - self.root_states[:, None, :3]
        if not self.global_obs:
            key_body_pos = convert_to_local_root_body_pos(self.root_states[:, 3:7], key_body_pos)
        key_body_pos = key_body_pos.reshape(self.num_envs, -1) # shape: (num_envs, num_key_bodies * 3)
        
        # Calculate logical foot contacts (2 dims)
        feet_contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        if hasattr(self, 'left_foot_indices_local') and len(self.left_foot_indices_local) > 0 and len(self.right_foot_indices_local) > 0:
            left_contact = torch.any(feet_contact[:, self.left_foot_indices_local], dim=1)
            right_contact = torch.any(feet_contact[:, self.right_foot_indices_local], dim=1)
            logical_contact = torch.stack([left_contact, right_contact], dim=1)
        else:
            logical_contact = feet_contact

        if self.cfg.domain_rand.domain_rand_general:
            priv_info = torch.cat((
                self.base_lin_vel, # 3 dims
                self.root_states[:, 2:3], # 1 dim
                key_body_pos, # num_bodies * 3 dims
                logical_contact, # 2 dims, foot contact
                self.mass_params_tensor,
                self.friction_coeffs_tensor,
                self.motor_strength[0] - 1, 
                self.motor_strength[1] - 1,
            ), dim=-1)
        else:
            priv_info = torch.zeros((self.num_envs, self.cfg.env.n_priv_info), device=self.device)
        
        obs_buf = torch.cat((
            mimic_obs,
            proprio_obs_buf,
        ), dim=-1)
        
        priv_obs_buf = torch.cat((
            priv_mimic_obs,
            proprio_obs_buf,
            priv_info,
        ), dim=-1)
        
        # Debug: Check dimensions on first call
        if not hasattr(self, '_dim_checked'):
            print(f"[T1 Obs Debug] priv_mimic_obs shape: {priv_mimic_obs.shape}")
            print(f"[T1 Obs Debug] proprio_obs_buf shape: {proprio_obs_buf.shape}")
            print(f"[T1 Obs Debug] priv_info shape: {priv_info.shape}")
            print(f"[T1 Obs Debug] priv_obs_buf shape: {priv_obs_buf.shape}")
            print(f"[T1 Obs Debug] Expected n_priv_obs_single: {self.cfg.env.n_priv_obs_single}")
            self._dim_checked = True
        
        self.privileged_obs_buf = priv_obs_buf
        
        if self.obs_type == 'priv':
            self.obs_buf = priv_obs_buf
        elif self.obs_type == 'student':
            self.obs_buf = torch.cat([obs_buf, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        
        if self.cfg.env.history_len > 0:
            self.privileged_obs_history_buf = torch.where(
                (self.episode_length_buf <= 1)[:, None, None], 
                torch.stack([priv_obs_buf] * self.cfg.env.history_len, dim=1),
                torch.cat([
                    self.privileged_obs_history_buf[:, 1:],
                    priv_obs_buf.unsqueeze(1)
                ], dim=1)
            )
            if self.obs_type == 'priv':
                self.obs_history_buf[:] = self.privileged_obs_history_buf[:]
            elif self.obs_type == 'student':
                self.obs_history_buf = torch.where(
                    (self.episode_length_buf <= 1)[:, None, None], 
                    torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
                    torch.cat([
                        self.obs_history_buf[:, 1:],
                        obs_buf.unsqueeze(1)
                    ], dim=1)
                )


############################################################################################################
##################################### Extra Reward Functions################################################
############################################################################################################

    def _reward_waist_dof_acc(self):
        waist_dof_idx = [8]
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt)[:, waist_dof_idx], dim=1)
    
    def _reward_waist_dof_vel(self):
        waist_dof_idx = [8]
        return torch.sum(torch.square(self.dof_vel[:, waist_dof_idx]), dim=1)
    
    def _reward_ankle_dof_acc(self):
        ankle_dof_idx = [13, 14, 19, 20]
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt)[:, ankle_dof_idx], dim=1)
    
    def _reward_ankle_dof_vel(self):
        ankle_dof_idx = [13, 14, 19, 20]
        return torch.sum(torch.square(self.dof_vel[:, ankle_dof_idx]), dim=1)
    
    def _reward_ankle_action(self):
        return torch.norm(self.action_history_buf[:, -1, [13, 14, 19, 20]], dim=1)
    
    def _reward_hip_dof_acc(self):
        # Hip indices: Left Hip (9, 10, 11) + Right Hip (15, 16, 17)
        hip_dof_idx = [9, 10, 11, 15, 16, 17]
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt)[:, hip_dof_idx], dim=1)
    
    def _reward_hip_dof_vel(self):
        # Hip indices: Left Hip (9, 10, 11) + Right Hip (15, 16, 17)
        hip_dof_idx = [9, 10, 11, 15, 16, 17]
        return torch.sum(torch.square(self.dof_vel[:, hip_dof_idx]), dim=1)
