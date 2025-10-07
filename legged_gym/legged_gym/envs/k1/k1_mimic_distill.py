#created by AI

from legged_gym.envs.base.humanoid_mimic import HumanoidMimic
from legged_gym.gym_utils.task_registry import task_registry
from legged_gym.envs.k1.k1_mimic_distill_config import K1MimicPrivCfg, K1MimicPrivCfgPPO, K1MimicStuRLCfg, K1MimicStuRLCfgDAgger

import torch


class K1MimicDistill(HumanoidMimic):
    """K1 Humanoid environment for motion imitation with teacher-student distillation."""
    
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # K1-specific initialization
        self.k1_specific_setup()
        
    def k1_specific_setup(self):
        """K1-specific environment setup."""
        # Add any K1-specific initialization here
        pass
        
    def _init_buffers(self):
        """Initialize K1-specific buffers."""
        super()._init_buffers()
        
        # K1 has 22 DOF
        assert self.num_dof == 22, f"K1 should have 22 DOF, got {self.num_dof}"
        
        # Override reference body positions buffer to match motion data dimensions
        # Motion data has 23 bodies, but K1 model has more rigid bodies
        num_motion_bodies = 23  # Number of bodies in the motion data
        self._ref_body_pos = torch.zeros((self.num_envs, num_motion_bodies, 3), dtype=torch.float, device=self.device)
        
        # Ensure reference root rotation is properly initialized
        self._ref_root_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self._ref_root_rot[:, 3] = 1.0  # Initialize as identity quaternion [0, 0, 0, 1]
        
        # CRITICAL FIX: Map robot key body indices to motion data indices
        print(f"[K1MimicDistill] ORIGINAL robot key_body_ids: {self._key_body_ids}")
        print(f"[K1MimicDistill] Robot key body names: {[self.body_names[i] for i in self._key_body_ids.cpu().numpy()]}")
        
        # Motion data body names (from TWIST dataset)
        motion_body_names = ['Trunk', 'Head_1', 'Head_2', 'Left_Arm_1', 'Left_Arm_2', 'Left_Arm_3', 
                           'left_hand_link', 'Right_Arm_1', 'Right_Arm_2', 'Right_Arm_3', 'right_hand_link', 
                           'Left_Hip_Pitch', 'Left_Hip_Roll', 'Left_Hip_Yaw', 'Left_Shank', 'Left_Ankle_Cross', 
                           'left_foot_link', 'Right_Hip_Pitch', 'Right_Hip_Roll', 'Right_Hip_Yaw', 'Right_Shank', 
                           'Right_Ankle_Cross', 'right_foot_link']
        
        # Map each robot key body name to motion data index
        robot_key_body_names = [self.body_names[i] for i in self._key_body_ids.cpu().numpy()]
        motion_key_indices = []
        
        for robot_key_name in robot_key_body_names:
            if robot_key_name in motion_body_names:
                motion_idx = motion_body_names.index(robot_key_name)
                motion_key_indices.append(motion_idx)
                print(f"[K1MimicDistill] Mapped '{robot_key_name}' -> motion index {motion_idx}")
            else:
                print(f"[K1MimicDistill] WARNING: '{robot_key_name}' not in motion data! Using fallback index 0")
                motion_key_indices.append(0)  # Safe fallback
        
        # DIRECTLY override _key_body_ids with motion-compatible indices
        self._key_body_ids = torch.tensor(motion_key_indices, device=self.device, dtype=torch.long)
        print(f"[K1MimicDistill] FIXED key_body_ids to motion indices: {self._key_body_ids}")
        print(f"[K1MimicDistill] All indices now in valid range [0, {len(motion_body_names)-1}]")
        
        # Store motion body names for reference
        self._motion_body_names = motion_body_names
        
    def check_termination(self):
        """Complete override of check_termination to use correct motion body indices."""
        # Contact force termination
        contact_force_termination = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.reset_buf = contact_force_termination
        
        # Height cutoff
        height_cutoff = torch.abs(self.root_states[:, 2] - self._ref_root_pos[:, 2]) > self.cfg.rewards.root_height_diff_threshold

        # Roll and pitch cutoff
        roll_cut = torch.abs(self.roll) > self.cfg.rewards.termination_roll
        pitch_cut = torch.abs(self.pitch) > self.cfg.rewards.termination_pitch
        self.reset_buf |= roll_cut
        self.reset_buf |= pitch_cut
        
        # Motion end
        motion_end = self.episode_length_buf * self.dt >= self._motion_lib.get_motion_length(self._motion_ids)
        self.reset_buf |= height_cutoff
        
        if self.viewer is None:
            self.reset_buf |= motion_end
        
        # Timeout
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        if self.viewer is None:
            self.time_out_buf |= motion_end
        
        self.reset_buf |= self.time_out_buf
        
        # Velocity too large
        vel_too_large = torch.norm(self.root_states[:, 7:10], dim=-1) > 5.
        self.reset_buf |= vel_too_large
        
                # Pose termination disabled for debugging
        # if self._pose_termination and len(self._key_body_ids_motion) > 0:
        #     # Use motion data body indices instead of robot body indices
        #     curr_body_pos_rel = self.body_pos[:, self.key_body_ids] - self.root_states[:, None, :3]
        #     tar_body_pos_rel = self._ref_body_pos[:, self._key_body_ids_motion] - self._ref_root_pos[:, None, :] 
        #     
        #     body_pos_diff = torch.norm(curr_body_pos_rel - tar_body_pos_rel, dim=-1)
        #     pose_termination = torch.any(body_pos_diff > self._pose_termination_dist, dim=1)
        #     self.reset_buf |= pose_termination
        
    def _reset_ref_motion(self, env_ids, motion_ids=None):
        """Reset reference motion for K1, handling shape mismatch."""
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
        
        # Handle potential shape mismatch between motion data and robot model
        print(f"Motion body_pos shape: {body_pos.shape}, ref_body_pos shape: {self._ref_body_pos.shape}")
        if body_pos.shape[1] != self._ref_body_pos.shape[1]:
            # Pad or truncate body_pos to match expected dimensions
            expected_bodies = self._ref_body_pos.shape[1]
            motion_bodies = body_pos.shape[1]
            print(f"Shape mismatch: motion has {motion_bodies} bodies, robot expects {expected_bodies}")
            
            if motion_bodies < expected_bodies:
                # Pad with zeros if motion data has fewer bodies
                padding = torch.zeros((body_pos.shape[0], expected_bodies - motion_bodies, 3), 
                                    device=body_pos.device, dtype=body_pos.dtype)
                body_pos = torch.cat([body_pos, padding], dim=1)
                print(f"Padded body_pos to shape: {body_pos.shape}")
            elif motion_bodies > expected_bodies:
                # Truncate if motion data has more bodies
                body_pos = body_pos[:, :expected_bodies, :]
                print(f"Truncated body_pos to shape: {body_pos.shape}")
        
        from legged_gym.envs.base.humanoid_char import convert_to_global_root_body_pos
        self._ref_body_pos[env_ids] = convert_to_global_root_body_pos(root_pos=root_pos, root_rot=root_rot, body_pos=body_pos)
        
        self._ref_root_pos[env_ids] = root_pos
        
        # Ensure root_rot is valid quaternion and normalize it
        if root_rot.shape[-1] == 4:
            # Normalize quaternions to avoid numerical issues
            root_rot_norm = torch.norm(root_rot, dim=-1, keepdim=True)
            root_rot_norm = torch.clamp(root_rot_norm, min=1e-8)  # Avoid division by zero
            root_rot = root_rot / root_rot_norm
            
            # Check for NaN or invalid values and replace with identity quaternion
            invalid_mask = torch.isnan(root_rot).any(dim=-1) | torch.isinf(root_rot).any(dim=-1)
            root_rot[invalid_mask] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=root_rot.device, dtype=root_rot.dtype)
            
        self._ref_root_rot[env_ids] = root_rot
        self._ref_root_vel[env_ids] = root_vel
        self._ref_root_ang_vel[env_ids] = root_ang_vel
        self._ref_dof_pos[env_ids] = dof_pos
        self._ref_dof_vel[env_ids] = dof_vel
        
    def _reward_tracking_root_pose(self):
        """Disabled for debugging CUDA issues."""
        return torch.zeros(self.num_envs, device=self.device)
        
    def _prepare_reward_function(self):
        """Prepare K1-specific reward functions."""
        super()._prepare_reward_function()
        
        # K1-specific reward setup can be added here
        
    def _compute_observations(self):
        """Compute observations for K1."""
        obs = super()._compute_observations()
        
        # Add any K1-specific observation processing here
        
        return obs
        
    def _compute_rewards(self):
        """Compute rewards for K1."""
        rewards = super()._compute_rewards()
        
        # Add any K1-specific reward computation here
        
        return rewards
        
    def _reset_envs(self, env_ids):
        """Reset environments for K1."""
        super()._reset_envs(env_ids)
        
        # Add any K1-specific reset logic here
        
    def step(self, actions):
        """Step the K1 environment."""
        return super().step(actions)
        
    def _process_rigid_shape_props(self, props, env_id):
        """Process rigid shape properties for K1."""
        # K1-specific physics property adjustments can be added here
        return super()._process_rigid_shape_props(props, env_id)
        
    def _process_dof_props(self, props, env_id):
        """Process DOF properties for K1."""
        # K1-specific DOF property adjustments can be added here
        return super()._process_dof_props(props, env_id)
        
    def _get_noise_scale_vec(self, cfg):
        """Get noise scale vector for K1."""
        noise_vec = super()._get_noise_scale_vec(cfg)
        
        # K1-specific noise scaling can be added here
        
        return noise_vec
        
    def _randomize_robot_props(self, env_ids):
        """Randomize robot properties for K1."""
        super()._randomize_robot_props(env_ids)
        
        # K1-specific randomization can be added here
        
    def _reward_ankle_dof_acc(self):
        """Disabled for debugging."""
        return torch.zeros(self.num_envs, device=self.device)
    
    def _reward_ankle_dof_vel(self):
        """Disabled for debugging."""
        return torch.zeros(self.num_envs, device=self.device)