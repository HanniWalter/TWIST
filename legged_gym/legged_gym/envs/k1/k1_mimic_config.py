#created by AI

from legged_gym.envs.base.humanoid_mimic_config import HumanoidMimicCfg, HumanoidMimicCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR

class K1MimicCfg(HumanoidMimicCfg):
    class env(HumanoidMimicCfg.env):
        tar_obs_steps = [1, 3, 5, 10]
        
        num_envs = 4096
        num_actions = 22  # K1 has 22 DOF
        n_priv = 0
        n_mimic_obs = 9 + 22  # CORRECTED: 9 base (3+2+3+1) + 22 DOF = 31 total per step
        # Total: mimic_obs(4*31=124) + base_ang_vel(3) + imu_obs(2) + dof_pos(22) + dof_vel(22) + action_history(22) = 195
        n_proprio = 195  # FIXED: Direct value to match actual observation size
        n_priv_latent = 4 + 1 + 2*num_actions
        extra_critic_obs = 3
        history_len = 10
        
        num_observations = n_proprio + n_priv_latent + history_len*n_proprio + n_priv + extra_critic_obs 
        num_privileged_obs = None

        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 10
        
        randomize_start_pos = True
        randomize_start_yaw = False
        
        history_encoding = True
        contact_buf_len = 10
        
        normalize_obs = True
        
        enable_early_termination = True
        pose_termination = True
        pose_termination_dist = 0.7
        root_tracking_termination_dist = 0.8
        rand_reset = True
        track_root = False
        
        # K1 specific DOF error weights (22 DOF) - hands excluded from DOF count
        dof_err_w = [0.5, 0.5,  # Head: yaw, pitch
                     1.0, 0.8, 0.8, 1.0,  # Left Arm: shoulder_pitch, shoulder_roll, elbow_pitch, elbow_yaw (no hand)
                     1.0, 0.8, 0.8, 1.0,  # Right Arm: shoulder_pitch, shoulder_roll, elbow_pitch, elbow_yaw (no hand)
                     1.0, 0.8, 0.8, 1.0, 0.5, 0.5,  # Left Leg: hip_pitch, hip_roll, hip_yaw, knee_pitch, ankle_pitch, ankle_roll
                     1.0, 0.8, 0.8, 1.0, 0.5, 0.5,  # Right Leg: hip_pitch, hip_roll, hip_yaw, knee_pitch, ankle_pitch, ankle_roll
                     ]
        
        global_obs = False
    
    class terrain(HumanoidMimicCfg.terrain):
        mesh_type = 'trimesh'
        # mesh_type = 'plane'
        height = [0, 0.00]
        horizontal_scale = 0.1
    
    class init_state(HumanoidMimicCfg.init_state):
        pos = [0, 0, 1.0]
        # K1 default joint angles
        default_joint_angles = {
            # Head
            'AAHead_yaw': 0.0,
            'Head_pitch': 0.0,
            
            # Left Arm
            'ALeft_Shoulder_Pitch': 0.0,
            'Left_Shoulder_Roll': 0.4,
            'Left_Elbow_Pitch': -1.2,
            'Left_Elbow_Yaw': 0.0,
            'Left_Hand_End_Ball': 0.0,
            
            # Right Arm  
            'ARight_Shoulder_Pitch': 0.0,
            'Right_Shoulder_Roll': -0.4,
            'Right_Elbow_Pitch': -1.2,
            'Right_Elbow_Yaw': 0.0,
            'Right_Hand_End_Ball': 0.0,
            
            # Left Leg
            'Left_Hip_Pitch': -0.2,
            'Left_Hip_Roll': 0.0,
            'Left_Hip_Yaw': 0.0,
            'Left_Knee_Pitch': 0.4,
            'Left_Ankle_Pitch': -0.2,
            'Left_Ankle_Roll': 0.0,
            
            # Right Leg
            'Right_Hip_Pitch': -0.2,
            'Right_Hip_Roll': 0.0,
            'Right_Hip_Yaw': 0.0,
            'Right_Knee_Pitch': 0.4,
            'Right_Ankle_Pitch': -0.2,
            'Right_Ankle_Roll': 0.0,
        }
    
    class control(HumanoidMimicCfg.control):
        # K1 motor parameters (estimated based on robot size and capabilities)
        stiffness = {
            # Head joints
            'AAHead_yaw': 20,
            'Head_pitch': 20,
            
            # Shoulder joints
            'ALeft_Shoulder_Pitch': 40,
            'Left_Shoulder_Roll': 40,
            'ARight_Shoulder_Pitch': 40,
            'Right_Shoulder_Roll': 40,
            
            # Elbow joints
            'Left_Elbow_Pitch': 30,
            'Left_Elbow_Yaw': 30,
            'Right_Elbow_Pitch': 30,
            'Right_Elbow_Yaw': 30,
            
            # Hand/gripper joints
            'Left_Hand_End_Ball': 10,
            'Right_Hand_End_Ball': 10,
            
            # Hip joints
            'Left_Hip_Pitch': 100,
            'Left_Hip_Roll': 100,
            'Left_Hip_Yaw': 100,
            'Right_Hip_Pitch': 100,
            'Right_Hip_Roll': 100,
            'Right_Hip_Yaw': 100,
            
            # Knee joints
            'Left_Knee_Pitch': 150,
            'Right_Knee_Pitch': 150,
            
            # Ankle joints
            'Left_Ankle_Pitch': 40,
            'Left_Ankle_Roll': 40,
            'Right_Ankle_Pitch': 40,
            'Right_Ankle_Roll': 40,
        }  # [N*m/rad]
        
        damping = {
            # Head joints
            'AAHead_yaw': 2,
            'Head_pitch': 2,
            
            # Shoulder joints
            'ALeft_Shoulder_Pitch': 4,
            'Left_Shoulder_Roll': 4,
            'ARight_Shoulder_Pitch': 4,
            'Right_Shoulder_Roll': 4,
            
            # Elbow joints
            'Left_Elbow_Pitch': 3,
            'Left_Elbow_Yaw': 3,
            'Right_Elbow_Pitch': 3,
            'Right_Elbow_Yaw': 3,
            
            # Hand/gripper joints
            'Left_Hand_End_Ball': 1,
            'Right_Hand_End_Ball': 1,
            
            # Hip joints
            'Left_Hip_Pitch': 2,
            'Left_Hip_Roll': 2,
            'Left_Hip_Yaw': 2,
            'Right_Hip_Pitch': 2,
            'Right_Hip_Roll': 2,
            'Right_Hip_Yaw': 2,
            
            # Knee joints
            'Left_Knee_Pitch': 4,
            'Right_Knee_Pitch': 4,
            
            # Ankle joints
            'Left_Ankle_Pitch': 2,
            'Left_Ankle_Roll': 2,
            'Right_Ankle_Pitch': 2,
            'Right_Ankle_Roll': 2,
        }  # [N*m*s/rad]
        
        action_scale = 0.5
        decimation = 10
        
    class sim(HumanoidMimicCfg.sim):
        dt = 0.002  # 1/500
        
    class normalization(HumanoidMimicCfg.normalization):
        clip_actions = 5.0
    
    class asset(HumanoidMimicCfg.asset):
        file = f'{LEGGED_GYM_ROOT_DIR}/../assets/booster_k1/K1_serial.urdf'
        
        # K1 specific body names
        torso_name: str = 'Trunk'  # K1 main body
        chest_name: str = 'Trunk'  # Same as torso for K1

        # Link names for reward computation
        thigh_name: str = 'Hip'     # Hip links
        shank_name: str = 'Link'    # Thigh/shin links  
        foot_name: str = 'foot_link'  # Foot links
        upper_arm_name: str = 'Arm_2'   # Upper arm links
        lower_arm_name: str = 'Arm_3'   # Lower arm/elbow links
        hand_name: list = ['left_hand_link', 'right_hand_link']  # Hand end effectors

        feet_bodies = ['left_foot_link', 'right_foot_link']
        n_lower_body_dofs: int = 12  # 6 DOF per leg

        penalize_contacts_on = ["Arm", "Hip", "Link"]  # Avoid contacts on arms, hips, links
        terminate_after_contacts_on = ['Trunk']  # Terminate if trunk contacts ground
        
        # K1 motor armature (estimated based on joint types)
        dof_armature = [0.001, 0.001,  # Head joints (small)
                       0.005, 0.005, 0.003, 0.003, 0.001,  # Left arm 
                       0.005, 0.005, 0.003, 0.003, 0.001,  # Right arm
                       0.01, 0.02, 0.01, 0.02, 0.004, 0.004,  # Left leg
                       0.01, 0.02, 0.01, 0.02, 0.004, 0.004]  # Right leg
        
        collapse_fixed_joints = False
    
    class rewards(HumanoidMimicCfg.rewards):
        regularization_names = [
            # Basic regularization terms can be added here
        ]
        regularization_scale = 1.0
        regularization_scale_range = [0.8,2.0]
        regularization_scale_curriculum = False
        regularization_scale_gamma = 0.0001
        
        class scales:
            tracking_joint_dof = 0.6
            tracking_joint_vel = 0.2
            tracking_root_pose = 0.6
            tracking_root_vel = 1.0
            tracking_keybody_pos = 2.0
            
            feet_slip = -0.1
            feet_contact_forces = -5e-4      
            feet_stumble = -1.25
            
            dof_pos_limits = -5.0
            dof_torque_limits = -1.0
            
            dof_vel = -1e-4
            dof_acc = -5e-8
            action_rate = -0.01
            
            feet_air_time = 5.0
            ang_vel_xy = -0.01

        min_dist = 0.1
        max_dist = 0.4
        max_knee_dist = 0.4
        feet_height_target = 0.2
        feet_air_time_target = 0.5
        only_positive_rewards = False
        tracking_sigma = 0.2
        tracking_sigma_ang = 0.125
        max_contact_force = 100
        soft_torque_limit = 0.95
        torque_safety_limit = 0.9
        root_height_diff_threshold = 0.2

    class domain_rand:
        domain_rand_general = True
        
        randomize_gravity = (True and domain_rand_general)
        gravity_rand_interval_s = 4
        gravity_range = (-0.1, 0.1)
        
        randomize_friction = (True and domain_rand_general)
        friction_range = [0.1, 2.]
        
        randomize_base_mass = (True and domain_rand_general)
        added_mass_range = [-3., 3]
        
        randomize_base_com = (True and domain_rand_general)
        added_com_range = [-0.05, 0.05]
        
        push_robots = (True and domain_rand_general)
        push_interval_s = 4
        max_push_vel_xy = 1.0
        
        push_end_effector = (True and domain_rand_general)
        push_end_effector_interval_s = 2
        max_push_force_end_effector = 20.0

        randomize_motor = (True and domain_rand_general)
        motor_strength_range = [0.8, 1.2]

        action_delay = (True and domain_rand_general)
        action_buf_len = 8
    
    class noise(HumanoidMimicCfg.noise):
        add_noise = True
        noise_increasing_steps = 3000
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.1
            lin_vel = 0.1
            ang_vel = 0.1
            gravity = 0.05
            imu = 0.1
        
    class motion(HumanoidMimicCfg.motion):
        motion_curriculum = True
        motion_curriculum_gamma = 0.01
        
        # K1 key bodies for motion tracking (hands, feet, elbows, knees, head)
        key_bodies = ["left_hand_link", "right_hand_link", "left_foot_link", "right_foot_link", 
                     "Left_Shank", "Right_Shank", "Left_Arm_3", "Right_Arm_3", "Head_2"]
        upper_key_bodies = ["left_hand_link", "right_hand_link", "Left_Arm_3", "Right_Arm_3", "Head_2"]
        
        motion_file = f"{LEGGED_GYM_ROOT_DIR}/motion_data_configs/twist_dataset.yaml"
        
        reset_consec_frames = 30


class K1MimicCfgPPO(HumanoidMimicCfgPPO):
    class policy(HumanoidMimicCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        priv_encoder_dims = [64, 20]
        activation = 'elu'
        tanh_encoder_output = False
        fix_action_std = False
        obs_context_len = 0
        
    class algorithm(HumanoidMimicCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 2e-4
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.008
        max_grad_norm = 1.
        dagger_update_freq = 20
        priv_reg_coef_schedual = [0, 0.1, 2000, 3000]
        priv_reg_coef_schedual_resume = [0, 0.1, 0, 1]
        normalizer_update_iterations = 3000

    class runner(HumanoidMimicCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        runner_class_name = 'OnPolicyRunner'
        num_steps_per_env = 24
        max_iterations = 20000

        save_interval = 100
        experiment_name = 'k1_mimic'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None