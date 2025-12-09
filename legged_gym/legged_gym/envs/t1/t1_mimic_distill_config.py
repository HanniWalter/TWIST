from legged_gym.envs.base.humanoid_mimic_config import HumanoidMimicCfg, HumanoidMimicCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR


class T1MimicPrivCfg(HumanoidMimicCfg):
    class env(HumanoidMimicCfg.env):
        tar_obs_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                         50, 55, 60, 65, 70, 75, 80, 85, 90, 95,]
        
        num_envs = 4096 * 7 // 8
        num_actions = 21
        obs_type = 'priv' # 'student'
        n_priv_latent = 4 + 1 + 2*num_actions
        extra_critic_obs = 3
        n_priv = 0
        
        n_proprio = 3 + 2 + 3*num_actions
        n_priv_mimic_obs = len(tar_obs_steps) * (8 + num_actions + 3*11) # Hardcode for now: 11 key bodies (see motion.key_bodies)
        n_mimic_obs = 8 + 21 # 21 for dof pos
        n_priv_info = 3 + 1 + 3*11 + 2 + 4 + 1 + 2*num_actions # base lin vel, root height, 11 key body pos, contact mask, priv latent
        history_len = 10
        
        n_obs_single = n_priv_mimic_obs + n_proprio + n_priv_info
        n_priv_obs_single = n_priv_mimic_obs + n_proprio + n_priv_info
        
        num_observations = n_obs_single

        num_privileged_obs = n_priv_obs_single

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
        rand_reset = True
        track_root = False
     
        # TODO: define weights for soccer purposes
        dof_err_w = [0.8, 0.8, 0.8, 1.0, # Left Arm
                     0.8, 0.8, 0.8, 1.0, # Right Arm
                     0.6, # waist
                     1.0, 0.8, 0.8, 1.0, 0.5, 0.5, # Left Leg
                     1.0, 0.8, 0.8, 1.0, 0.5, 0.5, # Right Leg
                     ]
        

        
        global_obs = False
        # global_obs = True
    
    class terrain(HumanoidMimicCfg.terrain):
        mesh_type = 'trimesh'
        # mesh_type = 'plane'
        # height = [0, 0.02]
        height = [0, 0.00]
        horizontal_scale = 0.1
    
    class init_state(HumanoidMimicCfg.init_state):
        pos = [0, 0, 1.0]
        default_joint_angles = {
            #'AAHead_yaw': 0.0,
            #'Head_pitch': 0.0,
            'Left_Shoulder_Pitch': 0.0,
            'Left_Shoulder_Roll': 0.0,
            'Left_Elbow_Pitch': 0.0,
            'Left_Elbow_Yaw': 0.0,
            'Right_Shoulder_Pitch': 0.0,
            'Right_Shoulder_Roll': 0.0,
            'Right_Elbow_Pitch': 0.0,
            'Right_Elbow_Yaw': 0.0,
            'Waist': 0.0,
            'Left_Hip_Pitch': 0.0,
            'Left_Hip_Roll': 0.0,
            'Left_Hip_Yaw': 0.0,
            'Left_Knee_Pitch': 0.0,
            'Left_Ankle_Pitch': 0.0,
            'Left_Ankle_Roll': 0.0,
            'Right_Hip_Pitch': 0.0,
            'Right_Hip_Roll': 0.0,
            'Right_Hip_Yaw': 0.0,
            'Right_Knee_Pitch': 0.0,
            'Right_Ankle_Pitch': 0.0,
            'Right_Ankle_Roll': 0.0,
        }
    
    class control(HumanoidMimicCfg.control):
        stiffness_pre = {"Hip": 100., "Knee": 100., "Ankle": 50., "Head": 20., "Shoulder": 20., "Elbow": 20., "Waist": 100.} # [N*m/rad]
        damping_pre = {"Hip": 2., "Knee": 2., "Ankle": 1., "Head": 2., "Shoulder": 2., "Elbow": 2., "Waist": 2.} # [N*m*s/rad]
        stiffness = {
            #"AAHead_yaw":            stiffness_pre["Head"],
            #"Head_pitch":            stiffness_pre["Head"],

            "Left_Shoulder_Pitch":   stiffness_pre["Shoulder"],
            "Left_Shoulder_Roll":    stiffness_pre["Shoulder"],
            "Left_Elbow_Pitch":      stiffness_pre["Elbow"],
            "Left_Elbow_Yaw":        stiffness_pre["Elbow"],

            "Waist":                 stiffness_pre["Hip"],

            "Left_Hip_Yaw":          stiffness_pre["Hip"],
            "Left_Hip_Roll":         stiffness_pre["Hip"],
            "Left_Hip_Pitch":        stiffness_pre["Hip"],
            "Left_Knee_Pitch":       stiffness_pre["Knee"],
            "Left_Ankle_Pitch":      stiffness_pre["Ankle"],
            "Left_Ankle_Roll":       stiffness_pre["Ankle"],

            "Right_Shoulder_Pitch":  stiffness_pre["Shoulder"],
            "Right_Shoulder_Roll":   stiffness_pre["Shoulder"],
            "Right_Elbow_Pitch":     stiffness_pre["Elbow"],
            "Right_Elbow_Yaw":       stiffness_pre["Elbow"],

            "Right_Hip_Yaw":         stiffness_pre["Hip"],
            "Right_Hip_Roll":        stiffness_pre["Hip"],
            "Right_Hip_Pitch":       stiffness_pre["Hip"],
            "Right_Knee_Pitch":      stiffness_pre["Knee"],
            "Right_Ankle_Pitch":     stiffness_pre["Ankle"],
            "Right_Ankle_Roll":      stiffness_pre["Ankle"],
        }

        damping = {
            #"AAHead_yaw":            damping_pre["Head"],
            #"Head_pitch":            damping_pre["Head"],

            "Left_Shoulder_Pitch":   damping_pre["Shoulder"],
            "Left_Shoulder_Roll":    damping_pre["Shoulder"],
            "Left_Elbow_Pitch":      damping_pre["Elbow"],
            "Left_Elbow_Yaw":        damping_pre["Elbow"],

            "Waist":                 damping_pre["Hip"],

            "Left_Hip_Yaw":          damping_pre["Hip"],
            "Left_Hip_Roll":         damping_pre["Hip"],
            "Left_Hip_Pitch":        damping_pre["Hip"],
            "Left_Knee_Pitch":       damping_pre["Knee"],
            "Left_Ankle_Pitch":      damping_pre["Ankle"],
            "Left_Ankle_Roll":       damping_pre["Ankle"],

            "Right_Shoulder_Pitch":  damping_pre["Shoulder"],
            "Right_Shoulder_Roll":   damping_pre["Shoulder"],
            "Right_Elbow_Pitch":     damping_pre["Elbow"],
            "Right_Elbow_Yaw":       damping_pre["Elbow"],

            "Right_Hip_Yaw":         damping_pre["Hip"],
            "Right_Hip_Roll":        damping_pre["Hip"],
            "Right_Hip_Pitch":       damping_pre["Hip"],
            "Right_Knee_Pitch":      damping_pre["Knee"],
            "Right_Ankle_Pitch":     damping_pre["Ankle"],
            "Right_Ankle_Roll":      damping_pre["Ankle"],
        }
        
        action_scale = 1
        decimation = 10  # 50 Hz Policy (Target: 50 Hz as per T1 spec)
        # decimation = 5  # 200 Hz (old)
    
    class sim(HumanoidMimicCfg.sim):
        dt = 0.002  # 500 Hz Motor (Target: 500 Hz as per T1 spec)
        # dt = 0.001 # 1000 Hz (old)
        
    class normalization(HumanoidMimicCfg.normalization):
        clip_actions = 5.0
    
    class asset(HumanoidMimicCfg.asset):
        file = f'{LEGGED_GYM_ROOT_DIR}/../assets/booster_t1/T1_serial_modified_3.urdf'
        
        # for both joint and link name
        #really unsure
        torso_name: str = 'Waist'  # humanoid pelvis part
        chest_name: str = 'Trunk'  # humanoid chest part
        
        # for link name
        thigh_name: str = 'Hip'
        shank_name: str = 'Shank'
        foot_name: str = 'foot_link'  # foot_pitch is not used - used to find feet by substring matching
        waist_name: list = ['Trunk']
        upper_arm_name: list = ['AR3', 'AL3']
        lower_arm_name: str = 'hand_link'
        hand_name: list = ['left_hand_tip', 'right_hand_tip']

        #left_feet_bodies = ['left_foot_link', 'left_inner_toe_link', 'left_outer_toe_link']
        #right_feet_bodies = ['right_foot_link', 'right_inner_toe_link', 'right_outer_toe_link']
        left_feet_bodies = ['left_foot_link']
        right_feet_bodies = ['right_foot_link']

        feet_bodies = left_feet_bodies + right_feet_bodies # exact names for force sensors
        n_lower_body_dofs: int = 12
        #TODO: weitermachen

        penalize_contacts_on = ["AR", "AL", "Hip", "Shank"]  # Fixed typo: Schank -> Shank
        terminate_after_contacts_on = ['Trunk']
        
        
        # ========================= Inertia =========================
        # Reference values from booster (see assets/booster_t1/refernces.txt)
        # Arms (Shoulder Pitch/Roll, Elbow Pitch/Yaw): 0.0282528
        # Waist: 0.0478125
        # Legs - Hip Pitch: 0.0523908, Hip Roll: 0.0478125, Hip Yaw: 0.0478125, Knee Pitch: 0.0636012
        # Feet (Ankle Pitch/Roll): 0.0339552
        
        # dof_armature for T1: 4 arm joints * 2 + 1 waist + 6 leg joints * 2 = 21 total
        # Order: Left arm (4) + Right arm (4) + Waist (1) + Left leg (6) + Right leg (6)
        dof_armature = [
            # Left Arm: Shoulder_Pitch, Shoulder_Roll, Elbow_Pitch, Elbow_Yaw
            0.0282528, 0.0282528, 0.0282528, 0.0282528,
            # Right Arm: Shoulder_Pitch, Shoulder_Roll, Elbow_Pitch, Elbow_Yaw
            0.0282528, 0.0282528, 0.0282528, 0.0282528,
            # Waist
            0.0478125,
            # Left Leg: Hip_Pitch, Hip_Roll, Hip_Yaw, Knee_Pitch, Ankle_Pitch, Ankle_Roll
            0.0523908, 0.0478125, 0.0478125, 0.0636012, 0.0339552, 0.0339552,
            # Right Leg: Hip_Pitch, Hip_Roll, Hip_Yaw, Knee_Pitch, Ankle_Pitch, Ankle_Roll
            0.0523908, 0.0478125, 0.0478125, 0.0636012, 0.0339552, 0.0339552,
        ]
        
        # ========================= Inertia =========================
        
        collapse_fixed_joints = False
    
    class rewards(HumanoidMimicCfg.rewards):
        regularization_names = [
                        # "feet_stumble",
                        # "feet_contact_forces",
                        # "lin_vel_z",
                        # "ang_vel_xy",
                        # "orientation",
                        # "dof_pos_limits",
                        # "dof_torque_limits",
                        # "collision",
                        # "torque_penalty",
                        # "thigh_torque_roll_yaw",
                        # "thigh_roll_yaw_acc",
                        # "dof_acc",
                        # "dof_vel",
                        # "action_rate",
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
            # tracking_keybody_pos = 0.6
            tracking_keybody_pos = 2.5
            
            # alive = 0.5

            feet_slip = -0.1
            feet_contact_forces = -5e-4      
            # collision = -10.0
            feet_stumble = -1.25
            
            dof_pos_limits = -5.0
            dof_torque_limits = -1.0
            
            dof_vel = -1e-4
            dof_acc = -1e-7
            action_rate = -0.01
            
            # feet_height = 5.0
            feet_air_time = 5.0
            
            
            ang_vel_xy = -0.01
            # orientation = -0.4
            
            # base_acc = -5e-7
            # orientation = -1.0
            
            # =========================
            # waist_dof_acc = -5e-8 * 2
            # waist_dof_vel = -1e-4 * 2
            
            ankle_dof_acc = -1e-7
            ankle_dof_vel = -2e-4
            
            hip_dof_acc = -2e-7
            hip_dof_vel = -1e-7
            
            # ankle_action = -0.02
            

        min_dist = 0.1
        max_dist = 0.4
        max_knee_dist = 0.4
        feet_height_target = 0.2
        feet_air_time_target = 0.5
        only_positive_rewards = False
        tracking_sigma = 0.2
        tracking_sigma_ang = 0.125
        max_contact_force = 500  # Forces above this value are penalized
        soft_torque_limit = 0.95
        torque_safety_limit = 0.9
        root_height_diff_threshold = 0.2

    class domain_rand:
        domain_rand_general = True # manually set this, setting from parser does not work;
        
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
        # push_end_effector = False
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
        #
        key_bodies = ["right_hand_tip", "left_hand_tip", "left_foot_link", "right_foot_link","right_outer_toe_link","left_outer_toe_link","right_inner_toe_link","left_inner_toe_link","Hip_Pitch_Right","Hip_Pitch_Left", "H2"] # 11 key bodies
        upper_key_bodies = ["right_hand_tip", "left_hand_tip", "H2"] #["right_hand_tip", "left_hand_tip", "AR3","AL3","H2"]
        
        motion_file = f"{LEGGED_GYM_ROOT_DIR}/motion_data_configs/twist_dataset.yaml"
        
        reset_consec_frames = 30
    

class T1MimicStuCfg(T1MimicPrivCfg):
    class env(T1MimicPrivCfg.env):
        tar_obs_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                         50, 55, 60, 65, 70, 75, 80, 85, 90, 95,]
        
        num_envs = 4096 * 7 // 8
        num_actions = 21
        obs_type = 'student'
        n_priv_latent = 4 + 1 + 2*num_actions
        extra_critic_obs = 3
        n_priv = 0
        
        n_proprio = 3 + 2 + 3*num_actions
        n_priv_mimic_obs = len(tar_obs_steps) * (8 + num_actions + 3*11) # Hardcode for now: 11 key bodies (see motion.key_bodies)
        n_mimic_obs = 8 + 21 # 21 for dof pos
        
        n_priv_info = 3 + 1 + 3*11 + 2 + 4 + 1 + 2*num_actions # base lin vel, root height, 11 key body pos, contact mask, priv latent
        history_len = 10
        
        n_obs_single = n_mimic_obs + n_proprio
        n_priv_obs_single = n_priv_mimic_obs + n_proprio + n_priv_info
        
        num_observations = n_obs_single * (history_len + 1)

        num_privileged_obs = n_priv_obs_single

class T1MimicStuRLCfg(T1MimicPrivCfg):
    class env(T1MimicPrivCfg.env):
        tar_obs_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                         50, 55, 60, 65, 70, 75, 80, 85, 90, 95,]
        
        num_envs = 4096 * 7 // 8
        num_actions = 21
        obs_type = 'student'
        n_priv_latent = 4 + 1 + 2*num_actions
        extra_critic_obs = 3
        n_priv = 0
        
        n_proprio = 3 + 2 + 3*num_actions
        n_priv_mimic_obs = len(tar_obs_steps) * (8 + num_actions + 3*11) # Hardcode for now: 11 key bodies (see motion.key_bodies)
        n_mimic_obs = 8 + 21 # 21 for dof pos
        
        n_priv_info = 3 + 1 + 3*11 + 2 + 4 + 1 + 2*num_actions # base lin vel, root height, 11 key body pos, contact mask, priv latent
        history_len = 10
        
        n_obs_single = n_mimic_obs + n_proprio
        n_priv_obs_single = n_priv_mimic_obs + n_proprio + n_priv_info
        
        num_observations = n_obs_single * (history_len + 1)

        num_privileged_obs = n_priv_obs_single
    
    class rewards(HumanoidMimicCfg.rewards):
        regularization_names = [
                        # "feet_stumble",
                        # "feet_contact_forces", 
                        # "lin_vel_z",
                        # "ang_vel_xy",
                        # "orientation",
                        # "dof_pos_limits",
                        # "dof_torque_limits",
                        # "collision",
                        # "torque_penalty",
                        # "thigh_torque_roll_yaw",
                        # "thigh_roll_yaw_acc",
                        # "dof_acc",
                        # "dof_vel",
                        # "action_rate",
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
            # tracking_keybody_pos = 0.6
            tracking_keybody_pos = 2.5
            
            # alive = 0.5

            feet_slip = -0.1 # same as teacher now
            feet_contact_forces = -5e-4      
            # collision = -10.0
            feet_stumble = -1.25
            
            dof_pos_limits = -5.0
            dof_torque_limits = -1.0
            
            dof_vel = -1e-4
            dof_acc = -1e-7
            action_rate = -0.01
            
            feet_air_time = 5.0
            
            
            ang_vel_xy = -0.01
            # orientation = -0.4
            
            # base_acc = -5e-7
            # orientation = -1.0
            
            # =========================
            # waist_dof_acc = -5e-8 * 2
            # waist_dof_vel = -1e-4 * 2
            
            ankle_dof_acc = -1e-7
            ankle_dof_vel = -2e-4
            
            hip_dof_acc = -2e-7
            hip_dof_vel = -1e-7
            
            # ankle_action = -0.02
            

        min_dist = 0.1
        max_dist = 0.4
        max_knee_dist = 0.4
        feet_height_target = 0.2
        feet_air_time_target = 0.5
        only_positive_rewards = False
        tracking_sigma = 0.2
        tracking_sigma_ang = 0.125
        max_contact_force = 500  # Forces above this value are penalized
        soft_torque_limit = 0.95
        torque_safety_limit = 0.9
        root_height_diff_threshold = 0.2

class T1MimicPrivCfgPPO(HumanoidMimicCfgPPO):
    seed = 1
    class runner(HumanoidMimicCfgPPO.runner):
        policy_class_name = 'ActorCriticMimic'
        algorithm_class_name = 'PPO'
        runner_class_name = 'OnPolicyRunnerMimic'
        max_iterations = 30_002 # number of policy updates

        # logging
        save_interval = 500 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
    
    class algorithm(HumanoidMimicCfgPPO.algorithm):
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        std_schedule = [1.0, 0.4, 4000, 1500]
        entropy_coef = 0.005
        
        # Transformer params
        # learning_rate = 1e-4 #1.e-3 #5.e-4
        # schedule = 'fixed' # could be adaptive, fixed
    
    class policy(HumanoidMimicCfgPPO.policy):
        action_std = [0.7] * 12 + [0.4] * 1 + [0.5] * 8
        init_noise_std = 1.0
        obs_context_len = 11
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = 'silu'
        layer_norm = True
        motion_latent_dim = 128
        


class T1MimicStuRLCfgDAgger(T1MimicStuRLCfg):
    seed = 1
    
    class teachercfg(T1MimicPrivCfgPPO):
        pass
    
    class runner(T1MimicPrivCfgPPO.runner):
        policy_class_name = 'ActorCriticMimic'
        algorithm_class_name = 'DaggerPPO'
        runner_class_name = 'OnPolicyDaggerRunner'
        max_iterations = 30_002
        warm_iters = 100
        
        # logging
        save_interval = 500
        experiment_name = 'test'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        
        teacher_experiment_name = 'test'
        teacher_proj_name = 't1_priv_mimic'
        teacher_checkpoint = -1
        eval_student = False

    class algorithm(HumanoidMimicCfgPPO.algorithm):
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        std_schedule = [1.0, 0.4, 4000, 1500]
        entropy_coef = 0.005
        
        dagger_coef_anneal_steps = 60000  # Total steps to anneal dagger_coef to dagger_coef_min
        
        dagger_coef = 0.1
        dagger_coef_min = 0.01  # Minimum value for dagger_coef
        # dagger_coef = 0.0
        # dagger_coef_min = 0.0  # Minimum value for dagger_coef

    class policy(HumanoidMimicCfgPPO.policy):
        action_std = [0.7] * 12 + [0.4] * 1 + [0.5] * 8
        init_noise_std = 1.0
        obs_context_len = 11
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = 'silu'
        layer_norm = True
        motion_latent_dim = 128