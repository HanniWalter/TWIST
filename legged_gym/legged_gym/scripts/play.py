# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os

from legged_gym.envs import *
from legged_gym.gym_utils import get_args, task_registry
import torch
import numpy as np
import faulthandler
from tqdm import tqdm
from termcolor import cprint

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="jit"):
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint

def set_play_cfg(env_cfg):
    env_cfg.env.num_envs = 2#2 if not args.num_envs else args.num_envs
    env_cfg.env.episode_length_s = 60
    # env_cfg.commands.resampling_time = 60
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True
    
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 5
    env_cfg.domain_rand.max_push_vel_xy = 2.5
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.action_delay = False
    
    if hasattr(env_cfg, "motion"):
        env_cfg.motion.motion_curriculum = False


def play(args):
    faulthandler.enable()
    log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    set_play_cfg(env_cfg)

    env_cfg.env.record_video = args.record_video
    env_cfg.env.rand_reset = False
    
    if_normalize = env_cfg.env.normalize_obs
    cprint(f"if_normalize: {if_normalize}", "green")
    
    if env_cfg.env.record_video:
        env_cfg.env.episode_length_s = 10

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    #ai says this needs to be set
    #env.enable_viewer_sync = True
    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True)

    if args.use_jit:
        path = os.path.join(log_pth, "traced")
        model, checkpoint = get_load_path(root=path, checkpoint=args.checkpoint)
        path = os.path.join(path, model)
        print("Loading jit for policy: ", path)
        policy_jit = torch.jit.load(path, map_location=env.device)
    else:
        policy = ppo_runner.get_inference_policy(device=env.device)
        if if_normalize:
            try:
                normalizer = ppo_runner.get_normalizer(device=env.device)
            except:
                print("No normalizer found")
                normalizer = None

    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device, requires_grad=False)

    if args.record_video:
        mp4_writers = []
        import imageio
        env.enable_viewer_sync = True
        # env.enable_viewer_sync = False
        for i in range(env.num_envs):
            video_name = args.proj_name + "-" + args.exptid +".mp4"
            run_name = log_pth.split("/")[-1]
            path = f"../../logs/videos_retarget/{run_name}"
            if not os.path.exists(path):
                os.makedirs(path)
            video_name = os.path.join(path, video_name)
            mp4_writer = imageio.get_writer(video_name, fps=int(1/env.dt))
            cprint(f"Recording video to {video_name}", "green")
            mp4_writers.append(mp4_writer)

    if args.record_log:
        import json
        run_name = log_pth.split("/")[-1]
        logs_dict = []
        dict_name = args.proj_name + "-" + args.exptid + ".json"
        path = f"../../logs/env_logs/{run_name}"
        if not os.path.exists(path):
            os.makedirs(path)
        dict_name = os.path.join(path, dict_name)
        
    
    if not (args.record_video or args.record_log):
        traj_length = 100*int(env.max_episode_length)
    else:
        traj_length = 2 * int(env.max_episode_length)
        
    env_id = env.lookat_id

    # Test with real-world debugging input ONCE before the main loop
    #twist_expected_output = [1.40007, 0.731191, 1.18439, -1.06781, 1.01839, 0.107984, 0.987685, 0.578106, -0.339, 0.380429, -0.055434, 0.595255, -0.417355, -0.198895, -0.299504, -0.37781, -0.236548, 0.703261, -0.172993, -0.0959824]
    
    # Create tensor from twist_input and run through policy
    #twist_obs = torch.tensor(twist_input, dtype=torch.float32, device=env.device).unsqueeze(0)  # Add batch dimension
    #cprint("=" * 80, "yellow")
    #cprint("DEBUGGING: Running policy with real-world twist input", "yellow")
    #cprint(f"Input shape: {twist_obs.shape}", "yellow")
    
    #if args.use_jit:
    #    print("using jit")
    #    twist_actions = policy_jit(twist_obs)
    #else:
    #    if if_normalize and normalizer is not None:
    #        print("normalizing twist obs")
    #        twist_obs_normalized = normalizer.normalize(twist_obs)
    #    else:
    #        print("not normalizing twist obs")
    #        twist_obs_normalized = twist_obs
    #    twist_actions = policy(twist_obs_normalized, hist_encoding=True)
    
    #cprint("=" * 80, "green")
    #cprint("POLICY OUTPUT with real-world input:", "green")
    #print(twist_actions.detach().cpu().numpy().tolist()[0])
    #cprint("=" * 80, "green")
    #cprint("EXPECTED OUTPUT:", "cyan")
    #print(twist_expected_output)
    #cprint("=" * 80, "cyan")


    #1000 steps
    for i in tqdm(range(traj_length)):
        if args.use_jit:
            print("using jit")
            actions = policy_jit(obs.detach())
            print("jit policy actions")
        else:
            if if_normalize and normalizer is not None:
                normalized_obs = normalizer.normalize(obs.detach())
            else:
                normalized_obs = obs.detach()
                print("unnormalized obs")
            actions = policy(normalized_obs, hist_encoding=True)
            #print(type(normalized_obs.detach()))
            #print(normalized_obs.detach().shape)
            #torch.Size([1, 1023])

            # ============================================================
            # OBSERVATION SPACE BREAKDOWN (1023 total dimensions)
            # ============================================================
            # Based on k1_mimic_distill_config.py (K1MimicStuCfg)
            # Structure: 11 frames stacked (1 current + 10 history)
            # Single frame observation = 93 dimensions
            # Total = 93 * 11 = 1023
            #
            # Per-frame breakdown (93 dims each):
            # ------------------------------------------------------------
            # MOTION TARGETS (mimic_obs) - indices [0:28]
            #   [0:1]    - Target root height (1)
            #   [1:4]    - Target root Euler angles (roll, pitch, yaw) (3)
            #   [4:7]    - Target root linear velocity (3)
            #   [7:8]    - Target root yaw angular velocity (1)
            #   [8:28]   - Target joint positions (20)
            #
            # PROPRIOCEPTION (proprio_obs_buf) - indices [28:93]
            #   [28:31]  - Base angular velocity (3)
            #   [31:33]  - IMU (roll, pitch) (2)
            #   [33:53]  - Joint positions (dof_pos - default) (20)
            #   [53:73]  - Joint velocities (dof_vel) (20)
            #   [73:93]  - Previous actions (20)
            #
            # HISTORY STRUCTURE:
            #   Frame 0  (most recent):  indices [0:93]
            #   Frame 1  (t-1):          indices [93:186]
            #   Frame 2  (t-2):          indices [186:279]
            #   ...
            #   Frame 10 (t-10):         indices [930:1023]
            # ============================================================
            #print(type(normalized_obs.detach()))
            #print(normalized_obs.detach().shape)
            #torch.Size([1, 1023])



            print("xxx policy actions")
        obs, _, rews, dones, infos = env.step(actions.detach())

        if args.record_video:
            imgs = env.render_record(mode='rgb_array')
            if imgs is not None:
                for i in range(env.num_envs):
                    mp4_writers[i].append_data(imgs[i])
                    
        if args.record_log:
            log_dict = env.get_episode_log()
            logs_dict.append(log_dict)
        
        # Interaction
        if env.button_pressed:
            print(f"env_id: {env.lookat_id:<{5}}")
    
    if args.record_video:
        for mp4_writer in mp4_writers:
            mp4_writer.close()
            
    if args.record_log:
        with open(dict_name, 'w') as f:
            json.dump(logs_dict, f)
    

if __name__ == '__main__':
    args = get_args()
    play(args)