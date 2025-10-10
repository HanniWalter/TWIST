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

import sys
from isaacgym import gymapi
from isaacgym import gymutil, gymtorch
import numpy as np
import torch
import time

# Base class for RL tasks
class BaseTask():

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.gym = gymapi.acquire_gym()

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless
        self.debug_real_time = getattr(cfg.env, "debug_real_time", False)
        self.debug_real_time_verbose = getattr(cfg.env, "debug_real_time_verbose", False)
        self._real_time_resync_threshold = getattr(cfg.env, "debug_real_time_resync_threshold", 0.1)
        self._real_time_next_tick = None
        self._real_time_start = None
        self._sim_time_accum = 0.0

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        if self.headless == True:
            self.graphics_device_id = -1

        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions
        
        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        else: 
            self.privileged_obs_buf = None
            # self.num_privileged_obs = self.num_obs

        self.extras = {}

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_F, "free_cam")
            for i in range(9):
                self.gym.subscribe_viewer_keyboard_event(
                self.viewer, getattr(gymapi, "KEY_"+str(i)), "lookat"+str(i))
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_LEFT_BRACKET, "prev_id")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_RIGHT_BRACKET, "next_id")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_SPACE, "pause")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_W, "vx_plus")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_S, "vx_minus")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_A, "left_turn")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_D, "right_turn")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_MINUS, "prev_motion")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_EQUAL, "next_motion")
        self.free_cam = False
        self.lookat_id = 0
        self.lookat_vec = torch.tensor([-0, 2, 1], requires_grad=False, device=self.device)
        self.button_pressed = False

    def get_observations(self):
        return self.obs_buf
    
    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def step(self, actions):
        raise NotImplementedError

    def lookat(self, i):
        look_at_pos = self.root_states[i, :3].clone()
        cam_pos = look_at_pos + self.lookat_vec
        self.set_camera(cam_pos, look_at_pos)

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()
            if not self.free_cam:
                self.lookat(self.lookat_id)
            # check for keyboard events
            evt_count = 0
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                    self._reset_real_time_clock()
                
                if not self.free_cam:
                    for i in range(9):
                        if evt.action == "lookat" + str(i) and evt.value > 0:
                            self.lookat(i)
                            self.lookat_id = i
                    if evt.action == "prev_id" and evt.value > 0:
                        self.lookat_id  = (self.lookat_id-1) % self.num_envs
                        self.lookat(self.lookat_id)
                    if evt.action == "next_id" and evt.value > 0:
                        self.lookat_id  = (self.lookat_id+1) % self.num_envs
                        self.lookat(self.lookat_id)
                    if evt.action == "vx_plus" and evt.value > 0:
                        self.commands[self.lookat_id, 0] += 0.1
                    if evt.action == "vx_minus" and evt.value > 0:
                        self.commands[self.lookat_id, 0] -= 0.1
                    if evt.action == "left_turn" and evt.value > 0:
                        self.commands[self.lookat_id, 2] -= 0.05
                    if evt.action == "right_turn" and evt.value > 0:
                        self.commands[self.lookat_id, 2] += 0.05
                    if evt.action == "next_motion" and evt.value > 0:
                        self._motion_ids[self.lookat_id] = (self._motion_ids[self.lookat_id] + 1) % self._motion_lib.num_motions()
                        print("curr motion name: ", self.motion_names[self._motion_ids[self.lookat_id]])
                    if evt.action == "prev_motion" and evt.value > 0:
                        self._motion_ids[self.lookat_id] = (self._motion_ids[self.lookat_id] - 1) % self._motion_lib.num_motions()
                        print("curr motion name: ", self.motion_names[self._motion_ids[self.lookat_id]])
                if evt.action == "free_cam" and evt.value > 0:
                    self.free_cam = not self.free_cam
                    if self.free_cam:
                        self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
                
                if evt.action == "pause" and evt.value > 0:
                    self.pause = True
                    self._reset_real_time_clock()
                    while self.pause:
                        time.sleep(0.1)
                        self.gym.draw_viewer(self.viewer, self.sim, True)
                        for pause_evt in self.gym.query_viewer_action_events(self.viewer):
                            if pause_evt.action == "pause" and pause_evt.value > 0:
                                self.pause = False
                                self._reset_real_time_clock()
                        if self.gym.query_viewer_has_closed(self.viewer):
                            sys.exit()
                if evt.value > 0:
                    evt_count += 1
            self.button_pressed = True if evt_count > 0 else False

                        
                
            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            self.gym.poll_viewer_events(self.viewer)
            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time and not self.debug_real_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)
            
            if not self.free_cam:
                p = self.gym.get_viewer_camera_transform(self.viewer, None).p
                cam_trans = torch.tensor([p.x, p.y, p.z], requires_grad=False, device=self.device)
                look_at_pos = self.root_states[self.lookat_id, :3].clone()
                self.lookat_vec = cam_trans - look_at_pos
            

    def _reset_real_time_clock(self):
        self._real_time_next_tick = None
        self._real_time_start = None
        self._sim_time_accum = 0.0

    def _sync_real_time(self, duration):
        if not (self.debug_real_time and self.viewer and self.enable_viewer_sync):
            return

        now = time.perf_counter()

        if self._real_time_next_tick is None:
            self._real_time_next_tick = now
            self._real_time_start = now
            self._sim_time_accum = 0.0

        self._real_time_next_tick += duration
        self._sim_time_accum += duration

        sleep_time = self._real_time_next_tick - now

        if sleep_time > 0:
            time.sleep(sleep_time)
            if self.debug_real_time_verbose:
                wake_drift = time.perf_counter() - self._real_time_next_tick
                if abs(wake_drift) > 1e-4:
                    print(f"[RealTime] woke with drift {wake_drift:.6f}s (dt={duration:.6f}s)")
        else:
            drift = -sleep_time
            if self.debug_real_time_verbose:
                sim_time = self._sim_time_accum
                real_time = now - self._real_time_start if self._real_time_start is not None else 0.0
                print(f"[RealTime] behind by {drift:.6f}s (dt={duration:.6f}s, sim={sim_time:.3f}s, real={real_time:.3f}s)")
            if drift > self._real_time_resync_threshold:
                if self.debug_real_time_verbose:
                    print(f"[RealTime] resync clock after drift {drift:.6f}s")
                now = time.perf_counter()
                self._real_time_next_tick = now
                self._real_time_start = now
                self._sim_time_accum = 0.0

