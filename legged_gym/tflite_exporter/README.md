# TFLite Exporter for TWIST Modified Student Policy

This directory contains tools to export trained TWIST **modified student** policies to TensorFlow Lite format for deployment on embedded devices.

> **Note**: This exporter is for the **modified student** architecture (`K1MimicStuRLCfg_modified`), which uses **future motion targets** instead of history frames.

## Quick Start

```bash
cd legged_gym/tflite_exporter
python tflite_exporter.py
```

This will convert `model/student_ready.pt` to `model/student_ready.tflite`.

## Model Verification

To verify model consistency between PyTorch and TFLite (and for C++ comparison):

```bash
# Run model verifier (generates CSV files for C++ verification)
python model_verifier.py

# Custom seed for reproducible random inputs
python model_verifier.py --seed 123

# Custom paths
python model_verifier.py --input model/student_ready.pt --output_dir verification_data
```

This generates CSV files in `verification_data/`:
- `input_*.csv` - Test inputs (all zeros + 10 random)
- `output_pytorch_*.csv` - PyTorch model outputs
- `output_tflite_*.csv` - TFLite model outputs
- `verification_data.csv` - Combined data for C++ comparison

Use these CSV files to verify the TFLite model produces identical outputs in your C++ environment.

## Usage

```bash
# Default (uses model/student_ready.pt -> model/student_ready.tflite)
python tflite_exporter.py

# Custom paths
python tflite_exporter.py --input /path/to/model.pt --output /path/to/model.tflite

# Short options
python tflite_exporter.py -i model.pt -o model.tflite
```

## Model Architecture

The exported TFLite model contains:
1. **Input Normalization**: Mean/std normalization learned during training
2. **Motion Encoder**: Encodes future motion targets into a latent representation
3. **Actor Network**: MLP that outputs action values

---

## Input Specification

**Input Shape**: `(batch_size, 1165)` (float32)

The input observation vector consists of **proprioceptive observations** (65 features) followed by **future motion targets** (1100 features):

```
obs = [proprio_obs (65), future_motion_targets (1100)]
```

### Observation Structure Overview

| Index Range | Size | Name | Description |
|-------------|------|------|-------------|
| 0-64 | 65 | **Proprioceptive Observation** | Current robot state |
| 65-1164 | 1100 | **Future Motion Targets** | 20 future timesteps × 55 features each |

---

### Proprioceptive Observation (65 features)

| Index | Size | Name | Description | Scale Factor |
|-------|------|------|-------------|--------------|
| 0-2 | 3 | `base_ang_vel` | Base angular velocity (x, y, z) | × obs_scales.ang_vel (0.25) |
| 3-4 | 2 | `imu_obs` | IMU roll and pitch angles | raw radians |
| 5-24 | 20 | `dof_pos` | Current joint positions (minus default, see table below) | × obs_scales.dof_pos (1.0) |
| 25-44 | 20 | `dof_vel` | Current joint velocities (see table below) | × obs_scales.dof_vel (0.05) |
| 45-64 | 20 | `last_actions` | Previous action commands (see table below) | raw |

**Note**: Ankle DOF velocities (indices 12, 13, 18, 19 within dof_vel) are zeroed out.

#### DOF Order (for `dof_pos`, `dof_vel`, and `last_actions`)

All three 20-element vectors use the same joint ordering:

| DOF Index | Obs Index (`dof_pos`) | Obs Index (`dof_vel`) | Obs Index (`last_actions`) | Joint Name | Description |
|-----------|----------------------|----------------------|---------------------------|------------|-------------|
| 0 | 5 | 25 | 45 | `ALeft_Shoulder_Pitch` | Left shoulder pitch |
| 1 | 6 | 26 | 46 | `Left_Shoulder_Roll` | Left shoulder roll |
| 2 | 7 | 27 | 47 | `Left_Elbow_Pitch` | Left elbow pitch |
| 3 | 8 | 28 | 48 | `Left_Elbow_Yaw` | Left elbow yaw |
| 4 | 9 | 29 | 49 | `ARight_Shoulder_Pitch` | Right shoulder pitch |
| 5 | 10 | 30 | 50 | `Right_Shoulder_Roll` | Right shoulder roll |
| 6 | 11 | 31 | 51 | `Right_Elbow_Pitch` | Right elbow pitch |
| 7 | 12 | 32 | 52 | `Right_Elbow_Yaw` | Right elbow yaw |
| 8 | 13 | 33 | 53 | `Left_Hip_Pitch` | Left hip pitch |
| 9 | 14 | 34 | 54 | `Left_Hip_Roll` | Left hip roll |
| 10 | 15 | 35 | 55 | `Left_Hip_Yaw` | Left hip yaw |
| 11 | 16 | 36 | 56 | `Left_Knee_Pitch` | Left knee pitch |
| 12 | 17 | 37 | 57 | `Left_Ankle_Pitch` | Left ankle pitch ⚠️ |
| 13 | 18 | 38 | 58 | `Left_Ankle_Roll` | Left ankle roll ⚠️ |
| 14 | 19 | 39 | 59 | `Right_Hip_Pitch` | Right hip pitch |
| 15 | 20 | 40 | 60 | `Right_Hip_Roll` | Right hip roll |
| 16 | 21 | 41 | 61 | `Right_Hip_Yaw` | Right hip yaw |
| 17 | 22 | 42 | 62 | `Right_Knee_Pitch` | Right knee pitch |
| 18 | 23 | 43 | 63 | `Right_Ankle_Pitch` | Right ankle pitch ⚠️ |
| 19 | 24 | 44 | 64 | `Right_Ankle_Roll` | Right ankle roll ⚠️ |

⚠️ **Ankle velocities are zeroed**: DOF indices 12, 13, 18, 19 have their `dof_vel` values set to 0.0 in the observation.

---

### Future Motion Targets (1100 features)

The model receives motion targets for **20 future timesteps**. Each timestep contains 55 features:

#### Future Timesteps

The `tar_obs_steps` define which future frames are included:
```python
tar_obs_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
```

These correspond to frames at approximately 0.02s, 0.1s, 0.2s, ... , 1.9s into the future (at 50Hz control rate).

#### Single Future Frame Structure (55 features per timestep)

| Index (within frame) | Size | Name | Description | Units |
|---------------------|------|------|-------------|-------|
| 0 | 1 | `root_height` | Target root height (z-coordinate) | meters |
| 1 | 1 | `roll` | Target roll angle | radians |
| 2 | 1 | `pitch` | Target pitch angle | radians |
| 3 | 1 | `yaw` | Target yaw angle | radians |
| 4-6 | 3 | `root_vel` | Target root linear velocity (x, y, z) in local frame | m/s |
| 7 | 1 | `root_ang_vel_yaw` | Target root angular velocity (yaw only) | rad/s |
| 8-27 | 20 | `dof_pos` | Target joint positions | radians |
| 28-54 | 27 | `key_body_pos` | Key body positions (9 bodies × 3 coords) | meters (relative) |

#### How Future Motion Targets Are Extracted from Training Data

The future motion targets come from **pre-recorded motion capture data** stored in `.pkl` files. During training, the simulation has access to the complete motion trajectory and can look ahead to future frames.

**Source Data Structure (`.pkl` files):**
```python
motion_data = {
    "root_pos": np.array,      # (num_frames, 3) - Root position [x, y, z]
    "root_rot": np.array,      # (num_frames, 4) - Root rotation quaternion [x, y, z, w]
    "dof_pos": np.array,       # (num_frames, 20) - Joint angles
    "local_body_pos": np.array, # (num_frames, num_bodies, 3) - Body positions relative to root
    "fps": int,                # Frame rate (typically 30 or 60 Hz)
    "link_body_list": list,    # Body link names
}
```

**Extraction Details:**

| Feature | Source Field | Transformation |
|---------|--------------|----------------|
| `root_height` | `root_pos[:, 2]` | Direct extraction (z-coordinate only) |
| `roll, pitch, yaw` | `root_rot` | Quaternion → Euler angles conversion |
| `root_vel` | `root_pos` | Finite difference: $v_t = \text{fps} \times (p_{t+1} - p_t)$, smoothed, rotated to local frame |
| `root_ang_vel_yaw` | `root_rot` | Quaternion diff → exponential map → smooth → local frame → z-component only |
| `dof_pos` | `dof_pos` | Direct extraction (with linear interpolation between frames) |
| `key_body_pos` | `local_body_pos` | Select 9 key bodies, positions are relative to root |

**Interpolation:** When querying times between motion capture frames, linear interpolation is used for positions/joints, and spherical linear interpolation (slerp) is used for quaternions.

**Important:** During deployment, you must provide these future motion targets from your motion planning system. The trained model expects to "see" the desired future trajectory to generate appropriate actions.

#### Future Motion Buffer Layout

| Timestep | Frame Index | Global Index Range | Time Offset |
|----------|-------------|-------------------|-------------|
| t+1 | 0 | 65-119 | +0.02s |
| t+5 | 1 | 120-174 | +0.10s |
| t+10 | 2 | 175-229 | +0.20s |
| t+15 | 3 | 230-284 | +0.30s |
| t+20 | 4 | 285-339 | +0.40s |
| t+25 | 5 | 340-394 | +0.50s |
| t+30 | 6 | 395-449 | +0.60s |
| t+35 | 7 | 450-504 | +0.70s |
| t+40 | 8 | 505-559 | +0.80s |
| t+45 | 9 | 560-614 | +0.90s |
| t+50 | 10 | 615-669 | +1.00s |
| t+55 | 11 | 670-724 | +1.10s |
| t+60 | 12 | 725-779 | +1.20s |
| t+65 | 13 | 780-834 | +1.30s |
| t+70 | 14 | 835-889 | +1.40s |
| t+75 | 15 | 890-944 | +1.50s |
| t+80 | 16 | 945-999 | +1.60s |
| t+85 | 17 | 1000-1054 | +1.70s |
| t+90 | 18 | 1055-1109 | +1.80s |
| t+95 | 19 | 1110-1164 | +1.90s |

**Total**: 65 (proprio) + 20 × 55 (future targets) = **1165 features**

---

## Deployment Reference (K1 Robot)

This section provides exact specifications for deploying the modified student policy on a real Booster K1 robot.

### 1. Policy Output DOF Order (20 Actions)

The policy outputs 20 action values in **URDF order** (as defined in `K1_serial_modified_slim.urdf`):

| Index | URDF Joint Name | Description |
|-------|-----------------|-------------|
| 0 | `ALeft_Shoulder_Pitch` | Left shoulder pitch |
| 1 | `Left_Shoulder_Roll` | Left shoulder roll |
| 2 | `Left_Elbow_Pitch` | Left elbow pitch |
| 3 | `Left_Elbow_Yaw` | Left elbow yaw |
| 4 | `ARight_Shoulder_Pitch` | Right shoulder pitch |
| 5 | `Right_Shoulder_Roll` | Right shoulder roll |
| 6 | `Right_Elbow_Pitch` | Right elbow pitch |
| 7 | `Right_Elbow_Yaw` | Right elbow yaw |
| 8 | `Left_Hip_Pitch` | Left hip pitch |
| 9 | `Left_Hip_Roll` | Left hip roll |
| 10 | `Left_Hip_Yaw` | Left hip yaw |
| 11 | `Left_Knee_Pitch` | Left knee pitch |
| 12 | `Left_Ankle_Pitch` | Left ankle pitch |
| 13 | `Left_Ankle_Roll` | Left ankle roll |
| 14 | `Right_Hip_Pitch` | Right hip pitch |
| 15 | `Right_Hip_Roll` | Right hip roll |
| 16 | `Right_Hip_Yaw` | Right hip yaw |
| 17 | `Right_Knee_Pitch` | Right knee pitch |
| 18 | `Right_Ankle_Pitch` | Right ankle pitch |
| 19 | `Right_Ankle_Roll` | Right ankle roll |

### 2. Policy Input DOF Order

**Input DOF order is the SAME as output** - no reindexing is applied.

The `reindex()` function in the K1 environment returns the vector unchanged (inherits from base class which is identity). The proprioceptive observations use the same URDF order as the actions.

### 3. Default Joint Positions

All default joint positions are **0.0 radians**:

```python
default_joint_angles = {
    'ALeft_Shoulder_Pitch': 0.0,
    'Left_Shoulder_Roll': 0.0,
    'Left_Elbow_Pitch': 0.0,
    'Left_Elbow_Yaw': 0.0,
    'ARight_Shoulder_Pitch': 0.0,
    'Right_Shoulder_Roll': 0.0,
    'Right_Elbow_Pitch': 0.0,
    'Right_Elbow_Yaw': 0.0,
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
```

### 4. Action Interpretation

| Parameter | Value |
|-----------|-------|
| `action_scale` | **1.0** |
| `clip_actions` | **5.0** (before scaling) |

**Formula:**
```python
# Clipping is applied BEFORE scaling
clip_limit = clip_actions / action_scale  # = 5.0 / 1.0 = 5.0
clipped_action = clip(raw_policy_output, -clip_limit, clip_limit)

# Target position calculation
target_position = default_dof_pos + action_scale * clipped_action
                = 0.0 + 1.0 * clipped_action
                = clipped_action  # Since defaults are 0 and scale is 1
```

**Important:** With `action_scale = 1.0` and `default_dof_pos = 0.0`, the policy output IS the target joint position in radians (after clipping to ±5.0 rad).

### 5. Joint Sign Conventions

**No sign flips** - the URDF joints match the policy output signs directly. The simulation uses the same URDF as deployment, so no sign conversion is needed.

### 6. Ankle Mechanism

The policy outputs `Ankle_Pitch` and `Ankle_Roll` directly. The URDF (`K1_serial_modified_slim.urdf`) **already abstracts the parallel mechanism** - you send pitch/roll commands, and the URDF/control handles the conversion to the actual crank-up/crank-down motors.

**Note:** Ankle DOF velocities are **zeroed out** in observations (indices 12, 13, 18, 19 within the dof_vel section):
```python
constant_ankle_dof_idx = [12, 13, 18, 19]
proprio_obs_buf[:, dof_vel_start_dim + ankle_idx] = 0.0
```

### 7. Observation Scaling

Apply these scales **before** feeding to the policy:

| Observation | Scale Factor | Formula |
|-------------|--------------|---------|
| `base_ang_vel` | 0.25 | `imu_gyro * 0.25` |
| `imu_obs` (roll, pitch) | 1.0 | raw radians |
| `dof_pos` | 1.0 | `(current_pos - default_pos) * 1.0` |
| `dof_vel` | 0.05 | `joint_velocity * 0.05` |
| `last_actions` | 1.0 | raw (from previous inference) |

### 8. Normalization

**Normalization IS baked into the TFLite model.** 

The exporter includes the learned mean/std from training:
```python
# Inside the TFLite model:
normalized_obs = (obs - mean) / (std + 1e-4)
normalized_obs = clamp(normalized_obs, -inf, inf)
```

You do **NOT** need to apply external normalization. Just apply the observation scales above and feed directly to the model.

### 9. Control Timing

| Parameter | Value | Description |
|-----------|-------|-------------|
| `sim_dt` | 0.002s (500 Hz) | Physics simulation timestep |
| `decimation` | 10 | Number of sim steps per policy step |
| **Policy frequency** | **50 Hz** | Policy runs at `500 Hz / 10 = 50 Hz` |
| **Policy period** | **20 ms** | Time between policy inferences |

**Important**: The policy expects to run at **50 Hz**. Running at different frequencies may affect behavior.

### 10. PD Controller Parameters (per joint)

The policy outputs target joint positions. You must apply a PD controller to compute motor torques:

```python
torque = Kp * (target_pos - current_pos) - Kd * current_vel
torque = clip(torque, -torque_limit, torque_limit)
```

| Index | Joint Name | Kp (N·m/rad) | Kd (N·m·s/rad) | Torque Limit (N·m) | Lower Limit (rad) | Upper Limit (rad) |
|-------|------------|--------------|----------------|--------------------|--------------------|-------------------|
| 0 | `ALeft_Shoulder_Pitch` | 4.0 | 1.0 | 10 | -3.316 | 1.22 |
| 1 | `Left_Shoulder_Roll` | 4.0 | 1.0 | 10 | -1.74 | 1.57 |
| 2 | `Left_Elbow_Pitch` | 4.0 | 1.0 | 10 | -2.27 | 2.27 |
| 3 | `Left_Elbow_Yaw` | 4.0 | 1.0 | 10 | -2.44 | 0.0 |
| 4 | `ARight_Shoulder_Pitch` | 4.0 | 1.0 | 10 | -3.316 | 1.22 |
| 5 | `Right_Shoulder_Roll` | 4.0 | 1.0 | 10 | -1.57 | 1.74 |
| 6 | `Right_Elbow_Pitch` | 4.0 | 1.0 | 10 | -2.27 | 2.27 |
| 7 | `Right_Elbow_Yaw` | 4.0 | 1.0 | 10 | 0.0 | 2.44 |
| 8 | `Left_Hip_Pitch` | 80.0 | 2.0 | 45 | -3.0 | 2.21 |
| 9 | `Left_Hip_Roll` | 80.0 | 2.0 | 30 | -0.4 | 1.57 |
| 10 | `Left_Hip_Yaw` | 80.0 | 2.0 | 30 | -1.0 | 1.0 |
| 11 | `Left_Knee_Pitch` | 80.0 | 2.0 | 45 | 0.0 | 2.23 |
| 12 | `Left_Ankle_Pitch` | 30.0 | 2.0 | 20 | -0.87 | 0.345 |
| 13 | `Left_Ankle_Roll` | 30.0 | 2.0 | 20 | -0.345 | 0.345 |
| 14 | `Right_Hip_Pitch` | 80.0 | 2.0 | 45 | -3.0 | 2.21 |
| 15 | `Right_Hip_Roll` | 80.0 | 2.0 | 30 | -1.57 | 0.4 |
| 16 | `Right_Hip_Yaw` | 80.0 | 2.0 | 30 | -1.0 | 1.0 |
| 17 | `Right_Knee_Pitch` | 80.0 | 2.0 | 45 | 0.0 | 2.23 |
| 18 | `Right_Ankle_Pitch` | 30.0 | 2.0 | 20 | -0.87 | 0.345 |
| 19 | `Right_Ankle_Roll` | 30.0 | 2.0 | 20 | -0.345 | 0.345 |

**Note**: These are the training values. You may need to tune Kp/Kd for your specific hardware.

### 11. Complete Deployment Code Example

```python
import numpy as np

# =============================================================================
# CONSTANTS
# =============================================================================
NUM_DOFS = 20
ACTION_SCALE = 1.0
CLIP_ACTIONS = 5.0
DEFAULT_DOF_POS = np.zeros(NUM_DOFS)
POLICY_FREQ_HZ = 50  # Policy runs at 50 Hz (every 20ms)

# Observation scales
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05

# Ankle indices (zero out velocities in observation)
ANKLE_IDX = [12, 13, 18, 19]

# PD Controller gains (Kp, Kd) per joint - from training config
KP = np.array([
    4.0, 4.0, 4.0, 4.0,      # Left arm: Shoulder_Pitch, Shoulder_Roll, Elbow_Pitch, Elbow_Yaw
    4.0, 4.0, 4.0, 4.0,      # Right arm: same
    80.0, 80.0, 80.0, 80.0, 30.0, 30.0,  # Left leg: Hip_P, Hip_R, Hip_Y, Knee, Ankle_P, Ankle_R
    80.0, 80.0, 80.0, 80.0, 30.0, 30.0,  # Right leg: same
], dtype=np.float32)

KD = np.array([
    1.0, 1.0, 1.0, 1.0,      # Left arm
    1.0, 1.0, 1.0, 1.0,      # Right arm
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0,  # Left leg
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0,  # Right leg
], dtype=np.float32)

# Torque limits per joint (N·m)
TORQUE_LIMITS = np.array([
    10, 10, 10, 10,          # Left arm
    10, 10, 10, 10,          # Right arm
    45, 30, 30, 45, 20, 20,  # Left leg: Hip_P, Hip_R, Hip_Y, Knee, Ankle_P, Ankle_R
    45, 30, 30, 45, 20, 20,  # Right leg
], dtype=np.float32)

# Joint position limits (rad) - [lower, upper]
JOINT_LIMITS = np.array([
    [-3.316, 1.22],   # 0: ALeft_Shoulder_Pitch
    [-1.74, 1.57],    # 1: Left_Shoulder_Roll
    [-2.27, 2.27],    # 2: Left_Elbow_Pitch
    [-2.44, 0.0],     # 3: Left_Elbow_Yaw
    [-3.316, 1.22],   # 4: ARight_Shoulder_Pitch
    [-1.57, 1.74],    # 5: Right_Shoulder_Roll
    [-2.27, 2.27],    # 6: Right_Elbow_Pitch
    [0.0, 2.44],      # 7: Right_Elbow_Yaw
    [-3.0, 2.21],     # 8: Left_Hip_Pitch
    [-0.4, 1.57],     # 9: Left_Hip_Roll
    [-1.0, 1.0],      # 10: Left_Hip_Yaw
    [0.0, 2.23],      # 11: Left_Knee_Pitch
    [-0.87, 0.345],   # 12: Left_Ankle_Pitch
    [-0.345, 0.345],  # 13: Left_Ankle_Roll
    [-3.0, 2.21],     # 14: Right_Hip_Pitch
    [-1.57, 0.4],     # 15: Right_Hip_Roll
    [-1.0, 1.0],      # 16: Right_Hip_Yaw
    [0.0, 2.23],      # 17: Right_Knee_Pitch
    [-0.87, 0.345],   # 18: Right_Ankle_Pitch
    [-0.345, 0.345],  # 19: Right_Ankle_Roll
], dtype=np.float32)

# =============================================================================
# FUNCTIONS
# =============================================================================

def build_proprio_obs(imu_gyro, imu_rpy, dof_pos, dof_vel, last_actions):
    """
    Build 65-dim proprioceptive observation.
    
    Args:
        imu_gyro: Angular velocity from IMU [wx, wy, wz] in rad/s
        imu_rpy: Roll, pitch, yaw from IMU [roll, pitch, yaw] in rad
        dof_pos: Current joint positions (20,) in rad
        dof_vel: Current joint velocities (20,) in rad/s
        last_actions: Previous policy output (20,)
    
    Returns:
        obs: Proprioceptive observation vector (65,)
    """
    obs = np.zeros(65, dtype=np.float32)
    
    # Base angular velocity (scaled)
    obs[0:3] = imu_gyro * ANG_VEL_SCALE
    
    # IMU roll, pitch (raw radians, NO yaw)
    obs[3:5] = imu_rpy[0:2]  # [roll, pitch]
    
    # DOF positions (offset from default, scaled)
    obs[5:25] = (dof_pos - DEFAULT_DOF_POS) * DOF_POS_SCALE
    
    # DOF velocities (scaled, with ankle velocities zeroed)
    dof_vel_scaled = dof_vel.copy() * DOF_VEL_SCALE
    dof_vel_scaled[ANKLE_IDX] = 0.0  # Zero out ankle velocities
    obs[25:45] = dof_vel_scaled
    
    # Last actions (raw)
    obs[45:65] = last_actions
    
    return obs

def policy_output_to_target_pos(raw_action):
    """
    Convert policy output to target joint positions.
    
    Args:
        raw_action: Raw policy output (20,)
    
    Returns:
        target_pos: Target joint positions (20,) in rad
    """
    # Clip actions
    clipped = np.clip(raw_action, -CLIP_ACTIONS, CLIP_ACTIONS)
    
    # Compute target positions
    target_pos = DEFAULT_DOF_POS + ACTION_SCALE * clipped
    
    # Optionally clip to joint limits for safety
    target_pos = np.clip(target_pos, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
    
    return target_pos

def compute_torques(target_pos, current_pos, current_vel):
    """
    Compute motor torques using PD control.
    
    Args:
        target_pos: Target joint positions from policy (20,) in rad
        current_pos: Current joint positions (20,) in rad
        current_vel: Current joint velocities (20,) in rad/s
    
    Returns:
        torques: Motor torques (20,) in N·m
    """
    # PD control: τ = Kp * (target - current) - Kd * velocity
    torques = KP * (target_pos - current_pos) - KD * current_vel
    
    # Clip to torque limits
    torques = np.clip(torques, -TORQUE_LIMITS, TORQUE_LIMITS)
    
    return torques

# =============================================================================
# MAIN CONTROL LOOP EXAMPLE
# =============================================================================

def run_control_loop(model, get_robot_state, send_torques, get_motion_targets):
    """
    Example control loop running at 50 Hz.
    
    Args:
        model: TFLite interpreter
        get_robot_state: Function returning (imu_gyro, imu_rpy, dof_pos, dof_vel)
        send_torques: Function to send torques to motors
        get_motion_targets: Function returning future motion targets (1100,)
    """
    import time
    
    last_actions = np.zeros(NUM_DOFS, dtype=np.float32)
    dt = 1.0 / POLICY_FREQ_HZ  # 20ms
    
    while True:
        t_start = time.time()
        
        # 1. Get robot state
        imu_gyro, imu_rpy, dof_pos, dof_vel = get_robot_state()
        
        # 2. Build observation
        proprio_obs = build_proprio_obs(imu_gyro, imu_rpy, dof_pos, dof_vel, last_actions)
        motion_targets = get_motion_targets()  # (1100,) future motion targets
        
        # 3. Concatenate to full observation
        obs = np.concatenate([proprio_obs, motion_targets]).astype(np.float32)
        obs = obs.reshape(1, -1)  # Add batch dimension
        
        # 4. Run inference
        model.set_tensor(model.get_input_details()[0]['index'], obs)
        model.invoke()
        raw_action = model.get_tensor(model.get_output_details()[0]['index']).squeeze()
        
        # 5. Convert to target positions
        target_pos = policy_output_to_target_pos(raw_action)
        
        # 6. Compute torques via PD control
        torques = compute_torques(target_pos, dof_pos, dof_vel)
        
        # 7. Send to motors
        send_torques(torques)
        
        # 8. Store for next iteration
        last_actions = raw_action.copy()
        
        # 9. Wait for next cycle
        elapsed = time.time() - t_start
        if elapsed < dt:
            time.sleep(dt - elapsed)
```

---

**Output Shape**: `(batch_size, 20)` (float32)

The output contains **20 action values** representing target joint position offsets.

### Output Processing Pipeline

The simulation applies these steps to the policy output:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  POLICY OUTPUT PROCESSING (what you must implement)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. raw_action = policy(observation)           # Shape: (20,)              │
│                                                                             │
│  2. clipped_action = clip(raw_action, -5.0, 5.0)                           │
│                                                                             │
│  3. target_pos = default_dof_pos + action_scale * clipped_action           │
│                = 0.0 + 1.0 * clipped_action                                │
│                = clipped_action  (since defaults=0, scale=1)               │
│                                                                             │
│  4. target_pos = clip(target_pos, joint_lower_limits, joint_upper_limits)  │
│     (optional but recommended for safety)                                   │
│                                                                             │
│  5. torque = Kp * (target_pos - current_pos) - Kd * current_vel            │
│                                                                             │
│  6. torque = clip(torque, -torque_limit, torque_limit)                     │
│                                                                             │
│  7. send torque to motors                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Joint Action Mapping

| Index | Joint Name | Description |
|-------|------------|-------------|
| 0 | `ALeft_Shoulder_Pitch` | Left shoulder pitch |
| 1 | `Left_Shoulder_Roll` | Left shoulder roll |
| 2 | `Left_Elbow_Pitch` | Left elbow pitch |
| 3 | `Left_Elbow_Yaw` | Left elbow yaw |
| 4 | `ARight_Shoulder_Pitch` | Right shoulder pitch |
| 5 | `Right_Shoulder_Roll` | Right shoulder roll |
| 6 | `Right_Elbow_Pitch` | Right elbow pitch |
| 7 | `Right_Elbow_Yaw` | Right elbow yaw |
| 8 | `Left_Hip_Pitch` | Left hip pitch |
| 9 | `Left_Hip_Roll` | Left hip roll |
| 10 | `Left_Hip_Yaw` | Left hip yaw |
| 11 | `Left_Knee_Pitch` | Left knee pitch |
| 12 | `Left_Ankle_Pitch` | Left ankle pitch |
| 13 | `Left_Ankle_Roll` | Left ankle roll |
| 14 | `Right_Hip_Pitch` | Right hip pitch |
| 15 | `Right_Hip_Roll` | Right hip roll |
| 16 | `Right_Hip_Yaw` | Right hip yaw |
| 17 | `Right_Knee_Pitch` | Right knee pitch |
| 18 | `Right_Ankle_Pitch` | Right ankle pitch |
| 19 | `Right_Ankle_Roll` | Right ankle roll |

### Action Interpretation

With `action_scale = 1.0` and `default_dof_pos = 0.0` for all joints:

```python
target_joint_position = 0.0 + 1.0 * clip(policy_output, -5.0, 5.0)
                      = clip(policy_output, -5.0, 5.0)  # in radians
```

**The policy output IS the target joint angle** (after clipping to ±5.0 rad).

---

## Key Bodies (for Motion Reference)

The model uses 9 key body positions for motion tracking:

| Index | Body Name | Description |
|-------|-----------|-------------|
| 0 | `left_hand_end_ball` | Left hand end effector |
| 1 | `right_hand_end_ball` | Right hand end effector |
| 2 | `left_foot_link` | Left foot |
| 3 | `right_foot_link` | Right foot |
| 4 | `right_outer_toe_link` | Right outer toe |
| 5 | `left_outer_toe_link` | Left outer toe |
| 6 | `right_inner_toe_link` | Right inner toe |
| 7 | `left_inner_toe_link` | Left inner toe |
| 8 | `Head_2` | Head |

---

## Example: Loading and Running the TFLite Model

### Python (with ai-edge-litert)

```python
import numpy as np
import ai_edge_litert.interpreter as tflite

# Load model
interpreter = tflite.Interpreter(model_path='model/student_ready.tflite')
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input (batch_size=1, features=1165)
# Structure: [proprio_obs (65), future_motion_targets (1100)]
observation = np.zeros((1, 1165), dtype=np.float32)

# Fill proprioceptive observations (indices 0-64)
# observation[0, 0:3] = base_ang_vel * 0.25
# observation[0, 3:5] = [roll, pitch]
# observation[0, 5:25] = (dof_pos - default_dof_pos)
# observation[0, 25:45] = dof_vel * 0.05
# observation[0, 45:65] = last_actions

# Fill future motion targets (indices 65-1164)
# 20 timesteps × 55 features each
# Each timestep: [root_h, roll, pitch, yaw, root_vel(3), root_ang_vel_yaw, dof_pos(20), key_body_pos(27)]

# Run inference
interpreter.set_tensor(input_details[0]['index'], observation)
interpreter.invoke()

# Get output actions
actions = interpreter.get_tensor(output_details[0]['index'])
print(f"Actions shape: {actions.shape}")  # (1, 20)
print(f"Actions: {actions}")
```

### C++ (with TensorFlow Lite)

```cpp
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

// Load model
auto model = tflite::FlatBufferModel::BuildFromFile("model/student_ready.tflite");
tflite::ops::builtin::BuiltinOpResolver resolver;
std::unique_ptr<tflite::Interpreter> interpreter;
tflite::InterpreterBuilder(*model, resolver)(&interpreter);
interpreter->AllocateTensors();

// Get input/output tensors
float* input = interpreter->typed_input_tensor<float>(0);
float* output = interpreter->typed_output_tensor<float>(0);

// Fill proprio observations (indices 0-64)
// input[0..2] = base_ang_vel * 0.25
// input[3..4] = roll, pitch
// input[5..24] = dof_pos - default_dof_pos
// input[25..44] = dof_vel * 0.05
// input[45..64] = last_actions

// Fill future motion targets (indices 65-1164)
// 20 timesteps × 55 features each
for (int step = 0; step < 20; step++) {
    int base = 65 + step * 55;
    // input[base + 0] = root_height
    // input[base + 1..3] = roll, pitch, yaw
    // input[base + 4..6] = root_vel
    // input[base + 7] = root_ang_vel_yaw
    // input[base + 8..27] = target_dof_pos
    // input[base + 28..54] = key_body_pos (9 bodies × 3)
}

// Run inference
interpreter->Invoke();

// Read output actions
for (int i = 0; i < 20; i++) {
    float action = output[i];
    // Apply action to robot...
}
```

---

## Coordinate Frames

- **Local Frame**: Robot-centric coordinate system
  - X: Forward
  - Y: Left
  - Z: Up
- All velocities are expressed in the **local (body) frame**
- Key body positions are relative to the robot's root (pelvis)

---

## Normalization

The model includes built-in input normalization. The normalization is applied as:

```python
normalized_obs = (obs - mean) / (std + eps)
normalized_obs = clamp(normalized_obs, -clip, clip)
```

Where:
- `mean`, `std`: Learned statistics from training
- `eps`: Small constant (1e-4) for numerical stability
- `clip`: Clipping value (default: inf)

**No external normalization is required** - just provide raw sensor data.

---

## File Structure

```
tflite_exporter/
├── README.md                 # This documentation
├── tflite_exporter.py        # Main exporter script
└── model/
    ├── student_ready.pt      # Input PyTorch checkpoint
    └── student_ready.tflite  # Output TFLite model
```

---

## Model Comparison

| Property | PyTorch (.pt) | TFLite (.tflite) |
|----------|---------------|------------------|
| File Size | ~27 MB | ~4.3 MB |
| Input Shape | (N, 1165) | (1, 1165) |
| Output Shape | (N, 20) | (1, 20) |
| Precision | float32 | float32 |
| Inference Device | GPU/CPU | CPU (optimized) |

---

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: rsl_rl**
   ```bash
   pip install -e ../../rsl_rl/
   ```

2. **ai-edge-torch not found**
   ```bash
   pip install ai-edge-torch
   ```

3. **Input shape mismatch**
   - Ensure input is `(1, 1165)` with dtype `float32`
   - Verify you're using the **modified student** model (future targets, no history)

4. **Output values seem wrong**
   - Verify normalization is handled internally by the model
   - Check action scaling before applying to robot

5. **Wrong model architecture**
   - This exporter is for `K1MimicStuRLCfg_modified` (future motion targets)
   - For the regular student with history, use the appropriate config

---

## Technical Details

### Network Architecture

```
Input (1165 features)
    │
    ├── Proprio (65) ──────────────────────────────────┐
    │                                                   │
    └── Future Motion Targets (1100) ──┐               │
                                       ▼               │
                            ┌─────────────────────┐    │
                            │   Normalization     │    │
                            └─────────────────────┘    │
                                       │               │
                                       ▼               │
                            ┌─────────────────────┐    │
                            │   Motion Encoder    │    │
                            │  Linear(55→60)      │    │
                            │  + SiLU             │    │
                            │  + Linear(60→128)   │    │
                            └─────────────────────┘    │
                                       │               │
                                       ▼               │
                            ┌─────────────────────┐    │
                            │   Concat            │◄───┘
                            │  [proprio, motion   │
                            │   first_step,       │
                            │   motion_latent]    │
                            └─────────────────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │   Actor Backbone    │
                            │  MLP [512, 512,     │
                            │       256, 128]     │
                            │  with LayerNorm     │
                            │  + SiLU             │
                            └─────────────────────┘
                                       │
                                       ▼
                            Output (20 actions)
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| num_observations | 1165 |
| n_proprio | 65 |
| n_priv_mimic_obs | 1100 |
| num_future_steps | 20 |
| features_per_step | 55 |
| motion_latent_dim | 128 |
| num_actions | 20 |
| actor_hidden_dims | [512, 512, 256, 128] |
| activation | SiLU |
| layer_norm | True |
| history_len | 0 (no history) |

---

## ✅ Deployment Checklist

Before deploying on real hardware, verify the following:

### Timing & Frequency
- [ ] Policy runs at **50 Hz** (every 20 ms)
- [ ] PD controller runs at higher frequency if needed (e.g., 500 Hz or 1 kHz)
- [ ] Latency from sensor read to torque command is minimized

### Observation Building
- [ ] Angular velocity from IMU is scaled by **0.25**
- [ ] Roll and pitch from IMU are in **radians** (not degrees)
- [ ] Joint positions are offset from default (which is 0.0 for all joints)
- [ ] Joint velocities are scaled by **0.05**
- [ ] **Ankle velocities (indices 12, 13, 18, 19) are zeroed** in observation
- [ ] `last_actions` stores the raw policy output from previous step (not clipped/scaled)
- [ ] Future motion targets are provided in correct format (20 steps × 55 features)

### Action Processing
- [ ] Raw policy output is clipped to **±5.0**
- [ ] Action scale is **1.0** (target = 0.0 + 1.0 × clipped_action)
- [ ] Target positions are optionally clipped to joint limits for safety

### PD Controller
- [ ] Kp values match training config (4.0 for arms, 80.0 for hip/knee, 30.0 for ankle)
- [ ] Kd values match training config (1.0 for arms, 2.0 for legs)
- [ ] Torques are clipped to per-joint limits
- [ ] PD formula: `τ = Kp * (target - current) - Kd * velocity`

### Safety
- [ ] Joint position limits are enforced
- [ ] Torque limits are enforced
- [ ] Emergency stop mechanism is in place
- [ ] Gradual startup (don't apply full actions immediately)

### Coordinate Frames
- [ ] IMU angular velocity is in **body frame** (local to robot)
- [ ] Roll/pitch angles use correct convention (X-forward, Y-left, Z-up)
- [ ] Future motion targets use local frame for velocities

### Initial State
- [ ] `last_actions` is initialized to zeros on startup
- [ ] Robot starts in a safe pose before enabling policy

---

## Common Pitfalls

1. **Wrong frequency**: Policy was trained at 50 Hz. Running faster/slower will affect behavior.

2. **Forgetting to zero ankle velocities**: The model was trained with ankle velocities always zero in observation.

3. **Wrong scaling**: 
   - Angular velocity: × 0.25
   - Joint velocity: × 0.05
   - Joint position: × 1.0 (no scaling, just offset from default)

4. **Using clipped actions as `last_actions`**: Store the *raw* policy output, not the clipped version.

5. **Missing PD controller**: The policy outputs position targets, not torques. You need PD control.

6. **Wrong joint order**: Make sure your motor indices match the URDF order.

7. **Degrees vs radians**: All angles are in **radians**.

8. **Missing future motion targets**: The model requires 1100 features of future motion data. Without valid motion targets, the model won't work properly.

---

## License

See the main TWIST repository LICENSE file.
