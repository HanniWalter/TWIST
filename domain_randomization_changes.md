# Domain Randomization Changes Analysis
Commit: `b2bfdd1236fc207d45907484227917b9e7ccce15`

## Quantitative Changes (Harder Values)
| Parameter | Old Value | New Value | Description |
|-----------|-----------|-----------|-------------|
| **Gravity Range** | `(-0.1, 0.1)` | `(-0.15, 0.15)` | Increased range of gravity perturbation |
| **Friction Range** | `[0.1, 2.]` | `[0.05, 3.]` | Wider range for friction coefficient |
| **Base Mass Range** | `[-3., 3.]` | `[-2.5, 2.5]` | Slightly narrower/shifted mass randomization |
| **Push Interval** | `4s` | `3.0s` | More frequent pushes |
| **Max Push Force (EE)** | `20.0` | `30.0` | Increased end-effector push force |
| **Motor Strength Range** | `[0.8, 1.2]` | `[0.7, 1.3]` | Wider range of motor strength factors |
| **DoF Pos Noise** | `0.01` | `0.02` | Increased joint position noise |
| **DoF Vel Noise** | `0.1` | `0.5` | Significantly increased joint velocity noise |
| **Lin Vel Noise** | `0.1` | `0.15` | Increased linear velocity noise |
| **Ang Vel Noise** | `0.1` | `0.3` | Increased angular velocity noise |
| **Gravity Noise** | `0.05` | `0.1` | Increased gravity vector noise |
| **IMU Noise** | `0.1` | `0.2` | Increased IMU noise |
| **Feet Slip** | `-0.1` | `-0.5` | Increased feet slip parameter (in `K1MimicStuRLCfg_future`) |
| **Root Height Threshold** | `0.2` | `0.5` | Relaxed height difference threshold for termination |
| **Torque Safety Limit** | `Scaled by 0.9` | `Removed` | Removed the 0.9 safety scaling factor when setting torque limits |

## Qualitative Changes (New Methods)
These changes introduce new types of randomization or simulation dynamics that were not present before.

### 1. Link Property Randomization
*   **Link Mass Randomization:** Added `randomize_link_mass` with range `[0.9, 1.1]`. Randomizes the mass of individual links, not just the base.
*   **Link CoM Randomization:** Added `randomize_link_com` with range `[-0.005, 0.005]`. Randomizes the Center of Mass position for links.

### 2. Joint Dynamics
*   **Joint Friction Randomization:** Added `randomize_joint_friction` with range `[0.0, 0.05]`. Adds random friction torque to joints.
*   **Stiffness & Damping Randomization:** Added `stiffness_multiplier_range` and `damping_multiplier_range` (both `[0.5, 1.5]`). Randomizes the PD gains or internal motor properties.

### 3. Sustained External Perturbations
Implemented a new **"sustained push"** mechanism that applies continuous force/torque for a duration, unlike the previous instantaneous impulse pushes.
*   **New Parameters:**
    *   `sustained_push_interval_s`: Interval for starting new pushes.
    *   `push_duration_s`: How long the push lasts.
    *   `max_push_force`: Maximum continuous force magnitude.
    *   `max_push_torque`: Maximum continuous torque magnitude.
*   **Implementation:** Added logic in `legged_robot.py` (`_update_push_parameters`, `_apply_sustained_push`) to apply these forces over multiple simulation steps.

### 4. Initial State Randomization
*   **Start Velocity Randomization:** Added logic in `legged_robot.py` to randomize initial linear (`7:9`) and angular (`10:13`) velocities if `randomize_start_vel` is true.
