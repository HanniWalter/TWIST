#!/usr/bin/env python3
"""
Model Verifier for TWIST Student Policy

This script verifies the consistency of the student policy model by:
1. Loading preset model inputs (all zeros and 10 random inputs)
2. Running inference on both student_ready.pt (PyTorch) and student_ready.tflite (TFLite)
3. Saving inputs and outputs to CSV files for comparison with C++ implementations

The CSV files can be used to verify that the TFLite model produces the same outputs
when running in a C++ environment.

Usage:
    python model_verifier.py
    python model_verifier.py --input model/student_ready.pt --seed 42
"""

import os
import sys
import argparse
import csv
import numpy as np
import torch
import torch.nn as nn

# Add rsl_rl to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'rsl_rl'))

from rsl_rl.modules.actor_critic_mimic import get_activation


class MotionEncoder(nn.Module):
    """Motion encoder for processing motion observations."""
    
    def __init__(self, activation_fn, input_size, tsteps, output_size):
        super().__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps
        
        channel_size = 20
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 3 * channel_size),
            self.activation_fn,
        )
        
        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=3 * channel_size, out_channels=2 * channel_size, kernel_size=8, stride=4),
                self.activation_fn,
                nn.Conv1d(in_channels=2 * channel_size, out_channels=channel_size, kernel_size=5, stride=1),
                self.activation_fn,
                nn.Conv1d(in_channels=channel_size, out_channels=channel_size, kernel_size=5, stride=1),
                self.activation_fn,
                nn.Flatten()
            )
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=3 * channel_size, out_channels=2 * channel_size, kernel_size=4, stride=2),
                self.activation_fn,
                nn.Conv1d(in_channels=2 * channel_size, out_channels=channel_size, kernel_size=2, stride=1),
                self.activation_fn,
                nn.Flatten()
            )
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=3 * channel_size, out_channels=2 * channel_size, kernel_size=6, stride=2),
                self.activation_fn,
                nn.Conv1d(in_channels=2 * channel_size, out_channels=channel_size, kernel_size=4, stride=2),
                self.activation_fn,
                nn.Flatten()
            )
        elif tsteps == 1:
            self.conv_layers = nn.Flatten()
        else:
            raise ValueError("tsteps must be 1, 10, 20 or 50")
        
        self.linear_output = nn.Linear(channel_size * 3, output_size)
    
    def forward(self, obs):
        nd = obs.shape[0]
        T = self.tsteps
        projection = self.encoder(obs.reshape([nd * T, -1]))
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return output


class ActorOnly(nn.Module):
    """Actor-only model for inference (no critic, no noise)."""
    
    def __init__(
        self,
        num_observations,
        num_motion_observations,
        num_motion_steps,
        motion_latent_dim,
        num_actions,
        actor_hidden_dims,
        activation,
        layer_norm=False,
    ):
        super().__init__()
        self.num_observations = num_observations
        self.num_motion_observations = num_motion_observations
        self.num_motion_steps = num_motion_steps
        self.num_single_motion_observations = int(num_motion_observations / num_motion_steps)
        self.num_actions = num_actions
        
        self.motion_encoder = MotionEncoder(
            activation,
            self.num_single_motion_observations,
            self.num_motion_steps,
            motion_latent_dim
        )
        
        actor_layers = []
        actor_layers.append(nn.Linear(
            self.num_observations - self.num_motion_observations + motion_latent_dim + self.num_single_motion_observations,
            actor_hidden_dims[0]
        ))
        actor_layers.append(activation)
        
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                if layer_norm and l == len(actor_hidden_dims) - 2:
                    actor_layers.append(nn.LayerNorm(actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        
        self.actor_backbone = nn.Sequential(*actor_layers)
    
    def forward(self, obs):
        motion_obs = obs[:, :self.num_motion_observations]
        motion_latent = self.motion_encoder(motion_obs)
        backbone_input = torch.cat([
            obs[:, self.num_motion_observations:],
            obs[:, :self.num_single_motion_observations],
            motion_latent
        ], dim=1)
        return self.actor_backbone(backbone_input)


class StudentPolicyForExport(nn.Module):
    """
    Full student policy for inference.
    Includes normalization and actor inference.
    """
    
    def __init__(
        self,
        num_observations,
        num_motion_observations,
        num_motion_steps,
        motion_latent_dim,
        num_actions,
        actor_hidden_dims,
        activation_name='silu',
        layer_norm=True,
    ):
        super().__init__()
        
        self.num_observations = num_observations
        activation = get_activation(activation_name)
        
        # Normalizer parameters (will be loaded from checkpoint)
        self.register_buffer('norm_mean', torch.zeros(num_observations))
        self.register_buffer('norm_std', torch.ones(num_observations))
        self.norm_eps = 1e-4
        self.norm_clip = float('inf')
        
        # Actor network
        self.actor = ActorOnly(
            num_observations=num_observations,
            num_motion_observations=num_motion_observations,
            num_motion_steps=num_motion_steps,
            motion_latent_dim=motion_latent_dim,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
        )
    
    def load_from_checkpoint(self, checkpoint_path):
        """Load model weights and normalizer from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Load normalizer parameters
        normalizer = checkpoint['normalizer']
        self.norm_mean.copy_(normalizer._mean.data)
        self.norm_std.copy_(normalizer._std.data)
        
        # Load actor weights (filter out critic and other non-actor weights)
        state_dict = checkpoint['model_state_dict']
        actor_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('actor.'):
                new_key = key  # Keep 'actor.' prefix since our model has self.actor
                actor_state_dict[new_key] = value
        
        self.load_state_dict(actor_state_dict, strict=False)
        print(f"Loaded PyTorch model from {checkpoint_path}")
    
    def normalize(self, x):
        """Normalize input observations."""
        norm_x = (x - self.norm_mean) / (self.norm_std + self.norm_eps)
        norm_x = torch.clamp(norm_x, -self.norm_clip, self.norm_clip)
        return norm_x
    
    def forward(self, obs):
        """Forward pass with normalization."""
        normalized_obs = self.normalize(obs)
        return self.actor(normalized_obs)


def infer_model_config(checkpoint_path):
    """Infer model configuration from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    normalizer = checkpoint['normalizer']
    
    # Infer dimensions from weights
    num_observations = normalizer._mean.shape[0]
    
    # Motion encoder input size (single motion observation)
    motion_enc_input = state_dict['actor.motion_encoder.encoder.0.weight'].shape[1]
    
    # Motion latent dim
    motion_latent_dim = state_dict['actor.motion_encoder.linear_output.weight'].shape[0]
    
    # Num actions (from last actor layer)
    num_actions = state_dict['actor.actor_backbone.9.weight'].shape[0]
    
    num_single_motion = motion_enc_input
    num_motion_observations = num_single_motion
    num_motion_steps = 1
    
    # Check for layer norm
    layer_norm = 'actor.actor_backbone.7.weight' in state_dict and state_dict['actor.actor_backbone.7.weight'].dim() == 1
    
    config = {
        'num_observations': num_observations,
        'num_motion_observations': num_motion_observations,
        'num_motion_steps': num_motion_steps,
        'motion_latent_dim': motion_latent_dim,
        'num_actions': num_actions,
        'actor_hidden_dims': [512, 512, 256, 128],
        'activation_name': 'silu',
        'layer_norm': layer_norm,
    }
    
    return config


def generate_test_inputs(num_observations, seed=42):
    """
    Generate test inputs for model verification.
    
    Returns:
        list: List of tuples (name, input_tensor) containing:
            - One all-zeros input
            - 10 random inputs with reproducible seeds
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    test_inputs = []
    
    # All zeros input
    zeros_input = torch.zeros(1, num_observations, dtype=torch.float32)
    test_inputs.append(('zeros', zeros_input))
    
    # 10 random inputs
    for i in range(10):
        # Use deterministic seed for each random input
        np.random.seed(seed + i + 1)
        torch.manual_seed(seed + i + 1)
        random_input = torch.randn(1, num_observations, dtype=torch.float32)
        test_inputs.append((f'random_{i+1}', random_input))
    
    return test_inputs


def run_pytorch_inference(model, test_inputs):
    """
    Run inference using PyTorch model.
    
    Returns:
        list: List of tuples (name, output_tensor)
    """
    model.eval()
    outputs = []
    
    with torch.no_grad():
        for name, input_tensor in test_inputs:
            output = model(input_tensor)
            outputs.append((name, output))
    
    return outputs


def run_tflite_inference(tflite_path, test_inputs):
    """
    Run inference using TFLite model.
    
    Returns:
        list: List of tuples (name, output_array)
    """
    # Try to import TFLite interpreter (from tflite-runtime or full tensorflow)
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite import Interpreter
        except ImportError:
            try:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter
            except ImportError:
                print("Warning: Neither tflite-runtime nor TensorFlow installed. Skipping TFLite inference.")
                return None
    
    # Load TFLite model
    interpreter = Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"TFLite model loaded from {tflite_path}")
    print(f"  Input shape: {input_details[0]['shape']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    
    outputs = []
    for name, input_tensor in test_inputs:
        # Convert to numpy and ensure correct dtype
        input_data = input_tensor.numpy().astype(np.float32)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        outputs.append((name, output_data))
    
    return outputs


def save_inputs_to_csv(test_inputs, output_dir, prefix='input'):
    """Save test inputs to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for name, input_tensor in test_inputs:
        csv_path = os.path.join(output_dir, f'{prefix}_{name}.csv')
        input_data = input_tensor.numpy().flatten()
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'value'])
            for i, val in enumerate(input_data):
                writer.writerow([i, f'{val:.8e}'])
        
        print(f"  Saved input to {csv_path}")


def save_outputs_to_csv(outputs, output_dir, prefix='output'):
    """Save model outputs to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for name, output_data in outputs:
        csv_path = os.path.join(output_dir, f'{prefix}_{name}.csv')
        
        # Handle both torch tensors and numpy arrays
        if isinstance(output_data, torch.Tensor):
            output_data = output_data.numpy()
        
        output_data = output_data.flatten()
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'value'])
            for i, val in enumerate(output_data):
                writer.writerow([i, f'{val:.8e}'])
        
        print(f"  Saved output to {csv_path}")


def save_combined_csv(test_inputs, pt_outputs, tflite_outputs, output_dir):
    """
    Save all inputs and outputs to a combined CSV for easy comparison.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    combined_path = os.path.join(output_dir, 'verification_data.csv')
    
    with open(combined_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Determine sizes
        sample_input = test_inputs[0][1].numpy().flatten()
        sample_pt_output = pt_outputs[0][1].numpy().flatten() if isinstance(pt_outputs[0][1], torch.Tensor) else pt_outputs[0][1].flatten()
        
        input_size = len(sample_input)
        output_size = len(sample_pt_output)
        
        # Write header
        header = ['test_name']
        header += [f'input_{i}' for i in range(input_size)]
        header += [f'pt_output_{i}' for i in range(output_size)]
        if tflite_outputs:
            header += [f'tflite_output_{i}' for i in range(output_size)]
        writer.writerow(header)
        
        # Write data for each test
        for i, (name, input_tensor) in enumerate(test_inputs):
            row = [name]
            row += [f'{v:.8e}' for v in input_tensor.numpy().flatten()]
            
            pt_out = pt_outputs[i][1]
            if isinstance(pt_out, torch.Tensor):
                pt_out = pt_out.numpy()
            row += [f'{v:.8e}' for v in pt_out.flatten()]
            
            if tflite_outputs:
                tflite_out = tflite_outputs[i][1]
                row += [f'{v:.8e}' for v in tflite_out.flatten()]
            
            writer.writerow(row)
    
    print(f"\n  Saved combined verification data to {combined_path}")


def compare_outputs(pt_outputs, tflite_outputs):
    """Compare PyTorch and TFLite outputs and print differences."""
    if tflite_outputs is None:
        print("\nSkipping comparison (TFLite outputs not available)")
        return
    
    print("\n" + "="*60)
    print("Output Comparison: PyTorch vs TFLite")
    print("="*60)
    
    for i, ((pt_name, pt_out), (tflite_name, tflite_out)) in enumerate(zip(pt_outputs, tflite_outputs)):
        if isinstance(pt_out, torch.Tensor):
            pt_out = pt_out.numpy()
        
        diff = np.abs(pt_out - tflite_out)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"\n{pt_name}:")
        print(f"  Max absolute difference: {max_diff:.8e}")
        print(f"  Mean absolute difference: {mean_diff:.8e}")
        
        if max_diff > 1e-4:
            print(f"  WARNING: Large difference detected!")


def main():
    parser = argparse.ArgumentParser(description='Verify student policy model outputs')
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='model/student_ready.pt',
        help='Path to input PyTorch checkpoint file (default: model/student_ready.pt)'
    )
    parser.add_argument(
        '--tflite', '-t',
        type=str,
        default=None,
        help='Path to TFLite model file (default: same as input with .tflite extension)'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='verification_data',
        help='Output directory for CSV files (default: verification_data)'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for reproducible random inputs (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Handle relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.isabs(args.input):
        pt_path = os.path.join(script_dir, args.input)
    else:
        pt_path = args.input
    
    if args.tflite is None:
        tflite_path = os.path.splitext(pt_path)[0] + '.tflite'
    elif not os.path.isabs(args.tflite):
        tflite_path = os.path.join(script_dir, args.tflite)
    else:
        tflite_path = args.tflite
    
    if not os.path.isabs(args.output_dir):
        output_dir = os.path.join(script_dir, args.output_dir)
    else:
        output_dir = args.output_dir
    
    # Check input files exist
    if not os.path.exists(pt_path):
        print(f"Error: PyTorch model file not found: {pt_path}")
        sys.exit(1)
    
    tflite_exists = os.path.exists(tflite_path)
    if not tflite_exists:
        print(f"Warning: TFLite model file not found: {tflite_path}")
        print("Will only run PyTorch inference.")
    
    print("="*60)
    print("Model Verifier for TWIST Student Policy")
    print("="*60)
    print(f"\nPyTorch model: {pt_path}")
    print(f"TFLite model:  {tflite_path}")
    print(f"Output dir:    {output_dir}")
    print(f"Random seed:   {args.seed}")
    print()
    
    # Infer model configuration
    print("Inferring model configuration...")
    config = infer_model_config(pt_path)
    print(f"  num_observations: {config['num_observations']}")
    print(f"  num_actions: {config['num_actions']}")
    print()
    
    # Create PyTorch model
    print("Loading PyTorch model...")
    model = StudentPolicyForExport(
        num_observations=config['num_observations'],
        num_motion_observations=config['num_motion_observations'],
        num_motion_steps=config['num_motion_steps'],
        motion_latent_dim=config['motion_latent_dim'],
        num_actions=config['num_actions'],
        actor_hidden_dims=config['actor_hidden_dims'],
        activation_name=config['activation_name'],
        layer_norm=config['layer_norm'],
    )
    model.load_from_checkpoint(pt_path)
    print()
    
    # Generate test inputs
    print(f"Generating test inputs (seed={args.seed})...")
    test_inputs = generate_test_inputs(config['num_observations'], seed=args.seed)
    print(f"  Generated {len(test_inputs)} test inputs")
    print()
    
    # Run PyTorch inference
    print("Running PyTorch inference...")
    pt_outputs = run_pytorch_inference(model, test_inputs)
    print(f"  Completed {len(pt_outputs)} inferences")
    print()
    
    # Run TFLite inference
    tflite_outputs = None
    if tflite_exists:
        print("Running TFLite inference...")
        tflite_outputs = run_tflite_inference(tflite_path, test_inputs)
        if tflite_outputs:
            print(f"  Completed {len(tflite_outputs)} inferences")
        print()
    
    # Save inputs to CSV
    print("Saving inputs to CSV files...")
    save_inputs_to_csv(test_inputs, output_dir)
    print()
    
    # Save PyTorch outputs to CSV
    print("Saving PyTorch outputs to CSV files...")
    save_outputs_to_csv(pt_outputs, output_dir, prefix='output_pytorch')
    print()
    
    # Save TFLite outputs to CSV
    if tflite_outputs:
        print("Saving TFLite outputs to CSV files...")
        save_outputs_to_csv(tflite_outputs, output_dir, prefix='output_tflite')
        print()
    
    # Save combined CSV
    print("Saving combined verification data...")
    save_combined_csv(test_inputs, pt_outputs, tflite_outputs, output_dir)
    
    # Compare outputs
    compare_outputs(pt_outputs, tflite_outputs)
    
    print("\n" + "="*60)
    print("Verification complete!")
    print("="*60)
    print(f"\nCSV files saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - input_*.csv: Test inputs (all zeros + 10 random)")
    print("  - output_pytorch_*.csv: PyTorch model outputs")
    if tflite_outputs:
        print("  - output_tflite_*.csv: TFLite model outputs")
    print("  - verification_data.csv: Combined data for C++ comparison")
    print("\nUse these CSV files to verify the model in your C++ environment.")


if __name__ == '__main__':
    main()
