#!/usr/bin/env python3
"""
TFLite Exporter for TWIST Student Policy

This script converts a trained student policy (student_ready.pt) to TensorFlow Lite format
for deployment on embedded devices.

Usage:
    python tflite_exporter.py --input model/student_ready.pt --output model/student_ready.tflite
"""

import os
import sys
import argparse
import torch
import torch.nn as nn

# Add rsl_rl to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'rsl_rl'))

from rsl_rl.modules.actor_critic_mimic import Actor, get_activation


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
    Full student policy for TFLite export.
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
        print(f"Loaded model from {checkpoint_path}")
        print(f"  Normalizer mean shape: {self.norm_mean.shape}")
        print(f"  Normalizer std shape: {self.norm_std.shape}")
    
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
    
    # Determine tsteps from conv layer kernel size
    if 'actor.motion_encoder.conv_layers.0.weight' in state_dict:
        conv1 = state_dict['actor.motion_encoder.conv_layers.0.weight']
        kernel_size = conv1.shape[2]
        if kernel_size == 6:
            num_motion_steps = 20
        elif kernel_size == 8:
            num_motion_steps = 50
        elif kernel_size == 4:
            num_motion_steps = 10
        else:
            num_motion_steps = 1
    else:
        num_motion_steps = 1
    
    # Calculate num_motion_observations
    num_single_motion = motion_enc_input
    num_motion_observations = num_single_motion * num_motion_steps
    
    # Verify with backbone input size
    backbone_input_size = state_dict['actor.actor_backbone.0.weight'].shape[1]
    expected_backbone_input = (num_observations - num_motion_observations) + num_single_motion + motion_latent_dim
    
    if backbone_input_size != expected_backbone_input:
        print(f"Warning: backbone input size mismatch!")
        print(f"  Expected: {expected_backbone_input}, Actual: {backbone_input_size}")
    
    # Infer actor hidden dims from layer weights
    actor_hidden_dims = []
    layer_idx = 0
    while f'actor.actor_backbone.{layer_idx}.weight' in state_dict:
        weight = state_dict[f'actor.actor_backbone.{layer_idx}.weight']
        if layer_idx == 0:
            actor_hidden_dims.append(weight.shape[0])
        else:
            # Check if this is the final output layer
            if weight.shape[0] != num_actions:
                actor_hidden_dims.append(weight.shape[0])
        layer_idx += 1
        # Skip activation layers (no weight), skip to next linear
        while f'actor.actor_backbone.{layer_idx}.weight' not in state_dict and layer_idx < 20:
            layer_idx += 1
    
    # Check for layer norm
    layer_norm = 'actor.actor_backbone.7.weight' in state_dict and state_dict['actor.actor_backbone.7.weight'].dim() == 1
    
    config = {
        'num_observations': num_observations,
        'num_motion_observations': num_motion_observations,
        'num_motion_steps': num_motion_steps,
        'motion_latent_dim': motion_latent_dim,
        'num_actions': num_actions,
        'actor_hidden_dims': [512, 512, 256, 128],  # Default from save_jit_stu_rlbc.py
        'activation_name': 'silu',
        'layer_norm': layer_norm,
    }
    
    print("Inferred model configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    return config


def export_to_onnx(model, output_path, num_observations):
    """Export PyTorch model to ONNX format."""
    model.eval()
    
    # Create sample input
    sample_input = torch.randn(1, num_observations)
    
    onnx_path = os.path.splitext(output_path)[0] + '.onnx'
    
    print(f"Exporting model to ONNX...")
    print(f"  Input shape: {sample_input.shape}")
    
    # Export to ONNX with opset 11 for better onnx-tf compatibility
    torch.onnx.export(
        model,
        sample_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['observations'],
        output_names=['actions'],
        dynamic_axes={
            'observations': {0: 'batch_size'},
            'actions': {0: 'batch_size'}
        }
    )
    
    print(f"Successfully exported to ONNX: {onnx_path}")
    file_size = os.path.getsize(onnx_path)
    print(f"  File size: {file_size / 1024:.2f} KB")
    
    return onnx_path


def export_to_tflite(model, output_path, num_observations):
    """Export PyTorch model to TFLite format via ONNX and TensorFlow."""
    model.eval()
    
    # Create sample input
    sample_input = torch.randn(1, num_observations)
    
    # First export to ONNX
    onnx_path = export_to_onnx(model, output_path, num_observations)
    
    print(f"\nConverting ONNX to TFLite...")
    
    try:
        # Try ai-edge-torch first (fastest, most compatible)
        import ai_edge_torch
        print("Using ai-edge-torch for conversion...")
        edge_model = ai_edge_torch.convert(model, (sample_input,))
        edge_model.export(output_path)
    except ImportError:
        # Fall back to onnx-tf + TensorFlow
        print("ai-edge-torch not available, using onnx-tf + TensorFlow...")
        try:
            import onnx
            from onnx_tf.backend import prepare
            import tensorflow as tf
            
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            
            # Convert to TensorFlow
            tf_rep = prepare(onnx_model)
            
            # Export to SavedModel
            saved_model_path = os.path.splitext(output_path)[0] + '_saved_model'
            tf_rep.export_graph(saved_model_path)
            
            # Convert SavedModel to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            
            # Save TFLite model
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            # Cleanup saved model directory
            import shutil
            if os.path.exists(saved_model_path):
                shutil.rmtree(saved_model_path)
                
        except Exception as e:
            print(f"TFLite conversion failed: {e}")
            print(f"ONNX model saved at: {onnx_path}")
            print("You can convert it to TFLite using other tools.")
            return
    
    print(f"Successfully exported to {output_path}")
    
    # Verify the exported model
    file_size = os.path.getsize(output_path)
    print(f"  File size: {file_size / 1024:.2f} KB")


def main():
    parser = argparse.ArgumentParser(description='Export student policy to TFLite format')
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='model/student_ready.pt',
        help='Path to input checkpoint file (default: model/student_ready.pt)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to output TFLite file (default: same as input with .tflite extension)'
    )
    
    args = parser.parse_args()
    
    # Handle relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.isabs(args.input):
        input_path = os.path.join(script_dir, args.input)
    else:
        input_path = args.input
    
    if args.output is None:
        output_path = os.path.splitext(input_path)[0] + '.tflite'
    elif not os.path.isabs(args.output):
        output_path = os.path.join(script_dir, args.output)
    else:
        output_path = args.output
    
    # Check input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print()
    
    # Infer model configuration from checkpoint
    config = infer_model_config(input_path)
    
    # Create model
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
    
    # Load weights
    model.load_from_checkpoint(input_path)
    
    # Test forward pass
    test_input = torch.randn(1, config['num_observations'])
    with torch.no_grad():
        test_output = model(test_input)
    print(f"Test forward pass: input {test_input.shape} -> output {test_output.shape}")
    print()
    
    # Export to TFLite
    export_to_tflite(model, output_path, config['num_observations'])


if __name__ == '__main__':
    main()
