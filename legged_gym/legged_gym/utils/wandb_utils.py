"""
Weights & Biases (W&B) utility functions for TWIST training
"""

import os
import yaml
import wandb
from pathlib import Path
from typing import Dict, Any, Optional


def load_wandb_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads the W&B configuration from the YAML file.
    
    Args:
        config_path: Path to the configuration file. If None, the default path is used.
        
    Returns:
        Dictionary with the W&B configuration
    """
    if config_path is None:
        # Search for the config file from the current directory
        current_dir = Path(__file__).parent
        config_path = current_dir.parent.parent.parent / "config" / "wandb_config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"⚠️  W&B configuration file not found: {config_path}")
        print("   Using default configuration (W&B disabled)")
        return get_default_config()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"❌ Error loading W&B configuration: {e}")
        print("   Using default configuration (W&B disabled)")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """
    Returns a default configuration if no configuration file is available.
    """
    return {
        'wandb': {
            'api_key': '',
            'entity': '',
            'project': 'twist_training',
            'mode': 'disabled',
            'dir': '../../logs',
            'settings': {
                'code_save': False,
                'git_save': False,
                'save_config_files': False
            }
        },
        'fallback': {
            'enable_local_logging': True,
            'local_log_dir': '../../logs'
        }
    }


def setup_wandb(project_name: str, 
                experiment_id: str,
                robot_type: str = 'unknown',
                force_disabled: bool = False,
                debug: bool = False) -> bool:
    """
    Initializes W&B based on the configuration.
    
    Args:
        project_name: Name of the W&B project
        experiment_id: Unique ID for this experiment
        robot_type: Type of robot (g1, k1, t1)
        force_disabled: Forces the deactivation of W&B
        debug: Debug mode activated
        
    Returns:
        True if W&B was successfully initialized, False otherwise
    """
    config = load_wandb_config()
    wandb_config = config.get('wandb', {})
    
    # Determine the mode
    mode = wandb_config.get('mode', 'disabled')
    
    if force_disabled or debug:
        mode = 'disabled'
        print("🔕 W&B was manually disabled")
    
    # Check API Key
    api_key = wandb_config.get('api_key', '').strip()
    if not api_key and mode != 'disabled':
        print("⚠️  No W&B API key found in configuration")
        print("   Add your API key to config/wandb_config.yaml")
        print("   Or use --no-wandb to disable W&B")
        mode = 'disabled'
    
    # Set API key as environment variable if available
    if api_key:
        os.environ['WANDB_API_KEY'] = api_key
    
    # Determine project and entity names
    entity = wandb_config.get('entity', '').strip() or None
    final_project_name = f"{robot_type}_{project_name}" if robot_type != 'unknown' else project_name
    
    try:
        # Initialize W&B
        wandb.init(
            project=final_project_name,
            name=experiment_id,
            entity=entity,
            mode=mode,
            dir=wandb_config.get('dir', '../../logs'),
            config={
                'robot_type': robot_type,
                'experiment_id': experiment_id,
            }
        )
        
        if mode != 'disabled':
            print(f"✅ W&B initialized:")
            print(f"   Project: {final_project_name}")
            print(f"   Experiment: {experiment_id}")
            print(f"   Mode: {mode}")
            if entity:
                print(f"   Entity: {entity}")
        
        return mode != 'disabled'
        
    except Exception as e:
        print(f"❌ Error during W&B initialization: {e}")
        print("   Training will continue without W&B")
        
        # Fallback: Disable W&B
        try:
            wandb.init(mode='disabled')
        except:
            pass
        
        return False


def save_config_files(robot_type: str, env_dir: str):
    """
    Saves configuration files to W&B if enabled.
    
    Args:
        robot_type: Type of robot (g1, k1, t1)
        env_dir: Directory with environment configurations
    """
    if wandb.run is None or wandb.run.mode == 'disabled':
        return
    
    config = load_wandb_config()
    if not config.get('wandb', {}).get('settings', {}).get('save_config_files', False):
        return
    
    try:
        config_file = f"{env_dir}/{robot_type}/{robot_type}_mimic_distill_config.py"
        if os.path.exists(config_file):
            wandb.save(config_file, policy="now")
            print(f"📁 Configuration file saved: {config_file}")
    except Exception as e:
        print(f"⚠️  Error saving configuration file: {e}")


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None):
    """
    Logs metrics to W&B if enabled.
    
    Args:
        metrics: Dictionary with metrics
        step: Optional step for the time series
    """
    if wandb.run is None or wandb.run.mode == 'disabled':
        return
    
    try:
        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)
    except Exception as e:
        print(f"⚠️  Error logging metrics: {e}")


def finish_wandb():
    """
    Properly terminates the W&B session.
    """
    try:
        if wandb.run is not None:
            wandb.finish()
    except Exception as e:
        print(f"⚠️  Error terminating W&B: {e}")


def get_wandb_status() -> str:
    """
    Returns the current W&B status.
    
    Returns:
        String with the status ("disabled", "online", "offline", etc.)
    """
    if wandb.run is None:
        return "not_initialized"
    return wandb.run.mode