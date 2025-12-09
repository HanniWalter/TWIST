#!/usr/bin/env python3
"""
List training runs and their checkpoints in the legged_gym/logs directory.
Usage: python list_runs.py [--robot ROBOT] [--type TYPE]
"""

import os
import argparse
from pathlib import Path
from datetime import datetime


def get_checkpoint_info(run_path):
    """Get information about checkpoints in a run directory."""
    checkpoints = []
    
    # Find all model checkpoint files
    for file in sorted(os.listdir(run_path)):
        if file.startswith("model_") and file.endswith(".pt"):
            checkpoint_num = file.replace("model_", "").replace(".pt", "")
            file_path = os.path.join(run_path, file)
            file_size = os.path.getsize(file_path)
            file_time = os.path.getmtime(file_path)
            
            checkpoints.append({
                'number': checkpoint_num,
                'file': file,
                'size_mb': file_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(file_time)
            })
    
    return checkpoints


def list_runs(logs_base_path, robot_type=None, run_type=None):
    """List all training runs with their checkpoint information."""
    
    if not os.path.exists(logs_base_path):
        print(f"Error: Logs directory not found: {logs_base_path}")
        return
    
    # Get all subdirectories in logs
    log_dirs = [d for d in os.listdir(logs_base_path) 
                if os.path.isdir(os.path.join(logs_base_path, d)) and d != 'wandb' and d != 'videos_retarget']
    
    # Map run_type to directory naming convention
    if run_type:
        if run_type == 'teacher':
            run_type = 'priv_mimic'
        elif run_type == 'student':
            run_type = 'stu_rl'
        log_dirs = [d for d in log_dirs if run_type in d]
    
    # Filter by robot if specified
    if robot_type:
        log_dirs = [d for d in log_dirs if d.startswith(robot_type)]
    
    if not log_dirs:
        print(f"No runs found matching criteria (robot: {robot_type}, type: {run_type})")
        return
    
    # Determine display type name
    display_type = None
    if run_type == 'priv_mimic':
        display_type = 'teacher'
    elif run_type == 'stu_rl':
        display_type = 'student'
    
    print(f"\n{'='*100}")
    print(f"Training Runs in: {logs_base_path}")
    if robot_type:
        print(f"Filtered by robot: {robot_type}")
    if display_type:
        print(f"Filtered by type: {display_type}")
    print(f"{'='*100}\n")
    
    # Process each log directory
    for log_dir in sorted(log_dirs):
        log_path = os.path.join(logs_base_path, log_dir)
        
        # Get all run directories in this log directory
        run_dirs = [d for d in os.listdir(log_path) 
                   if os.path.isdir(os.path.join(log_path, d))]
        
        if not run_dirs:
            continue
        
        # Determine if this is teacher or student
        type_label = ""
        if 'priv_mimic' in log_dir:
            type_label = " [TEACHER]"
        elif 'stu_rl' in log_dir:
            type_label = " [STUDENT]"
        
        print(f"\n📁 {log_dir}/{type_label}")
        print(f"{'-'*100}")
        
        total_runs = 0
        total_checkpoints = 0
        
        for run_dir in sorted(run_dirs):
            run_path = os.path.join(log_path, run_dir)
            checkpoints = get_checkpoint_info(run_path)
            
            if checkpoints:
                total_runs += 1
                total_checkpoints += len(checkpoints)
                
                # Get latest checkpoint
                latest = max(checkpoints, key=lambda x: x['modified'])
                
                # Format output
                print(f"  📊 {run_dir:<40} | "
                      f"Checkpoints: {len(checkpoints):3d} | "
                      f"Latest: model_{latest['number']}.pt | "
                      f"Modified: {latest['modified'].strftime('%Y-%m-%d %H:%M')}")
                
                # Show first few and last checkpoint numbers if there are many
                if len(checkpoints) > 5:
                    checkpoint_nums = [c['number'] for c in checkpoints]
                    display_nums = checkpoint_nums[:3] + ['...'] + checkpoint_nums[-2:]
                    print(f"      Checkpoints: {', '.join(display_nums)}")
                elif len(checkpoints) > 1:
                    checkpoint_nums = [c['number'] for c in checkpoints]
                    print(f"      Checkpoints: {', '.join(checkpoint_nums)}")
        
        if total_runs > 0:
            print(f"\n  Summary: {total_runs} runs, {total_checkpoints} total checkpoints")


def main():
    parser = argparse.ArgumentParser(
        description='List training runs and checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python list_runs.py                          # List all runs
  python list_runs.py --robot g1               # List only G1 runs
  python list_runs.py --type teacher           # List only teacher runs
  python list_runs.py --type student           # List only student runs
  python list_runs.py --robot t1 --type student # List T1 student runs
        """
    )
    
    parser.add_argument('--robot', type=str, choices=['g1', 't1', 'k1'],
                       help='Filter by robot type (g1, t1, k1)')
    parser.add_argument('--type', type=str, choices=['teacher', 'student', 'priv_mimic', 'stu_rl'],
                       help='Filter by run type (teacher/priv_mimic or student/stu_rl)')
    parser.add_argument('--logs-path', type=str, 
                       default='legged_gym/logs',
                       help='Path to logs directory (default: legged_gym/logs)')
    
    args = parser.parse_args()
    
    # Get absolute path
    script_dir = Path(__file__).parent
    logs_path = script_dir / args.logs_path
    
    list_runs(str(logs_path), args.robot, args.type)


if __name__ == "__main__":
    main()
