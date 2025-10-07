#!/bin/bash

# Default values
INIT=false
TEACHER=false
STUDENT=false
ROBOT="k1"
NO_WANDB=false
GMR_DATASET="/home/nao/Documents/trainingsdata/GMR/k1/training_data/"


# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --init)
            INIT=true
            shift
            ;;
        --teacher)
            TEACHER=true
            shift
            ;;
        --student)
            STUDENT=true
            shift
            ;;
        --GMR)
            GMR_DATASET="$2"
            shift 2
            ;;
        --robot)
            ROBOT="$2"
            shift 2
            ;;
        --no-wandb)
            NO_WANDB=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--init] [--teacher] [--student] [--robot ROBOT] [--no-wandb]"
            echo ""
            echo "Options:"
            echo "  --init        Initialize/setup the environment"
            echo "  --teacher     Train the teacher model"
            echo "  --student     Train the student model"
            echo "  --robot ROBOT Specify robot type (k1, g1, t1)"
            echo "  --no-wandb    Disable Weights & Biases logging"
            echo "  -h, --help    Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done


# Configuration - modify these values as needed
DEVICE="cuda:0"
ISSAC_GYM_PATH="$HOME/Documents/isaacgym"

# Initialize environment if requested
if [[ "$INIT" = true ]]; then
    echo "Removing existing twist environment..."
    conda env remove -n twist -y 2>/dev/null || true
    
    echo "Creating new twist environment with Python 3.8..."
    conda create -n twist python=3.8 -y
    
    # Source bashrc and activate environment
    source ~/.bashrc
    eval "$(conda shell.bash hook)"
    conda activate twist
    
    # Get the python and pip from the conda environment
    PYTHON_EXEC=$(conda run -n twist which python)
    PIP_EXEC=$(conda run -n twist which pip)

    echo "Using python from: $PYTHON_EXEC"
    echo "Using pip from: $PIP_EXEC"

    echo "Installing Python packages..."
    $PIP_EXEC install --no-cache-dir --upgrade pip setuptools
    $PIP_EXEC uninstall -y numpy
    $PIP_EXEC install --no-cache-dir "numpy>=2.0.0" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask dill gdown hydra-core imageio[ffmpeg] mujoco mujoco-python-viewer isaacgym-stubs pytorch-kinematics rich termcolor pyyaml
    $PIP_EXEC install --no-cache-dir redis[hiredis]
    $PIP_EXEC install --no-cache-dir pyttsx3 # for voice control

    echo "Installing IsaacGym..."
    if [ -d "$ISSAC_GYM_PATH/python" ]; then
        cd "$ISSAC_GYM_PATH/python" && $PIP_EXEC install --no-cache-dir -e .
        cd - > /dev/null
    else
        echo "Warning: IsaacGym path not found at $ISSAC_GYM_PATH/python"
        return 1
    fi

    # cd back here
    cd $HOME/Documents/TWIST
    echo "Installing rsl_rl..."
    if [ -d "rsl_rl" ]; then
        cd rsl_rl && $PIP_EXEC install --no-cache-dir -e . && cd ..
    else
        echo "Warning: rsl_rl directory not found"
        return 1
    fi
    
    echo "Installing legged_gym..."
    if [ -d "legged_gym" ]; then
        cd legged_gym && $PIP_EXEC install --no-cache-dir -e . && cd ..
    else
        echo "Warning: legged_gym directory not found"
        return 1
    fi
    
    echo "Installing pose..."
    if [ -d "pose" ]; then
        cd pose && $PIP_EXEC install --no-cache-dir -e . && cd ..
    else
        echo "Warning: pose directory not found"
    fi

else
    source ~/.bashrc
    eval "$(conda shell.bash hook)"
    conda activate twist
    PYTHON_EXEC=$(conda run -n twist which python)
fi

#cd to TWIST directory
cd $HOME/Documents/TWIST

if [ -d "$GMR_DATASET" ]; then
    echo "Using GMR dataset path: $GMR_DATASET"
    $PYTHON_EXEC create_dataset_yaml.py "$GMR_DATASET" "/home/nao/Documents/TWIST/legged_gym/motion_data_configs/twist_dataset.yaml"
else
    echo "Warning: Dataset path $GMR_DATASET does not exist."
    exit
fi

if [[ "$TEACHER" = true ]]; then
    echo "Starting teacher training for robot $ROBOT..."
    #print current directory
    echo "Current directory: $(pwd)"
    
    cd legged_gym/legged_gym/scripts

    # Set default values if not provided
    exptid="k1_teacher_$(date +%m%d_%H%M)"
    device="$DEVICE"

    task_name="g1_priv_mimic"
    proj_name="g1_priv_mimic"

    if [ "$ROBOT" = "t1" ]; then
        task_name="t1_priv_mimic"
        proj_name="t1_priv_mimic"
    elif [ "$ROBOT" = "k1" ]; then
        task_name="k1_priv_mimic"
        proj_name="k1_priv_mimic"
    else
        echo "Unknown robot type: $ROBOT. Supported types are G1, H1, T1, K1."
        echo "Using default G1 configuration."
    fi

    echo "Using task: $task_name and project: $proj_name"
    echo "Starting training with exptid: $exptid on device: $device"
    if [[ "$NO_WANDB" = true ]]; then
        echo "W&B logging is disabled"
    fi
    
    # Bereite die Trainings-Argumente vor
    TRAIN_ARGS="--task ${task_name} --proj_name ${proj_name} --exptid ${exptid} --device ${device}"
    
    # Füge --no-wandb hinzu falls gesetzt
    if [[ "$NO_WANDB" = true ]]; then
        TRAIN_ARGS="${TRAIN_ARGS} --no_wandb"
    fi
    
    # Run the training script
    $PYTHON_EXEC train.py $TRAIN_ARGS
                    # Uncomment these for additional options:
                    # --resume \
                    # --debug \
                    # --resumeid xxx
fi





conda deactivate
