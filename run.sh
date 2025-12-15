#!/bin/bash

# Default values
INIT=false
TRAIN_TEACHER=false
TRAIN_STUDENT=false
PLAY_TEACHER=false
PLAY_STUDENT=false
LIST_LOGS=false
PROJECT_NAME=""
EXPTID=""
TEACHER_EXPTID=""
ROBOT=""
NO_WANDB=false
NO_GMR=false
GMR_DATASET=""
DEBUG=false
RESUME=false
CHECKPOINT=-1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --init)
            INIT=true
            shift
            ;;
        --teacher)
            TRAIN_TEACHER=true
            shift
            ;;
        --train_teacher)
            TRAIN_TEACHER=true
            shift
            ;;
        --train_student)
            TRAIN_STUDENT=true
            shift
            ;;
        --student)
            TRAIN_STUDENT=true
            shift
            ;;
        --play_teacher)
            PLAY_TEACHER=true
            shift
            ;;
        --play_student)
            PLAY_STUDENT=true
            shift
            ;;
        --logs)
            LIST_LOGS=true
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --exptid)
            EXPTID="$2"
            shift 2
            ;;
        --teacher_exptid)
            TEACHER_EXPTID="$2"
            shift 2
            ;;
        --proj_name)
            PROJECT_NAME="$2"
            shift 2
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
        --debug)
            DEBUG=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--init] [--teacher] [--student] [--robot ROBOT] [--no-wandb]"
            echo ""
            echo "Options:"
            echo "  --init        Initialize/setup the environment"
            echo "  --teacher     Train the teacher model"
            echo "  --student     Train the student model"
            echo "  --play_teacher Play/evaluate trained teacher model"
            echo "  --play_student Play/evaluate trained student model"
            echo "  --logs        List all training runs and checkpoints (use with --robot to filter)"
            echo "  --resume      Resume training (use with --exptid to specify run, or auto-resume current)"
            echo "  --checkpoint N Load specific checkpoint number (default: -1 = latest)"
            echo "  --robot ROBOT Specify robot type (k1, g1, t1)"
            echo "  --exptid ID   Experiment ID (for resume: g1_teacher_1027_1548, for play: required)"
            echo "  --teacher_exptid ID Teacher experiment ID (required for student training)"
            echo "  --proj_name NAME  Project name (optional, defaults to robot-specific)"
            echo "  --no-wandb    Disable Weights & Biases logging"
            echo "  --debug       Enable debug mode (small environment, visible)"
            echo "  -h, --help    Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --teacher --robot g1                           # Train new G1 teacher"
            echo "  $0 --teacher --robot g1 --resume --exptid g1_teacher_1027_1548  # Resume specific run"
            echo "  $0 --logs --robot t1                              # List T1 training runs"
            exit 0
            ;;
        --no-gmr)
            NO_GMR=true
            shift
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

#clean pycache
find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf

# List training logs if requested
if [[ "$LIST_LOGS" = true ]]; then
    echo "Listing training runs..."
    
    # Build command with optional robot filter
    LOGS_CMD="python list_runs.py"
    if [ -n "$ROBOT" ]; then
        LOGS_CMD="$LOGS_CMD --robot $ROBOT"
    fi
    
    # Execute the command
    eval $LOGS_CMD
    exit 0
fi

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

    echo "Installing Python packages..."
    $PIP_EXEC install --no-cache-dir --upgrade pip setuptools
    $PIP_EXEC uninstall -y numpy
    $PIP_EXEC install --no-cache-dir "numpy==1.23.0" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask dill gdown hydra-core imageio[ffmpeg] mujoco mujoco-python-viewer isaacgym-stubs pytorch-kinematics rich termcolor pyyaml
    $PIP_EXEC install --no-cache-dir redis[hiredis]
    $PIP_EXEC install --no-cache-dir pyttsx3 # for voice control

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

# Set LD_LIBRARY_PATH for Isaac Gym to find Python libraries
CONDA_ENV_PATH=$(conda info --envs | grep twist | awk '{print $NF}')
export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:$LD_LIBRARY_PATH"

#cd to TWIST directory
cd $HOME/Documents/TWIST

# Configure Weights & Biases (auto-login from repo key or disable)
mkdir -p "$HOME/Documents/TWIST/logs"
if [[ "$NO_WANDB" = true ]]; then
    export WANDB_DISABLED=true
    export WANDB_MODE=disabled
    unset WANDB_API_KEY
    echo "W&B disabled via --no-wandb"
else
    KEY_FILE="$HOME/Documents/TWIST/wandb_key"
    if [[ -f "$KEY_FILE" ]]; then
        WANDB_KEY=$(tr -d ' \t\r\n' < "$KEY_FILE")
        if [[ -n "$WANDB_KEY" ]]; then
            export WANDB_API_KEY="$WANDB_KEY"
            export WANDB_LOGIN_SILENT=true
            export WANDB_SILENT=true
            export WANDB_DIR="$HOME/Documents/TWIST/logs"
            echo "W&B API key loaded from repo (silent login)."
        else
            echo "Warning: wandb_key file is empty. W&B may prompt for login."
        fi
    else
        echo "Warning: wandb_key file not found at $KEY_FILE. W&B may prompt for login. Use --no-wandb to disable."
    fi
fi

# check if GMR_DATASET is set and valid
if [ -z "$GMR_DATASET" ]; then
    GMR_DATASET_1="/home/nao/Documents/trainingsdata/GMR/"
    GMR_DATASET_2="/training_data/"
    if [ "$ROBOT" = "t1" ]; then
        GMR_DATASET="${GMR_DATASET_1}booster_t1_modified_2${GMR_DATASET_2}"
    elif [ "$ROBOT" = "k1" ]; then
        GMR_DATASET="${GMR_DATASET_1}booster_k1_modified${GMR_DATASET_2}"
    elif [ "$ROBOT" = "g1" ]; then
        GMR_DATASET="${GMR_DATASET_1}unitree_g1_slim${GMR_DATASET_2}"
    else
        echo "Unknown robot type: $ROBOT. Supported types are G1, T1, K1."
        exit 1
    fi
fi

if [[ "$NO_GMR" = true ]]; then
    GMR_DATASET="/home/nao/Documents/trainingsdata/TWIST/twist_motion_dataset/"
fi

if [ -d "$GMR_DATASET" ]; then
    echo "Using GMR dataset path: $GMR_DATASET"
    $PYTHON_EXEC create_dataset_yaml.py "$GMR_DATASET" "/home/nao/Documents/TWIST/legged_gym/motion_data_configs/twist_dataset.yaml"
else
    echo "Warning: Dataset path $GMR_DATASET does not exist."
    exit
fi

if [[ "$TRAIN_TEACHER" = true ]]; then
    echo "Starting teacher training for robot $ROBOT..."
    #print current directory
    echo "Current directory: $(pwd)"
    
    cd legged_gym/legged_gym/scripts

    # Set default values if not provided
    exptid="${ROBOT}_teacher_$(date +%m%d_%H%M)"
    device="$DEVICE"

    if [ "$ROBOT" = "t1" ]; then
        task_name="t1_priv_mimic"
        proj_name="t1_priv_mimic"
    elif [ "$ROBOT" = "k1" ]; then
        task_name="k1_priv_mimic"
        proj_name="k1_priv_mimic"
    elif [ "$ROBOT" = "g1" ]; then
        task_name="g1_priv_mimic"
        proj_name="g1_priv_mimic"
    else
        echo "Unknown robot type: $ROBOT. Supported types are G1, H1, T1, K1."
        return 1
    fi

    echo "Using task: $task_name and project: $proj_name"
    
    # Handle resume logic
    if [[ "$RESUME" = true ]]; then
        if [ -n "$EXPTID" ]; then
            echo "Resuming training from run: $EXPTID"
            exptid="$EXPTID"
        else
            echo "Resuming training from last checkpoint of current run"
        fi
        echo "Loading checkpoint: $CHECKPOINT (-1 = latest)"
    elif [ -n "$EXPTID" ]; then
        # If exptid is provided but not resume, use it as the new experiment name
        exptid="$EXPTID"
    fi
    
    echo "Starting training with exptid: $exptid on device: $device"
    if [[ "$NO_WANDB" = true ]]; then
        echo "W&B logging is disabled"
    fi
    
    # Bereite die Trainings-Argumente vor
    TRAIN_ARGS="--task ${task_name} --proj_name ${proj_name} --exptid ${exptid} --device ${device}"
    
    # Füge Resume-Parameter hinzu
    if [[ "$RESUME" = true ]]; then
        if [ -n "$EXPTID" ]; then
            TRAIN_ARGS="${TRAIN_ARGS} --resumeid ${EXPTID}"
        else
            TRAIN_ARGS="${TRAIN_ARGS} --resume"
        fi
        
        if [ "$CHECKPOINT" != "-1" ]; then
            TRAIN_ARGS="${TRAIN_ARGS} --checkpoint ${CHECKPOINT}"
        fi
    fi
    
    # Füge --no-wandb hinzu falls gesetzt
    if [[ "$NO_WANDB" = true ]]; then
        TRAIN_ARGS="${TRAIN_ARGS} --no_wandb"
    fi
    
    if [[ "$DEBUG" = true ]]; then
        TRAIN_ARGS="${TRAIN_ARGS} --debug"
    fi

    # Run the training script
    $PYTHON_EXEC train.py $TRAIN_ARGS
fi

if [[ "$TRAIN_STUDENT" = true ]]; then
    echo "Starting student training for robot $ROBOT..."
    #print current directory
    echo "Current directory: $(pwd)"
    
    cd legged_gym/legged_gym/scripts

    # Set default values if not provided
    exptid="${ROBOT}_student_$(date +%m%d_%H%M)"
    device="$DEVICE"

    if [ "$ROBOT" = "t1" ]; then
        task_name="t1_stu_rl"
        proj_name="t1_stu_rl"
    elif [ "$ROBOT" = "k1" ]; then
        task_name="k1_stu_rl_modified"
        proj_name="k1_stu_rl_modified"
    elif [ "$ROBOT" = "g1" ]; then
        task_name="g1_stu_rl"
        proj_name="g1_stu_rl"
    else
        echo "Unknown robot type: $ROBOT. Supported types are G1, T1, K1."
        return 1
    fi

    echo "Using task: $task_name and project: $proj_name"
    
    # Handle resume logic
    if [[ "$RESUME" = true ]]; then
        if [ -n "$EXPTID" ]; then
            echo "Resuming training from run: $EXPTID"
            exptid="$EXPTID"
        else
            echo "Resuming training from last checkpoint of current run"
        fi
        echo "Loading checkpoint: $CHECKPOINT (-1 = latest)"
    elif [ -n "$EXPTID" ]; then
        # If exptid is provided but not resume, use it as the new experiment name
        exptid="$EXPTID"
    fi
    
    echo "Starting training with exptid: $exptid on device: $device"
    if [[ "$NO_WANDB" = true ]]; then
        echo "W&B logging is disabled"
    fi
    
    # Bereite die Trainings-Argumente vor
    TRAIN_ARGS="--task ${task_name} --proj_name ${proj_name} --exptid ${exptid} --device ${device} --teacher_exptid ${TEACHER_EXPTID}"
    
    # Füge Resume-Parameter hinzu
    if [[ "$RESUME" = true ]]; then
        if [ -n "$EXPTID" ]; then
            TRAIN_ARGS="${TRAIN_ARGS} --resumeid ${EXPTID}"
        else
            TRAIN_ARGS="${TRAIN_ARGS} --resume"
        fi
        
        if [ "$CHECKPOINT" != "-1" ]; then
            TRAIN_ARGS="${TRAIN_ARGS} --checkpoint ${CHECKPOINT}"
        fi
    fi
    
    # Füge --no-wandb hinzu falls gesetzt
    if [[ "$NO_WANDB" = true ]]; then
        TRAIN_ARGS="${TRAIN_ARGS} --no_wandb"
    fi
    
    if [[ "$DEBUG" = true ]]; then
        TRAIN_ARGS="${TRAIN_ARGS} --debug"
    fi

    # Run the training script
    $PYTHON_EXEC train.py $TRAIN_ARGS
fi

if [[ "$PLAY_TEACHER" = true ]]; then
    echo "Starting teacher play for robot $ROBOT..."
    
    # Check GPU availability
    echo "Checking NVIDIA GPU status..."
    if nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
    else
        echo "Warning: nvidia-smi not available"
    fi
    
    cd legged_gym/legged_gym/scripts

    if [ "$ROBOT" = "t1" ]; then
        task_name="t1_priv_mimic"
        proj_name="t1_priv_mimic"
    elif [ "$ROBOT" = "k1" ]; then
        task_name="k1_priv_mimic"
        proj_name="k1_priv_mimic"
    elif [ "$ROBOT" = "g1" ]; then
        task_name="g1_priv_mimic"
        proj_name="g1_priv_mimic"
    else
        echo "Unknown robot type: $ROBOT. Supported types are G1, T1, K1."
        return 1
    fi

    # Use provided PROJECT_NAME or default to robot-specific project
    if [ -z "$PROJECT_NAME" ]; then
        PROJECT_NAME="$proj_name"
    fi

    # Check if EXPTID is provided
    if [ -z "$EXPTID" ]; then
        echo "Error: --exptid is required for playing teacher"
        echo "Usage: $0 --play_teacher --robot ROBOT --exptid EXPERIMENT_ID [--proj_name PROJECT]"
        exit 1
    fi

    echo "Playing task: $task_name with project: $PROJECT_NAME and experiment: $EXPTID"
    echo "Using device: $DEVICE"
    
    # Build play command with optional arguments
    PLAY_ARGS="--task ${task_name} --proj_name ${PROJECT_NAME} --exptid ${EXPTID} --device ${DEVICE} --num_envs 1"
    
    # Add record_video flag if not in debug mode
    if [[ "$DEBUG" != true ]]; then
        PLAY_ARGS="${PLAY_ARGS} --record_video"
    fi
    
    $PYTHON_EXEC play.py $PLAY_ARGS
fi

if [[ "$PLAY_STUDENT" = true ]]; then
    echo "Starting student play for robot $ROBOT..."
    
    # Check GPU availability
    echo "Checking NVIDIA GPU status..."
    if nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
    else
        echo "Warning: nvidia-smi not available"
    fi
    
    cd legged_gym/legged_gym/scripts

    if [ "$ROBOT" = "t1" ]; then
        task_name="t1_stu_rl"
        proj_name="t1_stu_rl"
    elif [ "$ROBOT" = "k1" ]; then
        task_name="k1_stu_rl_modified"
        proj_name="k1_stu_rl_modified"
    elif [ "$ROBOT" = "g1" ]; then
        task_name="g1_stu_rl"
        proj_name="g1_stu_rl"
    else
        echo "Unknown robot type: $ROBOT. Supported types are G1, T1, K1."
        return 1
    fi

    # Use provided PROJECT_NAME or default to robot-specific project
    if [ -z "$PROJECT_NAME" ]; then
        PROJECT_NAME="$proj_name"
    fi

    # Check if EXPTID is provided
    if [ -z "$EXPTID" ]; then
        echo "Error: --exptid is required for playing student"
        echo "Usage: $0 --play_student --robot ROBOT --exptid EXPERIMENT_ID --teacher_exptid TEACHER_EXPERIMENT_ID [--proj_name PROJECT]"
        exit 1
    fi

    # Check if TEACHER_EXPTID is provided
    if [ -z "$TEACHER_EXPTID" ]; then
        echo "Error: --teacher_exptid is required for playing student"
        echo "Usage: $0 --play_student --robot ROBOT --exptid EXPERIMENT_ID --teacher_exptid TEACHER_EXPERIMENT_ID [--proj_name PROJECT]"
        exit 1
    fi

    echo "Playing task: $task_name with project: $PROJECT_NAME and experiment: $EXPTID"
    echo "Using teacher experiment: $TEACHER_EXPTID"
    echo "Using device: $DEVICE"
    
    # Build play command with optional arguments
    PLAY_ARGS="--task ${task_name} --proj_name ${PROJECT_NAME} --exptid ${EXPTID} --teacher_exptid ${TEACHER_EXPTID} --device ${DEVICE} --num_envs 1"
    
    # Add record_video flag if not in debug mode
    if [[ "$DEBUG" != true ]]; then
        PLAY_ARGS="${PLAY_ARGS} --record_video"
    fi
    
    $PYTHON_EXEC play.py $PLAY_ARGS
fi

conda deactivate
