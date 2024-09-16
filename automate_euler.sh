# Set variables for the Euler cluster
USERNAME="hgraef"  # Ensure USERNAME is not empty
EULER_HOST="${USERNAME}@euler.ethz.ch"
REMOTE_BASE_DIR="/cluster/home/${USERNAME}/deep_learning"
LOCAL_PROJECT_DIR="/Users/heikograef/development/SkinSafeAI/EulerTrainer"
LOCAL_SCRIPTS_PATH="${LOCAL_PROJECT_DIR}/scripts"
VENV_DIR="${REMOTE_BASE_DIR}/deep_env"
ACTIVATE_SCRIPT="${VENV_DIR}/bin/activate"  # Path to the activate script

# Debugging: Check if REMOTE_BASE_DIR is assigned correctly
echo "REMOTE_BASE_DIR is set to: '${REMOTE_BASE_DIR}'"

# Ensure the remote base directory exists
ssh ${EULER_HOST} << EOF
if [ ! -d "${REMOTE_BASE_DIR}" ]; then
    echo "Creating remote base directory: ${REMOTE_BASE_DIR}"
    mkdir -p "${REMOTE_BASE_DIR}" || { echo "Failed to create remote base directory"; exit 1; }
fi
EOF

echo "Remote base directory created successfully!"

# Log in to Euler and create necessary directories
echo "Creating scripts directory on Euler..."
ssh ${EULER_HOST} "mkdir -p ${REMOTE_BASE_DIR}/scripts"

# Upload ONLY Python scripts to Euler (modify this to include only the files you need)
echo "Uploading Python scripts..."
scp -r ${LOCAL_SCRIPTS_PATH}/*.py ${EULER_HOST}:${REMOTE_BASE_DIR}/scripts/

# Log into Euler to set up the environment and create the job script
echo "Setting up environment and job script on Euler..."
ssh ${EULER_HOST} << 'EOF'
set -e  # Exit on any error

# Define the VENV_DIR explicitly
VENV_DIR="/cluster/home/hgraef/deep_learning/deep_env"
ACTIVATE_SCRIPT="${VENV_DIR}/bin/activate"  # Path to the activate script

# Load the necessary stack and compiler modules
module load stack/2024-06
module load gcc/12.2.0
module load cuda/11.8.0
module load cudnn/8.9.7.29-12
module load python/3.10.13

# Ensure the virtual environment directory exists and venv is created
if [ ! -d "${VENV_DIR}" ]; then
    echo "Attempting to create virtual environment in: ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}" || { echo "Failed to create venv in ${VENV_DIR}. Exiting..."; exit 1; }
fi

# Verify if the 'bin/activate' script exists
if [ -f "${ACTIVATE_SCRIPT}" ]; then
    echo "Virtual environment activation script found at ${ACTIVATE_SCRIPT}"
else
    echo "Activation script not found! Virtual environment creation may have failed."
    exit 1
fi

# Activate the virtual environment
source ${ACTIVATE_SCRIPT} || { echo "Failed to activate venv"; exit 1; }
EOF
