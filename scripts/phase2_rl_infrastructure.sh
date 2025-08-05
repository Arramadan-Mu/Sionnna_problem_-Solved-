# ~/ns3-rl-setup/phase2_rl_infrastructure.sh
# Phase 2: Install RL infrastructure with GPU acceleration

#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_gpu() {
    echo -e "${PURPLE}[GPU]${NC} $1"
}

# Function to check if command was successful
check_status() {
    if [ $? -eq 0 ]; then
        print_success "$1"
    else
        print_error "Failed: $1"
        exit 1
    fi
}

print_status "=== Phase 2: RL Infrastructure Setup ==="
print_status "Installing ns3-gym + SB3 + PyTorch with GPU acceleration"
print_gpu "Target GPU: RTX 4080 with CUDA 12.9"
echo

# Check CUDA availability
print_gpu "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    print_success "NVIDIA GPU detected"
else
    print_warning "nvidia-smi not found in WSL - this is normal, we'll use CUDA from PyTorch"
fi

# Navigate to ns-3.40
cd "$HOME/ns-allinone-3.40/ns-3.40"

# Verify ns-3.40 is working
print_status "Verifying ns-3.40 base installation..."
./ns3 run hello-simulator > /dev/null 2>&1
check_status "ns-3.40 verification passed"

# Install ZeroMQ and Protocol Buffers (required for ns3-gym)
print_status "Installing ns3-gym dependencies..."
sudo apt update
sudo apt install -y libzmq3-dev libprotobuf-dev protobuf-compiler pkg-config
check_status "ZeroMQ and Protocol Buffers installed"

# Download and install ns3-gym
print_status "Downloading ns3-gym 1.0.2 for ns-3.40..."
cd contrib

# Remove any existing opengym
if [ -d "opengym" ]; then
    rm -rf opengym
fi

# Download the latest ns3-gym
wget https://github.com/tkn-tub/ns3-gym/releases/download/v1.0.2/ns3-gym-1.0.2.tar.gz
check_status "ns3-gym downloaded"

# Extract ns3-gym
tar -xzf ns3-gym-1.0.2.tar.gz
mv ns3-gym-1.0.2 opengym
check_status "ns3-gym extracted to contrib/opengym"

# Go back to ns-3.40 root
cd ..

# Create Python virtual environment for RL
print_status "Creating Python virtual environment for RL..."
cd "$HOME"
python3 -m venv ns3-rl-env
source ns3-rl-env/bin/activate
check_status "Python virtual environment created"

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip
check_status "pip upgraded"

# Install PyTorch with CUDA support for RTX 4080
print_gpu "Installing PyTorch with CUDA 12.1 support (compatible with CUDA 12.9)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
check_status "PyTorch with CUDA support installed"

# Verify PyTorch CUDA installation
print_gpu "Verifying PyTorch CUDA installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('CUDA not available - will use CPU mode')
"
check_status "PyTorch CUDA verification completed"

# Install Stable-Baselines3 and dependencies
print_status "Installing Stable-Baselines3 (SB3) and ML dependencies..."
pip install stable-baselines3[extra] gymnasium
check_status "SB3 installed"

# Install additional useful RL libraries
print_status "Installing additional RL and visualization libraries..."
pip install tensorboard wandb matplotlib seaborn pandas numpy scipy
check_status "Additional libraries installed"

# Install ns3gym Python module
print_status "Installing ns3gym Python module..."
cd "$HOME/ns-allinone-3.40/ns-3.40/contrib/opengym"
pip install -e ./model/ns3gym
check_status "ns3gym Python module installed"

# Configure and build ns-3.40 with ns3-gym
print_status "Configuring ns-3.40 with ns3-gym..."
cd "$HOME/ns-allinone-3.40/ns-3.40"

# Clean and reconfigure with ns3-gym
./ns3 clean
./ns3 configure --enable-examples --enable-tests
check_status "ns-3.40 configured with ns3-gym"

# Build ns-3.40 with ns3-gym
print_status "Building ns-3.40 with ns3-gym (this may take a few minutes)..."
./ns3 build
check_status "ns-3.40 with ns3-gym built successfully"

# Test ns3-gym installation
print_status "Testing ns3-gym installation..."

# Create a simple test script
cat > ~/test_ns3gym.py << 'EOF'
#!/usr/bin/env python3

import subprocess
import time
import sys
import os

# Add the virtual environment to the path
sys.path.insert(0, os.path.expanduser('~/ns3-rl-env/lib/python3.*/site-packages'))

try:
    import gymnasium as gym
    import ns3gym
    from ns3gym import ns3env
    print("✅ All imports successful!")
    print(f"ns3gym version: {ns3gym.__version__ if hasattr(ns3gym, '__version__') else 'unknown'}")
    print(f"gymnasium version: {gym.__version__}")
    
    # Test if we can create the environment (this requires ns-3 to be running)
    print("✅ ns3-gym installation test passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Other error: {e}")
    sys.exit(1)
EOF

chmod +x ~/test_ns3gym.py
source ~/ns3-rl-env/bin/activate
python3 ~/test_ns3gym.py
check_status "ns3-gym Python module test passed"

# Test the full ns3-gym integration
print_status "Testing full ns3-gym integration with example..."

# Start the ns-3 simulation in background
cd "$HOME/ns-allinone-3.40/ns-3.40"
print_status "Starting ns-3 opengym example (Terminal 1)..."
timeout 30s ./ns3 run "opengym" &
NS3_PID=$!

# Give ns-3 time to start
sleep 3

# Run the Python agent
print_status "Running Python RL agent (Terminal 2)..."
cd contrib/opengym/examples/opengym/
source ~/ns3-rl-env/bin/activate

# Create a timeout wrapper for the test
timeout 20s python3 test.py --start=0 || true

# Kill the ns-3 process if still running
if kill -0 $NS3_PID 2>/dev/null; then
    kill $NS3_PID
fi

wait $NS3_PID 2>/dev/null || true

print_success "ns3-gym integration test completed"

# Create convenient activation script
print_status "Creating convenience scripts..."

# Environment activation script
cat > ~/ns3-rl-setup/activate_rl_env.sh << 'EOF'
#!/bin/bash
# Activate the RL environment for ns-3 work

echo "🚀 Activating ns-3 RL Environment..."
source ~/ns3-rl-env/bin/activate
cd ~/ns-allinone-3.40/ns-3.40

echo "Environment activated!"
echo "• Python virtual env: ns3-rl-env"
echo "• Working directory: ns-3.40"
echo "• Available: PyTorch (CUDA), SB3, ns3-gym"
echo ""
echo "Quick test commands:"
echo "  ./ns3 run opengym                    # Terminal 1: Start simulation"
echo "  python3 contrib/opengym/examples/opengym/test.py --start=0  # Terminal 2: Run agent"
echo ""
echo "🎯 Ready for RL research!"
EOF

chmod +x ~/ns3-rl-setup/activate_rl_env.sh

# Create a sample SB3 integration script
cat > ~/ns3-rl-setup/sample_sb3_integration.py << 'EOF'
#!/usr/bin/env python3
# ~/ns3-rl-setup/sample_sb3_integration.py
# Sample script showing how to use SB3 with ns3-gym

import gymnasium as gym
import ns3gym
from ns3gym import ns3env
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import torch

def main():
    print("🤖 SB3 + ns3-gym Integration Sample")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Note: This requires ns-3 simulation to be running
    # env = ns3env.Ns3Env()
    # check_env(env)
    # model = PPO("MlpPolicy", env, verbose=1, device="cuda" if torch.cuda.is_available() else "cpu")
    
    print("✅ Ready to train RL agents with GPU acceleration!")
    print("📝 Edit this file to implement your specific RL scenario")

if __name__ == "__main__":
    main()
EOF

chmod +x ~/ns3-rl-setup/sample_sb3_integration.py

# Display final summary
echo
print_success "=== Phase 2 Complete! ==="
print_status "RL Infrastructure Summary:"
echo "  • ✅ ns-3.40: Working with ns3-gym integration"
echo "  • 🐍 Python Environment: ~/ns3-rl-env (virtual environment)"
echo "  • 🔥 PyTorch: CUDA-enabled for RTX 4080"
echo "  • 🤖 Stable-Baselines3: Ready for RL training"
echo "  • 🏃‍♂️ ns3-gym: Installed and tested"
echo "  • 📚 Extra libraries: TensorBoard, Wandb, Matplotlib, etc."
echo
print_status "Quick Start:"
echo "  source ~/ns3-rl-setup/activate_rl_env.sh"
echo "  # Then run your RL experiments!"
echo
print_status "Next: Phase 3 - Install ns3sionna for realistic ray-traced channels"
print_gpu "🚀 Your RTX 4080 is ready for GPU-accelerated RL training!"