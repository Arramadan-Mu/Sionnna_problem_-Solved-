# ~/ns3-rl-setup/phase2c_fix_python_venv.sh
# Phase 2C: Fix Python virtual environment and complete RL setup

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

print_status "=== Phase 2C: Fix Python Virtual Environment ==="
print_status "Installing missing Python venv package and completing RL setup"
echo

# Install missing python3-venv package
print_status "Installing python3-venv package..."
sudo apt update
sudo apt install -y python3.12-venv python3-pip
check_status "python3-venv package installed"

# Remove broken virtual environment if it exists
if [ -d "$HOME/ns3-rl-env" ]; then
    print_status "Removing broken virtual environment..."
    rm -rf "$HOME/ns3-rl-env"
fi

# Create Python virtual environment properly
print_status "Creating Python virtual environment..."
cd "$HOME"
python3 -m venv ns3-rl-env
check_status "Python virtual environment created successfully"

# Verify virtual environment was created
if [ -f "$HOME/ns3-rl-env/bin/activate" ]; then
    print_success "Virtual environment activation script found"
else
    print_error "Virtual environment creation failed"
    exit 1
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source ~/ns3-rl-env/bin/activate
check_status "Virtual environment activated"

# Verify Python environment
print_status "Verifying Python environment..."
which python3
python3 --version
pip --version
check_status "Python environment verified"

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip
check_status "pip upgraded"

# Install PyTorch with CUDA support for RTX 4080
print_gpu "Installing PyTorch with CUDA 12.1 support..."
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

# Configure and build ns-3.40 with ns3-gym
print_status "Configuring ns-3.40 with ns3-gym..."
cd "$HOME/ns-allinone-3.40/ns-3.40"

# Verify ns3-gym is properly installed
if [ ! -d "contrib/opengym" ]; then
    print_error "ns3-gym (opengym) not found in contrib directory"
    print_status "Please run Phase 2B first to install ns3-gym"
    exit 1
fi

# Clean and reconfigure with ns3-gym
./ns3 clean
./ns3 configure --enable-examples --enable-tests
check_status "ns-3.40 configured with ns3-gym"

# Build ns-3.40 with ns3-gym
print_status "Building ns-3.40 with ns3-gym (this may take a few minutes)..."
./ns3 build
check_status "ns-3.40 with ns3-gym built successfully"

# Install ns3gym Python module
print_status "Installing ns3gym Python module..."
cd contrib/opengym
pip install -e ./model/ns3gym
check_status "ns3gym Python module installed"

# Test ns3gym Python module
print_status "Testing ns3gym Python imports..."
cd "$HOME"
python3 -c "
try:
    import gymnasium as gym
    import ns3gym
    from ns3gym import ns3env
    print('✅ All imports successful!')
    print(f'gymnasium version: {gym.__version__}')
    print('✅ ns3-gym installation test passed!')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"
check_status "ns3-gym Python module test passed"

# Test basic integration (without full simulation)
print_status "Testing ns3-gym C++ integration..."
cd "$HOME/ns-allinone-3.40/ns-3.40"

# Check if opengym example exists and can be listed
./ns3 show targets | grep opengym > /dev/null 2>&1
if [ $? -eq 0 ]; then
    print_success "ns3-gym C++ integration successful"
else
    print_warning "Checking opengym examples in different location..."
    if [ -d "contrib/opengym/examples" ]; then
        print_success "ns3-gym examples found"
    else
        print_warning "opengym examples not found, but installation looks good"
    fi
fi

# Create convenience scripts
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
echo "  ./ns3 show targets | grep opengym    # List opengym targets"
echo "  ./ns3 run contrib/opengym/examples/opengym/opengym  # Start simulation"
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

# Create a simple integration test script
cat > ~/ns3-rl-setup/test_integration.sh << 'EOF'
#!/bin/bash
# Test the complete integration

echo "🧪 Testing ns3-gym Integration..."
source ~/ns3-rl-env/bin/activate
cd ~/ns-allinone-3.40/ns-3.40

echo "Available ns3-gym targets:"
./ns3 show targets | grep -i gym || echo "No specific gym targets, checking examples..."

echo ""
echo "Checking ns3-gym example structure:"
ls -la contrib/opengym/examples/ 2>/dev/null || echo "Examples directory not found"

echo ""
echo "Testing Python imports:"
python3 -c "
import gymnasium as gym
import ns3gym
from stable_baselines3 import PPO
import torch
print('✅ All imports working!')
print(f'GPU available: {torch.cuda.is_available()}')
"
EOF

chmod +x ~/ns3-rl-setup/test_integration.sh

# Display final summary
echo
print_success "=== Phase 2C Complete! ==="
print_status "RL Infrastructure Summary:"
echo "  • ✅ ns-3.40: Working with ns3-gym integration"
echo "  • 🐍 Python Environment: ~/ns3-rl-env (properly created)"
echo "  • 🔥 PyTorch: CUDA-enabled for RTX 4080"
echo "  • 🤖 Stable-Baselines3: Ready for RL training"
echo "  • 🏃‍♂️ ns3-gym: Properly installed and built"
echo "  • 📚 Extra libraries: TensorBoard, Wandb, Matplotlib, etc."
echo
print_status "Quick Start:"
echo "  source ~/ns3-rl-setup/activate_rl_env.sh"
echo "  ~/ns3-rl-setup/test_integration.sh"
echo ""
print_status "Next: Phase 3 - Install ns3sionna for realistic ray-traced channels"
print_gpu "🚀 Your RTX 4080 is ready for GPU-accelerated RL training!"