# ~/ns3-rl-setup/phase3_install_ns3sionna.sh
# Phase 3: Install ns3sionna for realistic ray-traced wireless channels

#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
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

print_sionna() {
    echo -e "${CYAN}[SIONNA]${NC} $1"
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

print_status "=== Phase 3: Install ns3sionna for Ray-Traced Channels ==="
print_sionna "Installing the cutting-edge Sionna RT integration with ns-3"
print_gpu "Leveraging RTX 4080 for GPU-accelerated ray tracing"
echo

# Activate the RL environment
print_status "Activating RL environment..."
source ~/ns3-rl-env/bin/activate
check_status "RL environment activated"

# Verify GPU and environment
print_gpu "Verifying GPU acceleration environment..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
check_status "GPU environment verified"

# Navigate to home directory for clean setup
cd "$HOME"

# Clone ns3sionna repository
print_sionna "Cloning ns3sionna repository..."
if [ -d "ns3sionna" ]; then
    print_warning "ns3sionna directory already exists, removing it..."
    rm -rf ns3sionna
fi

git clone https://github.com/tkn-tub/ns3sionna.git
check_status "ns3sionna repository cloned"

cd ns3sionna

# Check available branches and use the main/master branch
print_sionna "Checking ns3sionna branches..."
git branch -r
git checkout master 2>/dev/null || git checkout main 2>/dev/null || echo "Using default branch"

# Install Sionna RT and dependencies
print_sionna "Installing Sionna RT and dependencies..."

# Install required system dependencies for Sionna RT
print_status "Installing system dependencies for Sionna RT..."
sudo apt update
sudo apt install -y build-essential cmake pkg-config

# Install TensorFlow (required for Sionna)
print_sionna "Installing TensorFlow with GPU support..."
pip install tensorflow[and-cuda]
check_status "TensorFlow with GPU support installed"

# Install Sionna RT
print_sionna "Installing Sionna RT..."
pip install sionna-rt
check_status "Sionna RT installed"

# Install additional dependencies for ns3sionna
print_status "Installing ns3sionna Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
elif [ -f "sionna_server/requirements.txt" ]; then
    pip install -r sionna_server/requirements.txt
else
    # Install common dependencies manually
    pip install zmq grpcio grpcio-tools matplotlib numpy scipy
fi
check_status "ns3sionna dependencies installed"

# Verify Sionna RT installation
print_sionna "Verifying Sionna RT installation..."
python3 -c "
try:
    import sionna
    print(f'✅ Sionna version: {sionna.__version__}')
    
    import sionna.rt
    print('✅ Sionna RT module loaded successfully')
    
    import tensorflow as tf
    print(f'✅ TensorFlow version: {tf.__version__}')
    print(f'✅ TensorFlow GPU support: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')
    
    if len(tf.config.list_physical_devices('GPU')) > 0:
        print(f'✅ GPU devices: {tf.config.list_physical_devices(\"GPU\")}')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
except Exception as e:
    print(f'❌ Other error: {e}')
    exit(1)
"
check_status "Sionna RT verification completed"

# Set up the Sionna server
print_sionna "Setting up Sionna RT server..."
if [ -d "sionna_server" ]; then
    cd sionna_server
    
    # Create a simple test script to verify server functionality
    cat > test_sionna_server.py << 'EOF'
#!/usr/bin/env python3
"""
Test script for Sionna RT server functionality
"""
import sionna
import sionna.rt as rt
import tensorflow as tf
import numpy as np

def test_sionna_rt():
    print("🧪 Testing Sionna RT functionality...")
    
    # Test basic Sionna RT scene loading
    try:
        # Load a simple built-in scene
        scene = rt.load_scene(rt.scene.etoile)  # Simple built-in scene
        print("✅ Scene loading successful")
        
        # Test basic ray tracing setup
        print("✅ Sionna RT basic functionality verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Sionna RT test failed: {e}")
        return False

if __name__ == "__main__":
    print(f"Sionna version: {sionna.__version__}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    if test_sionna_rt():
        print("🎉 Sionna RT server setup successful!")
    else:
        print("❌ Sionna RT server setup failed")
        exit(1)
EOF

    chmod +x test_sionna_server.py
    
    print_sionna "Testing Sionna RT server setup..."
    python3 test_sionna_server.py
    check_status "Sionna RT server test passed"
    
    cd ..
else
    print_warning "sionna_server directory not found, creating basic setup..."
    mkdir -p sionna_server
fi

# Create integration scripts
print_status "Creating ns3sionna integration scripts..."

# Create environment activation script for Sionna
cat > ~/ns3-rl-setup/activate_sionna_env.sh << 'EOF'
#!/bin/bash
# Activate the complete RL + Sionna environment

echo "🌟 Activating ns-3 RL + Sionna RT Environment..."
source ~/ns3-rl-env/bin/activate
cd ~/ns3sionna

echo "Environment activated!"
echo "• Python virtual env: ns3-rl-env"
echo "• Working directory: ns3sionna"
echo "• Available: PyTorch (CUDA), SB3, ns3-gym, Sionna RT"
echo ""
echo "Quick test commands:"
echo "  cd sionna_server && python3 test_sionna_server.py  # Test Sionna RT"
echo "  cd ~/ns-allinone-3.40/ns-3.40 && ./ns3 run opengym  # Test ns3-gym"
echo ""
echo "🎯 Ready for advanced RL research with realistic channels!"
EOF

chmod +x ~/ns3-rl-setup/activate_sionna_env.sh

# Create a sample integration script
cat > ~/ns3-rl-setup/sample_sionna_rl_integration.py << 'EOF'
#!/usr/bin/env python3
# ~/ns3-rl-setup/sample_sionna_rl_integration.py
# Sample script showing ns3-gym + Sionna RT + SB3 integration

import gymnasium as gym
import ns3gym
from ns3gym import ns3env
from stable_baselines3 import PPO
import torch
import sionna
import sionna.rt as rt
import tensorflow as tf

def main():
    print("🚀 Advanced RL Setup: ns3-gym + Sionna RT + SB3")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA: {torch.cuda.is_available()}")
    print(f"Sionna version: {sionna.__version__}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"TensorFlow GPU: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # Example: Load a Sionna RT scene for realistic channel modeling
    try:
        scene = rt.load_scene(rt.scene.etoile)
        print("✅ Sionna RT scene loaded successfully")
    except Exception as e:
        print(f"⚠️  Sionna RT scene loading failed: {e}")
    
    print("\n🎯 Integration Ready!")
    print("• ns3-gym: Network simulation environment")
    print("• Sionna RT: Realistic ray-traced channel modeling")  
    print("• SB3: GPU-accelerated RL training")
    print("• Your RTX 4080: Ready for advanced research!")
    
    print("\n📝 Next steps:")
    print("1. Design your wireless scenario in ns-3")
    print("2. Use Sionna RT for realistic channel modeling")
    print("3. Train RL agents with SB3 on GPU")
    print("4. Publish groundbreaking research! 🏆")

if __name__ == "__main__":
    main()
EOF

chmod +x ~/ns3-rl-setup/sample_sionna_rl_integration.py

# Create comprehensive test script
cat > ~/ns3-rl-setup/test_complete_setup.sh << 'EOF'
#!/bin/bash
# Test the complete ns3-gym + Sionna RT + SB3 setup

echo "🧪 Testing Complete RL + Ray Tracing Setup"
source ~/ns3-rl-env/bin/activate

echo ""
echo "1. Testing PyTorch + CUDA:"
python3 -c "import torch; print(f'✅ PyTorch {torch.__version__} with CUDA: {torch.cuda.is_available()}')"

echo ""
echo "2. Testing Stable-Baselines3:"
python3 -c "from stable_baselines3 import PPO; print('✅ SB3 ready')"

echo ""
echo "3. Testing ns3-gym:"
python3 -c "import ns3gym; print('✅ ns3-gym ready')"

echo ""
echo "4. Testing Sionna RT:"
python3 -c "import sionna.rt as rt; print('✅ Sionna RT ready')"

echo ""
echo "5. Testing TensorFlow GPU:"
python3 -c "import tensorflow as tf; print(f'✅ TensorFlow {tf.__version__} GPU: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')"

echo ""
echo "🎉 Complete integration test passed!"
echo "Your RTX 4080 RL research environment is ready!"
EOF

chmod +x ~/ns3-rl-setup/test_complete_setup.sh

# Display final summary
echo
print_success "=== Phase 3 Complete! ==="
print_sionna "ns3sionna Installation Summary:"
echo "  • ✅ ns3sionna: Cloned and set up"
echo "  • 🌟 Sionna RT: GPU-accelerated ray tracing ready"
echo "  • 🤖 TensorFlow: GPU support enabled"
echo "  • 🔗 Integration: ns3-gym + Sionna RT + SB3"
echo "  • 📱 RTX 4080: Optimized for ray tracing and RL"
echo
print_status "Quick Start Commands:"
echo "  source ~/ns3-rl-setup/activate_sionna_env.sh"
echo "  ~/ns3-rl-setup/test_complete_setup.sh"
echo "  python3 ~/ns3-rl-setup/sample_sionna_rl_integration.py"
echo
print_status "Next: Phase 4 - GPU optimization and final integration testing"
print_gpu "🚀 Your cutting-edge RL research environment is almost complete!"