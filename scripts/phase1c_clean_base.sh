# ~/ns3-rl-setup/phase1c_clean_base.sh  
# Phase 1C: Get a clean working ns-3.40 base installation

#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Function to check if command was successful
check_status() {
    if [ $? -eq 0 ]; then
        print_success "$1"
    else
        print_error "Failed: $1"
        exit 1
    fi
}

print_status "=== Phase 1C: Clean ns-3.40 Base Installation ==="
print_status "Getting a clean working ns-3.40 foundation for RL setup"
echo

# Navigate to home directory
cd "$HOME"

# Create a safe backup location for nr modules
print_status "Creating backup directory for 5G-LENA modules..."
mkdir -p ~/5g-lena-modules-backup

# Navigate to ns-3.40
cd "$HOME/ns-allinone-3.40/ns-3.40"

# Backup and completely remove any nr-related modules
print_status "Backing up and removing all nr-related modules..."
if [ -d "src/nr" ]; then
    cp -r src/nr ~/5g-lena-modules-backup/nr-original
    rm -rf src/nr
    print_status "Original nr module backed up and removed"
fi

if [ -d "src/nr_temp" ]; then
    cp -r src/nr_temp ~/5g-lena-modules-backup/nr-temp
    rm -rf src/nr_temp  
    print_status "Temporary nr module backed up and removed"
fi

# Clean everything thoroughly
print_status "Cleaning all build artifacts..."
./ns3 clean
rm -rf build cmake-cache

# Test clean ns-3.40 configuration
print_status "Configuring clean ns-3.40 base..."
./ns3 configure --enable-examples --enable-tests

if [ $? -eq 0 ]; then
    print_success "Clean ns-3.40 configuration successful!"
    
    # Build clean ns-3.40
    print_status "Building clean ns-3.40 base (this may take a few minutes)..."
    ./ns3 build
    
    if [ $? -eq 0 ]; then
        print_success "Clean ns-3.40 build successful!"
        
        # Test the installation
        print_status "Testing clean ns-3.40 installation..."
        ./ns3 run hello-simulator
        check_status "Basic ns-3.40 test passed"
        
        # Test a few more examples to ensure everything works
        print_status "Testing additional examples..."
        ./ns3 run first
        check_status "First example test passed"
        
        ./ns3 run second
        check_status "Second example test passed"
        
        print_success "ns-3.40 base installation is working perfectly!"
        
    else
        print_error "ns-3.40 build failed even without nr modules"
        exit 1
    fi
else
    print_error "ns-3.40 configuration failed even without nr modules"
    exit 1
fi

# Create a simple 5G-LENA installation script for later
print_status "Creating 5G-LENA reinstallation script for later use..."
cat > ~/ns3-rl-setup/install_5g_lena_later.sh << 'EOF'
#!/bin/bash
# Script to install 5G-LENA after RL setup is complete

echo "Installing 5G-LENA into working ns-3.40..."
cd ~/ns-allinone-3.40/ns-3.40/src

# Clone fresh 5G-LENA
git clone https://gitlab.com/cttc-lena/nr.git

# Try different branches for compatibility
cd nr
if git show-ref --verify --quiet refs/remotes/origin/5g-lena-v3.0.x; then
    echo "Using 5g-lena-v3.0.x branch..."
    git checkout 5g-lena-v3.0.x
else
    echo "Using master branch..."
    git checkout master
fi

# Go back and reconfigure
cd ../../
./ns3 clean
./ns3 configure --enable-examples --enable-tests

if [ $? -eq 0 ]; then
    ./ns3 build
    if [ $? -eq 0 ]; then
        echo "5G-LENA successfully installed!"
        ./ns3 run cttc-nr-demo
    else
        echo "5G-LENA build failed, but ns-3.40 base still works"
    fi
else
    echo "Configuration failed with 5G-LENA, removing it..."
    rm -rf src/nr
    ./ns3 clean
    ./ns3 configure --enable-examples --enable-tests
    ./ns3 build
fi
EOF

chmod +x ~/ns3-rl-setup/install_5g_lena_later.sh

# Display final status
echo
print_success "=== Phase 1C Complete! ==="
print_status "Status Summary:"
echo "  • ✅ ns-3.40 base: Clean and working perfectly"
echo "  • 📦 5G-LENA modules: Safely backed up in ~/5g-lena-modules-backup/"
echo "  • 🔧 5G-LENA reinstall script: ~/ns3-rl-setup/install_5g_lena_later.sh"
echo "  • 🚀 Ready for: Phase 2 (ns3-gym + SB3/PyTorch installation)"
echo
print_status "Next step: Run Phase 2 to install ns3-gym with GPU-accelerated RL environment"
print_warning "Note: We'll add 5G-LENA back after the RL infrastructure is working"