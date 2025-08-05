# ~/ns3-rl-setup/phase1_migrate_to_ns3_40.sh
# Phase 1: Migrate from ns-3-dev to stable ns-3.40 while preserving 5G-LENA

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

print_status "=== Phase 1: ns-3.40 Migration Script ==="
print_status "This script will migrate your ns-3-dev to stable ns-3.40"
print_status "while preserving your 5G-LENA (NR) module setup"
echo

# Check current setup
print_status "Checking current ns-3 setup..."
if [ ! -d "$HOME/ns-3-dev" ]; then
    print_error "ns-3-dev directory not found at $HOME/ns-3-dev"
    print_error "Please make sure your ns-3-dev installation is in the home directory"
    exit 1
fi

if [ ! -d "$HOME/ns-3-dev/src/nr" ]; then
    print_error "5G-LENA (NR) module not found in $HOME/ns-3-dev/src/nr"
    print_error "Please make sure 5G-LENA is properly installed"
    exit 1
fi

print_success "Found existing ns-3-dev with 5G-LENA module"

# Create backup of current setup
print_status "Creating backup of current ns-3-dev setup..."
BACKUP_DIR="$HOME/ns3-dev-backup-$(date +%Y%m%d_%H%M%S)"
cp -r "$HOME/ns-3-dev" "$BACKUP_DIR"
check_status "Backup created at $BACKUP_DIR"

# Navigate to home directory
cd "$HOME"

# Download ns-3.40
print_status "Downloading ns-3.40 stable release..."
if [ ! -f "ns-allinone-3.40.tar.bz2" ]; then
    wget https://www.nsnam.org/releases/ns-allinone-3.40.tar.bz2
    check_status "Downloaded ns-3.40"
else
    print_warning "ns-3.40 archive already exists, using existing file"
fi

# Extract ns-3.40
print_status "Extracting ns-3.40..."
if [ ! -d "ns-allinone-3.40" ]; then
    tar xf ns-allinone-3.40.tar.bz2
    check_status "Extracted ns-3.40"
else
    print_warning "ns-allinone-3.40 directory already exists"
fi

# Navigate to ns-3.40
cd ns-allinone-3.40/ns-3.40

# Check if 5G-LENA is already installed in ns-3.40
if [ -d "src/nr" ]; then
    print_warning "5G-LENA already exists in ns-3.40, removing old version"
    rm -rf src/nr
fi

# Copy 5G-LENA from backup
print_status "Migrating 5G-LENA module to ns-3.40..."
cp -r "$BACKUP_DIR/src/nr" src/
check_status "5G-LENA module copied to ns-3.40"

# Update 5G-LENA to latest version (optional but recommended)
print_status "Updating 5G-LENA to latest version..."
cd src/nr
git fetch origin
git checkout master
git pull origin master
check_status "5G-LENA updated to latest version"

# Go back to ns-3.40 root
cd ../..

# Clean any previous build artifacts
print_status "Cleaning previous build artifacts..."
./ns3 clean
check_status "Build artifacts cleaned"

# Configure ns-3.40 with examples and tests
print_status "Configuring ns-3.40 with 5G-LENA..."
./ns3 configure --enable-examples --enable-tests
check_status "ns-3.40 configured successfully"

# Build ns-3.40 with 5G-LENA
print_status "Building ns-3.40 with 5G-LENA (this may take several minutes)..."
./ns3 build
check_status "ns-3.40 with 5G-LENA built successfully"

# Test the installation
print_status "Testing ns-3.40 installation..."
./ns3 run hello-simulator
check_status "Basic ns-3.40 test passed"

print_status "Testing 5G-LENA module..."
./ns3 run cttc-nr-demo
check_status "5G-LENA test passed"

# Create symlink for easy access
print_status "Creating convenient symlink..."
cd "$HOME"
if [ -L "ns-3.40" ]; then
    rm ns-3.40
fi
ln -s ns-allinone-3.40/ns-3.40 ns-3.40
check_status "Symlink created: ~/ns-3.40"

# Verify final setup
print_status "Verifying final setup..."
if [ -d "$HOME/ns-3.40/src/nr" ]; then
    print_success "5G-LENA module successfully migrated"
fi

if [ -x "$HOME/ns-3.40/ns3" ]; then
    print_success "ns-3.40 build system ready"
fi

# Display summary
echo
print_success "=== Phase 1 Migration Complete! ==="
print_status "Summary:"
echo "  • Original ns-3-dev backed up to: $BACKUP_DIR"
echo "  • ns-3.40 installed at: $HOME/ns-allinone-3.40/ns-3.40"
echo "  • Convenient symlink: $HOME/ns-3.40"
echo "  • 5G-LENA (NR) module successfully migrated and updated"
echo
print_status "Next steps:"
echo "  • Run Phase 2 script to install ns3-gym with SB3/PyTorch"
echo "  • Test commands:"
echo "    cd ~/ns-3.40"
echo "    ./ns3 run hello-simulator"
echo "    ./ns3 run cttc-nr-demo"
echo
print_warning "Note: Your original ns-3-dev is safely backed up and unchanged"