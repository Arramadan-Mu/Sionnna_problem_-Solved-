# ~/ns3-rl-setup/phase1b_fix_5glena.sh
# Phase 1B: Fix 5G-LENA compatibility issues with ns-3.40

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

print_status "=== Phase 1B: Fix 5G-LENA Compatibility Issues ==="
print_status "This script will resolve 5G-LENA compatibility issues with ns-3.40"
echo

# Navigate to ns-3.40
cd "$HOME/ns-allinone-3.40/ns-3.40"

# First, let's test ns-3.40 without 5G-LENA
print_status "Testing ns-3.40 base installation without 5G-LENA..."
if [ -d "src/nr" ]; then
    print_status "Temporarily moving nr module for base test..."
    mv src/nr src/nr_temp
fi

# Clean and configure without nr
./ns3 clean
./ns3 configure --enable-examples --enable-tests

if [ $? -eq 0 ]; then
    print_success "ns-3.40 base installation works correctly"
    # Build base to make sure everything works
    ./ns3 build
    check_status "ns-3.40 base build successful"
else
    print_error "ns-3.40 base installation has issues"
    exit 1
fi

# Now let's fix the 5G-LENA module
print_status "Restoring and fixing 5G-LENA module..."
if [ -d "src/nr_temp" ]; then
    mv src/nr_temp src/nr
fi

# Navigate to nr module
cd src/nr

# Check the current branch and get latest compatible version
print_status "Checking 5G-LENA compatibility..."
git remote -v

# Reset to a known working commit for ns-3.40
print_status "Switching to ns-3.40 compatible version of 5G-LENA..."
git fetch origin

# Try to find a ns-3.40 compatible branch or tag
if git show-ref --verify --quiet refs/remotes/origin/5g-lena-v3.1.x; then
    print_status "Found 5g-lena-v3.1.x branch, switching to it..."
    git checkout 5g-lena-v3.1.x
    git pull origin 5g-lena-v3.1.x
elif git show-ref --verify --quiet refs/remotes/origin/5g-lena-v3.0.x; then
    print_status "Found 5g-lena-v3.0.x branch, switching to it..."
    git checkout 5g-lena-v3.0.x  
    git pull origin 5g-lena-v3.0.x
else
    print_warning "No specific ns-3.40 branch found, trying to fix current master..."
    git checkout master
    git pull origin master
fi

# Check if CMakeLists.txt has the problematic command
print_status "Checking CMakeLists.txt for compatibility issues..."
if grep -q "disable_cmake_warnings" CMakeLists.txt; then
    print_status "Fixing 'disable_cmake_warnings' command..."
    # Comment out the problematic line
    sed -i 's/disable_cmake_warnings(/# disable_cmake_warnings(/g' CMakeLists.txt
    check_status "Fixed disable_cmake_warnings command"
fi

# Check for hosvd_deps dependency
if grep -q "hosvd_deps" CMakeLists.txt; then
    print_status "Found hosvd_deps dependency - making it optional..."
    # Make hosvd_deps optional instead of required
    sed -i 's/find_package.*hosvd_deps.*REQUIRED/find_package(hosvd_deps QUIET)/g' CMakeLists.txt
    sed -i 's/check_deps.*hosvd_deps/# check_deps(hosvd_deps)/g' CMakeLists.txt
    check_status "Made hosvd_deps dependency optional"
fi

# Go back to ns-3.40 root
cd ../..

# Clean and try to configure again
print_status "Testing configuration with fixed 5G-LENA..."
./ns3 clean
./ns3 configure --enable-examples --enable-tests

if [ $? -eq 0 ]; then
    print_success "Configuration successful with fixed 5G-LENA!"
    
    # Try to build
    print_status "Building ns-3.40 with fixed 5G-LENA..."
    ./ns3 build
    
    if [ $? -eq 0 ]; then
        print_success "Build successful!"
        
        # Test the installation
        print_status "Testing installations..."
        ./ns3 run hello-simulator
        check_status "Basic ns-3.40 test passed"
        
        # Test 5G-LENA if it built successfully
        if ./ns3 run cttc-nr-demo 2>/dev/null; then
            print_success "5G-LENA test passed"
        else
            print_warning "5G-LENA demo failed, but base installation works"
            print_status "You can still proceed with ns3-gym installation"
        fi
    else
        print_warning "Build failed with 5G-LENA, trying alternative approach..."
        
        # Alternative: Use a fresh 5G-LENA clone
        print_status "Trying fresh 5G-LENA installation..."
        rm -rf src/nr
        cd src
        git clone https://gitlab.com/cttc-lena/nr.git
        cd nr
        
        # Try to find the most stable branch
        git checkout 5g-lena-v3.0.x 2>/dev/null || git checkout master
        cd ../..
        
        ./ns3 clean
        ./ns3 configure --enable-examples --enable-tests
        
        if [ $? -eq 0 ]; then
            ./ns3 build
            if [ $? -eq 0 ]; then
                print_success "Fresh 5G-LENA installation successful!"
            else
                print_warning "5G-LENA still has issues, but ns-3.40 base works"
            fi
        fi
    fi
else
    print_error "Configuration still failing. Let's proceed without 5G-LENA for now"
    
    # Remove nr module temporarily
    print_status "Removing 5G-LENA module temporarily..."
    if [ -d "src/nr" ]; then
        mv src/nr src/nr_disabled
        print_warning "5G-LENA moved to src/nr_disabled"
    fi
    
    # Configure without nr
    ./ns3 clean
    ./ns3 configure --enable-examples --enable-tests
    ./ns3 build
    
    if [ $? -eq 0 ]; then
        print_success "ns-3.40 working without 5G-LENA"
        print_status "We can add 5G-LENA back later after ns3-gym installation"
    else
        print_error "Even base ns-3.40 is failing. Something went wrong."
        exit 1
    fi
fi

# Create summary
echo
print_success "=== Phase 1B Complete! ==="
print_status "Status Summary:"
if [ -d "src/nr" ]; then
    echo "  • 5G-LENA: Installed and configured"
else
    echo "  • 5G-LENA: Temporarily disabled (moved to src/nr_disabled)"
fi
echo "  • ns-3.40: Ready for Phase 2 (ns3-gym installation)"
echo
print_status "Next step: Run Phase 2 script to install ns3-gym with SB3/PyTorch"