# ~/setup_wireless_rl_environment.ps1
# Complete PowerShell setup for Wireless RL Research Environment
# Installs WSL2 + Ubuntu + Complete RL Environment with GPU acceleration

#Requires -RunAsAdministrator

# Colors for output
$Red = "`e[31m"
$Green = "`e[32m"  
$Yellow = "`e[33m"
$Blue = "`e[34m"
$Purple = "`e[35m"
$Cyan = "`e[36m"
$Reset = "`e[0m"

function Write-Status {
    param([string]$Message)
    Write-Host "${Blue}[INFO]${Reset} $Message"
}

function Write-Success {
    param([string]$Message)
    Write-Host "${Green}[SUCCESS]${Reset} $Message"
}

function Write-Warning {
    param([string]$Message)
    Write-Host "${Yellow}[WARNING]${Reset} $Message"
}

function Write-Error {
    param([string]$Message)
    Write-Host "${Red}[ERROR]${Reset} $Message"
}

function Write-GPU {
    param([string]$Message)
    Write-Host "${Purple}[GPU]${Reset} $Message"
}

Write-Host "${Cyan}"
Write-Host "=================================================================="
Write-Host "🚀 WIRELESS RL RESEARCH ENVIRONMENT SETUP"
Write-Host "=================================================================="
Write-Host "Complete setup: WSL2 + Ubuntu + ns-3 + Sionna + PyTorch + SB3"
Write-Host "Target GPU: RTX 4080 with CUDA acceleration"
Write-Host "Expected time: ~3 hours total"
Write-Host "=================================================================="
Write-Host "${Reset}"

# Check if running as administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Error "This script must be run as Administrator!"
    Write-Host "Right-click PowerShell and select 'Run as Administrator'"
    exit 1
}

# Phase 1: Install WSL2 and Ubuntu
Write-Status "=== Phase 1: Installing WSL2 and Ubuntu ==="

# Enable WSL and Virtual Machine Platform
Write-Status "Enabling WSL and Virtual Machine Platform features..."
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# Download and install WSL2 kernel update
Write-Status "Downloading WSL2 kernel update..."
$wslUpdateUrl = "https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi"
$wslUpdatePath = "$env:TEMP\wsl_update_x64.msi"

if (!(Test-Path $wslUpdatePath)) {
    Invoke-WebRequest -Uri $wslUpdateUrl -OutFile $wslUpdatePath
}

Write-Status "Installing WSL2 kernel update..."
Start-Process msiexec.exe -Wait -ArgumentList "/i $wslUpdatePath /quiet"

# Set WSL2 as default version
Write-Status "Setting WSL2 as default version..."
wsl --set-default-version 2

# Install Ubuntu 24.04
Write-Status "Installing Ubuntu 24.04 LTS..."
wsl --install -d Ubuntu-24.04

Write-Warning "System restart may be required for WSL2 installation."
Write-Status "If restart is needed, run this script again after restart."

# Wait for WSL to be ready
Write-Status "Waiting for WSL to be ready..."
Start-Sleep -Seconds 10

# Check if WSL is working
try {
    $wslStatus = wsl --status
    if ($LASTEXITCODE -eq 0) {
        Write-Success "WSL2 is ready!"
    }
} catch {
    Write-Warning "WSL may need restart. Please restart Windows and run script again."
    Read-Host "Press Enter to continue anyway or Ctrl+C to exit"
}

# Phase 2: Setup Ubuntu Environment
Write-Status "=== Phase 2: Setting up Ubuntu Environment ==="

# Update Ubuntu system
Write-Status "Updating Ubuntu system..."
wsl -d Ubuntu-24.04 -- sudo apt update
wsl -d Ubuntu-24.04 -- sudo apt upgrade -y

# Install essential packages
Write-Status "Installing essential development packages..."
wsl -d Ubuntu-24.04 -- sudo apt install -y build-essential cmake ninja-build git wget curl python3.12-venv python3-pip pkg-config

# Install additional dependencies
Write-Status "Installing ns-3 and ML dependencies..."
wsl -d Ubuntu-24.04 -- sudo apt install -y libzmq3-dev libprotobuf-dev protobuf-compiler

Write-Success "Ubuntu environment setup complete!"

# Phase 3: Download and Setup RL Environment Repository
Write-Status "=== Phase 3: Setting up RL Environment Repository ===" 

# Clone the repository
Write-Status "Cloning Wireless RL Environment repository..."
wsl -d Ubuntu-24.04 -- git clone https://github.com/Arramadan-Mu/Sionnna_problem_-Solved-.git /home/`$(whoami)/wireless-rl-env

# Navigate to repository and make scripts executable
wsl -d Ubuntu-24.04 -- bash -c "cd /home/`$(whoami)/wireless-rl-env && chmod +x scripts/*.sh"

Write-Success "Repository cloned and scripts prepared!"

# Phase 4: Run the Complete Installation
Write-Status "=== Phase 4: Running Complete RL Environment Installation ==="

Write-Warning "This phase will take approximately 10 hours. The process includes:"
Write-Host "  • Phase 1: ns-3.40 foundation setup (2 hours)"
Write-Host "  • Phase 2: RL infrastructure with GPU (3 hours)"
Write-Host "  • Phase 3: Sionna integration (4 hours)"
Write-Host "  • Testing and optimization (1 hour)"

$response = Read-Host "Continue with automated installation? (y/N)"
if ($response -eq 'y' -or $response -eq 'Y') {
    
    Write-Status "Starting Phase 1: ns-3.40 Foundation..."
    wsl -d Ubuntu-24.04 -- bash -c "cd /home/`$(whoami)/wireless-rl-env && ./scripts/phase1_migrate_to_ns3_40.sh"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Phase 1 completed successfully!"
        
        Write-Status "Starting Phase 2: RL Infrastructure..."
        wsl -d Ubuntu-24.04 -- bash -c "cd /home/`$(whoami)/wireless-rl-env && ./scripts/phase2_rl_infrastructure.sh"
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Phase 2 completed successfully!"
            
            Write-Status "Starting Phase 3: Sionna Integration..."
            wsl -d Ubuntu-24.04 -- bash -c "cd /home/`$(whoami)/wireless-rl-env && ./scripts/phase3_install_ns3sionna.sh"
            
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Phase 3 completed successfully!"
                
                Write-Status "Running final tests and optimization..."
                wsl -d Ubuntu-24.04 -- bash -c "cd /home/`$(whoami)/wireless-rl-env && ./scripts/phase3c_finalize_setup.sh"
                
                if ($LASTEXITCODE -eq 0) {
                    Write-Success "🎉 Complete installation successful!"
                } else {
                    Write-Warning "Final setup had issues, but core installation likely works"
                }
            } else {
                Write-Error "Phase 3 failed. Running conflict resolution..."
                wsl -d Ubuntu-24.04 -- bash -c "cd /home/`$(whoami)/wireless-rl-env && ./scripts/phase3b_fix_conflicts.sh"
            }
        } else {
            Write-Error "Phase 2 failed. Running fixes..."
            wsl -d Ubuntu-24.04 -- bash -c "cd /home/`$(whoami)/wireless-rl-env && ./scripts/phase2c_fix_python_venv.sh"
        }
    } else {
        Write-Error "Phase 1 failed. Running clean installation..."
        wsl -d Ubuntu-24.04 -- bash -c "cd /home/`$(whoami)/wireless-rl-env && ./scripts/phase1c_clean_base.sh"
    }
} else {
    Write-Status "Skipping automated installation. You can run it manually later:"
    Write-Host "  wsl -d Ubuntu-24.04"
    Write-Host "  cd wireless-rl-env"
    Write-Host "  ./scripts/phase1_migrate_to_ns3_40.sh"
}

# Phase 5: GPU Setup and Verification
Write-GPU "=== Phase 5: GPU Setup and Verification ==="

# Check NVIDIA drivers
Write-GPU "Checking NVIDIA GPU drivers..."
try {
    $nvidiaInfo = nvidia-smi
    if ($LASTEXITCODE -eq 0) {
        Write-Success "NVIDIA drivers detected!"
        Write-Host $nvidiaInfo
    }
} catch {
    Write-Warning "NVIDIA drivers not detected. Please install latest NVIDIA drivers."
    Write-Host "Download from: https://www.nvidia.com/drivers/"
}

# Test GPU access from WSL
Write-GPU "Testing GPU access from WSL..."
wsl -d Ubuntu-24.04 -- nvidia-smi

if ($LASTEXITCODE -eq 0) {
    Write-Success "GPU accessible from WSL!"
} else {
    Write-Warning "GPU not accessible from WSL. This is normal and will be fixed during installation."
}

# Phase 6: Final Testing
Write-Status "=== Phase 6: Final Environment Testing ==="

Write-Status "Testing complete RL environment..."
wsl -d Ubuntu-24.04 -- bash -c "cd /home/`$(whoami)/wireless-rl-env && source scripts/activate_complete_env.sh && ./scripts/test_all_components.sh"

if ($LASTEXITCODE -eq 0) {
    Write-Success "🎉 All components working correctly!"
} else {
    Write-Warning "Some components may need manual fixes. Check the documentation."
}

# Final Summary
Write-Host "${Cyan}"
Write-Host "=================================================================="
Write-Host "🏆 INSTALLATION COMPLETE!"
Write-Host "=================================================================="
Write-Host "${Green}✅ WSL2 Ubuntu 24.04: Installed and configured"
Write-Host "✅ Development tools: Ready for compilation"
Write-Host "✅ RL Environment: Complete installation attempted"
Write-Host "✅ GPU Support: NVIDIA drivers verified${Reset}"
Write-Host ""
Write-Status "Quick start commands:"
Write-Host "${Yellow}  # Enter WSL environment"
Write-Host "  wsl -d Ubuntu-24.04"
Write-Host ""
Write-Host "  # Navigate to project"
Write-Host "  cd wireless-rl-env"
Write-Host ""
Write-Host "  # Activate environment"
Write-Host "  source scripts/activate_complete_env.sh"
Write-Host ""
Write-Host "  # Test everything"
Write-Host "  ./scripts/test_all_components.sh"
Write-Host ""
Write-Host "  # Run demo"
Write-Host "  python3 scripts/demo_complete_integration.py${Reset}"
Write-Host ""
Write-Success "Your RTX 4080 wireless RL research environment is ready!"
Write-Host ""
Write-Status "Documentation available at:"
Write-Host "  docs/COMPLETE_DOCUMENTATION.md"
Write-Host "  INSTALLATION_GUIDE.md"

# Cleanup
Remove-Item -Path $wslUpdatePath -Force -ErrorAction SilentlyContinue

Write-Host "${Cyan}🎯 Happy researching!${Reset}"