# Installation Guide

**Complete RL Research Environment Setup - 10 Hours Total**

## 🎯 **Overview**

This guide will take you through the systematic installation of a complete wireless RL research environment. The process is divided into phases to handle complexity and ensure reliability.

## ⚡ **Quick Installation (Automated)**

```bash
# Clone repository
git clone https://github.com/Arramadan-Mu/Sionnna_problem_-Solved-.git
cd Sionnna_problem_-Solved-

# Make scripts executable
chmod +x scripts/*.sh

# Run complete installation (10 hours)
./scripts/install_complete_environment.sh

# Test installation
source scripts/activate_complete_env.sh
./scripts/test_all_components.sh
```

## 📋 **Manual Installation (Step-by-Step)**

### **Prerequisites**

- **OS**: Ubuntu 20.04+ or WSL2 with Ubuntu
- **GPU**: NVIDIA GPU with CUDA support
- **Storage**: 100GB available space
- **Internet**: Stable connection for downloads

### **Phase 1: Foundation Setup (2 hours)**

**Goal**: Establish stable ns-3.40 base

```bash
# Run Phase 1 installation
./scripts/phase1_migrate_to_ns3_40.sh

# If 5G-LENA compatibility issues:
./scripts/phase1b_fix_5glena.sh

# For clean base installation:
./scripts/phase1c_clean_base.sh
```

**Verification**:
```bash
cd ~/ns-allinone-3.40/ns-3.40
./ns3 run hello-simulator
```

### **Phase 2: RL Infrastructure (3 hours)**

**Goal**: Install ns3-gym + SB3 + PyTorch with GPU acceleration

```bash
# Run Phase 2 installation
./scripts/phase2_rl_infrastructure.sh

# If ns3-gym issues:
./scripts/phase2b_fix_ns3gym.sh

# If Python environment issues:
./scripts/phase2c_fix_python_venv.sh
```

**Verification**:
```bash
source ~/ns3-rl-env/bin/activate
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### **Phase 3: Advanced Channel Modeling (4 hours)**

**Goal**: Integrate Sionna RT for realistic wireless channels

```bash
# Run Phase 3 installation
./scripts/phase3_install_ns3sionna.sh

# Fix CUDA conflicts:
./scripts/phase3b_fix_conflicts.sh

# Finalize setup:
./scripts/phase3c_finalize_setup.sh
```

**Verification**:
```bash
source ~/ns3-rl-setup/activate_complete_env.sh
python3 ~/ns3-rl-setup/demo_complete_integration.py
```

### **Phase 4: Testing & Optimization (1 hour)**

**Goal**: Validate complete integration and optimize performance

```bash
# Run comprehensive tests
./scripts/test_all_components.sh

# Performance optimization
./scripts/optimize_gpu_performance.sh
```

## 🔧 **Component Details**

### **Installed Software Versions**

| Component | Version | Purpose |
|-----------|---------|---------|
| Ubuntu | 24.04 LTS | Operating system |
| ns-3 | 3.40 | Network simulation |
| Python | 3.12.3 | Runtime environment |
| PyTorch | 2.5.1+cu121 | GPU-accelerated ML |
| TensorFlow | 2.19.0 | CPU-optimized ML |
| Sionna | 1.1.0 | Wireless modeling |
| SB3 | 2.7.0 | RL algorithms |

### **Directory Structure After Installation**

```
/home/username/
├── ns3-rl-env/                 # Python virtual environment
├── ns-allinone-3.40/           # ns-3 installation
│   └── ns-3.40/
│       ├── contrib/opengym/    # ns3-gym integration
│       └── src/nr/             # 5G-LENA (optional)
├── ns3sionna/                  # Sionna RT integration  
├── ns3-rl-setup/               # Setup scripts and utilities
└── 5g-lena-modules-backup/     # Backup directory
```

## 🚀 **GPU Optimization**

### **RTX 4080 Specific Settings**

```bash
# Optimal batch sizes for 12GB VRAM
export PPO_BATCH_SIZE=2048
export DQN_BUFFER_SIZE=100000
export SAC_BATCH_SIZE=512

# Memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### **Performance Tuning**

```python
# In your training scripts
import torch
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
torch.cuda.empty_cache()               # Free unused memory
```

## 🔍 **Troubleshooting**

### **Common Issues**

#### **CUDA Memory Errors**
```bash
# Solution: Reduce batch size or clear cache
python3 -c "import torch; torch.cuda.empty_cache()"
```

#### **Import Conflicts**
```bash
# Solution: Verify virtual environment
source ~/ns3-rl-env/bin/activate
which python3
```

#### **ns3-gym Connection Issues**
```bash
# Solution: Check ports and restart simulation
sudo lsof -i :5000
./ns3 run opengym
```

### **GPU Not Detected**

```bash
# Check NVIDIA drivers
nvidia-smi

# Verify PyTorch CUDA
python3 -c "import torch; print(torch.cuda.get_device_name(0))"

# Reinstall PyTorch if needed
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 📊 **Validation Checklist**

After installation, verify these components:

- [ ] **ns-3.40**: `./ns3 run hello-simulator`
- [ ] **PyTorch GPU**: CUDA available and device detected
- [ ] **SB3**: Can create PPO model with GPU device
- [ ] **ns3-gym**: Python imports successful
- [ ] **Sionna**: Core modules importable
- [ ] **Integration**: Demo script runs without errors

## ⏱️ **Installation Timeline**

```
Phase 1 (Foundation)     ██████░░░░ 2h
Phase 2 (RL Training)    ████████░░ 3h  
Phase 3 (Channel Model)  ██████████ 4h
Testing & Optimization   ██░░░░░░░░ 1h
═══════════════════════════════════
Total Project Time:     ██████████ 10h
```

## 🎯 **Next Steps**

1. **Explore Examples**: Try the demo scripts in `examples/`
2. **Read Documentation**: Review the complete technical guide
3. **Start Research**: Implement your wireless RL scenarios
4. **Optimize Performance**: Fine-tune for your specific use case

## 🆘 **Getting Help**

If you encounter issues:

1. **Check the troubleshooting section** above
2. **Run diagnostic scripts**: `./scripts/diagnose_issues.sh`
3. **Review logs**: Check installation logs in `/tmp/ns3-rl-install.log`
4. **Open an issue**: Describe your setup and error messages

---

**🎉 Installation complete! Your cutting-edge RL research environment is ready!**