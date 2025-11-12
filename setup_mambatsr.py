"""
MambaTSR Setup Script
Automatically clone and setup MambaTSR repository for Plant Disease Classification
"""

import os
import sys
import subprocess
import platform

def run_command(cmd, description, check=True):
    """Run shell command with progress indication"""
    print(f"\n{'='*70}")
    print(f"▶ {description}")
    print(f"{'='*70}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    
    if result.returncode != 0 and check:
        print(f"\n❌ ERROR: {description} failed!")
        if result.stderr:
            print(result.stderr)
        return False
    
    print(f"✅ {description} completed successfully!")
    return True


def check_git():
    """Check if git is installed"""
    try:
        subprocess.run(['git', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print(f"❌ Python 3.11+ required. Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def check_cuda():
    """Check if CUDA is available"""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA toolkit found:")
            print(result.stdout)
            return True
    except FileNotFoundError:
        pass
    
    print("⚠️  CUDA toolkit not found. GPU training may not work.")
    return False


def clone_mambatsr():
    """Clone MambaTSR repository"""
    if os.path.exists('MambaTSR'):
        print("📁 MambaTSR folder already exists.")
        response = input("Do you want to remove and re-clone? (y/n): ")
        if response.lower() == 'y':
            import shutil
            shutil.rmtree('MambaTSR')
        else:
            print("✅ Using existing MambaTSR folder.")
            return True
    
    return run_command(
        'git clone https://github.com/VIDAR-Vision/MambaTSR.git',
        'Cloning MambaTSR repository'
    )


def install_pytorch_nightly():
    """Install PyTorch nightly with CUDA support"""
    print("\n" + "="*70)
    print("PyTorch Installation Options:")
    print("="*70)
    print("1. PyTorch Nightly with CUDA 12.4 (Recommended for RTX 5060 Ti)")
    print("2. PyTorch Stable with CUDA 12.1")
    print("3. Skip (already installed)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        cmd = 'pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124'
        return run_command(cmd, 'Installing PyTorch nightly (CUDA 12.4)')
    elif choice == '2':
        cmd = 'pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121'
        return run_command(cmd, 'Installing PyTorch stable (CUDA 12.1)')
    else:
        print("✅ Skipping PyTorch installation.")
        return True


def install_dependencies():
    """Install required Python packages"""
    packages = [
        'timm',
        'matplotlib',
        'tqdm',
        'pillow',
        'numpy',
        'setuptools',
        'wheel'
    ]
    
    cmd = f'pip install {" ".join(packages)}'
    return run_command(cmd, 'Installing Python dependencies')


def compile_selective_scan():
    """Compile selective_scan CUDA kernels"""
    if not os.path.exists('MambaTSR/kernels/selective_scan'):
        print("❌ selective_scan folder not found!")
        return False
    
    # Check if already compiled
    if os.path.exists('MambaTSR/kernels/selective_scan/build'):
        print("📦 selective_scan already compiled.")
        response = input("Do you want to recompile? (y/n): ")
        if response.lower() != 'y':
            print("✅ Using existing compilation.")
            return True
    
    # Patch setup.py for compute_90
    setup_path = 'MambaTSR/kernels/selective_scan/setup.py'
    
    print("\n" + "="*70)
    print("Patching setup.py for CUDA forward compatibility")
    print("="*70)
    
    try:
        with open(setup_path, 'r') as f:
            content = f.read()
        
        # Check if already patched
        if 'compute_90' in content:
            print("✅ setup.py already patched for compute_90")
        else:
            # Add compute_90 support
            if "'nvcc'" in content:
                # Find nvcc extra_compile_args section
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if "'nvcc':" in line or '"nvcc":' in line:
                        # Find the last gencode line
                        j = i + 1
                        while j < len(lines) and 'gencode' in lines[j]:
                            j += 1
                        # Insert compute_90 line before the last gencode
                        if j > i + 1:
                            indent = len(lines[j-1]) - len(lines[j-1].lstrip())
                            new_line = ' ' * indent + "'-gencode', 'arch=compute_90,code=sm_90',"
                            lines.insert(j, new_line)
                            print(f"✅ Added compute_90 support at line {j+1}")
                            break
                
                # Write back
                with open(setup_path, 'w') as f:
                    f.write('\n'.join(lines))
                
                print("✅ setup.py patched successfully!")
            else:
                print("⚠️  Could not automatically patch setup.py")
                print("Please manually add this line to nvcc extra_compile_args:")
                print("  '-gencode', 'arch=compute_90,code=sm_90',")
    
    except Exception as e:
        print(f"⚠️  Error patching setup.py: {e}")
    
    # Compile
    original_dir = os.getcwd()
    os.chdir('MambaTSR/kernels/selective_scan')
    
    success = run_command(
        'python setup.py install',
        'Compiling selective_scan CUDA kernels (this may take a few minutes)'
    )
    
    os.chdir(original_dir)
    return success


def verify_installation():
    """Verify MambaTSR installation"""
    print("\n" + "="*70)
    print("Verifying Installation")
    print("="*70)
    
    verification_script = """
import sys
import torch

print("✓ Python:", sys.version.split()[0])
print("✓ PyTorch:", torch.__version__)
print("✓ CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("✓ CUDA version:", torch.version.cuda)
    print("✓ GPU:", torch.cuda.get_device_name(0))
    print("✓ GPU memory:", f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Test MambaTSR import
sys.path.insert(0, 'MambaTSR')
try:
    from models.vmamba import VSSM
    print("✓ MambaTSR VSSM import: OK")
    
    # Try to create model
    model = VSSM(patch_size=4, in_chans=3, num_classes=10, depths=[2,2,9,2], dims=[96,192,384,768])
    print("✓ VSSM model creation: OK")
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test selective_scan
    try:
        import selective_scan_cuda
        print("✓ selective_scan CUDA kernels: AVAILABLE")
    except ImportError:
        print("⚠ selective_scan CUDA kernels: NOT FOUND")
        print("  Model will fall back to PyTorch implementation (slower)")
    
except Exception as e:
    print(f"✗ MambaTSR verification failed: {e}")
    sys.exit(1)

print("\\n" + "="*70)
print("✅ All checks passed! MambaTSR is ready to use.")
print("="*70)
"""
    
    with open('_verify_mambatsr.py', 'w') as f:
        f.write(verification_script)
    
    result = run_command(
        f'{sys.executable} _verify_mambatsr.py',
        'Running verification tests',
        check=False
    )
    
    # Cleanup
    if os.path.exists('_verify_mambatsr.py'):
        os.remove('_verify_mambatsr.py')
    
    return result


def main():
    """Main setup function"""
    print("\n" + "="*70)
    print("MambaTSR Setup Script")
    print("Plant Disease Classification Project")
    print("="*70)
    
    # Check prerequisites
    print("\n📋 Checking prerequisites...")
    
    if not check_python_version():
        sys.exit(1)
    
    if not check_git():
        print("❌ Git not found! Please install Git first:")
        print("   https://git-scm.com/downloads")
        sys.exit(1)
    
    print("✅ Git found")
    
    has_cuda = check_cuda()
    if not has_cuda:
        response = input("\nContinue without CUDA? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Setup steps
    steps = [
        ("Clone MambaTSR", clone_mambatsr),
        ("Install PyTorch", install_pytorch_nightly),
        ("Install dependencies", install_dependencies),
        ("Compile CUDA kernels", compile_selective_scan),
        ("Verify installation", verify_installation)
    ]
    
    print("\n" + "="*70)
    print("Setup Plan:")
    print("="*70)
    for i, (step_name, _) in enumerate(steps, 1):
        print(f"{i}. {step_name}")
    
    response = input("\nProceed with setup? (y/n): ")
    if response.lower() != 'y':
        print("Setup cancelled.")
        sys.exit(0)
    
    # Execute steps
    for step_name, step_func in steps:
        if not step_func():
            print(f"\n❌ Setup failed at step: {step_name}")
            print("\nPlease check the error messages above and try again.")
            sys.exit(1)
    
    # Final message
    print("\n" + "="*70)
    print("✅ Setup Complete!")
    print("="*70)
    print("\n🎉 MambaTSR is ready to use!")
    print("\n📚 Next steps:")
    print("   1. Prepare your dataset in Data/PlantVillage/PlantVillage-Dataset-master/")
    print("   2. Run training: python train_mambatsr_plantvillage.py")
    print("   3. Check docs/ for detailed guides")
    print("\n💡 For help, see: docs/guides/MAMBATSR_SETUP_GUIDE.md")
    print("="*70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
