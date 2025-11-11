# ====================================================================
# SCRIPT DỌN DẸP WORKSPACE - AN TOÀN
# ====================================================================
# Tạo folder structure mới mà KHÔNG ẢNH HƯỞNG đến:
#   - MambaTSR/ (code gốc)
#   - Data/ (dataset)
#   - models/ (trained models)
#   - train_mambatsr_plantvillage.py (main script)
#   - .venv/ (Python environment)
# ====================================================================

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "CLEANUP WORKSPACE - MambaTSR Project" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

# Set working directory
$BaseDir = "G:\Dataset"
Set-Location $BaseDir

Write-Host "Current directory: $BaseDir" -ForegroundColor Yellow
Write-Host ""

# ====================================================================
# 1. TẠO FOLDER STRUCTURE MỚI
# ====================================================================

Write-Host "[1/5] Creating new folder structure..." -ForegroundColor Green

$NewFolders = @(
    "docs/reports",
    "docs/guides", 
    "docs/presentations",
    "scripts/tests",
    "scripts/shell",
    "notebooks",
    "archive"
)

foreach ($folder in $NewFolders) {
    $path = Join-Path $BaseDir $folder
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
        Write-Host "  ✅ Created: $folder" -ForegroundColor Gray
    } else {
        Write-Host "  ⏭️  Exists: $folder" -ForegroundColor DarkGray
    }
}

Write-Host ""

# ====================================================================
# 2. DI CHUYỂN MARKDOWN FILES
# ====================================================================

Write-Host "[2/5] Moving Markdown documentation files..." -ForegroundColor Green

# Reports
$ReportFiles = @(
    "BAO_CAO_VISION_MAMBA_2.md",
    "BAO_CAO_VAN_DE_KY_THUAT.md",
    "TRAINING_RESULTS_REPORT.md",
    "CODE_SUMMARY_FOR_TEACHER.md"
)

foreach ($file in $ReportFiles) {
    if (Test-Path $file) {
        Move-Item -Path $file -Destination "docs/reports/" -Force
        Write-Host "  ✅ Moved: $file → docs/reports/" -ForegroundColor Gray
    }
}

# Guides
$GuideFiles = @(
    "HOW_I_DID_IT.md",
    "MAMBATSR_SETUP_GUIDE.md",
    "TRAINING_GUIDE.md",
    "QUICK_START.md",
    "SOLUTION_WSL2_PYTORCH_NIGHTLY.md",
    "DANG_CHUNG_PYTORCH_CUDA_COMPATIBILITY.md",
    "MAMBATSR_RTX5060TI_FINAL_STATUS.md"
)

foreach ($file in $GuideFiles) {
    if (Test-Path $file) {
        Move-Item -Path $file -Destination "docs/guides/" -Force
        Write-Host "  ✅ Moved: $file → docs/guides/" -ForegroundColor Gray
    }
}

# Presentations
$PresentationFiles = @(
    "PRESENTATION_FOR_TEACHER.md",
    "QUICK_REFERENCE_FOR_PRESENTATION.md",
    "FINAL_SUMMARY.md"
)

foreach ($file in $PresentationFiles) {
    if (Test-Path $file) {
        Move-Item -Path $file -Destination "docs/presentations/" -Force
        Write-Host "  ✅ Moved: $file → docs/presentations/" -ForegroundColor Gray
    }
}

Write-Host ""

# ====================================================================
# 3. DI CHUYỂN TEST SCRIPTS
# ====================================================================

Write-Host "[3/5] Moving test and utility scripts..." -ForegroundColor Green

# Test scripts
$TestFiles = @(
    "check_gpu_compute.py",
    "check_mambatsr_env.py",
    "check_versions.py",
    "test_cuda.py",
    "test_direct_import.py",
    "test_import_simple.py",
    "test_mambatsr_model.py",
    "test_selective_scan.py",
    "test_step_imports.py",
    "test_train_pipeline.py",
    "verify_selective_scan.py"
)

foreach ($file in $TestFiles) {
    if (Test-Path $file) {
        Move-Item -Path $file -Destination "scripts/tests/" -Force
        Write-Host "  ✅ Moved: $file → scripts/tests/" -ForegroundColor Gray
    }
}

# Shell scripts
$ShellFiles = @(
    "run_training.sh",
    "start_training.sh",
    "START_TRAINING.bat"
)

foreach ($file in $ShellFiles) {
    if (Test-Path $file) {
        Move-Item -Path $file -Destination "scripts/shell/" -Force
        Write-Host "  ✅ Moved: $file → scripts/shell/" -ForegroundColor Gray
    }
}

Write-Host ""

# ====================================================================
# 4. DI CHUYỂN NOTEBOOKS
# ====================================================================

Write-Host "[4/5] Moving Jupyter notebooks..." -ForegroundColor Green

$NotebookFiles = @(
    "Plant_Disease_EDA.ipynb",
    "Plant_Disease_MambaTSR.ipynb",
    "Plant_Disease_MambaTSR_Colab.ipynb",
    "Plant_Disease_YOLOv4_Ensemble.ipynb"
)

foreach ($file in $NotebookFiles) {
    if (Test-Path $file) {
        Move-Item -Path $file -Destination "notebooks/" -Force
        Write-Host "  ✅ Moved: $file → notebooks/" -ForegroundColor Gray
    }
}

Write-Host ""

# ====================================================================
# 5. DI CHUYỂN MỤC ARCHIVE
# ====================================================================

Write-Host "[5/5] Moving archived/unused files..." -ForegroundColor Green

$ArchiveFiles = @(
    "training.log",
    "cuda-keyring_1.1-1_all.deb",
    "README_MAMBATSR_STATUS.md"
)

foreach ($file in $ArchiveFiles) {
    if (Test-Path $file) {
        Move-Item -Path $file -Destination "archive/" -Force
        Write-Host "  ✅ Moved: $file → archive/" -ForegroundColor Gray
    }
}

# Delete __pycache__ if exists
if (Test-Path "__pycache__") {
    Remove-Item -Path "__pycache__" -Recurse -Force
    Write-Host "  🗑️  Deleted: __pycache__/" -ForegroundColor Gray
}

Write-Host ""

# ====================================================================
# 6. TẠO README CHO CÁC FOLDER
# ====================================================================

Write-Host "[BONUS] Creating README files..." -ForegroundColor Green

# docs/README.md
$DocsReadme = @"
# Documentation

## 📁 Structure

- **reports/** - Báo cáo kỹ thuật và phân tích
- **guides/** - Hướng dẫn setup và troubleshooting
- **presentations/** - Tài liệu trình bày

## 📋 Files

### Reports
- ``BAO_CAO_VISION_MAMBA_2.md`` - Nghiên cứu Vision Mamba 2
- ``TRAINING_RESULTS_REPORT.md`` - Kết quả training chi tiết
- ``CODE_SUMMARY_FOR_TEACHER.md`` - Tóm tắt code cho thầy

### Guides
- ``HOW_I_DID_IT.md`` - Technical deep dive
- ``MAMBATSR_SETUP_GUIDE.md`` - Setup instructions
- ``TRAINING_GUIDE.md`` - Training guide

### Presentations
- ``PRESENTATION_FOR_TEACHER.md`` - 16-slide presentation
- ``QUICK_REFERENCE_FOR_PRESENTATION.md`` - Q&A cheat sheet
"@

Set-Content -Path "docs/README.md" -Value $DocsReadme -Encoding UTF8
Write-Host "  ✅ Created: docs/README.md" -ForegroundColor Gray

# scripts/README.md
$ScriptsReadme = @"
# Scripts

## 📁 Structure

- **tests/** - Test scripts và verification
- **shell/** - Shell scripts để chạy training

## 🧪 Test Scripts

- ``check_*.py`` - Environment verification
- ``test_*.py`` - Unit tests
- ``verify_*.py`` - Component validation

## 🐚 Shell Scripts

- ``run_training.sh`` - WSL training script
- ``start_training.sh`` - Linux training script
- ``START_TRAINING.bat`` - Windows batch script
"@

Set-Content -Path "scripts/README.md" -Value $ScriptsReadme -Encoding UTF8
Write-Host "  ✅ Created: scripts/README.md" -ForegroundColor Gray

Write-Host ""

# ====================================================================
# 7. SUMMARY
# ====================================================================

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "✅ CLEANUP COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "📂 NEW STRUCTURE:" -ForegroundColor Yellow
Write-Host "  G:\Dataset/" -ForegroundColor White
Write-Host "  ├── MambaTSR/           ✅ KHÔNG THAY ĐỔI" -ForegroundColor Green
Write-Host "  ├── Data/               ✅ KHÔNG THAY ĐỔI" -ForegroundColor Green
Write-Host "  ├── models/             ✅ KHÔNG THAY ĐỔI" -ForegroundColor Green
Write-Host "  ├── .venv/              ✅ KHÔNG THAY ĐỔI" -ForegroundColor Green
Write-Host "  ├── train_mambatsr_plantvillage.py  ✅ KHÔNG THAY ĐỔI" -ForegroundColor Green
Write-Host "  ├── generate_training_plots.py      ✅ KHÔNG THAY ĐỔI" -ForegroundColor Green
Write-Host "  ├── docs/               📝 15 MD files" -ForegroundColor Cyan
Write-Host "  ├── scripts/            🧪 14 scripts" -ForegroundColor Cyan
Write-Host "  ├── notebooks/          📓 4 notebooks" -ForegroundColor Cyan
Write-Host "  └── archive/            📦 3 old files" -ForegroundColor DarkGray
Write-Host ""

Write-Host "🎯 NEXT STEPS:" -ForegroundColor Yellow
Write-Host "  1. Verify training script still works:" -ForegroundColor White
Write-Host "     python train_mambatsr_plantvillage.py --help" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. Check if paths are correct:" -ForegroundColor White
Write-Host "     - MambaTSR/ should be accessible" -ForegroundColor Gray
Write-Host "     - Data/ should be accessible" -ForegroundColor Gray
Write-Host "     - models/ should contain .pth files" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. Update .gitignore if needed" -ForegroundColor White
Write-Host ""

Write-Host "💡 TIP: All important files are UNTOUCHED!" -ForegroundColor Green
Write-Host "    Only documentation and test files were organized." -ForegroundColor Gray
Write-Host ""

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
