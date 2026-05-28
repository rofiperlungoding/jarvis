# JARVIS Installer Build Script
# Prerequisites: PyInstaller installed, Inno Setup 6 installed (iscc.exe on PATH or at default location)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

Write-Host "=== JARVIS Installer Build ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Generate icon if missing
$icon = Join-Path $PSScriptRoot "jarvis.ico"
if (-not (Test-Path $icon) -or (Get-Item $icon).Length -lt 1000) {
    Write-Host "[1/4] Generating icon..." -ForegroundColor Yellow
    python (Join-Path $PSScriptRoot "generate_icon.py")
} else {
    Write-Host "[1/4] Icon exists ($((Get-Item $icon).Length / 1KB) KB)" -ForegroundColor Green
}

# Step 2: PyInstaller build
Write-Host "[2/4] Running PyInstaller..." -ForegroundColor Yellow
$spec = Join-Path $PSScriptRoot "jarvis.spec"
Push-Location $Root
try {
    pyinstaller --noconfirm --clean $spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
$distDir = Join-Path $Root "dist\JARVIS"
$exePath = Join-Path $distDir "JARVIS.exe"
if (-not (Test-Path $exePath)) { throw "Expected $exePath not found after PyInstaller" }
$distSize = (Get-ChildItem -Path $distDir -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "   Bundle size: $([math]::Round($distSize, 1)) MB" -ForegroundColor Green

# Step 3: Inno Setup compile
Write-Host "[3/4] Running Inno Setup..." -ForegroundColor Yellow
$issFile = Join-Path $PSScriptRoot "jarvis_setup.iss"
$iscc = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if (-not (Test-Path $iscc)) {
    $iscc = "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
}
if (-not (Test-Path $iscc)) {
    # Try PATH
    $iscc = (Get-Command iscc.exe -ErrorAction SilentlyContinue).Source
}
if (-not $iscc -or -not (Test-Path $iscc)) {
    Write-Host "   Inno Setup not found. Skipping installer creation." -ForegroundColor Red
    Write-Host "   Install from: https://jrsoftware.org/isdl.php" -ForegroundColor Red
    Write-Host "   The PyInstaller bundle is ready at: $distDir" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "=== Build partial (no installer) ===" -ForegroundColor Yellow
    exit 0
}
& $iscc $issFile
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed with exit code $LASTEXITCODE" }

# Step 4: Report
$outputDir = Join-Path $PSScriptRoot "output"
$installer = Get-ChildItem -Path $outputDir -Filter "JARVIS-Setup-*.exe" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($installer) {
    $installerSize = $installer.Length / 1MB
    Write-Host ""
    Write-Host "[4/4] Done!" -ForegroundColor Green
    Write-Host "   Installer: $($installer.FullName)" -ForegroundColor Cyan
    Write-Host "   Size: $([math]::Round($installerSize, 1)) MB" -ForegroundColor Cyan
} else {
    Write-Host "[4/4] Installer file not found in $outputDir" -ForegroundColor Red
}
