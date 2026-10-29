$ErrorActionPreference = "Stop"

$pythonExe = "C:\Users\haruh\anaconda3\python.exe"
$projectDir = "C:\Users\haruh\Documents\Codex\flow_velocity_gui\auto_velocity_prototype"
$outputDir = Join-Path $projectDir "all_patterns_step1_temporal_v2_20260831"
$rawDir = Join-Path $outputDir "raw"
$calibrationMap = Join-Path $outputDir "calibration_map.json"
$framesRoot = (Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $outputDir "frames_root.txt")).Trim()
$env:MPLCONFIGDIR = Join-Path $outputDir ".matplotlib"

Set-Location -LiteralPath $projectDir

& $pythonExe (Join-Path $projectDir "run_all_unlabeled_sequences.py") `
    $framesRoot `
    --output-dir $rawDir `
    --step 1 `
    --max-gap 4 `
    --minimum-confidence 0.85 `
    --workers 12 `
    --progress-every 100 `
    --include-manual `
    --calibration-map $calibrationMap

if ($LASTEXITCODE -ne 0) {
    throw "Step-1 image-pair estimation failed with exit code $LASTEXITCODE"
}

& $pythonExe (Join-Path $projectDir "apply_temporal_v2_step1.py")
if ($LASTEXITCODE -ne 0) {
    throw "Temporal-v2 postprocessing failed with exit code $LASTEXITCODE"
}
