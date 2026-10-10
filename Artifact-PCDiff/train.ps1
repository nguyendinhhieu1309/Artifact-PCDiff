param(
    [int]$BatchSize = 8,
    [int]$Epochs = 4000,
    [int]$Workers = 0,
    [double]$LearningRate = 0.0001,
    [int]$Seed = 42,
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$TrainCsv = Join-Path $ProjectRoot '..\data\precol\train.csv'
if (-not (Test-Path -LiteralPath $TrainCsv -PathType Leaf)) {
    throw "Training CSV not found: $TrainCsv"
}

$PythonArgs = @(
    '-m', 'Artifact-PCDiff.train_completion',
    '--path', '..\data\precol\train.csv',
    '--dataset', 'precol',
    '--bs', $BatchSize.ToString(),
    '--workers', $Workers.ToString(),
    '--niter', $Epochs.ToString(),
    '--num_points', '9012',
    '--num_nn', '820',
    '--lr', $LearningRate.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    '--manualSeed', $Seed.ToString()
)

$ExitCode = 0
Push-Location $ProjectRoot
try {
    & $Python @PythonArgs
    $ExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

if ($ExitCode -ne 0) {
    exit $ExitCode
}
