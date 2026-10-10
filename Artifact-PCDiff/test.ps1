param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$ModelPath,
    [int]$BatchSize = 1,
    [int]$Workers = 0,
    [string]$EvalPath = 'Artifact-PCDiff/results',
    [int]$NumEnsembles = 1,
    [ValidateSet('ddpm', 'ddim')]
    [string]$SamplingMethod = 'ddpm',
    [int]$SamplingSteps = 1000,
    [int]$Seed = 48,
    [int]$Gpu = 0,
    [switch]$ComputeMetrics,
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) {
    throw "Checkpoint not found: $ModelPath"
}
$ResolvedModelPath = (Resolve-Path -LiteralPath $ModelPath).Path
$ProjectRoot = $PSScriptRoot
$TestCsv = Join-Path $ProjectRoot '..\data\precol\test.csv'
if (-not (Test-Path -LiteralPath $TestCsv -PathType Leaf)) {
    throw "Test CSV not found: $TestCsv"
}
if ([System.IO.Path]::IsPathRooted($EvalPath)) {
    $ResolvedEvalPath = [System.IO.Path]::GetFullPath($EvalPath)
}
else {
    $ResolvedEvalPath = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $EvalPath))
}

$PythonArgs = @(
    '-m', 'Artifact-PCDiff.test_completion',
    '--path', '..\data\precol\test.csv',
    '--dataset', 'precol',
    '--model', $ResolvedModelPath,
    '--eval_path', $ResolvedEvalPath,
    '--bs', $BatchSize.ToString(),
    '--workers', $Workers.ToString(),
    '--num_ens', $NumEnsembles.ToString(),
    '--sampling_method', $SamplingMethod,
    '--sampling_steps', $SamplingSteps.ToString(),
    '--manualSeed', $Seed.ToString(),
    '--gpu', $Gpu.ToString()
)

$ExitCode = 0
Push-Location $ProjectRoot
try {
    & $Python @PythonArgs
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -eq 0 -and $ComputeMetrics) {
        & $Python -m Artifact-PCDiff.evaluate_artifact_metrics `
            --csv '..\data\precol\test.csv' `
            --results-dir (Join-Path $ResolvedEvalPath 'syn') `
            --output-csv (Join-Path $ResolvedEvalPath 'artifact_metrics.csv')
        $ExitCode = $LASTEXITCODE
    }
}
finally {
    Pop-Location
}

if ($ExitCode -ne 0) {
    exit $ExitCode
}
