param(
    [switch]$TestBaseline,
    [switch]$ConvertCSV,
    [switch]$GenerateScenario1,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunArgs
)
$ErrorActionPreference = 'Stop'
$baselinePython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $baselinePython)) {
    $baselinePython = (Get-Command python -ErrorAction Stop).Source
}
$baselineExitCode = 0
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
        if ($TestBaseline) {
            & $baselinePython -m unittest discover -s demo_py/tests -t . -v
        } elseif ($ConvertCSV) {
            & $baselinePython -m demo_py.Convert_CSV @RunArgs
        } elseif ($GenerateScenario1) {
            & $baselinePython -m demo_py.Generate_Scenario.generate_scenario1 @RunArgs
        } else {
            & $baselinePython -m demo_py.run @RunArgs
        }
        $baselineExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $baselineExitCode
