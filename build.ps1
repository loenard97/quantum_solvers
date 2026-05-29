$ErrorActionPreference = "Stop"

$pythonExe = (Get-Command python).Source

if (-not $pythonExe) {
    throw "Python not found in PATH"
}

$pythonInclude = python -c "from sysconfig import get_paths; print(get_paths()['include'])"
$pythonLibDir  = python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR') or sysconfig.get_config_var('LIBPL'))"
$pythonVersion = python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')"

$pybindIncludes = python -m pybind11 --includes

Write-Host "Python executable: $pythonExe"
Write-Host "Python include:    $pythonInclude"
Write-Host "Python libs:       $pythonLibDir"
Write-Host "Python version:    $pythonVersion"
Write-Host ""

$cmd = @(
    "g++",
    "-O3",
    "-shared",
    "-static",
    "-I./pybind11/include",
    "-I./eigen",
    "-I$pythonInclude",
    $pybindIncludes,
    "./src/_bindings.cpp",
    "-L$pythonLibDir",
    "-lpython$pythonVersion",
    "-o",
    "./solvers/_core_windows.pyd"
)

Write-Host "Running build:"
Write-Host ($cmd -join " ")
Write-Host ""

& $cmd[0] $cmd[1..($cmd.Length - 1)]

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Build successful."
} else {
    throw "Build failed."
}