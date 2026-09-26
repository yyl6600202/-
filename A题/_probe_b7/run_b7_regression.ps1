$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$attachmentRoot = Get-ChildItem -LiteralPath $projectRoot -Directory |
    Where-Object {
        (Test-Path (Join-Path $_.FullName 'code\contest_io.py')) -and
        (Test-Path (Join-Path $_.FullName 'data\config.txt'))
    } |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $attachmentRoot) {
    throw 'Could not locate the A题 attachment directory'
}

# Run Python with ASCII relative paths. Windows PowerShell 5 can corrupt
# non-ASCII process arguments even when the files themselves are UTF-8.
Push-Location $attachmentRoot
try {
    $graph = '..\_probe_b7\minimal_graph.json'
    $plan = '..\_probe_b7\minimal_graph_multicore_res.json'
    $config = 'data\config.txt'

foreach ($problem in 1..3) {
    $evaluator = "code\multicore_cut_evaluate_problem_$problem.py"
    $result = "..\_probe_b7\problem_${problem}_res.json"
    $trace = "..\_probe_b7\problem_${problem}_trace.json"
    $log = "..\_probe_b7\problem_${problem}_log.txt"

    python $evaluator $graph $plan --config $config -o $result --trace-output $trace --log-output $log
    if ($LASTEXITCODE -ne 0) {
        throw "Problem $problem evaluator exited with code $LASTEXITCODE"
    }

    $data = Get-Content -LiteralPath $result -Raw | ConvertFrom-Json
    if ($data.makespan -ne 6 -or
        $data.data_movement_bytes.original_graph_copy_bytes -ne 32 -or
        $data.data_movement_bytes.scheduled_copy_bytes -ne 32 -or
        $data.data_movement_bytes.added_copy_bytes -ne 0) {
        throw "Problem $problem failed the official B.7 expected values"
    }
    if ($problem -eq 3 -and $data.cache_stats.hits -ne 0) {
        throw 'Problem 3 B.7 cache hit count was expected to be 0'
    }
    Write-Output "PASS problem=$problem makespan=$($data.makespan) added_copy_bytes=$($data.data_movement_bytes.added_copy_bytes)"
}
} finally {
    Pop-Location
}
