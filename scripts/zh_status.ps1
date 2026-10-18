<#
Watch the Chinese corpus pipeline without asking anyone.

    pwsh scripts/zh_status.ps1                  # refresh every 5s
    pwsh scripts/zh_status.ps1 -Interval 10     # slower
    pwsh scripts/zh_status.ps1 -Target 40000    # different goal
    pwsh scripts/zh_status.ps1 -Once            # print once and exit

Ctrl-C to stop. Nothing here touches the running job: it only reads files and
asks nvidia-smi what the card is doing.
#>

param(
    [int]$Interval = 5,
    [int]$Target = 20000,
    [switch]$Once
)

$root  = Split-Path -Parent $PSScriptRoot
$synth = Join-Path $root "data/zh/raw/synth.jsonl"
$topics = Join-Path $root "data/zh/raw/topics.jsonl"
$trainBin = Join-Path $root "data/zh/train.bin"
$corpus = Join-Path $root "data/zh/raw/corpus.txt"

<# The writer holds synth.jsonl open for append, so a plain ReadLines throws
   "used by another process". Open it with ReadWrite sharing instead. #>
function Read-LinesShared([string]$p) {
    if (-not (Test-Path $p)) { return @() }
    try {
        $fs = [System.IO.File]::Open($p, [System.IO.FileMode]::Open,
              [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
    } catch { return @() }
    $sr = New-Object System.IO.StreamReader($fs)
    $lines = @()
    while ($null -ne ($l = $sr.ReadLine())) { $lines += $l }
    $sr.Close(); $fs.Close()
    return $lines
}

function Count-Lines([string]$p) {
    return (Read-LinesShared $p).Count
}

function Get-Size([string]$p) {
    if (-not (Test-Path $p)) { return "-" }
    return "{0:N1} MB" -f ((Get-Item $p).Length / 1MB)
}

function Get-Gpu {
    $out = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>$null
    if (-not $out) { return "GPU  n/a" }
    $f = ($out -split ',') | ForEach-Object { $_.Trim() }
    # nvidia-smi reports "62 %", not "62": strip it before adding our own.
    $u = ($f[0] -replace '%', '').Trim()
    return "GPU  ${u}% util   VRAM $($f[1]) / $($f[2])"
}

function Get-LogTail([string]$pattern, [int]$n = 3) {
    $f = Get-ChildItem $env:TEMP -Filter $pattern -ErrorAction SilentlyContinue |
         Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $f) { return @() }
    $lines = Read-LinesShared $f.FullName
    if ($lines.Count -eq 0) { return @() }
    $take = [Math]::Min($n, $lines.Count)
    return $lines[($lines.Count - $take)..($lines.Count - 1)] |
           Where-Object { $_ -notmatch 'Loading weights' }
}

function Get-LastParagraph {
    $lines = Read-LinesShared $synth
    if ($lines.Count -eq 0) { return $null }
    $last = $lines[-1]
    if (-not $last) { return $null }
    try {
        $j = $last | ConvertFrom-Json
        return "[{0}] {1}" -f $j.topic, $j.text
    } catch { return $last }
}

# The writer's own progress line ("9216/21337 kept=...") is authoritative: the
# run resumes from however many paragraphs already exist, so the file count and
# the job's denominator differ. Prefer it unless -Target was passed explicitly.
$logTarget = 0
$tl = Get-LogTail "synth_full*.log" 1
if ($tl -and $tl -match '(\d+)/(\d+)') { $logTarget = [int]$Matches[2] }
$eff = if ($Target -gt 0) { $Target }
       elseif ($logTarget -gt 0) { $logTarget } else { 20000 }

$startCount = Count-Lines $synth
$startTime  = Get-Date
$prevCount  = $startCount
$prevTime   = $startTime

while ($true) {
    # Clear-Host throws "handle is invalid" when stdout is redirected.
    try { Clear-Host } catch { Write-Host ("-" * 46) }
    $now    = Get-Date
    $count  = Count-Lines $synth
    $dt     = ($now - $prevTime).TotalSeconds
    $speed  = if ($dt -gt 0.5) { ($count - $prevCount) / $dt } else { 0 }
    $avg    = if (($now - $startTime).TotalSeconds -gt 1) {
                  ($count - $startCount) / ($now - $startTime).TotalSeconds } else { 0 }

    Write-Host "===== ESP32-AI 中文语料流水线 =====" -ForegroundColor Cyan
    Write-Host ("时间   {0:HH:mm:ss}   刷新 {1}s   (Ctrl-C 退出)" -f $now, $Interval)

    $pct = if ($eff -gt 0) { 100.0 * $count / $eff } else { 0 }
    Write-Host ""
    Write-Host "[合成]  $count / $eff  ($('{0:N1}' -f $pct)%)"
    Write-Host ("        速度 {0:N2} 段/秒 (瞬时)   {1:N2} 段/秒 (均值)" -f $speed, $avg)
    if ($speed -gt 0.01 -and $count -lt $eff) {
        $eta = [TimeSpan]::FromSeconds(($eff - $count) / $speed)
        Write-Host ("        剩余 {0:hh\:mm\:ss}   预计完成 {1:HH:mm}" -f $eta, ($now + $eta))
    } elseif ($count -ge $eff) {
        Write-Host "        已完成目标数量" -ForegroundColor Green
    }

    Write-Host ""
    Write-Host (Get-Gpu)

    $para = Get-LastParagraph
    if ($para) {
        Write-Host ""
        Write-Host "[最新样本]" -ForegroundColor Yellow
        $show = if ($para.Length -gt 110) { $para.Substring(0, 110) + "..." } else { $para }
        Write-Host "  $show"
    }

    Write-Host ""
    Write-Host "[文件]"
    Write-Host ("        主题池   {0} 条" -f (Count-Lines $topics))
    Write-Host ("        合成     {0} 段  ({1})" -f $count, (Get-Size $synth))
    Write-Host ("        语料     {0}" -f (Get-Size $corpus))
    Write-Host ("        train.bin {0}" -f (Get-Size $trainBin))

    foreach ($spec in @(@("synth_full*.log", "合成日志"), @("train*.log", "训练日志"),
                        @("sample*.log", "采样输出"))) {
        $tail = Get-LogTail $spec[0] 3
        if ($tail) {
            Write-Host ""
            Write-Host "[$($spec[1])] $($spec[0])" -ForegroundColor Cyan
            foreach ($l in $tail) { Write-Host "  $l" }
        }
    }

    if ($Once) { break }
    $prevCount = $count
    $prevTime  = $now
    Start-Sleep -Seconds $Interval
}
