<#
Watch the Chinese corpus pipeline without asking anyone.

    pwsh scripts/zh_status.ps1                  # refresh every 5s
    pwsh scripts/zh_status.ps1 -Interval 10     # slower
    pwsh scripts/zh_status.ps1 -Target 40000    # different goal

Target is inferred when not given: the writer's own "N/M" log line first,
then 44096 for the Qwen3.5-2B re-distillation run (synth_story_q35.jsonl),
then 20000 for the legacy explanatory run.
    pwsh scripts/zh_status.ps1 -Once            # print once and exit

Ctrl-C to stop. Nothing here touches the running job: it only reads files and
asks nvidia-smi what the card is doing.
#>

param(
    [int]$Interval = 5,
    [int]$Target = 0,
    [switch]$Once
)

$root  = Split-Path -Parent $PSScriptRoot
$synth = Join-Path $root "data/zh/raw/synth.jsonl"
$story = Join-Path $root "data/zh/raw/synth_story.jsonl"
$q35   = Join-Path $root "data/zh/raw/synth_story_q35.jsonl"
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
    # Only logs touched recently: a stale run's log shows "done:" and reads
    # like the current job finished when it belongs to a run from days ago.
    $f = Get-ChildItem $env:TEMP -Filter $pattern -ErrorAction SilentlyContinue |
         Where-Object { $_.LastWriteTime -gt (Get-Date).AddHours(-6) } |
         Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $f) { return @() }
    $lines = Read-LinesShared $f.FullName
    if ($lines.Count -eq 0) { return @() }
    $take = [Math]::Min($n, $lines.Count)
    return $lines[($lines.Count - $take)..($lines.Count - 1)] |
           Where-Object { $_ -notmatch 'Loading weights' }
}

function Get-LastParagraph([string]$p) {
    $lines = Read-LinesShared $p
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
$tl = Get-LogTail "synth_*full*.log" 1
if ($tl -and $tl -match '(\d+)/(\d+)') { $logTarget = [int]$Matches[2] }
# But a resumed run's "N/M" counts only this invocation's todo: the paragraphs
# already in the file appear in neither N nor M. The writer prints
# "resuming: K paragraphs already written" once at startup, so the full job
# is K + M (the q35 re-distillation restarted at K=10113 of 44096). Scan the
# whole log, not the tail: the startup line scrolls out of a 3-line window
# within minutes and out of any fixed window within the hour.
$resumed = 0
$logFile = Get-ChildItem $env:TEMP -Filter "synth_*full*.log" -ErrorAction SilentlyContinue |
           Where-Object { $_.LastWriteTime -gt (Get-Date).AddHours(-6) } |
           Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($logFile) {
    foreach ($l in (Read-LinesShared $logFile.FullName)) {
        if ($l -match 'resuming: (\d+)') { $resumed = [int]$Matches[1]; break }
    }
}
$eff = if ($Target -gt 0) { $Target }
       elseif ($logTarget -gt 0) { $logTarget + $resumed }
       elseif (Test-Path $q35) { 44096 } else { 20000 }

$startCount = Count-Lines $synth
$startTime  = Get-Date
$prevCount  = $startCount
$prevTime   = $startTime

while ($true) {
    # Clear-Host throws "handle is invalid" when stdout is redirected.
    try { Clear-Host } catch { Write-Host ("-" * 46) }
    $now    = Get-Date
    # Whichever corpus file exists later in the pipeline is the active job;
    # right now that is the story run, earlier it was the explanatory one.
    $active = if (Test-Path $q35) { $q35 } elseif (Test-Path $story) { $story } else { $synth }
    $count  = Count-Lines $active
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

    $para = Get-LastParagraph $active
    if ($para) {
        Write-Host ""
        Write-Host "[最新样本]" -ForegroundColor Yellow
        $show = if ($para.Length -gt 110) { $para.Substring(0, 110) + "..." } else { $para }
        Write-Host "  $show"
    }

    Write-Host ""
    Write-Host "[文件]"
    Write-Host ("        主题池   {0} 条" -f (Count-Lines $topics))
    Write-Host ("        说明文   {0} 段  ({1})" -f (Count-Lines $synth), (Get-Size $synth))
    Write-Host ("        故事     {0} 段  ({1})" -f (Count-Lines $story), (Get-Size $story))
    if (Test-Path $q35) {
        Write-Host ("        Qwen3.5  {0} 段  ({1})" -f (Count-Lines $q35), (Get-Size $q35))
    }
    Write-Host ("        语料     {0}" -f (Get-Size $corpus))
    Write-Host ("        train.bin {0}" -f (Get-Size $trainBin))

    foreach ($spec in @(@("synth_*full*.log", "合成日志"), @("train*.log", "训练日志"),
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
