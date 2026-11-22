# Generates the PNG/SVG of each diagram (.mmd) and the PDF with every view (architecture.pdf).
# Needs Node.js (for mermaid-cli, fetched by npx the first time) and Microsoft Edge.
# Usage, in PowerShell, inside this folder:   .\generate_pdf.ps1
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$cli = '@mermaid-js/mermaid-cli@11.4.2'
$views = @(
    @{ f = '00_overview'; t = '0. Overview of the system';
       d = 'Each technique is a module of its own that reads the same data in data/. garch_t and ewma work at portfolio level and write backtest_summary.csv with the same columns. realized_caviar runs the central comparison: Realized-CAViaR, EWMA-N, EWMA-t and GARCH-t on the same stock, the same targets and the same calibration calendar. The shared library common/ (model interface, RiskMetrics baseline, jump filter) is compiled into the three engines.' },
    @{ f = '01_ewma_data_flow'; t = '1. EWMA: data flow';
       d = 'From the price file to the report: the data-quality layer decides what the engine receives and the shared jump filter the models'' view of each bar; the engine runs the three models (EWMA-N (ref), EWMA-N, EWMA-t) behind the common interface; the backtest compares them with what happened.' },
    @{ f = '02_ewma_modules'; t = '2. EWMA: modules';
       d = 'The code in layers: each module uses only those of the layers below it. Each box is a pair include/ewmar/&lt;name&gt;.hpp + src/&lt;name&gt;.cpp in ewma/source/.' },
    @{ f = '03_ewma_15min_cycle'; t = '3. EWMA: 15-minute cycle';
       d = 'What happens at each closed bar. The update takes a few microseconds; the end-of-day re-estimation a few milliseconds (15 stocks).' },
    @{ f = '04_garch_t_modules'; t = '4. GARCH-t: modules';
       d = 'The code of garch_t, from the #includes of garch_t/source/. The model layer is portfolio (IntradayRiskEngine, the GARCH-t/EVT model) with models, which puts it and the shared baseline behind the common interface; revaluation revalues the options by Monte Carlo off the bar thread; data_quality applies the shared jump filter to a bar panel.' },
    @{ f = '05_realized_caviar'; t = '5. Realized-CAViaR';
       d = 'Third technique and central comparison (C++20): Realized-CAViaR estimates the 1% return quantile by a recursion on the aggregate return R and the realized variance RV; EWMA-N, EWMA-t and GARCH-t of one stock, on returns divided by the intraday profile, are calibrated on the same calendar and scored on the same targets. In shared-data mode the jump filter runs over every stock first. Latency measured in batches.' }
)
$sections = ''
foreach ($v in $views) {
    npx -y $cli -i "$($v.f).mmd" -o "$($v.f).png" -s 3 -b white | Out-Null
    npx -y $cli -i "$($v.f).mmd" -o "$($v.f).svg" -b white | Out-Null
    $svg = [IO.File]::ReadAllText("$PSScriptRoot\$($v.f).svg") -replace '^<\?xml[^>]*>', ''
    $head = if ($sections -eq '') { '<h1>Architecture of the intraday risk engine</h1>' } else { '' }
    $sections += "<section>$head<h2>$($v.t)</h2><p>$($v.d)</p><div class=`"fig`">$svg</div></section>`n"
}
$html = @"
<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Architecture - intraday risk engine</title>
<style>
@page { size: A4 landscape; margin: 14mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; color: #222; }
section { page-break-after: always; height: 178mm; display: flex; flex-direction: column; }
section:last-child { page-break-after: auto; }
h1 { font-size: 20pt; margin: 0 0 4mm; } h2 { font-size: 15pt; margin: 0 0 2mm; }
p { font-size: 10.5pt; margin: 0 0 4mm; color: #444; }
.fig { flex: 1; display: flex; align-items: center; justify-content: center; min-height: 0; }
.fig svg { max-width: 100%; max-height: 100%; width: auto !important; height: auto; }
</style></head><body>
$sections
</body></html>
"@
$tmp = Join-Path $env:TEMP 'architecture.html'
[IO.File]::WriteAllText($tmp, $html, [Text.UTF8Encoding]::new($false))
$edge = @("${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe", "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe") |
    Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $edge) { throw 'Microsoft Edge not found' }
$out = Join-Path $PSScriptRoot 'architecture.pdf'
Start-Process -FilePath $edge -Wait -ArgumentList @('--headless', '--disable-gpu', '--no-pdf-header-footer',
    "--print-to-pdf=`"$out`"", "file:///$($tmp -replace '\\', '/')")
Write-Host "PDF written: $out"
