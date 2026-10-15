# Gera os PNG/SVG de cada diagrama (.mmd) e o PDF com todas as vistas (arquitetura.pdf).
# Requisitos: Node.js (para o mermaid-cli, baixado pelo npx na primeira vez) e o Microsoft Edge.
# Uso, no PowerShell, dentro desta pasta:   .\gerar_pdf.ps1
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$cli = '@mermaid-js/mermaid-cli@11.4.2'
$views = @(
    @{ f = '00_visao_geral'; t = '0. Visão geral do sistema';
       d = 'Cada técnica é um módulo próprio que lê os mesmos dados de data/. O garch_t e o ewma são independentes entre si e gravam o backtest_summary.csv com as mesmas colunas; o realized_caviar reutiliza parte do código do garch_t, trabalha com uma ação por execução e grava os próprios arquivos. A comparação do TCC é qualidade estatística versus custo computacional.' },
    @{ f = '01_fluxo_de_dados'; t = '1. EWMA: fluxo de dados';
       d = 'Do arquivo de preços até o relatório: os preços passam pela camada de qualidade de dados, que decide o que o motor pode usar; o motor produz as previsões dos três modelos (EWMA-N (ref), EWMA-N e EWMA-t); o backtest as compara com o que aconteceu.' },
    @{ f = '02_modulos'; t = '2. EWMA: módulos';
       d = 'Organização do código em camadas: cada módulo usa apenas os das camadas abaixo dele. Cada caixa corresponde a um par include/ewmar/&lt;nome&gt;.hpp + src/&lt;nome&gt;.cpp em ewma/source/.' },
    @{ f = '03_ciclo_15min'; t = '3. EWMA: ciclo de 15 minutos';
       d = 'O que acontece a cada barra fechada. A atualização leva cerca de 1 µs; a re-estimação do fim do dia, cerca de 4 ms (15 ações).' },
    @{ f = '04_modulos_garch_t'; t = '4. GARCH-t: módulos';
       d = 'Organização do código do garch_t, levantada a partir dos #include de garch_t/source/. O núcleo é o portfolio (IntradayRiskEngine); synthetic gera o mercado fictício usado por todos os módulos; revaluation reavalia as opções por Monte Carlo.' },
    @{ f = '05_realized_caviar'; t = '5. Realized-CAViaR';
       d = 'Terceira técnica (C++20): estima diretamente o quantil de 1% do retorno (VaR 99%) por uma recursão com o retorno agregado R e a variância realizada RV. Os coeficientes são estimados uma vez, fora do caminho crítico (perda pinball); a atualização online só aplica a recursão. A caixa da direita é o código reutilizado do garch_t.' }
)
$sections = ''
foreach ($v in $views) {
    npx -y $cli -i "$($v.f).mmd" -o "$($v.f).png" -s 3 -b white | Out-Null
    npx -y $cli -i "$($v.f).mmd" -o "$($v.f).svg" -b white | Out-Null
    $svg = [IO.File]::ReadAllText("$PSScriptRoot\$($v.f).svg") -replace '^<\?xml[^>]*>', ''
    $head = if ($sections -eq '') { '<h1>Arquitetura do motor de risco intraday</h1>' } else { '' }
    $sections += "<section>$head<h2>$($v.t)</h2><p>$($v.d)</p><div class=`"fig`">$svg</div></section>`n"
}
$html = @"
<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><title>Arquitetura - motor de risco intraday</title>
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
$tmp = Join-Path $env:TEMP 'arquitetura.html'
[IO.File]::WriteAllText($tmp, $html, [Text.UTF8Encoding]::new($false))
$edge = @("${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe", "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe") |
    Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $edge) { throw 'Microsoft Edge não encontrado' }
$out = Join-Path $PSScriptRoot 'arquitetura.pdf'
Start-Process -FilePath $edge -Wait -ArgumentList @('--headless', '--disable-gpu', '--no-pdf-header-footer',
    "--print-to-pdf=`"$out`"", "file:///$($tmp -replace '\\', '/')")
Write-Host "PDF gerado: $out"
