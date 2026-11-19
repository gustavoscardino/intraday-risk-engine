# Painel: VaR intradiário

Site estático (sem etapa de build) que apresenta o TCC. Ele tem seis abas:

- **Visão geral**
- **Como funcionam:** gráficos interativos de cada modelo.
- **Replay ao vivo:** o motor C++ compilado para WebAssembly roda no navegador.
- **Comparação:** os resultados nativos das 30 ações, com Diebold–Mariano.
- **Arquitetura e qualidade:** diagramas, ATAM e os motores de carteira.
- **Método e limitações**

O painel tem duas línguas, português (padrão) e inglês, com um botão no topo.

## Rodar localmente

Os módulos ES e o WebAssembly não abrem por `file://`; é preciso um servidor HTTP. Na raiz do repositório:

```bash
python -m http.server 8080 -d dashboard
```

Depois, abra <http://localhost:8080>.

## Publicar no GitHub Pages

O workflow [`.github/workflows/pages.yml`](../.github/workflows/pages.yml) publica a pasta `dashboard/` (sem `tools/`) a cada push na `main` que mexa nela. Ele também pode ser disparado à mão, em *Actions → Publish dashboard → Run workflow*.

Configuração única, no GitHub: *Settings → Pages → Build and deployment → Source: **GitHub Actions***. O site fica em `https://<usuário>.github.io/intraday-risk-engine/`.

## Publicar no Vercel (alternativa)

1. Importe o repositório no Vercel (*Add New → Project*).
2. Em **Root Directory**, escolha `dashboard`.
3. Em **Framework Preset**, use `Other`. Deixe **Build Command** e **Install Command** vazios e **Output Directory** como `.` (o padrão).
4. Clique em *Deploy*.

O `vercel.json` define o tipo `application/wasm` e o cache dos arquivos do motor. O `.vercelignore` tira a pasta `tools/` do site publicado. Cada push na branch configurada gera um novo deploy.

## Estrutura

```text
dashboard/
├── index.html              página única com as seis abas
├── css/style.css
├── js/
│   ├── app.js              roteador por hash (#overview, #how, ...), carrega cada aba sob demanda
│   ├── i18n.js, strings.js textos em PT e EN; t('chave', {variáveis})
│   ├── data.js             lê data/*.json
│   ├── engine.js           cliente do worker: runComparison(), tailFactors()
│   ├── engine.worker.js    carrega engine/rcaviar.js (WebAssembly) num Web Worker
│   ├── sources.js          ReplaySource e Player (a LiveSource do nível C segue a mesma interface)
│   ├── charts.js           cores dos modelos e utilitários do ECharts
│   └── tabs/               uma aba por arquivo
├── engine/                 motor compilado: rcaviar.js, .wasm e .data (os CSVs de preços embutidos)
├── data/                   resultados nativos exportados (comparação, ATAM, carteira)
├── assets/                 diagramas de arquitetura (SVG) e favicon
├── vendor/                 ECharts 5.5.0 (licença Apache-2.0, ECHARTS_LICENSE.txt)
├── tools/                  scripts que geram engine/ e data/ (não são publicados)
├── vercel.json
└── LEVEL_C.md              plano para dados ao vivo
```

## Atualizar os resultados

Os arquivos em `data/` são gerados a partir das saídas dos motores nativos. Depois de mudar código ou dados:

```bat
realized_caviar\run_comparison.bat      :: comparação nas 30 ações (realized_caviar/comparison/)
atam\run.bat                            :: avaliação ATAM (atam/results/), ~10 min
python dashboard\tools\export_data.py   :: gera dashboard/data/*.json e copia os diagramas
```

O `export_data.py` também lê os backtests dos motores de carteira (`ewma/results`, `garch_t/results`) e grava em `data/meta.json` a data e o commit atual do repositório, que aparecem no rodapé.

## Recompilar o motor WebAssembly

Só é preciso quando mudar o código C++ (`realized_caviar`, `garch_t`, `ewma`) ou os CSVs de preços. É necessário o [Emscripten](https://emscripten.org/docs/getting_started/downloads.html); o motor foi compilado com a versão 4.0.10.

```powershell
# Windows (PowerShell). O padrão é o emsdk em %USERPROFILE%\emsdk
powershell -ExecutionPolicy Bypass -File dashboard\tools\build_engine.ps1 -Emsdk C:\caminho\do\emsdk
```

```bash
# Linux/macOS, com o emsdk ativado (source emsdk_env.sh)
dashboard/tools/build_engine.sh
```

A API exposta ao JavaScript fica em `realized_caviar/source/wasm/engine_api.cpp`:

- `tickers(dataset)`
- `run(dataset, ticker, windowDays, confidence, lambda, diurnal, exposure, benchmarkUpdates)`
- `tail(nu, confidence)`

Todas devolvem JSON. O resultado do WebAssembly foi conferido contra o motor nativo: as mesmas exceções e a mesma perda pinball.

## Próximo nível

Veja [LEVEL_C.md](LEVEL_C.md) para ligar o motor a um feed ao vivo.
