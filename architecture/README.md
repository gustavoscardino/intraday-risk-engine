# Arquitetura do motor de risco intraday

Seis vistas do sistema: uma visão geral com todos os módulos e o detalhe de cada técnica.

| Vista | Pergunta | Arquivo |
|---|---|---|
| 0. Visão geral | quais são as partes do sistema e como se ligam? | [`00_visao_geral.mmd`](00_visao_geral.mmd) |
| 1. EWMA: fluxo de dados | o que acontece com um preço desde o arquivo até o VaR? | [`01_fluxo_de_dados.mmd`](01_fluxo_de_dados.mmd) |
| 2. EWMA: módulos | como o código do `ewma` está organizado? | [`02_modulos.mmd`](02_modulos.mmd) |
| 3. EWMA: ciclo de 15 minutos | o que o motor faz a cada barra? | [`03_ciclo_15min.mmd`](03_ciclo_15min.mmd) |
| 4. GARCH-t: módulos | como o código do `garch_t` está organizado? | [`04_modulos_garch_t.mmd`](04_modulos_garch_t.mmd) |
| 5. Realized-CAViaR | como o `realized_caviar` funciona e o que reutiliza? | [`05_realized_caviar.mmd`](05_realized_caviar.mmd) |

Os diagramas estão em [Mermaid](https://mermaid.js.org): são texto, ficam versionados no git e
aparecem desenhados no GitHub e no VS Code (extensão *Markdown Preview Mermaid Support*).

O PDF `arquitetura.pdf` (para o drive compartilhado) e os PNG/SVG desta pasta são gerados por
`.\gerar_pdf.ps1` a partir dos `.mmd`; rode-o de novo depois de editar um diagrama.

---

## 0. Visão geral

Cada técnica é um módulo próprio que lê os mesmos dados de `data/`. O `garch_t` e o `ewma` são
independentes entre si e gravam o `backtest_summary.csv` com as mesmas colunas. O `realized_caviar`
reutiliza parte do código do `garch_t` (compila `math`, `optimize`, `risk`, `backtest` e
`market_data` a partir das fontes dele), trabalha com uma ação por execução e grava os próprios
arquivos (`forecasts.csv`, `report.txt`). A comparação do TCC é qualidade estatística versus custo
computacional.

```mermaid
flowchart LR
    subgraph Dados["data/ (compartilhado pelos módulos)"]
        direction TB
        SY["synthetic/<br/>15 ações fictícias · 150 pregões<br/>carteira · fills · opções<br/>feed limpo e feed com falhas"]
        B3["b3_real/<br/>15 ações da B3 · 60 pregões<br/>(Yahoo Finance)"]
        SY ~~~ B3
    end

    subgraph Tecnicas["Técnicas de VaR (um módulo por técnica)"]
        direction TB
        GT["garch_t/ · C++17<br/>GARCH(1,1)-t por ação · carteira<br/>correlação dinâmica · cauda EVT<br/>opções por Monte Carlo"]
        EW["ewma/ · C++17<br/>EWMA-N (ref) · EWMA-N · EWMA-t · carteira<br/>camada de qualidade de dados"]
        RC["realized_caviar/ · C++20<br/>Realized-CAViaR · uma ação por execução<br/>quantil de 1% direto (VaR 99%)<br/>reutiliza math · risk · backtest · market_data do garch_t"]
        GT ~~~ EW ~~~ RC
    end

    subgraph Resultados["results/ de cada módulo"]
        direction TB
        RS["garch_t · ewma<br/>risk_15m.csv · risk_overnight.csv<br/>backtest_summary.csv (mesmas colunas)"]
        BS["realized_caviar<br/>forecasts.csv · report.txt<br/>Kupiec · Christoffersen · pinball"]
        RS ~~~ BS
    end

    CMP["Comparação no TCC<br/>qualidade estatística × custo computacional"]

    Dados --> Tecnicas --> Resultados --> CMP
```

## 1. EWMA: fluxo de dados

Do arquivo de preços até o relatório. Os preços passam primeiro pela camada de qualidade de dados,
que decide o que o motor pode usar; o motor produz as previsões dos três modelos; o backtest as
compara com o que aconteceu.

```mermaid
flowchart LR
    subgraph Entrada["data/"]
        direction TB
        P["prices_15m.csv<br/>(ordem de chegada)"]
        A["assets.csv · portfolio_initial.csv<br/>fills.csv · options.csv"]
    end

    subgraph DQ["Camada de qualidade de dados"]
        direction TB
        I["Ingestão<br/>DUPLICATE · OUT_OF_ORDER · DELAY"]
        G["Painel de barras<br/>GAP · STALE"]
        S["Regra SPIKE<br/>|z| > 6, confirmada na barra seguinte"]
        M["dq_score e modo<br/>NORMAL · DEGRADED · FREEZE"]
        I --> G --> S --> M
    end

    subgraph Motor["Motor EWMA"]
        direction TB
        R["Retornos<br/>1ª barra sem o gap · ÷√k"]
        U["Perfil diurno (U)"]
        E["Filtros EWMA<br/>ref λ = 0,94 · λ estimado"]
        F["VaR / ES<br/>EWMA-N (ref) · EWMA-N · EWMA-t"]
        O["Risco overnight"]
        R --> U --> E --> F
        E --> O
    end

    subgraph Saida["results/"]
        direction TB
        B["Backtest<br/>Kupiec · Christoffersen<br/>Berkowitz · ES Z2"]
        C["report.txt e CSVs<br/>dq_events · alerts · dq_detection"]
        B --> C
    end

    Entrada --> DQ --> Motor --> Saida
```

## 2. EWMA: módulos

Organização do código em camadas: cada módulo usa apenas os das camadas abaixo dele. Cada caixa é
um par `include/ewmar/<nome>.hpp` + `src/<nome>.cpp` em `ewma/source/`.

```mermaid
flowchart TB
    subgraph Apps["Programas (apps/)"]
        APP1["ewma_risk<br/>replay + relatório"]
        APP2["benchmark<br/>velocidade"]
    end

    REP["replay<br/>ciclo de 15 min + backtest de todas as previsões"]

    subgraph Nucleo["Núcleo"]
        ENG["engine<br/>on_bar · refit · compute · compute_overnight"]
        DQ["data_quality<br/>regras · dq_score · modos · detecção"]
        BT["backtest<br/>testes estatísticos"]
    end

    subgraph Modelo["Modelo"]
        EW["ewma<br/>covariância · estimação de λ"]
        SE["seasonality<br/>perfil U"]
        RK["risk<br/>fatores N e t · ν · Black–Scholes"]
    end

    subgraph Base["Base"]
        MD["market_data<br/>CSV · calendário B3 · painel · retornos"]
        MA["math<br/>Φ · t de Student · Brent · números aleatórios"]
    end

    APP1 --> REP
    APP2 --> ENG
    REP --> ENG & DQ & BT
    REP --> MD
    ENG --> EW & SE & RK
    DQ --> MD
    BT --> RK
    EW & RK & BT --> MA
```

## 3. EWMA: ciclo de 15 minutos

O que acontece a cada barra fechada. A atualização leva cerca de 1 µs; a re-estimação do fim do
dia, cerca de 4 ms (15 ações).

```mermaid
flowchart LR
    A([Barra t fecha])

    subgraph Dados["Dados e qualidade"]
        direction TB
        B["Decide os preços suspeitos da barra t−1<br/>reverteu → bad tick, preço excluído<br/>não reverteu → choque real, mantido"]
        C["Pontua a previsão anterior<br/>perda da barra t−1 vs VaR previsto"]
        D["Aplica as ordens executadas (fills)<br/>atualiza posições"]
        E["Calcula os retornos<br/>aplica a regra SPIKE"]
        F["Calcula dq_score e o modo"]
        B --> C --> D --> E --> F
    end

    subgraph Modelo["Modelo e previsão"]
        direction TB
        G{FREEZE?}
        H["Não atualiza o EWMA<br/>mantém o último VaR/ES válido"]
        I["Atualiza os 3 filtros EWMA"]
        J{Fim do dia?}
        K["Re-estima perfil U, λ e ν na janela<br/>previsão overnight"]
        L["Prevê σ, VaR e ES da barra t+1<br/>EWMA-N (ref) · EWMA-N · EWMA-t"]
        G -- sim --> H --> J
        G -- não --> I --> J
        J -- sim --> K --> L
        J -- não --> L
    end

    Z([Próxima barra: volta ao início])
    A --> Dados --> Modelo --> Z
```

## 4. GARCH-t: módulos

Organização do código do `garch_t` (módulo do colega), levantada a partir dos `#include` de
`garch_t/source/`. Cada caixa é um par `include/garcht/<nome>.hpp` + `src/<nome>.cpp`. O
núcleo é o `portfolio` (`IntradayRiskEngine`); `synthetic` gera o mercado fictício usado por
todos os módulos, e `revaluation` reavalia as opções por Monte Carlo.

```mermaid
flowchart TB
    subgraph Apps["Programas (apps/)"]
        IR["intraday_risk<br/>replay + relatório"]
        MC["monte_carlo<br/>estudo em vários mercados"]
        GD["generate_data<br/>gera o mercado sintético"]
        FS["fit_series<br/>ajusta um GARCH a uma série"]
        BM["benchmark<br/>velocidade"]
    end

    REP["replay<br/>ciclo de 15 min + backtest de todas as previsões"]

    subgraph Nucleo["Núcleo"]
        PF["portfolio<br/>IntradayRiskEngine: on_bar · refit · compute<br/>correlação EWMA/DCC · cauda EVT · overnight"]
        BT["backtest<br/>testes estatísticos"]
        RV["revaluation<br/>opções: Black–Scholes e Monte Carlo"]
        SY["synthetic<br/>gerador do mercado fictício"]
    end

    subgraph Modelo["Modelo"]
        SE["seasonality<br/>perfil U"]
        GA["garch<br/>GARCH(1,1)-t: estimação e filtro"]
        RK["risk<br/>VaR / ES: Normal, t, EVT"]
    end

    BASE["Base (usada por todas as camadas acima)<br/>market_data · math · optimize · random<br/>CSV e calendário B3 · funções especiais · BFGS · números aleatórios"]

    IR --> REP
    MC --> REP & SY
    GD --> SY
    FS --> GA
    BM --> PF
    REP --> PF & BT & RV & SY
    PF --> SE & GA & RK
    BT & RV --> RK
    SY --> RV
    RK --> GA
    GA --> BASE
```

## 5. Realized-CAViaR

Terceira técnica (C++20). Em vez de estimar a volatilidade e supor uma distribuição, estima
diretamente o quantil de 1% do retorno (VaR 99%) por uma recursão que usa o retorno agregado R e a
variância realizada RV da janela anterior. Os coeficientes são estimados uma vez, fora do caminho
crítico, minimizando a perda pinball; a atualização online só aplica a recursão. Há dois modos de
entrada: preços a cada 5 s (simulação própria ou CSV) ou uma ação dos dados compartilhados de
`data/`, em barras de 15 min. A caixa da direita mostra o código que vem do `garch_t`.

```mermaid
flowchart LR
    subgraph Entrada["Entradas (um modo por execução)"]
        direction TB
        HF["Alta frequência<br/>preços a cada 5 s<br/>simulação interna ou --prices CSV"]
        SH["Dados compartilhados<br/>data/synthetic · data/b3_real<br/>uma ação (--ticker), barras de 15 min"]
        HF ~~~ SH
    end

    subgraph Lib["Biblioteca rcaviar"]
        direction TB
        DA["data<br/>leitura e simulação de preços<br/>janelas: R, RV e alvo futuro"]
        SD["shared_data<br/>janelas de 1 h (4 barras) sobre data/<br/>sem atravessar o pernoite"]
        FIT["model · fit (offline)<br/>minimiza a perda pinball<br/>Nelder-Mead com vários pontos iniciais"]
        FL["model · Filter (online)<br/>q(t+1) = β0 + β1·q(t) + β2·|R(t)| + β3·√RV(t)<br/>4 coeficientes + 1 quantil"]
        DA --> FIT
        SD --> FIT
        FIT --> FL
    end

    subgraph Reuso["Reutilizado do garch_t (compilado das fontes)"]
        direction TB
        GMD["market_data<br/>CSV · grade de barras · retornos"]
        GBT["backtest<br/>Kupiec · Christoffersen"]
        GMA["math · optimize · risk"]
        GMD ~~~ GBT ~~~ GMA
    end

    subgraph Saida["results/"]
        direction TB
        VAR["VaR da posição comprada<br/>−E · expm1(q)"]
        OUT["forecasts.csv · report.txt<br/>exceções · Kupiec · Christoffersen<br/>pinball · latência p50/p95/p99"]
        VAR --> OUT
    end

    Entrada --> Lib --> Saida
    Lib -. usa .-> Reuso
```

---

Detalhes de cada componente (fórmulas, regras e parâmetros) estão em
[`ewma/README.md`](../ewma/README.md), [`garch_t/README.md`](../garch_t/README.md) e
[`realized_caviar/README.md`](../realized_caviar/README.md).