# Arquitetura do motor de risco intraday (módulo EWMA)

Três vistas complementares do sistema, cada uma respondendo a uma pergunta:

| Vista | Pergunta | Arquivo |
|---|---|---|
| 1. Fluxo de dados | o que acontece com um preço desde o arquivo até o VaR? | [`01_fluxo_de_dados.mmd`](01_fluxo_de_dados.mmd) |
| 2. Módulos | como o código está organizado? | [`02_modulos.mmd`](02_modulos.mmd) |
| 3. Ciclo de 15 minutos | o que o motor faz a cada barra? | [`03_ciclo_15min.mmd`](03_ciclo_15min.mmd) |

Os diagramas estão em [Mermaid](https://mermaid.js.org): são texto, ficam versionados no git e
aparecem desenhados no GitHub e no VS Code (extensão *Markdown Preview Mermaid Support*). Para
colocar no texto do TCC, cole o conteúdo de um `.mmd` em [mermaid.live](https://mermaid.live) e
exporte em PNG ou SVG.

---

## 1. Fluxo de dados

Do arquivo de preços até o relatório. Os preços passam primeiro pela camada de qualidade de dados,
que decide o que o motor pode usar; o motor produz as previsões dos três modelos; o backtest as
compara com o que aconteceu.

```mermaid
flowchart LR
    subgraph Entrada["data/"]
        P["prices_15m.csv<br/>(ordem de chegada)"]
        A["assets.csv · portfolio_initial.csv<br/>fills.csv · options.csv"]
    end

    subgraph DQ["Camada de qualidade de dados"]
        I["Ingestão<br/>DUPLICATE · OUT_OF_ORDER · DELAY"]
        G["Painel de barras<br/>GAP · STALE"]
        S["Regra SPIKE<br/>|z| > 6, confirmada na barra seguinte"]
        M["dq_score e modo<br/>NORMAL · DEGRADED · FREEZE"]
    end

    subgraph Motor["Motor EWMA"]
        R["Retornos<br/>1ª barra sem o gap · ÷√k"]
        U["Perfil diurno (U)"]
        E["Filtros EWMA<br/>ref λ = 0,94 · λ estimado"]
        F["VaR / ES<br/>EWMA-N (ref) · EWMA-N · EWMA-t"]
        O["Risco overnight"]
    end

    subgraph Saida["results/"]
        B["Backtest<br/>Kupiec · Christoffersen · Berkowitz · ES Z2"]
        C["report.txt e CSVs<br/>dq_events · alerts · dq_detection"]
    end

    P --> I --> G --> R --> S --> M --> E
    A --> F
    U --> E --> F --> B --> C
    E --> O --> B
    M -. alertas .-> C
```

## 2. Módulos

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

## 3. Ciclo de 15 minutos

O que acontece a cada barra fechada. A atualização leva cerca de 1 µs; a re-estimação do fim do
dia, cerca de 4 ms (15 ações).

```mermaid
flowchart TD
    A([Barra t fecha]) --> B["Decide os preços suspeitos da barra t−1<br/>reverteu → bad tick, preço excluído<br/>não reverteu → choque real, mantido"]
    B --> C["Pontua a previsão anterior<br/>perda da barra t−1 vs VaR previsto"]
    C --> D["Aplica as ordens executadas (fills)<br/>atualiza posições"]
    D --> E["Calcula os retornos<br/>aplica a regra SPIKE"]
    E --> F["Calcula dq_score e o modo"]
    F --> G{FREEZE?}
    G -- sim --> H["Não atualiza o EWMA<br/>mantém o último VaR/ES válido"]
    G -- não --> I["Atualiza os 3 filtros EWMA"]
    H --> J{Fim do dia?}
    I --> J
    J -- sim --> K["Re-estima perfil U, λ e ν na janela<br/>previsão overnight"]
    J -- não --> L
    K --> L["Prevê σ, VaR e ES da barra t+1<br/>EWMA-N (ref) · EWMA-N · EWMA-t"]
    L --> A
```

---

Detalhes de cada componente (fórmulas, regras e parâmetros) estão em
[`ewma/README.md`](../ewma/README.md).
