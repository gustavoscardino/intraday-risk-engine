# Roteiro da apresentação: avaliação de arquitetura (ATAM)

Roteiro para a reunião de acompanhamento. Cada slide indica a figura (em `results/figures/`, PNG
16:9) e os números da execução de 15/11/2026, que estão em `results/report.txt`. Se o
`run.bat` for rodado de novo, os números de tempo mudam um pouco; os demais são determinísticos.

---

## 1. O que o professor pediu e como aplicamos

- ATAM (Architecture Tradeoff Analysis Method), do SEI / Carnegie Mellon.
- Cada motor é uma arquitetura: **EWMA** e **GARCH-t** são motores de carteira; **Realized-CAViaR** é
  um motor de ativo único que roda os três modelos pareados.
- Os quatro passos pedidos: business drivers, régua de qualidade e riscos, cenários, avaliação.
- Tudo é medido por código (`atam/run.bat`) e pode ser refeito quando os modelos mudarem.

## 2. Business drivers

- Objetivo: VaR 99% intradiário e a pergunta de pesquisa (o CAViaR prevê melhor a um custo aceitável?).
- Partes interessadas: mesa de risco, nós como pesquisadores, orientador e banca.
- Restrições: C++, processo único, calibração offline, dados de 15 min, um semestre.
- Atributos em ordem: comparabilidade e corretude temporal, robustez a dados ruins,
  reprodutibilidade, desempenho, modificabilidade.

## 3. A régua (utility tree)

**Figura:** `atam_utility_tree.png`

- 11 cenários com medida de resposta e prioridade (importância, dificuldade).
- Os de prioridade (A,A) e (A,M) foram analisados primeiro.
- **Pedido ao professor:** validar a régua e os limites (slide 11).

## 4. Resultado geral

**Figura:** `atam_scorecard.png`

- Nenhuma arquitetura atende tudo:
  - EWMA atende 6 de 10.
  - GARCH-t atende 4 de 10 (o P1 fica no limite).
  - Realized-CAViaR atende 5 de 8 (P1 e P2 não se aplicam a ele).
- Cada motor é forte onde os outros são fracos: o EWMA em robustez, o CAViaR em modificabilidade e na
  grade de tempo alvo.

## 5. Corretude: não há vazamento de futuro, mas há duas armadilhas

**Figura:** `c1_no_lookahead.png`

- **Não-risco N1:** trocar todos os preços depois de um corte T não muda nada antes de T nos três
  motores (982 mil células comparadas). O mesmo teste muda 100% das previsões depois de T, então ele
  pegaria um vazamento se houvesse.
- **Risco R-1:** a janela de estimação padrão é "metade das sessões do arquivo". Num feed ao vivo
  isso faz as previsões dependerem de quanto dado vem depois. Correção: fixar `--window-days`.
- **Risco R-2 (EWMA):** a última barra do arquivo é tratada como fechamento do pregão. Isso troca a
  previsão overnight pendente por um registro degenerado.
- **Não-risco N2:** os três motores são determinísticos (726 mil células, 0 diferenças).

## 6. Comparabilidade: os modelos veem o mesmo mercado?

**Figura:** `cp1_comparability.png`

- **N3:** com a mesma política de qualidade de dados, o baseline EWMA(0,94) é idêntico nos dois
  motores de carteira em 2517 de 2517 previsões.
- **R-3:** no padrão, o EWMA liga a camada de qualidade de dados e os outros não. Por isso, a
  comparação estatística precisa fixar uma política só.
- **S-1:** nos dados reais da B3, uma barra com 6 de 15 ações sem negócio faz os motores divergirem
  em 139 previsões (até 12,6%). A causa é a política de dado ausente: um mantém a variância, o outro
  imputa por regressão.
- **N5:** o CAViaR lê o mercado com a mesma convenção de retornos (0 diferenças) e compara os três
  modelos sobre os mesmos alvos.
- **S-3:** a comparação pareada começa às 11:00, então a primeira hora do pregão fica de fora.

## 7. Robustez: proteger contra tick ruim sem esconder choque real

**Figuras:** `r1_degraded_feed.png` e `r2_genuine_shocks.png`

- **Tradeoff T-1 (EWMA):** a camada de qualidade derruba as previsões que desviam mais de 5% de
  8,34% para 0,04%. O custo é afastar o VaR em até 2,1% por uma barra quando o choque é real.
- **Tradeoff T-2 (GARCH-t):** o corte em 6σ só reduz o dano de 26,4% para 13,6% e ainda amortece
  choques reais (VaR × 0,962). A regra com confirmação do EWMA faz melhor nos dois lados.
- **Risco R-6:** o CAViaR não tem camada de qualidade, e 24,7% das previsões por ativo desviam no feed
  com falhas.
- **Risco R-8, o achado mais sério:** o crash das 10:15 seguido de 2 barras de parada **nunca entra**
  nos atributos do CAViaR, porque as janelas precisam de 4 barras seguidas. Os modelos de base no
  mesmo motor enxergam o crash.

## 8. Desempenho e custo computacional

**Figuras:** `p1_p4_model_stage.png` (principal) e `p1_p2_latency.png`

- **N6:** sob a mesma fronteira de medição, o CAViaR é o modelo mais barato (27 ns por atualização,
  40 bytes de estado), contra 45 ns / 240 B do EWMA-N e 40 ns / 280 B do GARCH-t. Isso responde o
  lado de custo da pergunta de pesquisa.
- **R-7:** os motores medem latência com fronteiras diferentes. Para comparar custo, só vale o
  benchmark pareado.
- **N4:** tudo está em microssegundos contra barras de 15 min; desempenho não é o gargalo.
- **S-2:** no GARCH-t, recalibrar a cauda quando entra uma ordem acontece dentro do ciclo da barra
  (até 1,3 ms contra 4 µs de mediana).

## 9. Mapa de tradeoffs (síntese)

**Figura:** `atam_tradeoffs.png`

- 12 decisões de arquitetura × 7 atributos, cada célula com o número medido.
- Este é o slide para discutir: cada linha é uma decisão que podemos manter, mudar ou mover.

## 10. O que fazemos nas próximas duas semanas

1. Fixar `--window-days` e uma política de qualidade de dados única nas comparações (R-1, R-3).
2. Levar a regra de qualidade de dados (suspeita confirmada na barra seguinte) para o leitor
   compartilhado `garch_t/market_data`, que os três motores compilam. Isso protege todos (R-6, T-1, T-2).
3. CAViaR: tratar janelas incompletas depois de uma parada, para que o choque entre nos atributos (R-8).
4. Monografia: declarar a grade de 15 minutos usada nos experimentos (R-5) e escrever o capítulo
   ATAM com os quadros gerados em `results/latex/`.
5. Usar o benchmark pareado para o custo computacional (R-7).

## 11. Perguntas para o orientador

- A régua (cenários, prioridades e limites) está adequada?
- Qual política de qualidade de dados usar na comparação estatística: ligada nos três ou desligada nos três?
- Podemos reportar a grade de 15 minutos, em vez de 5 s / 1 min / 5 min, como a configuração dos experimentos?
- Que limite faz sentido para memória/estado (P4)?
