# Nível C: VaR intradiário com dados ao vivo

O painel está hoje no **nível B**. O motor C++ compilado para WebAssembly roda no navegador sobre um histórico gravado, e a aba *Replay* reproduz o resultado barra a barra. O **nível C** liga o mesmo motor a um feed de mercado ao vivo. A cada barra que fecha, o motor calcula o VaR dos quatro modelos, e cada previsão é pontuada quando o retorno-alvo chega.

Este documento descreve como implementar o nível C sem refazer o que já existe.

## O que já está pronto para isso

| Peça | Onde | Papel no nível C |
|---|---|---|
| Interface de fonte de dados | `dashboard/js/sources.js` (`ReplaySource`) | `size`, `record(i)`, `subscribe(fn)`. Uma `LiveSource` com a mesma interface alimenta as mesmas telas. |
| Janelas incrementais | `rcaviar::StreamingWindows` (`realized_caviar/source/include/rcaviar/data.hpp`) | `push(Price)` devolve a janela (retorno agregado e variância realizada da última hora) quando uma barra fecha. |
| Filtro online do CAViaR | `rcaviar::Filter::update(R, RV)` | Atualiza o quantil em O(1). |
| Filtros online do EWMA e do GARCH-t | módulos `ewma` e `garch_t` (já usados em `replay.cpp`) | Atualizam a variância em O(1). |
| Calibração diária | `rcaviar::fit`, a estimação de λ e ν do EWMA e o ajuste do GARCH-t, chamados por `run_replay` | Rodam uma vez por dia sobre a janela móvel. |
| Build WebAssembly | `dashboard/tools/build_engine.*`, `realized_caviar/source/wasm/engine_api.cpp` | Basta acrescentar a nova API. |

## Passo 1: uma sessão incremental no motor

Hoje `run_replay` recebe o histórico inteiro e devolve tudo de uma vez. O nível C precisa de um objeto com estado, exposto via embind:

```cpp
// realized_caviar/source/wasm/engine_api.cpp (novo)
class StreamingSession {
public:
    StreamingSession(double confidence, double exposure, int window_days, bool diurnal);
    // Calibra os quatro modelos sobre o histórico recente (barras de 15 min, timestamps UTC).
    std::string calibrate(const std::string& bars_json);
    // Uma barra fechada: pontua a previsão anterior e devolve a próxima
    // {t, R, RV, q:[4], var:[4], exceptions:[4]}.
    std::string push(double timestamp, double price);
    void begin_session();              // abertura do pregão (zera as janelas)
    std::string recalibrate();         // fim do dia: refaz a calibração na janela móvel
};
EMSCRIPTEN_BINDINGS(streaming) {
    emscripten::class_<StreamingSession>("StreamingSession")
        .constructor<double, double, int, bool>()
        .function("calibrate", &StreamingSession::calibrate)
        .function("push", &StreamingSession::push)
        .function("beginSession", &StreamingSession::begin_session)
        .function("recalibrate", &StreamingSession::recalibrate);
}
```

A lógica é a mesma do laço de `run_replay` (`realized_caviar/source/src/replay.cpp`), separada em duas partes: a calibração (o que roda no fim de cada dia) e o passo de uma barra. Para garantir que nada muda, um teste deve reproduzir `run_replay` alimentando a sessão barra a barra e comparar as previsões (a mesma verificação feita entre o WASM e o nativo no nível B).

## Passo 2: uma `LiveSource` no navegador

```js
// dashboard/js/sources.js (novo)
export class LiveSource {
  constructor({ symbol, interval = '15m', session }) { /* ... */ }
  async start() {
    const history = await fetchHistory(this.symbol, this.interval); // REST
    await this.session.calibrate(history);
    this.ws = new WebSocket(streamUrl(this.symbol, this.interval));
    this.ws.onmessage = (ev) => {
      const bar = parseBar(ev.data);
      if (!bar.closed) return;                                       // só barras fechadas
      this.records.push(await this.session.push(bar.time, bar.close));
      this.emit(this.records.length - 1);
    };
  }
  get size() { return this.records.length; }
  record(i) { return this.records[i]; }
  // subscribe/emit: iguais aos da ReplaySource
}
```

O worker (`engine.worker.js`) passa a manter uma instância de `StreamingSession` e a responder às mensagens `calibrate` e `push`. A aba *Replay* ganha um seletor de fonte ("histórico" ou "ao vivo"); o gráfico e os contadores já funcionam com qualquer fonte que siga a interface.

## Passo 3: de onde vêm os dados

| Opção | Custo | Funciona direto do navegador? | Observação |
|---|---|---|---|
| Cripto, por exemplo o WebSocket público da Binance (`<símbolo>@kline_15m` ou `@kline_1m`) e o REST `/api/v3/klines` | gratuito | sim | Mercado 24/7: não há abertura nem fechamento. O perfil intradiário e a "sessão" precisam ser redefinidos (por exemplo, sessões de 24 h em UTC). |
| B3 em tempo real (provedores como Cedro, Nelogica ou o Market Data da B3) | pago | geralmente não | Exige um servidor intermediário para guardar a chave e repassar o feed. Mantém a comparação com o nível B. |
| B3 com atraso (por exemplo, 15 min) | gratuito ou barato | depende | Bom para demonstração: o motor não sabe que o dado está atrasado. |

Para o TCC, a recomendação é começar por cripto: valida o fluxo inteiro sem custo. Com isso pronto, trocar para a B3 muda só o `fetchHistory` e o `streamUrl`.

Com barras de 1 minuto (ou mais finas), a variância realizada do CAViaR passa a usar 15 ou mais retornos por barra-alvo, em vez de 4. Isso aproxima o experimento da hipótese do trabalho, a variância realizada de alta frequência.

## Passo 4: qualidade dos dados ao vivo

O feed ao vivo vai ter as falhas que a avaliação ATAM simulou: barras faltando, duplicadas, fora de ordem, preço congelado e bad ticks. Antes de chamar `push`, a `LiveSource` deve aplicar as mesmas regras da camada de qualidade do módulo `ewma`:

- descartar timestamps repetidos ou que voltam no tempo;
- marcar lacunas e chamar `beginSession()` depois de uma lacuna longa, como o `reset_on_gap` do `StreamingWindows`;
- rejeitar saltos de preço implausíveis.

Mostrar na tela um indicador do estado do feed (ok, lacuna ou reconectando) evita interpretar uma falha de dado como um evento de risco.

## Alternativa: motor nativo num servidor

Em vez de rodar o motor no navegador, um processo nativo pode assinar o feed e publicar as previsões por WebSocket ou SSE para todos os visitantes.

- **Vantagens:** uma única calibração serve todos os usuários, a chave do provedor fica no servidor, o histórico das previsões é gravado e a latência é a do motor nativo.
- **Desvantagens:** exige um servidor sempre ligado (o Vercel serve páginas estáticas e funções de curta duração, que não seguram um WebSocket aberto; seria preciso Fly.io, Railway ou uma VM), e o custo é contínuo.

A `LiveSource` ficaria igual. Só a origem dos registros muda: em vez de chamar o worker, ela lê o WebSocket do servidor.

## Riscos e cuidados

- **Tempo de calibração:** a calibração diária dos quatro modelos leva cerca de 1 s por ação no navegador. Isso é aceitável, mas deve rodar no worker para não travar a página. Isso já acontece no nível B.
- **Fuso e calendário:** todas as datas do motor são UTC. A definição de pregão (B3) ou de sessão (cripto) precisa ser explícita.
- **Reconexão:** na volta de uma queda, buscar as barras perdidas pelo REST antes de retomar o `push`.
- **Avaliação estatística:** os testes (Kupiec, Christoffersen, Diebold–Mariano) continuam válidos, mas com poucas previsões ao vivo têm pouco poder. Mostrar o intervalo de confiança da taxa de exceção, como o painel já faz.
