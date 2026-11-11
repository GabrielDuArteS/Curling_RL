# Curling RL

Ambiente 2D de **curling para Aprendizado por Reforço**, desenvolvido em Python com [Gymnasium](https://gymnasium.farama.org/), [NumPy](https://numpy.org/) e [Pygame Community Edition](https://pyga.me/).

O projeto modela uma partida entre duas equipes (Vermelha e Azul). O agente escolhe parâmetros contínuos para lançar uma pedra e controlar a varredura enquanto ela se movimenta. O ambiente simula o atrito do gelo, o deslocamento das pedras e as colisões, acompanha a pontuação e apresenta uma visualização da partida.

> **Status:** protótipo acadêmico em desenvolvimento. A classe Gymnasium, o modelo físico simplificado, as regras simplificadas de pontuação, a interface Pygame e uma política aleatória para demonstração estão presentes. Um algoritmo de aprendizado (por exemplo, PPO ou SAC) e um pipeline de treinamento ainda não estão incluídos.

## Instalação

Requisitos:

- Python 3.10 ou superior;
- pip.

Clone o repositório e entre na pasta do projeto:

```bash
git clone https://github.com/SEU_USUARIO/curling-rl.git
cd curling-rl
```

Crie e ative um ambiente virtual.

**Windows (PowerShell):**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux ou macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependências:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Ambiente

O ambiente é implementado pela classe `CurlingEnv`, localizada em `env/curling_env.py`, que estende `gymnasium.Env`.

Seguindo a organização da documentação do Racecar Gym, os espaços são apresentados separadamente em **Observation Space** e **Action Space**. Nesta versão, ambos são espaços contínuos do tipo `gymnasium.spaces.Box`: a observação é um vetor de 57 valores e a ação é um vetor de quatro valores. Diferentemente do Racecar Gym, este projeto ainda não utiliza um `Dict` de agentes nem implementa uma API PettingZoo.

A configuração padrão utiliza:

| Parâmetro | Valor | Descrição |
|---|---:|---|
| Equipes | 2 | Vermelha (`0`) e Azul (`1`). |
| Ends | `8` | Quantidade de rodadas em uma partida. |
| Pedras por equipe/end | `4` | Quatro pedras por equipe em cada end. |
| Pedras por end | `8` | Oito lançamentos no total, somando as duas equipes. |
| Comprimento da pista | `26.0 m` | Comprimento usado na simulação física. |
| Largura da pista | `4.75 m` | Largura usada na simulação física. |
| Passo físico (`physics_dt`) | `0.01 s` | Intervalo interno de integração da física. |
| Intervalo de decisão (`decision_dt`) | `0.15 s` | Intervalo entre decisões do agente durante o movimento. |

Os parâmetros podem ser alterados ao instanciar `CurlingEnv`.

### Fluxo de uma jogada

1. A equipe da vez seleciona o ângulo, a velocidade inicial e a rotação da pedra.
2. O ambiente cria a pedra e inicia seu movimento.
3. Enquanto a pedra está em movimento, o agente fornece a intensidade da varredura.
4. A física atualiza posição e velocidade, considerando atrito, varredura, rotação e possíveis colisões.
5. Quando as pedras param, o ambiente encerra o lançamento e passa a vez à outra equipe.
6. Depois que todas as pedras do end são lançadas, a função de pontuação calcula o resultado e atualiza o placar.
7. A partida termina após o número configurado de ends.

A implementação atual alterna os lançamentos entre Vermelha e Azul. O atributo `hammer_team` registra a equipe com o hammer, mas ainda não determina qual equipe inicia o end seguinte nem garante que a equipe com hammer realize o último lançamento. Esse comportamento é uma simplificação pendente em relação ao modelo de partida descrito no projeto.

### Resumo dos espaços

| Espaço | Tipo Gymnasium | Formato | Tipo de dado |
|---|---|---:|---|
| Observation Space | `Box` | `(57,)` | `float32` |
| Action Space | `Box` | `(4,)` | `float32` |

## Observações (*Observation Space*)

O espaço de observações é definido por um único `gymnasium.spaces.Box`:

```python
observation_space = Box(
    low=-inf,
    high=inf,
    shape=(57,),
    dtype=float32,
)
```

Na configuração padrão, existem quatro pedras por equipe, totalizando oito pedras por end. Cada pedra ocupa sete posições no vetor. A última posição informa qual equipe deve jogar.

| Chave | Espaço / dimensão | Padrão | Descrição |
|---|---|---|---|
| Posição X | 8 valores (1 por pedra) | `x / length` | Coordenada longitudinal da pedra, normalizada pelo comprimento da pista. |
| Posição Y | 8 valores (1 por pedra) | `y / width` | Coordenada lateral da pedra, normalizada pela largura da pista. |
| Velocidade X | 8 valores (1 por pedra) | `vx / 5.0` | Componente longitudinal da velocidade, normalizada por `5.0`. |
| Velocidade Y | 8 valores (1 por pedra) | `vy / 5.0` | Componente lateral da velocidade, normalizada por `5.0`. |
| Equipe proprietária | 8 valores (1 por pedra) | `0` ou `1` | Identifica a equipe proprietária: `0` = Vermelha; `1` = Azul. |
| Estado ativa | 8 valores (1 por pedra) | `0` ou `1` | Indica se a pedra está em movimento. |
| Estado em jogo | 8 valores (1 por pedra) | `0` ou `1` | Indica se a pedra continua dentro dos limites da pista. |
| Equipe da vez | 1 valor | `0` ou `1` | Identifica a equipe responsável pela próxima ação. |

As sete primeiras características são repetidas para cada uma das oito pedras possíveis. Assim, a dimensão total é:

```python
observation_size = (2 * stones_per_team * 7) + 1
# Configuração padrão: (2 * 4 * 7) + 1 = 57
```

A ordem dos dados é a ordem em que as pedras foram lançadas. Pedras ainda não lançadas ocupam posições zeradas no vetor. O tipo de dado é `numpy.float32`.

**Observação parcial:** o vetor entregue ao agente não inclui diretamente rotação, varredura, atrito do gelo, placar acumulado, end atual, hammer ou quantidade de pedras restantes. Essas informações não devem ser confundidas com a observação. Parte delas é disponibilizada no dicionário `info`, enquanto o atrito permanece interno à simulação.

> O `Box` da observação utiliza limites `-inf` e `inf`. O Gymnasium aceita essa definição, mas `check_env()` pode emitir avisos recomendando limites finitos. Esses avisos não significam, por si só, que a validação falhou.

## Ações (*Action Space*)

O espaço de ações é um único `gymnasium.spaces.Box` contínuo com quatro componentes:

```python
action_space = Box(
    low=[-1.0, 0.0, -1.0, 0.0],
    high=[1.0, 1.0, 1.0, 1.0],
    shape=(4,),
    dtype=float32,
)
```

A ação é normalizada nos intervalos indicados na tabela. As linhas abaixo representam componentes do mesmo vetor `Box`, e não quatro espaços independentes.

| Chave | Espaço | Descrição |
|---|---|---|
| `angle` | `[-1, 1]` | Ângulo normalizado. O ambiente converte o valor para uma variação angular de aproximadamente `-12°` a `+12°`. |
| `speed` | `[0, 1]` | Velocidade inicial normalizada. É convertida para uma velocidade entre `1.2` e `5.0` unidades de velocidade do modelo. |
| `spin` | `[-1, 1]` | Valor de rotação aplicado à pedra e utilizado pelo modelo físico. |
| `sweep` | `[0, 1]` | Intensidade da varredura. `0` representa ausência de varredura e `1` representa intensidade máxima. |

Exemplo de ação válida:

```python
import numpy as np

action = np.array(
    [0.5, 0.8, -0.3, 0.7],
    dtype=np.float32,
)
```

A ação possui quatro componentes em todas as chamadas a `step()`, mas o ambiente utiliza os componentes de acordo com a fase atual:

- **Lançamento (`launch`):** utiliza ângulo, velocidade e rotação (`action[0:3]`).
- **Movimento/varredura (`sweep`):** utiliza a intensidade de varredura (`action[3]`) e avança a simulação física.

## Estado e informações (*State*)

Além da observação, `reset()` e `step()` retornam um dicionário `info` com informações auxiliares sobre a partida.

No retorno de `step()`, estão disponíveis:

| Chave | Tipo | Descrição |
|---|---|---|
| `phase` | `str` | Fase atual: `launch` ou `sweep`. |
| `current_team` | `int` | Equipe responsável pela próxima jogada (`0` = Vermelha; `1` = Azul). |
| `hammer_team` | `int` | Equipe que possui o hammer. |
| `score` | `dict` | Pontuação acumulada de cada equipe. |
| `end_scores` | `list[dict]` | Pontuação de cada equipe nos ends já concluídos. |
| `end` | `int` | Número do end atual, começando em `1`. |
| `sim_time` | `float` | Tempo acumulado da simulação física. |
| `friction_hidden` | `bool` | Indica que o atrito não é exposto diretamente ao agente. |
| `stones_remaining` | `dict` | Quantidade de pedras que cada equipe ainda pode lançar no end atual. |

O `info` de `reset()` contém a fase, equipe da vez, hammer, end inicial e contadores de pedras restantes.

## Recompensa e pontuação

### Regra de pontuação implementada

A função `score_end()` em `env/rules.py` utiliza uma versão simplificada da pontuação do curling:

1. Desconsidera pedras que saíram de jogo (`in_play=False`).
2. Considera válidas apenas as pedras cujo centro esteja dentro do raio da casa definido por `HOUSE_RADIUS = 1.83`.
3. Identifica a equipe cuja pedra válida está mais próxima do centro.
4. Concede à equipe vencedora um ponto por pedra que esteja mais próxima do centro do que a pedra válida adversária mais próxima.
5. Se nenhuma pedra válida estiver dentro da casa, o end termina sem pontos.

Essa regra é uma aproximação para o protótipo e não reproduz todas as particularidades do regulamento oficial do curling.

### Recompensa: especificação MDP e implementação atual

Na especificação conceitual do projeto, a recompensa é definida por end e é competitiva, com valores opostos para as duas equipes:

\[
r_{Vermelha} = p_{Vermelha} - p_{Azul}
\]

\[
r_{Azul} = p_{Azul} - p_{Vermelha}
\]

Na implementação Gymnasium atual, a API expõe uma única recompensa por chamada a `step()`:

- durante a partida, a recompensa é `0.0`;
- quando o último end termina, a recompensa terminal é `score[0] - score[1]`, isto é, a diferença final vista da perspectiva da equipe Vermelha.

Portanto, a recompensa implementada ainda **não corresponde integralmente** à recompensa por end e às recompensas individuais opostas descritas no MDP. Essa diferença deve ser considerada ao desenvolver o treinamento multiagente.

## APIs disponíveis

O ambiente implementa a API padrão de `gymnasium.Env` e é instanciado diretamente pela classe `CurlingEnv`.

| Método | Descrição |
|---|---|
| `__init__(...)` | Configura pista, parâmetros da partida, espaços de ação/observação e motor físico. |
| `reset(seed=None, options=None)` | Reinicia o episódio e retorna `(observation, info)`. |
| `step(action)` | Executa uma ação e retorna `(observation, reward, terminated, truncated, info)`. |
| `render()` | Renderiza a partida no modo selecionado (`human` ou `rgb_array`). |
| `close()` | Encerra os recursos do Pygame. |
| `set_training_info(...)` | Atualiza algoritmo, episódio, passos e estado exibidos no cabeçalho da interface. |

Parâmetros principais do construtor:

| Parâmetro | Padrão | Descrição |
|---|---:|---|
| `render_mode` | `None` | `None`, `"human"` ou `"rgb_array"`. |
| `stones_per_team` | `4` | Quantidade de pedras por equipe em cada end. |
| `max_ends` | `8` | Quantidade de ends da partida. |
| `decision_dt` | `0.15` | Intervalo entre decisões do agente, em segundos. |
| `physics_dt` | `0.01` | Intervalo de integração da física, em segundos. |
| `seed` | `None` | Semente para a aleatoriedade do ambiente. |
| `algorithm_name` | `"."` | Texto inicial exibido como nome do algoritmo na interface. |
| `episode_number` | `0` | Número do episódio exibido na interface. |

O ambiente ainda não está registrado com um identificador para uso via `gymnasium.make()`. Utilize a classe diretamente.

### Gymnasium API

Exemplo mínimo de uso sem abrir a janela gráfica:

```python
from env.curling_env import CurlingEnv

# Cria o ambiente.
env = CurlingEnv(
    render_mode=None,
    stones_per_team=4,
    max_ends=8,
    seed=42,
)

# Inicializa uma partida.
observation, info = env.reset(seed=42)

terminated = False
truncated = False

while not (terminated or truncated):
    # Seleciona uma ação válida aleatória.
    action = env.action_space.sample()

    # Executa a ação no ambiente.
    observation, reward, terminated, truncated, info = env.step(action)

print("Placar final:", info["score"])
env.close()
```

## Exemplos

O arquivo `main.py` executa uma demonstração visual com `render_mode="human"`. A política `RandomAgent` amostra ações aleatórias do espaço de ações e é compartilhada pelas duas equipes nos turnos alternados.

Execute a demonstração na raiz do projeto:

```bash
python main.py
```

A política aleatória serve para verificar o fluxo da partida e a interface. Ela não realiza aprendizado, não é treinada e não deve ser interpretada como um agente competitivo.

## Testes

Os testes automatizados ficam em `test_env.py`. Para executá-los:

```bash
python -m pytest -v
```

A suíte atual contém verificações de:

- compatibilidade básica com a API Gymnasium usando `check_env`;
- formato e validade do vetor de observação;
- lançamento e avanço do movimento de uma pedra;
- registro da intensidade de varredura;
- conclusão de um episódio.

A suíte deve ser executada depois da instalação das dependências. A demonstração visual deve ser verificada separadamente em uma sessão com suporte a janela gráfica.

## Estrutura do projeto

```text
curling-rl/
├── agents/
│   ├── __init__.py
│   └── random_agent.py
├── env/
│   ├── __init__.py
│   ├── curling_env.py
│   ├── ice.py
│   ├── physics.py
│   └── rules.py
├── training/             # reservado para treinamento
├── visualization/        # reservado para visualizações adicionais
├── main.py               # demonstração visual introdutória
├── test_env.py           # testes do ambiente
├── requirements.txt
├── .gitignore
└── README.md
```

## Observações e limitações

- A simulação física é aproximada e utiliza parâmetros simplificados; seus resultados não representam uma simulação oficial ou calibrada do esporte.
- A observação é parcial e não inclui o mapa de atrito do gelo.
- A interface acompanha a posse do hammer, mas a ordem dos lançamentos ainda não é determinada por essa posse.
- O ambiente é uma única classe `gymnasium.Env` com alternância de equipes. Ainda não implementa uma interface multiagente PettingZoo nem fornece recompensas individuais simultâneas para cada equipe.
- O exemplo `RandomAgent` é apenas uma política de demonstração. Não há treinamento PPO, SAC ou outro algoritmo incluído nesta versão.
- `training/` e `visualization/` estão reservados para etapas futuras.
- O projeto ainda não inclui um arquivo `LICENSE`. Escolha e adicione uma licença antes de conceder permissões de uso, modificação ou redistribuição.

## Referências

- [Gymnasium — documentação da classe `Env`](https://gymnasium.farama.org/api/env/)
- [Gymnasium — criação de ambientes personalizados](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)
- [Racecar Gym](https://github.com/axelbr/racecar_gym) — referência de organização da documentação do repositório.
