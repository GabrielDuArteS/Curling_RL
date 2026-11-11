
import numpy as np
from gymnasium.utils.env_checker import check_env

from env.curling_env import CurlingEnv


def test_gymnasium():
    print("\n[1/5] Verificando compatibilidade com Gymnasium...")

    env = CurlingEnv(
        render_mode=None,
        stones_per_team=2,
        max_ends=1,
        seed=42,
    )

    check_env(env, skip_render_check=True)

    print("OK: API do Gymnasium validada.")
    env.close()


def test_observation():
    print("\n[2/5] Testando observações...")

    env = CurlingEnv(
        stones_per_team=2,
        max_ends=1,
        seed=42,
    )

    observation, info = env.reset(seed=42)

    assert env.observation_space.contains(observation), (
        "A observação inicial não pertence ao observation_space."
    )

    assert np.all(np.isfinite(observation)), (
        "A observação contém NaN ou infinito."
    )

    assert info["phase"] == "launch"

    print("OK: formato e valores da observação válidos.")
    print("Tamanho da observação:", observation.shape)

    env.close()


def test_launch_and_motion():
    print("\n[3/5] Testando lançamento e movimento...")

    env = CurlingEnv(
        stones_per_team=2,
        max_ends=1,
        seed=42,
    )

    env.reset(seed=42)

    # Lança uma pedra com velocidade alta e sem rotação.
    launch_action = np.array(
        [0.0, 1.0, 0.0, 0.0],
        dtype=np.float32,
    )

    observation, reward, terminated, truncated, info = (
        env.step(launch_action)
    )

    assert len(env.stones) == 1, (
        "O lançamento não criou exatamente uma pedra."
    )

    assert env.phase == "sweep", (
        "O ambiente não entrou na fase de varredura."
    )

    initial_position = env.stones[0]["pos"].copy()

    # A fase seguinte controla a varredura e avança a física.
    sweep_action = np.array(
        [0.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )

    env.step(sweep_action)

    final_position = env.stones[0]["pos"]

    assert np.all(np.isfinite(final_position)), (
        "A posição da pedra contém valores inválidos."
    )

    assert not np.allclose(initial_position, final_position), (
        "A pedra não se movimentou após o avanço da física."
    )

    print("OK: lançamento e movimento funcionaram.")
    print("Posição inicial da verificação:", initial_position)
    print("Posição após a física:", final_position)

    env.close()


def test_sweep_input():
    print("\n[4/5] Testando controle de varredura...")

    env = CurlingEnv(
        stones_per_team=2,
        max_ends=1,
        seed=123,
    )

    env.reset(seed=123)

    env.step(
        np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    )

    sweep_action = np.array(
        [0.0, 0.0, 0.0, 0.8],
        dtype=np.float32,
    )

    env.step(sweep_action)

    assert np.isclose(env.last_sweep, 0.8), (
        "A intensidade da varredura não foi registrada."
    )

    assert np.isclose(env.stones[0]["sweep"], 0.8), (
        "A intensidade da varredura não foi atribuída à pedra."
    )

    print("OK: a ação de varredura foi registrada.")

    env.close()


def test_complete_episode():
    print("\n[5/5] Testando episódio completo...")

    env = CurlingEnv(
        stones_per_team=2,
        max_ends=1,
        seed=7,
    )

    observation, info = env.reset(seed=7)

    max_steps = 10000

    for step in range(max_steps):
        if env.phase == "launch":
            # Ângulo central, velocidade moderada e sem rotação.
            action = np.array(
                [0.0, 0.65, 0.0, 0.0],
                dtype=np.float32,
            )
        else:
            # Mantém a varredura desligada neste teste.
            action = np.array(
                [0.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            )

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        assert env.observation_space.contains(observation), (
            f"Observação inválida no passo {step}."
        )

        assert np.all(np.isfinite(observation)), (
            f"Valores inválidos na observação no passo {step}."
        )

        if terminated or truncated:
            break

    assert env.done, (
        "O episódio não terminou dentro do limite de passos."
    )

    assert env.throws[0] == env.stones_per_team
    assert env.throws[1] == env.stones_per_team

    print("OK: episódio concluído.")
    print("Passos executados:", step + 1)
    print("Placar final:", env.score)
    print("Recompensa final:", reward)

    env.close()


if __name__ == "__main__":
    test_gymnasium()
    test_observation()
    test_launch_and_motion()
    test_sweep_input()
    test_complete_episode()

    print("\nTodos os testes passaram!")
