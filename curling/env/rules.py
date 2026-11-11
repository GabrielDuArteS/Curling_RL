
"""Regras de pontuação de um end de curling."""

import numpy as np


# Raio externo da casa, em metros.
HOUSE_RADIUS = 1.83


def score_end(stones, target, house_radius=HOUSE_RADIUS):
    """
    Calcula a pontuacao de um end de curling.

    Regras:
    - Pedras fora de jogo nao contam.
    - Pedras fora da casa nao pontuam.
    - A equipe com a pedra valida mais proxima do centro vence o end.
    - A equipe vencedora recebe um ponto por cada pedra que esteja
      mais perto do centro do que a pedra valida adversaria mais praxima.
    - Se nenhuma equipe tiver pedras validas, o end termina sem pontos.

    Args:
        stones: lista de pedras do ambiente.
        target: posicao [x, y] do centro do alvo.
        house_radius: raio valido da casa, em metros.

    Returns:
        dict: {0: pontos_vermelha, 1: pontos_azul}
    """
    target = np.asarray(target, dtype=float)

    if house_radius <= 0:
        raise ValueError("house_radius deve ser positivo.")

    # Considera apenas pedras em jogo e dentro da casa.
    valid_stones = []

    for stone in stones:
        if not stone.get("in_play", True):
            continue

        position = np.asarray(stone["pos"], dtype=float)
        distance = float(np.linalg.norm(position - target))

        if distance <= house_radius:
            valid_stones.append(
                {
                    "team": int(stone["team"]),
                    "distance": distance,
                }
            )

    # Nenhuma pedra dentro da casa: end sem pontuacao.
    if not valid_stones:
        return {0: 0, 1: 0}

    # Ordena as pedras válidas pela distância ao centro.
    valid_stones.sort(key=lambda stone: stone["distance"])

    # A equipe da pedra mais próxima é a vencedora.
    winner = valid_stones[0]["team"]
    opponent = 1 - winner

    # Distancia da pedra adversaria mais proxima que esteja na casa.
    opponent_distances = [
        stone["distance"]
        for stone in valid_stones
        if stone["team"] == opponent
    ]

    if opponent_distances:
        opponent_best = min(opponent_distances)
    else:
        opponent_best = float("inf")

    # Conta somente as pedras vencedoras mais proximas do centro
    # do que a melhor pedra valida adversaria.
    
    points = sum(
        1
        for stone in valid_stones
        if stone["team"] == winner
        and stone["distance"] < opponent_best
    )

    return {
        winner: points,
        opponent: 0,
    }
