
"""Motor físico 2D para curling.

O ambiente conhece o atrito do gelo, mas não o expõe ao agente.
A rotação produz um efeito lateral aproximado (curl).
A varredura reduz o atrito médio em uma região à frente da pedra.
"""

import numpy as np


class CurlingPhysics:
    """Simulação física das pedras de curling."""

    def __init__(
        self,
        ice,
        dt=0.01,
        sweep_length=1.0,
        sweep_effect=0.45,
        stone_radius=0.145,
        restitution=0.82,
        curl_strength=0.18,
        spin_decay=0.025,
        stop_speed=0.015,
    ):
        self.ice = ice
        self.dt = float(dt)
        self.sweep_length = float(sweep_length)
        self.sweep_effect = float(sweep_effect)
        self.stone_radius = float(stone_radius)
        self.restitution = float(restitution)
        self.curl_strength = float(curl_strength)
        self.spin_decay = float(spin_decay)
        self.stop_speed = float(stop_speed)

        if self.dt <= 0:
            raise ValueError("dt deve ser positivo.")

        if self.sweep_length <= 0:
            raise ValueError("sweep_length deve ser positivo.")

        if not 0 <= self.restitution <= 1:
            raise ValueError(
                "restitution deve estar entre 0 e 1."
            )

        if self.stone_radius <= 0:
            raise ValueError("stone_radius deve ser positivo.")

        if self.stop_speed < 0:
            raise ValueError("stop_speed não pode ser negativo.")

    @staticmethod
    def _speed(stone):
        """Retorna o módulo da velocidade da pedra."""
        return float(np.linalg.norm(stone["vel"]))

    def _swept(self, stone, x=None, y=None):
        """Verifica se a varredura está ativa durante o movimento."""
        if stone.get("sweep", 0.0) <= 0:
            return False

        return self._speed(stone) >= self.stop_speed

    def _friction_ahead(self, stone):
        """Estima o atrito na região de gelo à frente da pedra."""

        pos = stone["pos"]
        vel = stone["vel"]
        speed = self._speed(stone)

        if speed < 1e-8:
            return float(
                self.ice.coefficient_at(
                    float(pos[0]),
                    float(pos[1]),
                )
            )

        direction = vel / speed
        normal = np.array(
            [-direction[1], direction[0]],
            dtype=float,
        )

        longitudinal = np.linspace(
            0.10,
            self.sweep_length,
            5,
        )

        lateral = (-0.15, 0.0, 0.15)
        coefficients = []

        for distance in longitudinal:
            for offset in lateral:
                point = (
                    pos
                    + direction * distance
                    + normal * offset
                )

                point_x = float(
                    np.clip(
                        point[0],
                        0.0,
                        self.ice.length,
                    )
                )

                point_y = float(
                    np.clip(
                        point[1],
                        0.0,
                        self.ice.width,
                    )
                )

                mu = self.ice.coefficient_at(
                    point_x,
                    point_y,
                )

                coefficients.append(float(mu))

        return float(np.mean(coefficients))



    def _effective_friction(self, stone, is_controlled):
        """Calcula o atrito efetivo utilizado na simulação."""

        if is_controlled and self._swept(stone):
            mu = self._friction_ahead(stone)

            intensity = float(
                np.clip(
                    stone.get("sweep", 0.0),
                    0.0,
                    1.0,
                )
            )

            return mu * (
                1.0 - self.sweep_effect * intensity
            )

        x, y = stone["pos"]

        return float(
            self.ice.coefficient_at(
                float(x),
                float(y),
            )
        )

    def _remove_if_out_of_bounds(self, stone):
        """Retira de jogo uma pedra que ultrapassou os limites."""

        if not stone.get("in_play", True):
            stone["active"] = False
            stone["vel"][:] = 0.0
            return True

        x, y = stone["pos"]
        radius = self.stone_radius

        outside = (
            x <= 0.0
            or x >= self.ice.length
            or y <= radius
            or y >= self.ice.width - radius
        )

        if outside:
            stone["in_play"] = False
            stone["active"] = False
            stone["vel"][:] = 0.0
            return True

        return False

    def _collisions(self, stones, active_stone_index=None):
        """Resolve colisões e registra pedras atingidas pelo lançamento."""

        min_dist = 2.0 * self.stone_radius

        for i, a in enumerate(stones):
            if not a.get("in_play", True):
                continue

            for j in range(i + 1, len(stones)):
                b = stones[j]

                if not b.get("in_play", True):
                    continue

                delta = b["pos"] - a["pos"]
                dist = float(np.linalg.norm(delta))

                # As pedras não estão em contato.
                if dist >= min_dist:
                    continue

                # Registra o impacto entre a pedra lançada e outra pedra.
                if active_stone_index is not None:
                    hit_index = None

                    if i == active_stone_index:
                        hit_index = j
                    elif j == active_stone_index:
                        hit_index = i

                    if hit_index is not None:
                        launched_stone = stones[
                            active_stone_index
                        ]

                        hit_list = launched_stone.setdefault(
                            "hit_stones",
                            [],
                        )

                        if hit_index not in hit_list:
                            hit_list.append(hit_index)

                # Determina a normal da colisão.
                if dist > 1e-9:
                    normal = delta / dist
                else:
                    relative_velocity = (
                        b["vel"] - a["vel"]
                    )

                    relative_speed = float(
                        np.linalg.norm(relative_velocity)
                    )

                    if relative_speed > 1e-9:
                        normal = (
                            -relative_velocity
                            / relative_speed
                        )
                    else:
                        normal = np.array(
                            [1.0, 0.0],
                            dtype=float,
                        )

                # Corrige a sobreposição entre os centros.
                overlap = min_dist - dist

                a["pos"] -= normal * (overlap * 0.5)
                b["pos"] += normal * (overlap * 0.5)

                # Calcula a velocidade relativa na direção da normal.
                relative_normal_speed = float(
                    np.dot(
                        b["vel"] - a["vel"],
                        normal,
                    )
                )

                # Aplica impulso somente se as pedras estiverem
                # se aproximando.
                if relative_normal_speed < 0:
                    impulse = (
                        -(1.0 + self.restitution)
                        * relative_normal_speed
                        / 2.0
                    )

                    a["vel"] -= impulse * normal
                    b["vel"] += impulse * normal

                    # Pedras anteriormente paradas podem voltar
                    # a se mover após uma colisão.
                    if self._speed(a) >= self.stop_speed:
                        a["active"] = True

                    if self._speed(b) >= self.stop_speed:
                        b["active"] = True

                # Não prendemos as pedras artificialmente dentro da
                # pista. Se o contato ou a correção da colisão as
                # levar para fora dos limites, elas saem de jogo.
                self._remove_if_out_of_bounds(a)
                self._remove_if_out_of_bounds(b)



    def _advance_stone(self, stone, is_controlled):
        """Integra um passo físico de uma pedra."""

        if not stone.get("in_play", True) or not stone["active"]:
            return

        velocity = stone["vel"]
        speed = self._speed(stone)

        if speed < self.stop_speed:
            stone["vel"][:] = 0.0
            stone["active"] = False
            return

        direction = velocity / speed

        # Atrito contrário ao movimento.
        mu = self._effective_friction(
            stone,
            is_controlled,
        )

        friction_acceleration = 9.81 * mu
        acceleration = -friction_acceleration * direction

        # Efeito lateral aproximado causado pela rotação.
        spin = float(
            np.clip(
                stone.get("spin", 0.0),
                -1.0,
                1.0,
            )
        )

        perpendicular = np.array(
            [-direction[1], direction[0]],
            dtype=float,
        )

        curl_acceleration = (
            self.curl_strength
            * spin
            * min(speed / 3.0, 1.0)
        )

        acceleration += (
            curl_acceleration * perpendicular
        )

        # Integração semi-implícita.
        new_velocity = velocity + acceleration * self.dt

        # Impede que o atrito faça a pedra inverter artificialmente
        # a direção longitudinal do movimento.
        if np.dot(new_velocity, direction) <= 0:
            new_velocity[:] = 0.0

        stone["vel"] = new_velocity
        stone["pos"] += stone["vel"] * self.dt

        # Decaimento da rotação.
        stone["spin"] *= np.exp(
            -self.spin_decay * self.dt
        )

        # Remove a pedra imediatamente se sair da pista.
        if self._remove_if_out_of_bounds(stone):
            return

        # Encerra o movimento quando a velocidade fica muito baixa.
        if self._speed(stone) < self.stop_speed:
            stone["vel"][:] = 0.0
            stone["active"] = False



    def advance(self, stones, steps, active_stone_index):
        """Avança a física e resolve colisões em cada subpasso."""

        for _ in range(int(steps)):
            # Atualiza todas as pedras em movimento.
            for idx, stone in enumerate(stones):
                is_controlled = (
                    idx == active_stone_index
                )

                self._advance_stone(
                    stone,
                    is_controlled,
                )

            # Resolve colisões também com pedras inicialmente paradas.
            self._collisions(
                stones,
                active_stone_index,
            )

            # Atualiza o estado de movimento após os impactos.
            for stone in stones:
                if not stone.get("in_play", True):
                    stone["active"] = False
                    stone["vel"][:] = 0.0
                    continue

                if self._speed(stone) < self.stop_speed:
                    stone["vel"][:] = 0.0
                    stone["active"] = False
