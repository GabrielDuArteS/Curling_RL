
"""Ambiente Gymnasium de curling com placar e contadores no Pygame.

Ações:
    [ângulo normalizado, velocidade normalizada, rotação, varredura]

Equipes:
    0 = Vermelha
    1 = Azul

O atrito do gelo não é revelado na observação do agente.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .ice import Ice
from .physics import CurlingPhysics
from .rules import score_end


class CurlingEnv(gym.Env):
    """Ambiente de curling para treinamento por reinforcement learning."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

    RED = (194, 55, 55)
    BLUE = (45, 85, 190)
    BACKGROUND = (20, 29, 42)
    PANEL = (36, 49, 67)
    TEXT = (235, 241, 248)
    MUTED = (167, 181, 198)
    ICE_COLOR = (218, 232, 241)

    def __init__(
        self,
        render_mode=None,
        stones_per_team=4,
        max_ends=8,
        decision_dt=0.15,
        physics_dt=0.01,
        seed=None,
        algorithm_name=".",
        episode_number=0,
    ):
        super().__init__()

        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(
                "render_mode deve ser None, 'human' ou 'rgb_array'"
            )

        self.render_mode = render_mode
        self.stones_per_team = int(stones_per_team)
        self.max_ends = int(max_ends)
        self.decision_dt = float(decision_dt)
        self.physics_dt = float(physics_dt)

        self.algorithm_name = str(algorithm_name)
        self.episode_number = int(episode_number)
        self.training_status = "Pronto"
        self.training_steps = 0

        if self.stones_per_team < 1 or self.max_ends < 1:
            raise ValueError(
                "stones_per_team e max_ends devem ser positivos"
            )

        if self.physics_dt <= 0 or self.decision_dt <= 0:
            raise ValueError(
                "Os intervalos de tempo devem ser positivos"
            )

        self.steps_per_decision = int(
            round(self.decision_dt / self.physics_dt)
        )

        if not np.isclose(
            self.steps_per_decision * self.physics_dt,
            self.decision_dt,
        ):
            raise ValueError(
                "decision_dt deve ser múltiplo de physics_dt"
            )

        # Dimensões físicas da pista, em metros.
        self.length = 26.0
        self.width = 4.75
        self.target = np.array(
            [24.0, self.width / 2],
            dtype=float,
        )

        self.action_space = spaces.Box(
            low=np.array([-1, 0, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        # Cada pedra ocupa sete valores na observação.
        self.max_stones = 2 * self.stones_per_team
        obs_size = self.max_stones * 7 + 1

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )

        self.ice = Ice(self.length, self.width, seed=seed)
        self.physics = CurlingPhysics(
            self.ice,
            dt=self.physics_dt,
        )

        self._pygame = None
        self.window = None
        self.clock = None
        self._throw_start_positions = {}

        # Dados do placar.
        self.end_scores = []
        self.hammer_team = 1
        self.last_end_result = {0: 0, 1: 0}

        # Janela.
        self.window_width = 1440
        self.window_height = 850

        self.reset(seed=seed)

    # ---------------------------------------------------------
    # Configuração da interface
    # ---------------------------------------------------------

    def set_training_info(
        self,
        algorithm=None,
        episode=None,
        steps=None,
        status=None,
    ):
        """Atualiza os dados exibidos no cabeçalho."""

        if algorithm is not None:
            self.algorithm_name = str(algorithm)

        if episode is not None:
            self.episode_number = int(episode)

        if steps is not None:
            self.training_steps = int(steps)

        if status is not None:
            self.training_status = str(status)

    # ---------------------------------------------------------
    # Contadores de pedras restantes
    # ---------------------------------------------------------

    def _stones_remaining(self, team):
        """Retorna quantas pedras a equipe ainda pode lançar no end."""

        return max(
            0,
            self.stones_per_team - self.throws[team],
        )

    # ---------------------------------------------------------
    # Reinício do episódio
    # ---------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.ice.rng = np.random.default_rng(seed)

        self.ice.reset()
        self.physics = CurlingPhysics(
            self.ice,
            dt=self.physics_dt,
        )

        self.stones = []
        self.team_to_throw = 0
        self.throws = {0: 0, 1: 0}
        self.score = {0: 0, 1: 0}

        self.end_index = 0
        self.phase = "launch"
        self.active_stone_index = None
        self.last_sweep = 0.0
        self.sim_time = 0.0
        self.done = False

        self.end_scores = []
        self.hammer_team = 1
        self.last_end_result = {0: 0, 1: 0}
        self._throw_start_positions = {}

        return self._observation(), {
            "phase": self.phase,
            "current_team": self.team_to_throw,
            "hammer_team": self.hammer_team,
            "end": self.end_index + 1,
            "stones_remaining": {
                0: self._stones_remaining(0),
                1: self._stones_remaining(1),
            },
        }

    # ---------------------------------------------------------
    # Observação do agente
    # ---------------------------------------------------------

    def _observation(self):
        stone_data = np.zeros(
            (self.max_stones, 7),
            dtype=np.float32,
        )

        for i, stone in enumerate(self.stones[:self.max_stones]):
            stone_data[i] = [
                stone["pos"][0] / self.length,
                stone["pos"][1] / self.width,
                stone["vel"][0] / 5.0,
                stone["vel"][1] / 5.0,
                float(stone["team"]),
                float(stone.get("active", False)),
                float(stone.get("in_play", True)),
            ]

        current_team = np.array(
            [float(self.team_to_throw)],
            dtype=np.float32,
        )

        return np.concatenate(
            [stone_data.flatten(), current_team]
        ).astype(np.float32)

    # ---------------------------------------------------------
    # Física e lançamentos
    # ---------------------------------------------------------

    def _restore_hit_stones(self, stone_index):
        if not 0 <= stone_index < len(self.stones):
            return

        launched_stone = self.stones[stone_index]

        for hit_index in launched_stone.get("hit_stones", []):
            if not 0 <= hit_index < len(self.stones):
                continue

            if hit_index not in self._throw_start_positions:
                continue

            hit_stone = self.stones[hit_index]
            hit_stone["pos"] = (
                self._throw_start_positions[hit_index].copy()
            )
            hit_stone["vel"][:] = 0.0
            hit_stone["active"] = False
            hit_stone["in_play"] = True

    def _launch(self, action):
        self._throw_start_positions = {
            i: stone["pos"].copy()
            for i, stone in enumerate(self.stones)
        }

        angle = float(action[0]) * np.deg2rad(12)
        speed = 1.2 + float(action[1]) * 3.8

        y0 = self.width / 2 + float(
            self.np_random.uniform(-0.03, 0.03)
        )

        y0 = float(
            np.clip(
                y0,
                self.physics.stone_radius,
                self.width - self.physics.stone_radius,
            )
        )

        velocity = np.array(
            [
                speed * np.cos(angle),
                speed * np.sin(angle),
            ],
            dtype=float,
        )

        self.stones.append(
            {
                "pos": np.array([0.8, y0], dtype=float),
                "vel": velocity,
                "spin": float(np.clip(action[2], -1.0, 1.0)),
                "team": self.team_to_throw,
                "active": True,
                "in_play": True,
                "sweep": 0.0,
                "hit_stones": [],
            }
        )

        self.active_stone_index = len(self.stones) - 1
        self.throws[self.team_to_throw] += 1
        self.phase = "sweep"

    def _finish_throw(self):
        """Finaliza um lançamento e contabiliza o end quando completo."""

        self.phase = "launch"
        self.active_stone_index = None
        self.last_sweep = 0.0

        self.team_to_throw = 1 - self.team_to_throw

        all_thrown = all(
            self.throws[team] >= self.stones_per_team
            for team in (0, 1)
        )

        if not all_thrown:
            return

        result = score_end(self.stones, self.target)

        result = {
            0: int(result.get(0, 0)),
            1: int(result.get(1, 0)),
        }

        self.last_end_result = result
        self.end_scores.append(result.copy())

        for team in (0, 1):
            self.score[team] += result[team]

        # A equipe que pontua perde o hammer no end seguinte.
        # Em um end sem pontos, o hammer permanece com a mesma equipe.
        if result[0] > result[1]:
            self.hammer_team = 1
        elif result[1] > result[0]:
            self.hammer_team = 0

        self.end_index += 1

        if self.end_index >= self.max_ends:
            self.done = True
        else:
            self.stones = []
            self.throws = {0: 0, 1: 0}
            self._throw_start_positions = {}

    # ---------------------------------------------------------
    # Transição Gymnasium
    # ---------------------------------------------------------

    def step(self, action):
        if self.done:
            raise RuntimeError(
                "Episódio encerrado; chame reset() antes de step()."
            )

        action = np.asarray(action, dtype=np.float32)

        if not self.action_space.contains(action):
            action = np.clip(
                action,
                self.action_space.low,
                self.action_space.high,
            ).astype(np.float32)

        reward = 0.0

        if self.phase == "launch":
            self._launch(action)
        else:
            launched_index = self.active_stone_index

            self.last_sweep = float(action[3])
            self.stones[launched_index]["sweep"] = self.last_sweep

            self.physics.advance(
                self.stones,
                self.steps_per_decision,
                launched_index,
            )

            self.sim_time += self.decision_dt

            if not self.stones[launched_index].get("in_play", True):
                self._restore_hit_stones(launched_index)

            any_moving = any(
                stone.get("active", False)
                and stone.get("in_play", True)
                for stone in self.stones
            )

            if not any_moving:
                self._finish_throw()

        terminated = self.done
        truncated = False

        if terminated:
            reward = float(self.score[0] - self.score[1])

        info = {
            "phase": self.phase,
            "current_team": self.team_to_throw,
            "hammer_team": self.hammer_team,
            "score": dict(self.score),
            "end_scores": [s.copy() for s in self.end_scores],
            "end": min(self.end_index + 1, self.max_ends),
            "sim_time": self.sim_time,
            "friction_hidden": True,
            "stones_remaining": {
                0: self._stones_remaining(0),
                1: self._stones_remaining(1),
            },
        }

        if self.render_mode == "human":
            self.render()

        return (
            self._observation(),
            reward,
            terminated,
            truncated,
            info,
        )

    # ---------------------------------------------------------
    # Funções auxiliares de desenho
    # ---------------------------------------------------------

    def _draw_text(
        self,
        surface,
        text,
        position,
        font,
        color=None,
        align="left",
    ):
        color = color or self.TEXT
        image = font.render(str(text), True, color)
        rect = image.get_rect()

        if align == "center":
            rect.center = position
        elif align == "right":
            rect.midright = position
        else:
            rect.topleft = position

        surface.blit(image, rect)

    def _draw_panel(self, pg, surface, rect, color=None):
        pg.draw.rect(
            surface,
            color or self.PANEL,
            rect,
            border_radius=10,
        )

    def _draw_stone_counter(
        self,
        pg,
        surface,
        x,
        y,
        width,
        team,
        font,
    ):
        """
        Desenha o contador de pedras restantes alinhado à área
        das colunas dos ends.
        """

        color = self.RED if team == 0 else self.BLUE
        team_name = "V" if team == 0 else "A"
        remaining = self._stones_remaining(team)

        # Célula no mesmo estilo visual dos cabeçalhos dos ends.
        cell = pg.Rect(
            round(x),
            round(y),
            max(1, round(width) - 3),
            17,
        )

        pg.draw.rect(
            surface,
            (48, 64, 84),
            cell,
            border_radius=4,
        )

        # Pequeno marcador colorido da equipe.
        marker_x = cell.x + 10
        marker_y = cell.centery
        pg.draw.circle(
            surface,
            color,
            (marker_x, marker_y),
            4,
        )

        # Quantidade restante e total por equipe.
        counter_text = (
            f"{team_name} {remaining}/{self.stones_per_team}"
        )

        self._draw_text(
            surface,
            counter_text,
            (cell.centerx + 4, cell.centery),
            font,
            self.TEXT,
            align="center",
        )

    def _draw_scoreboard(self, pg, surface, font, small_font):
        """Desenha o placar, os ends e os contadores de pedras."""

        margin = 24
        header_y = 104

        # Painel mais alto para comportar as duas linhas de contadores.
        header_h = 220

        panel_rect = pg.Rect(
            margin,
            header_y,
            self.window_width - margin * 2,
            header_h,
        )
        self._draw_panel(pg, surface, panel_rect)

        # Bloco END ATUAL.
        current_rect = pg.Rect(
            margin + 12,
            header_y + 12,
            145,
            header_h - 24,
        )

        pg.draw.rect(
            surface,
            (48, 64, 84),
            current_rect,
            border_radius=8,
        )

        current_end = min(self.end_index + 1, self.max_ends)

        if self.done:
            current_end = self.max_ends

        self._draw_text(
            surface,
            "END ATUAL",
            current_rect.center,
            small_font,
            self.MUTED,
            align="center",
        )

        self._draw_text(
            surface,
            f"{current_end}/{self.max_ends}",
            (current_rect.centerx, current_rect.centery + 17),
            font,
            self.TEXT,
            align="center",
        )

        hammer_name = (
            "Vermelha" if self.hammer_team == 0 else "Azul"
        )

        self._draw_text(
            surface,
            f"Hammer: {hammer_name}",
            (current_rect.centerx, current_rect.bottom - 24),
            small_font,
            self.TEXT,
            align="center",
        )

        # Tabela: equipe, hammer, ends e total.
        table_x = current_rect.right + 14
        table_y = header_y + 12
        table_w = panel_rect.right - 12 - table_x

        name_w = max(104, int(table_w * 0.115))
        hammer_w = max(78, int(table_w * 0.085))
        total_w = max(78, int(table_w * 0.085))

        remaining_width = table_w - name_w - hammer_w - total_w
        end_w = remaining_width / self.max_ends

        # Título.
        self._draw_text(
            surface,
            "PLACAR POR END",
            (table_x, table_y + 2),
            small_font,
            self.MUTED,
        )

        # Duas linhas de contadores acima dos números dos ends.
        counter_y_red = table_y + 25
        counter_y_blue = table_y + 44

        ends_start_x = table_x + name_w + hammer_w

        for end_number in range(self.max_ends):
            x = ends_start_x + end_number * end_w

            self._draw_stone_counter(
                pg,
                surface,
                x,
                counter_y_red,
                end_w,
                team=0,
                font=small_font,
            )

            self._draw_stone_counter(
                pg,
                surface,
                x,
                counter_y_blue,
                end_w,
                team=1,
                font=small_font,
            )

        # Cabeçalho numérico dos ends, abaixo dos contadores.
        label_y = table_y + 64
        row_h = 45
        row_gap = 7

        x = ends_start_x

        for end_number in range(1, self.max_ends + 1):
            cell = pg.Rect(
                round(x),
                label_y,
                max(1, round(end_w) - 3),
                25,
            )

            pg.draw.rect(
                surface,
                (48, 64, 84),
                cell,
                border_radius=4,
            )

            self._draw_text(
                surface,
                str(end_number),
                cell.center,
                small_font,
                self.TEXT,
                align="center",
            )

            x += end_w

        # Cabeçalho TOTAL.
        total_x = ends_start_x + self.max_ends * end_w

        total_header = pg.Rect(
            round(total_x),
            label_y,
            total_w - 2,
            25,
        )

        pg.draw.rect(
            surface,
            (80, 94, 112),
            total_header,
            border_radius=4,
        )

        self._draw_text(
            surface,
            "TOTAL",
            total_header.center,
            small_font,
            self.TEXT,
            align="center",
        )

        # Linhas das equipes.
        for team in (0, 1):
            row_y = label_y + 30 + team * (row_h + row_gap)

            team_color = self.RED if team == 0 else self.BLUE
            team_name = "Vermelha" if team == 0 else "Azul"

            # Nome.
            name_rect = pg.Rect(
                table_x,
                row_y,
                name_w - 4,
                row_h,
            )

            pg.draw.rect(
                surface,
                team_color,
                name_rect,
                border_radius=5,
            )

            self._draw_text(
                surface,
                team_name,
                name_rect.center,
                small_font,
                (255, 255, 255),
                align="center",
            )

            # Hammer.
            hammer_rect = pg.Rect(
                table_x + name_w,
                row_y,
                hammer_w - 4,
                row_h,
            )

            has_hammer = self.hammer_team == team

            hammer_color = (
                (232, 237, 245)
                if has_hammer
                else (48, 64, 84)
            )

            pg.draw.rect(
                surface,
                hammer_color,
                hammer_rect,
                border_radius=5,
            )

            hammer_text = "HAMMER" if has_hammer else "—"
            hammer_text_color = (
                (26, 39, 55)
                if has_hammer
                else self.MUTED
            )

            self._draw_text(
                surface,
                hammer_text,
                hammer_rect.center,
                small_font,
                hammer_text_color,
                align="center",
            )

            # Pontuação por end.
            x = ends_start_x

            for end_number in range(self.max_ends):
                cell = pg.Rect(
                    round(x),
                    row_y,
                    max(1, round(end_w) - 3),
                    row_h,
                )

                if end_number < len(self.end_scores):
                    points = self.end_scores[end_number][team]

                    cell_color = (
                        (255, 229, 229)
                        if team == 0
                        else (224, 234, 255)
                    )

                    point_text = str(points)
                    text_color = (29, 40, 54)
                else:
                    cell_color = (245, 247, 250)
                    point_text = "—"
                    text_color = (140, 151, 164)

                pg.draw.rect(
                    surface,
                    cell_color,
                    cell,
                    border_radius=5,
                )

                self._draw_text(
                    surface,
                    point_text,
                    cell.center,
                    font,
                    text_color,
                    align="center",
                )

                x += end_w

            # Total acumulado.
            total_rect = pg.Rect(
                round(total_x),
                row_y,
                total_w - 2,
                row_h,
            )

            pg.draw.rect(
                surface,
                (215, 222, 232),
                total_rect,
                border_radius=5,
            )

            self._draw_text(
                surface,
                str(self.score[team]),
                total_rect.center,
                font,
                (23, 35, 50),
                align="center",
            )

    def _draw_ice(self, pg, surface, font, small_font):
        """Desenha a pista abaixo do placar."""

        margin_x = 36
        top_y = 350

        available_h = self.window_height - top_y - 30
        available_w = self.window_width - margin_x * 2

        aspect = self.length / self.width

        lane_w = available_w
        lane_h = int(lane_w / aspect)

        if lane_h > available_h:
            lane_h = available_h
            lane_w = int(lane_h * aspect)

        lane_x = (self.window_width - lane_w) // 2
        lane_y = top_y + max(0, (available_h - lane_h) // 2)

        lane = pg.Rect(lane_x, lane_y, lane_w, lane_h)

        pg.draw.rect(
            surface,
            self.ICE_COLOR,
            lane,
            border_radius=5,
        )

        def px(x):
            return lane.x + int((x / self.length) * lane.width)

        def py(y):
            return lane.y + int((y / self.width) * lane.height)

        # Linhas de referência da pista.
        line_color = (247, 250, 252)
        red_line = (204, 70, 70)

        line_positions = [
            (2.0, line_color),
            (4.0, red_line),
            (6.0, line_color),
            (20.0, line_color),
            (22.0, red_line),
            (24.0, line_color),
        ]

        for x_pos, color in line_positions:
            pg.draw.line(
                surface,
                color,
                (px(x_pos), lane.top),
                (px(x_pos), lane.bottom),
                max(2, lane.width // 450),
            )

        # Casas nas extremidades.
        house_radius = max(12, int(lane.height * 0.31))

        for center_x in (4.0, 24.0):
            cx = px(center_x)
            cy = py(self.width / 2)

            for radius, color in [
                (house_radius, (190, 65, 65)),
                (int(house_radius * 0.72), (248, 250, 252)),
                (int(house_radius * 0.44), (52, 91, 177)),
                (int(house_radius * 0.17), (248, 250, 252)),
            ]:
                pg.draw.circle(
                    surface,
                    color,
                    (cx, cy),
                    max(2, radius),
                )

        # Pedras.
        stone_radius = max(5, int(lane.height * 0.045))

        for stone in self.stones:
            if not stone.get("in_play", True):
                continue

            x = px(float(stone["pos"][0]))
            y = py(float(stone["pos"][1]))

            color = self.RED if stone["team"] == 0 else self.BLUE

            pg.draw.circle(
                surface,
                color,
                (x, y),
                stone_radius,
            )

            pg.draw.circle(
                surface,
                (31, 39, 50),
                (x, y),
                stone_radius,
                2,
            )

            pg.draw.circle(
                surface,
                (255, 255, 255),
                (x, y),
                max(2, stone_radius // 3),
            )

        # Legenda.
        legend_y = min(self.window_height - 18, lane.bottom + 18)

        pg.draw.circle(
            surface,
            self.RED,
            (lane.left + 10, legend_y),
            5,
        )

        self._draw_text(
            surface,
            "Equipe vermelha",
            (lane.left + 22, legend_y - 9),
            small_font,
            self.MUTED,
        )

        pg.draw.circle(
            surface,
            self.BLUE,
            (lane.left + 155, legend_y),
            5,
        )

        self._draw_text(
            surface,
            "Equipe azul",
            (lane.left + 167, legend_y - 9),
            small_font,
            self.MUTED,
        )

    # ---------------------------------------------------------
    # Renderização principal
    # ---------------------------------------------------------

    def render(self):
        if self.render_mode is None:
            return None

        if self._pygame is None:
            import pygame

            self._pygame = pygame
            pygame.init()

            pygame.display.set_caption("Curling RL — Simulação")

            self.window = pygame.display.set_mode(
                (self.window_width, self.window_height),
                pygame.RESIZABLE,
            )

            self.clock = pygame.time.Clock()

        pg = self._pygame

        if self.render_mode == "human" and self.window is not None:
            self.window_width, self.window_height = self.window.get_size()

        surface = pg.Surface(
            (self.window_width, self.window_height)
        )

        surface.fill(self.BACKGROUND)

        title_font = pg.font.Font(None, 34)
        font = pg.font.Font(None, 25)
        small_font = pg.font.Font(None, 19)

        # Cabeçalho.
        self._draw_text(
            surface,
            "CURLING RL",
            (24, 20),
            title_font,
        )

        status_color = (
            (88, 195, 130)
            if self.training_status.lower()
            in ("treinando", "em execução")
            else (230, 190, 85)
        )

        status_text = (
            f"Algoritmo: {self.algorithm_name}    "
            f"Episódio: {self.episode_number}    "
            f"Passos: {self.training_steps}    "
            f"Estado: {self.training_status}"
        )

        self._draw_text(
            surface,
            status_text,
            (24, 61),
            small_font,
            self.MUTED,
        )

        team_name = (
            "Vermelha" if self.team_to_throw == 0 else "Azul"
        )

        team_color = (
            self.RED if self.team_to_throw == 0 else self.BLUE
        )

        self._draw_text(
            surface,
            f"Equipe da vez: {team_name}",
            (self.window_width - 24, 23),
            font,
            team_color,
            align="right",
        )

        # Placar e contadores.
        self._draw_scoreboard(
            pg,
            surface,
            font,
            small_font,
        )

        # Pista.
        self._draw_ice(
            pg,
            surface,
            font,
            small_font,
        )

        if self.render_mode == "human":
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.close()
                    return None

                if event.type == pg.VIDEORESIZE:
                    self.window_width = max(1000, event.w)
                    self.window_height = max(650, event.h)

                    self.window = pg.display.set_mode(
                        (self.window_width, self.window_height),
                        pg.RESIZABLE,
                    )

            self.window.blit(surface, (0, 0))
            pg.display.flip()
            self.clock.tick(self.metadata["render_fps"])

            return None

        return np.transpose(
            pg.surfarray.array3d(surface),
            (1, 0, 2),
        )

    def close(self):
        if self._pygame is not None:
            self._pygame.quit()
            self._pygame = None
            self.window = None
            self.clock = None
