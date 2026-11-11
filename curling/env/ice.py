"""Mapa de atrito interno: usado pela física, nunca incluído na observação."""
import numpy as np

class Ice:
    def __init__(self, length=26.0, width=4.75, cell_size=0.5,
                 mu_min=0.010, mu_max=0.040, seed=None):
        self.length, self.width, self.cell_size = float(length), float(width), float(cell_size)
        self.mu_min, self.mu_max = float(mu_min), float(mu_max)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        nx = int(np.ceil(self.length / self.cell_size)) + 1
        ny = int(np.ceil(self.width / self.cell_size)) + 1
        self.friction = self.rng.uniform(self.mu_min, self.mu_max, size=(nx, ny)).astype(np.float32)

    def coefficient_at(self, x, y):
        ix = int(np.clip(round(x / self.cell_size), 0, self.friction.shape[0] - 1))
        iy = int(np.clip(round(y / self.cell_size), 0, self.friction.shape[1] - 1))
        return float(self.friction[ix, iy])
