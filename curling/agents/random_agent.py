class RandomAgent:
    """Política aleatória compartilhada, útil apenas para validar o ambiente."""
    def __init__(self, action_space): self.action_space = action_space
    def act(self, observation): return self.action_space.sample()
