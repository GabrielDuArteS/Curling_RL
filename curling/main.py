from env.curling_env import CurlingEnv
from agents.random_agent import RandomAgent

def main():
    env = CurlingEnv(render_mode="human", stones_per_team=2, max_ends=8)
    policy = RandomAgent(env.action_space)  # a mesma política controla as duas equipes
    obs, info = env.reset(seed=7)
    terminated = truncated = False
    while not (terminated or truncated):
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
    print("Placar final:", info["score"], "| recompensa terminal:", reward)
    env.close()

if __name__ == "__main__": main()
