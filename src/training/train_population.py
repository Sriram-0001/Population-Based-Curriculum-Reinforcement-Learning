from src.agents.ppo_agent import build_agent


def train_population():
    print("Training EASY agent...")
    easy = build_agent(1.0)
    easy.learn(200_000)

    print("Training MID agent...")
    mid = build_agent(0.1)
    mid.policy.load_state_dict(easy.policy.state_dict())
    mid.learn(200_000)

    print("Training HARD agent...")
    hard = build_agent(0.0)
    hard.policy.load_state_dict(mid.policy.state_dict())
    hard.learn(400_000)

    return hard
