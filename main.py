from src.training.train_population import train_population
from src.evaluation.evaluate import evaluate

agent = train_population()
reward, success = evaluate(agent)

print("\nFINAL RESULTS")
print(f"Average Reward: {reward:.2f}")
print(f"Success Rate : {success*100:.1f}%")
