from env.simple_env import SimpleEnv
from model.basal_ganglia import BasalGangliaModel
from agent.rl_agent import RLAgent
import matplotlib.pyplot as plt

def main():
    env = SimpleEnv()
    brain = BasalGangliaModel()
    agent = RLAgent(brain)

    all_rewards = []
    for episode in range(1000):
        state = env.reset()
        done = False
        total_reward = 0

        if episode < 2:
            print("fEpisode {episode}: Initial state = {state}")

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if episode < 2:
                print(f"State: {state}, Action: {action}, Reward: {reward}")
            
            if reward == 1:
                print(f"Congrats! Agent reached the goal in episode {episode}")
        
        all_rewards.append(total_reward)
        print(f"Episode {episode}: Reward = {total_reward}")
    plt.plot(all_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Learning Progress')
    plt.show()

if __name__ == "__main__":
    main()