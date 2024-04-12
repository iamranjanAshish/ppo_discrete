import gym
import numpy as np
import matplotlib.pyplot as plt
from ppo import PPO

plt.style.use('fivethirtyeight')
rng = np.random.default_rng()

MAX_EPI = 5000
GAMMA = 0.98
C_LR = 0.001
A_LR = 0.0003
EPSILON = 0.2
EPOCHS = 4
MAX_SIZE = 5
LAM = 0.95

if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    ppo = PPO(GAMMA, LAM, EPSILON, C_LR, A_LR, 4, 2, 1, (64, 64), (64, 32), EPOCHS, MAX_SIZE)

    best_score = env.reward_range[0]
    score_history = []

    for i in range(MAX_EPI):
        obs = env.reset()[0]
        done = False
        truncated = False
        score = 0
        step = 0
        while not done and not truncated:
            action = ppo.choose_action(obs)
            new_state, reward, done, truncated, info = env.step(action)
            ppo.remember(obs, action, reward, new_state, done)
            ppo.learn()
            obs = new_state
            score += reward
            step += 1
   
        score_history.append(score)
        avg_score = np.mean(score_history[-10:])

        if avg_score > best_score:
            best_score = avg_score
            ppo.save_model()

        print('episode : {} score : {} average score {}'.format(i, score, avg_score))

    filename = 'pendulum.png'
    figure_file = 'plots/' + filename

    plt.plot(list(range(MAX_EPI)), score_history)
    plt.show()
