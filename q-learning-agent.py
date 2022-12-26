# Go through the above tutorial and try it out for different values of the parameters (learning rate and discount rate).
# Comment on the influence the above parameters have on how fast q-learning can converge. 
# Plot the necessary graphs to justify your answer.

import numpy as np
import gym
import random

def main():

    # create Taxi environment
    env = gym.make('Taxi-v3')

    # initialize q-table
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size)) # Initialize q-table with zeros

    print(qtable)

    # hyperparameters
    learning_rate = 0.9 # How easily the agent should accept new information over previously learnt information
    discount_rate = 0.8 # how much the agent should take into consideration the rewards it could receive in the future versus its immediate reward
    epsilon = 1.0 # Exploration rate (How much the agent should explore the environment before exploiting what it has learnt)
    decay_rate= 0.005 # How quickly the agent should stop exploring and start exploiting

    # training variables
    num_episodes = 1000
    max_steps = 99 # per episode

    # training
    for episode in range(num_episodes):

        # reset the environment
        state = env.reset()
        done = False

        for s in range(max_steps):

            # exploration-exploitation tradeoff
            if random.uniform(0,1) < epsilon:
                # explore
                action = env.action_space.sample() # Choose a random action
            else:
                # exploit
                action = np.argmax(qtable[state,:]) # Choose the action with the highest q-value

            # take action and observe reward
            new_state, reward, done, info = env.step(action)

            # Q-learning algorithm
            qtable[state,action] = qtable[state,action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state,:])-qtable[state,action])

            # Update to our new state
            state = new_state

            # if done, finish episode
            if done == True:
                break

        # Decrease epsilon (Beacause as we train the agent, we want it to exploit more while exploring less)
        epsilon = np.exp(-decay_rate*episode)

    print(f"Training completed over {num_episodes} episodes")
    input("Press Enter to watch trained agent...")

    # watch trained agent
    state = env.reset()
    done = False
    rewards = 0

    for s in range(max_steps):

        print(f"TRAINED AGENT")
        print("Step {}".format(s+1))

        action = np.argmax(qtable[state,:])
        new_state, reward, done, info = env.step(action)
        rewards += reward
        env.render()
        print(f"score: {rewards}")
        state = new_state

        if done == True:
            break

    env.close()

if __name__ == "__main__":
    main()