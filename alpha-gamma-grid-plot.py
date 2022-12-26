# Go through the above tutorial and try it out for different values of the parameters (learning rate and discount rate).
# Comment on the influence the above parameters have on how fast q-learning can converge. 
# Plot the necessary graphs to justify your answer.

import numpy as np
import gym
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

x_cords = []
y_cords = []

def taxi_env(learning_rate, discount_rate):

    # create Taxi environment
    env = gym.make('Taxi-v3')

    # hyperparameters
    epsilon = 1.0 # Exploration rate (How much the agent should explore the environment before exploiting what it has learnt)
    decay_rate= 0.005 # How quickly the agent should stop exploring and start exploiting

    # initialize q-table
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size)) # Initialize q-table with zeros

    # training variables
    num_episodes = 1000
    max_steps = 99 # per episode

    # training
    for episode in range(num_episodes):

        cumml_reward = 0
        count = 0

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
            
            cumml_reward += reward

            # Q-learning algorithm
            qtable[state,action] = qtable[state,action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state,:])-qtable[state,action])

            # Update to our new state
            state = new_state

            count += 1

            # if done, finish episode
            if done == True:
                break
        
        learnings.append(learning_rate)
        discounts.append(discount_rate)
        x_cords.append(episode)
        y_cords.append(cumml_reward)

        # Decrease epsilon (Beacause as we train the agent, we want it to exploit more while exploring less)
        epsilon = np.exp(-decay_rate*episode)

    env.close()

if __name__ == "__main__":

    learning_rates = [1, 0.8, 0.6, 0.4, 0.2]
    discount_rates = [1, 0.8, 0.6, 0.4, 0.2]

    learnings = []
    discounts = []
    x_cords = []
    y_cords = []

    for learning_rate in learning_rates:
        for discount_rate in discount_rates:
            print(learning_rate,discount_rate)
            
            taxi_env(learning_rate, discount_rate)
            
    lists = [learnings, discounts, x_cords, y_cords]
    df = pd.concat([pd.Series(x) for x in lists], axis =1)

    df.rename(columns = {0:'Learning Rate'}, inplace = True)
    df.rename(columns = {1:'Discount Rate'}, inplace = True)
    df.rename(columns = {2:'Episode Number'}, inplace = True)
    df.rename(columns = {3:'Cumulative Reward'}, inplace = True)

    sea = sns.FacetGrid(df, col = 'Learning Rate', row ='Discount Rate')
    sea.map(sns.lineplot, 'Episode Number', 'Cumulative Reward', color = ".3")
    plt.show()