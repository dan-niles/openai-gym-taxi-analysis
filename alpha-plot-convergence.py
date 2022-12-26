# Go through the above tutorial and try it out for different values of the parameters (learning rate and discount rate).
# Comment on the influence the above parameters have on how fast q-learning can converge. 
# Plot the necessary graphs to justify your answer.

import numpy as np
import gym
import random
import matplotlib.pyplot as plt

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
    episodes_to_conv = 0
    for episode in range(num_episodes):

        cumml_reward = 0
        count = 0
        max_change = 0

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

            # old q value to be recorded to comapre later
            old_q_val = qtable[state,action]

            # Q-learning algorithm
            qtable[state,action] = qtable[state,action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state,:])-qtable[state,action])

            # change in the value of qtable
            change = abs(qtable[state,action] - old_q_val)

            if change > max_change:
                max_change = change

            # Update to our new state
            state = new_state

            count += 1

            # if done, finish episode
            if done == True:
                break
        
        x_cords.append(episode)
        y_cords.append(cumml_reward)

        if max_change > 0.000000000001:
            episodes_to_conv += 1
        else:
            break

        # Decrease epsilon (Beacause as we train the agent, we want it to exploit more while exploring less)
        epsilon = np.exp(-decay_rate*episode)
    
    return episodes_to_conv

    # print(f"Training completed over {num_episodes} episodes")
    # input("Press Enter to watch trained agent...")

    # # watch trained agent
    # state = env.reset()
    # done = False
    # rewards = 0

    # for s in range(max_steps):

    #     print(f"TRAINED AGENT")
    #     print("Step {}".format(s+1))

    #     action = np.argmax(qtable[state,:])
    #     new_state, reward, done, info = env.step(action)
    #     rewards += reward
    #     env.render()
    #     print(f"score: {rewards}")
    #     state = new_state

    #     if done == True:
    #         break

    env.close()

if __name__ == "__main__":

    learning_rates = [1, 0.8, 0.6, 0.4, 0.2, 0.1, 0]
    discount_rate = 0.8
    converging_numbers = []

    for learning_rate in learning_rates:
        print(learning_rate)

        episodes_to_converge = taxi_env(learning_rate, discount_rate)
        # print(episodes_to_converge)
        converging_numbers.append(episodes_to_converge)

    # plt.scatter(x_cords, y_cords)
    print(learning_rates)
    print(converging_numbers)

    # myline = np.linspace(1, len(learning_rates), 1)

    plt.plot(learning_rates, converging_numbers)
    
    # plt.plot(x_cords, y_cords, label = "alpha = " + str(learning_rate))
    plt.xlabel('Learning rate applied')
    plt.ylabel('Number of episodes required for convergence')

    plt.title('Convergence vs Learning Rate')
    # plt.legend()
    plt.show()
    