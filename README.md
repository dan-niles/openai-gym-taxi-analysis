# OpenAI Gym Taxi-v3 Environment Analysis

## Background

This analysis was performed for a group project of CS3613: Introduction to Artificial Intelligence, based OpenAI Gym's Taxi environment.

### Introduction

OpenAI’s Gym is a standard API for reinforcement learning, containing a diverse collection of reference environments. “Taxi” is an environment used to test RL algorithms, where the agent’s goal is to pick up passengers and drop them off at the destination in the least amount of moves. The goal of this assignment is to find how the convergence of the q-learning algorithm used to train the taxi agent, is influenced by parameters such as the learning rate and discount rate.

## Methodology

The Reinforcement Learning process using q-learning was applied and tested under multiple values for the learning rate and discount rate parameters. The values applied for each parameter are as follows,

- Learning rate = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
- Discount rate = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}

## Getting Started

Make sure you have Python 3.8 or higher installed. Then, install the required packages using the following command.

```bash
pip install gym numpy matplotlib
```

## Contribution

- 200110P - N. Y. D. De Silva
- 200421U - D. A. Niles
- 200490D - V. P. Pussewela
- 200730P - W. A. D. O. R. Wijesooriya

## References

- Official documentation for the Taxi environemnt : https://www.gymlibrary.dev/environments/toy_text/taxi/
- Tutorial on OpenAI Gym Taxi : https://www.gocoder.one/blog/rl-tutorial-with-openai-gym
