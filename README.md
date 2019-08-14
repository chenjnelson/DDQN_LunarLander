## DDQN agent in Lunar Lander

This repo explores strategies around various reinforcement learning techniques, specifically Q-learning. Q-learning can be used to solve a wide range of tasks such as playing video games or stock-trading.

The environment I wanted to solve with Rainbow was [LunarLander](https://gym.openai.com/envs/LunarLander-v2/)**, as part of OpenAI's gym. 

The goal is to develop an agent piloting a spacecraft to arrive to a landing pad. The environment is an 8-dimensional continuous state space (x, y, vx, vy, ?, v?, left-leg, right-leg) with 4 discrete actions (do nothing, fire left engine, fire main engine, fire right engine). Various points are assigned based on the quality of the landing, with the problem being deemed solved once an average score of 200 points is met after 100 consecutive runs. I used DDQN with experience replay to effectively solve the problem.

![Sample of a successful episode](https://user-images.githubusercontent.com/1076706/33915900-ea25fd06-df5a-11e7-9c7a-71dafc04a770.gif)

### Background

Q-learning is a widely-known algorithm in model-free RL. An RL problem is set up where an agent explores an environment enough to learn about a certain behavior to maximize a reward. Q-learning is the agent coming up with Q values for a state-action (s,a) pair to arrive at a policy (the behavior), p. The policy recommends discrete actions at each time step to arrive to a goal.

![](https://latex.codecogs.com/gif.latex?Q%5E*%28s%2C%20a%29%20%3D%20r_0%20&plus;%20%5Cgamma%20%28r_1%20&plus;%20%5Cgamma%20r_2%20&plus;%20%5Cgamma%5E2%20r_3%20&plus;%20...%29%20%3D%20r_0%20&plus;%20%5Cgamma%20%5Cmax_a%20Q%5E*%28s%27%2C%20a%29)

The Bellman equation defines the optimal policy, which maximizes the long-term expected reward. Gamma denotes the weight to place future rewards. alpha (not shown above) is the learning rate at which the agent updates their Q-values. When it is time to test the model, all the agent has to do is look at it's Q-values given a state-action pair and take the corresponding best action.

A couple of enhancements have followed suit to improve this framework, some of which are outlined here***:

**Deep Q-network (DQN)**: high-dimensional state-action pairs cause combinatoric explosions to the Q-value lookup. A fix to this is to use a neural network to approximate the Q-function. 

**Experience Replay**: Q-learning has been proven to converge, but DQN representation is known to be unstable. Part of this is because the agent experiences an environment in a sequential manner, with states being highly correlated to each other. Experience replay is used to decouple the samples that arrive by storing transitions into a memory buffer, sampling a batch, then performing gradient descent to update Q-values.

**Double DQN (DDQN)**: Researchers have found that DQN also overestimates values, because of the *max* aspect. Updating the Q-values amplifies the differences greatly over the rewards which can cause divergence. A second network is introduced: the Q network is used to select the best action, and the Target network is used to estimate the value. Splitting also removes oscillation (of the highly correlated and sequential samples) and allows more stable Q-values for the algorithm to converge. 

#### Instructions
My code is in Python 3.7, and the external libraries used were `numpy`, `gym`, and `keras`. 

The code can be ran either to train models or evaluate a single one.

```python run.py```

There are quite a few parameters to tune as well. Some settings I decided on for this exercise were:

* gamma on [.9, .99, .999]
* alpha on [.0001, .001, .01]
* number of training episodes as 5000
* epsilon_min as .005
* epsilon_decay as .999
* minibatch of 30
* epochs of 1
* ADAM optimizer
* linear activation
* 2 hidden layers, 64 inputs each
* MSE for computing loss (MAE/Huber can also be used, error clipping is a very well discussed topic)

##

#####*Dependencies for LunarLander are a bit messy to install and set up on Windows OS, but can be done below:
* Install Python
* Install Build Tools
  * Download and Install Build Tools for Visual Studio 2017
    * Check Visual C++ Build Tools workload
    * In the Installation Details pane on the right, the Windows 10 SDK is the only optional package that needs to be checked; if not already
* Install Swig
  * Download Swig. Use the one with the prebuilt exe from the Swig Download Page
  * Extract Swig download to a location
  * Add to your `PATH`
* Install OpenAi Gym with support for Lunar Lander
  * Open shell with the environment variables configured to use Visual Studio Build Tools: `Open Start Menu > Visual Studio 2017 Folder > [x64/x86] Native Tools Command Prompt for VS 2017`
  * Configure pip/binutils to use Visual Studio by setting an environment variable: `set DISTUTILS_USE_SDK=1`
  * Install OpenAi Gym with Box2d support

##### Native Tools Command Prompt for VS 2017

```pip install box2d-py```

##### Base env with Anaconda

`conda install -c conda-forge swig=3.0.12`

`pip install gym[all]`

`pip install tensorflow`

`pip install keras`

`pip install numpy`

No need to use the same shell or keep build tools on your computer after installation. All that is required is to keep Swig and the PATH entry.

##

\*\*\*As an aside, the **[Rainbow](https://arxiv.org/pdf/1710.02298.pdf)** algorithm combines several different additional enhancements (Dueling DQN, Noisy Nets, Distributional-RL, N-step, DDQN, Prioritized Experience Replay) of Q-learning to create a model that for many tasks can achieve superhuman performance. 









