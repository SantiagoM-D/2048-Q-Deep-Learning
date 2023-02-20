import numpy as np
import pandas as pd
import re
import random
import tensorflow as tf
import time
from collections import deque, namedtuple
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE
from selenium import webdriver
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup as bs

MEMORY_SIZE = 100_000     # size of memory buffer.
GAMMA = 0.995             # discount factor.
ALPHA = 1e-3              # learning rate.
NUM_STEPS_FOR_UPDATE = 4  # perform a learning update every C time steps.

SEED = 0  # Seed for the pseudo-random number generator.
MINIBATCH_SIZE = 64  # Mini-batch size.
TAU = 1e-3  # Soft update parameter.
E_DECAY = 0.995  # ε-decay rate for the ε-greedy policy.
E_MIN = 0.01  # Minimum ε value for the ε-greedy policy.

def get_num(a,b):
    return 'tile-position-'+str(a)+'-'+str(b)

def get_matrix(driver):
    content = driver.page_source
    soup = bs(content,"html.parser")
    game = [0 for i in range(16)]
    for i in range(1,5):
        for j in range(1,5):
            a = 4*(i-1)+j-1
            try: game[a] = int(soup.select('div'+'.'+get_num(j,i)+'.'+'tile-merged')[0].text)
            except:
                try: game[a]=int(soup.find_all('div',attrs={'class':get_num(j,i)})[0].text)
                except: game[a]=0
    return game

def move(a,driver):
    actions = ActionChains(driver)
    if a == 0: actions.send_keys(Keys.ARROW_UP).perform()
    elif a == 1: actions.send_keys(Keys.ARROW_DOWN).perform()
    elif a == 2: actions.send_keys(Keys.ARROW_LEFT).perform()
    elif a == 3: actions.send_keys(Keys.ARROW_RIGHT).perform()

def get_score(driver):
    content = driver.page_source
    soup = bs(content,"html.parser")
    score_text = soup.find_all('div',attrs={'class':'score-container'})[0].text
    return int(re.findall(r'^\D*(\d+)', score_text)[0])

def kepp_playing(driver):
    content = driver.page_source
    soup = bs(content,"html.parser")
    try: 
        end_text = soup.find_all('div',attrs={'class':'game-message game-over'})[0].text
        if end_text == 'Game over!Keep going\nTry again':
            end = True
        else:
            end = False
    except:
        end = False
    return end

def step(a,driver):

    initial_state = get_matrix(driver)
    initial_score = get_score(driver)
    action = a
    move(a,driver)
    next_state = get_matrix(driver)
    next_score = get_score(driver)
    reward = -(initial_score - next_score)
    done = kepp_playing(driver)
    return (initial_state, action, next_state, reward, done)

"""interaccion con el ambiente"""

driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())
driver.get("https://play2048.co/")


print(step(1,driver))

driver.close()


'''Deep Q-Learning
1 - Target Network'''
# Create the Q-Network.
q_network = Sequential([
    ### START CODE HERE ### 
    Input(shape=16),                      
    Dense(units=128, activation='relu'),            
    Dense(units=128, activation='relu'),
    Dense(units=128, activation='relu'),            
    Dense(units=4, activation='linear'),
    ### END CODE HERE ### 
    ])

# Create the target Q^-Network.
target_q_network = Sequential([
    ### START CODE HERE ### 
    Input(shape=16),                       
    Dense(units=128, activation='relu'),            
    Dense(units=128, activation='relu'),
    Dense(units=128, activation='relu'),            
    Dense(units=4, activation='linear'), 
    ### END CODE HERE ###
    ])

optimizer = Adam(learning_rate=ALPHA)

'''Deep Q-Learning
2 - Experience Replay'''

experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

'''Deep Q-Learning Algorithm with Experience Replay'''

def compute_loss(experiences, gamma, q_network, target_q_network):
    """ 
    Calculates the loss.

    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.
      q_network: (tf.keras.Sequential) Keras model for predicting the q_values
      target_q_network: (tf.keras.Sequential) Keras model for predicting the targets

    Returns:
      loss: (TensorFlow Tensor(shape=(0,), dtype=int32)) the Mean-Squared Error between
            the y targets and the Q(s,a) values.
    """


    # Unpack the mini-batch of experience tuples.
    states, actions, rewards, next_states, done_vals = experiences

    # Compute max Q^(s,a).
    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)

    # Set y = R if episode terminates, otherwise set y = R + γ max Q^(s,a).
    y_targets = rewards + (gamma * max_qsa * (1 - done_vals))

    # Get the q_values.
    q_values = q_network(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                tf.cast(actions, tf.int32)], axis=1))

    # Calculate the loss.
    loss = MSE(y_targets, q_values)

    return loss

'''Update the Network Weights'''

def update_target_network(q_network, target_q_network):
    """
    Updates the weights of the target Q-Network using a soft update.
    
    The weights of the target_q_network are updated using the soft update rule:
    
                    w_target = (TAU * w) + (1 - TAU) * w_target
    
    where w_target are the weights of the target_q_network, TAU is the soft update
    parameter, and w are the weights of the q_network.
    
    Args:
        q_network (tf.keras.Sequential): 
            The Q-Network. 
        target_q_network (tf.keras.Sequential):
            The Target Q-Network.
    """

    for target_weights, q_net_weights in zip(
        target_q_network.weights, q_network.weights
    ):
        target_weights.assign(TAU * q_net_weights + (1.0 - TAU) * target_weights)

@tf.function
def agent_learn(experiences, gamma):
    """
    Updates the weights of the Q networks.
    
    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.
    
    """
    
    # Calculate the loss.
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, q_network, target_q_network)

    # Get the gradients of the loss with respect to the weights.
    gradients = tape.gradient(loss, q_network.trainable_variables)
    
    # Update the weights of the q_network.
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

    # update the weights of target q_network.
    update_target_network(q_network, target_q_network)

'''Train the Agent'''

def get_action(q_values, epsilon=0.0):
    """
    Returns an action using an ε-greedy policy.

    This function will return an action according to the following rules:
        - With probability epsilon, it will return an action chosen at random.
        - With probability (1 - epsilon), it will return the action that yields the
        maximum Q value in q_values.
    
    Args:
        q_values (tf.Tensor):
            The Q values returned by the Q-Network. For the Lunar Lander environment
            this TensorFlow Tensor should have a shape of [1, 4] and its elements should
            have dtype=tf.float32. 
        epsilon (float):
            The current value of epsilon.

    Returns:
       An action (numpy.int64). For the Lunar Lander environment, actions are
       represented by integers in the closed interval [0,3].
    """

    if random.random() > epsilon:
        return np.argmax(q_values.numpy()[0])
    else:
        return random.choice(np.arange(4))

def check_update_conditions(t, num_steps_upd, memory_buffer):
    """
    Determines if the conditions are met to perform a learning update.

    Checks if the current time step t is a multiple of num_steps_upd and if the
    memory_buffer has enough experience tuples to fill a mini-batch (for example, if the
    mini-batch size is 64, then the memory buffer should have more than 64 experience
    tuples in order to perform a learning update).
    
    Args:
        t (int):
            The current time step.
        num_steps_upd (int):
            The number of time steps used to determine how often to perform a learning
            update. A learning update is only performed every num_steps_upd time steps.
        memory_buffer (deque):
            A deque containing experiences. The experiences are stored in the memory
            buffer as namedtuples: namedtuple("Experience", field_names=["state",
            "action", "reward", "next_state", "done"]).

    Returns:
       A boolean that will be True if conditions are met and False otherwise. 
    """

    if (t + 1) % num_steps_upd == 0 and len(memory_buffer) > MINIBATCH_SIZE:
        return True
    else:
        return False

def get_experiences(memory_buffer):
    """
    Returns a random sample of experience tuples drawn from the memory buffer.

    Retrieves a random sample of experience tuples from the given memory_buffer and
    returns them as TensorFlow Tensors. The size of the random sample is determined by
    the mini-batch size (MINIBATCH_SIZE). 
    
    Args:
        memory_buffer (deque):
            A deque containing experiences. The experiences are stored in the memory
            buffer as namedtuples: namedtuple("Experience", field_names=["state",
            "action", "reward", "next_state", "done"]).

    Returns:
        A tuple (states, actions, rewards, next_states, done_vals) where:

            - states are the starting states of the agent.
            - actions are the actions taken by the agent from the starting states.
            - rewards are the rewards received by the agent after taking the actions.
            - next_states are the new states of the agent after taking the actions.
            - done_vals are the boolean values indicating if the episode ended.

        All tuple elements are TensorFlow Tensors whose shape is determined by the
        mini-batch size and the given Gym environment. For the Lunar Lander environment
        the states and next_states will have a shape of [MINIBATCH_SIZE, 8] while the
        actions, rewards, and done_vals will have a shape of [MINIBATCH_SIZE]. All
        TensorFlow Tensors have elements with dtype=tf.float32.
    """

    experiences = random.sample(memory_buffer, k=MINIBATCH_SIZE)
    states = tf.convert_to_tensor(
        np.array([e.state for e in experiences if e is not None]), dtype=tf.float32
    )
    actions = tf.convert_to_tensor(
        np.array([e.action for e in experiences if e is not None]), dtype=tf.float32
    )
    rewards = tf.convert_to_tensor(
        np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32
    )
    next_states = tf.convert_to_tensor(
        np.array([e.next_state for e in experiences if e is not None]), dtype=tf.float32
    )
    done_vals = tf.convert_to_tensor(
        np.array([e.done for e in experiences if e is not None]).astype(np.uint8),
        dtype=tf.float32,
    )
    return (states, actions, rewards, next_states, done_vals)

def get_new_eps(epsilon):
    """
    Updates the epsilon value for the ε-greedy policy.
    
    Gradually decreases the value of epsilon towards a minimum value (E_MIN) using the
    given ε-decay rate (E_DECAY).

    Args:
        epsilon (float):
            The current value of epsilon.

    Returns:
       A float with the updated value of epsilon.
    """

    return max(E_MIN, E_DECAY * epsilon)

start = time.time()

num_episodes = 2000
max_num_timesteps = 1000

total_point_history = []

num_p_av = 10    # number of total points to use for averaging.
epsilon = 1.0     # initial ε value for ε-greedy policy.

# Create a memory buffer D with capacity N.
memory_buffer = deque(maxlen=MEMORY_SIZE)

# Set the target network weights equal to the Q-Network weights.
target_q_network.set_weights(q_network.get_weights())

#start the environment

game_driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())
game_driver.get("https://play2048.co/")

for i in range(num_episodes):

    # Reset the environment to the initial state and get the initial state.

    state = get_matrix(game_driver)
    total_points = 0
    
    for t in range(max_num_timesteps):
        
        # From the current state S choose an action A using an ε-greedy policy.
        state_qn = np.expand_dims(state, axis=0)  # state needs to be the right shape for the q_network.
        q_values = q_network(state_qn)
        action = get_action(q_values, epsilon)
        
        # Take action A and receive reward R and the next state S'.
        _, _, next_state, reward, done = step(action,game_driver)
        
        # Store experience tuple (S,A,R,S') in the memory buffer.
        # We store the done variable as well for convenience.
        memory_buffer.append(experience(state, action, reward, next_state, done))
        
        # Only update the network every NUM_STEPS_FOR_UPDATE time steps.
        update = check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer)
        
        if update:
            # Sample random mini-batch of experience tuples (S,A,R,S') from D.
            experiences = get_experiences(memory_buffer)
            
            # Set the y targets, perform a gradient descent step,
            # and update the network weights.
            agent_learn(experiences, GAMMA)
        
        state = next_state.copy()
        total_points += reward

        if done:
            if  i % 4 == 0:
                game_driver.close()
                game_driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())
                game_driver.get("https://play2048.co/")
                #restart = game_driver.find_element(By.CLASS_NAME, "restart-button")
                #restart.click()
                break
            else:
                restart = game_driver.find_element(By.CLASS_NAME, "restart-button")
                restart.click()
                break

            
    total_point_history.append(total_points)
    av_latest_points = np.mean(total_point_history[-num_p_av:])
    
    # Update the ε value.
    epsilon = get_new_eps(epsilon)

    print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}", end="")

    if (i+1) % num_p_av == 0:
        print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")

    # We will consider that the environment is solved if we get an
    # average of 200 points in the last 100 episodes.
    if av_latest_points >= 100000.0:
        print(f"\n\nEnvironment solved in {i+1} episodes!")
        q_network.save('2048_model.h5')
        break
        
tot_time = time.time() - start

print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")
