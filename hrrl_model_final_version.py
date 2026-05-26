import numpy as np
import matplotlib.pyplot as plt

# Model Parameters
h_opt = 36 # optimal temperature (degrees)
t_shiver = 2 # increase in temperature (degrees) due to shivering 
t_cold = 4 # decrease in temperature (degrees) due to the cold shower

gamma = 0.97 # discounting value
alpha = 0.2 # learning rate

n_states = 3 # state 0 -> initial state, state 1 -> bell state, state 2 -> cold shower state
n_actions = 2 # action 0 -> no shivering, action 1 -> shivering

time_units = 1200 # steps per simulation
n_runs = 1000 # number of simulations

# Reward function as a change in the distance from the optimal homiostatic point
def d(x):
    return (h_opt - x) ** 2

def reward(temp_before, temp_after):
    return d(temp_before) - d(temp_after)

class TemperatureEnvironment:

    def __init__(self):
        self.state = 0
        self.temperature = h_opt

    def transition_state(self, current_state):

        if current_state == 0:
            # stochastic transition
            # with prob = 0.9 the rat remains to the initial state
            if np.random.rand() < 0.1:
                return 1
            else:
                return 0

        elif current_state == 1:
            return 2

        elif current_state == 2:
            # temperature recovers to optimal
            self.temperature = h_opt
            return 0

    def step(self, action):

        old_temp = self.temperature
        current_state = self.state
        if action == 1:
            self.temperature += t_shiver

        if current_state == 2:
            # temperature drops by l due to cold shower
            self.temperature -= t_cold 

        r = reward(old_temp, self.temperature)
        # next state
        next_state = self.transition_state(current_state)
        self.state = next_state
        return next_state, r

# Action Selection
def softmax(x):
    x = x - np.max(x)  # numerical stability
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def sample_action(Q, state, temperature=1.0):
    probs = softmax(Q[state] / temperature)
    return np.random.choice(len(probs), p=probs)

def main():

    # Store the chosen actions for each time step of each simulation
    counts = np.zeros((n_states, time_units, n_actions))
    
    # Main loop
    for run in range(n_runs):
        
        visit_index = np.zeros(n_states, dtype=int)
        env = TemperatureEnvironment()
        Q = np.zeros((n_states, n_actions))
        state = 0
        action = sample_action(Q, state)
        # buffer for 2-step SARSA
        buffer = [] 

        for t in range(time_units):
            ts = visit_index[state]
            # record stats (event-time indexing)
            counts[state, ts, action] += 1
            # step environment
            next_state, r = env.step(action)
            next_action = sample_action(Q, next_state)
            # store transition
            buffer.append((state, action, r))
            # keep only last 3 transitions (needed for 2-step return)
            if len(buffer) > 3:
                buffer.pop(0)

            # 2-step SARSA update
            if len(buffer) == 3:
                (s0, a0, r0), (s1, a1, r1), (s2, a2, r2) = buffer
                td_target = r0 + gamma * r1 + (gamma ** 2) * Q[s2, a2]
                td_error = td_target - Q[s0, a0]
                Q[s0, a0] += alpha * td_error
        
            visit_index[state] += 1
            state = next_state
            action = next_action

    # Convert counts to probabilities
    probabilities = np.zeros((n_states, time_units, n_actions))

    for s in range(n_states):
        for t in range(visit_index[s]):
            total = counts[s, t, 0] + counts[s, t, 1]
            if total > 0:
                probabilities[s, t, 0] = counts[s, t, 0] / total
                probabilities[s, t, 1] = counts[s, t, 1] / total

    # Plot
    fig, axes = plt.subplots(n_states, 1, figsize=(10, 4 * n_states))

    state_names = {
        0: "Initial state",
        1: "Bell state",
        2: "Cold Shower state"
    }

    action_names = {
        0: "NO_SHIVER",
        1: "SHIVER"
    }

    for s in range(n_states):
        T = visit_index[s]
        x = range(T)

        axes[s].scatter(
            x,
            probabilities[s, :T, 0],
            s=10,
            label=action_names[0]
        )
        axes[s].scatter(
            x,
            probabilities[s, :T, 1],
            s=10,
            label=action_names[1]
        )

        axes[s].set_title(state_names[s])
        axes[s].set_xlabel("Trials")
        axes[s].set_ylabel("Action Probabilities")
        axes[s].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()