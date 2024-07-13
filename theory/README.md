## On-Policy RL Pseudocode
### Initialization

1. Initialize policy parameters `θ`.
2. Initialize learning rate `α`.
3. Initialize discount factor `γ`.

### Learning loop
```python 
for episode in range(1, N_episodes + 1):
    Initialize state S 
    episode_rewards = []
    episode_states = []
    episode_actions = []
    
    while not done:  # will exist when the maximum step limit is reached, or the task-specific condition is satisfied.
        Choose action A based on policy π(A|S; θ)
        Take action A, observe reward R and next state S'
        
        episode_rewards.append(R)
        episode_states.append(S)
        episode_actions.append(A)

        S = S'
        
    G = 0  # Return (total discounted reward)
    policy_gradient = 0
    
    for t in reversed(range(len(episode_rewards))):
        G = G * γ + episode_rewards[t]
        
        # Compute the gradient of the log probability of the action
        policy_gradient += (G - baseline) * ∇θ log π(episode_actions[t] | episode_states[t]; θ)
    
    # Update policy parameters
    θ = θ + α * policy_gradient
```

##  Off-policy RL
### Initialization

1. Initialize Q-values `Q(s, a)` arbitrarily (e.g., to zero).
2. Initialize learning rate `α`.
3. Initialize discount factor `γ`.
```python 
for episode in range(1, N_episodes + 1):
    Initialize state S
    done = False

    while not done:
        Choose action A using ε-greedy policy based on Q-values Q(s, a)
        Take action A, observe reward R and next state S'
        
        Q(S, A) = Q(S, A) + α * (R + γ * max_a' Q(S', a') - Q(S, A))
        
        S = S'
        if S is terminal:
            done = True
```
