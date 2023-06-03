import numpy as np

num_states = 5
num_actions = 2
rewards = np.array([[0,0],[0,1],[1,0],[0,0],[0,0]])


q_table = np.zeros((num_states,num_actions))

learning_rate = 0.1
discount_factor = 0.9
num_epi = 1000
max_step = 100


for num in range(num_epi):
    state = 0
    for i in range(max_step):
        if np.random.rand()<0.1:
            action = np.random.randint(num_actions)
        else:
            action = np.argmax(q_table[state])
        
        next_state = state+1
        reward = rewards[state,action]

        q_table[state,action]+= learning_rate*(reward+discount_factor*np.max(q_table[next_state])-q_table[state,action])

        state = next_state

        if state == num_states -1:
            break

print("Q-table")
print(q_table) 