import numpy as np

ratings = np.array([
            [5,4,3,2],
            [1, 2, 4, 0],
            [0, 5, 3, 2],
            [4, 0, 1, 3],
            ])

num_users,num_items = ratings.shape
q_table = np.zeros((num_users,num_items))

learning_rate = 0.1
discount_rate = 0.9
num_epi = 1000
max_step = 10

for user in range(num_epi):
    state = np.random.randint(num_users)
    for i in range(max_step):
        if np.random.rand()<0.1:
            action = np.random.randint(num_items)
        else:
            action = np.argmax(q_table[state])
        
        next_state = state +1
        reward = ratings[state,action]

        q_table[state,action] += learning_rate*(reward+discount_rate*np.max(q_table[state])-q_table[state,action])



print(q_table)

def recommend_items(user, q_table):
    recommended_item = np.argmax(q_table[user])
    return recommended_item

user_id = 0 
recommended_item = recommend_items(user_id, q_table)
print("Recommended item for user", user_id+1, ":", recommended_item+1)
