
import numpy as np
class UCB:
    def __init__(self,epsilon,num_actions):
        self.num_actions=num_actions
        self.action_count=[0]*num_actions
        self.action_values=[0.0]*num_actions
        self.timestep=0

    def choose_action(self):
        self.timestep+=1
        exploration_term=np.sqrt(np.log(self.timestep)/(self.action_count+1e-5))
        ucb_value=self.action_values+exploration_term
        action=np.argmax(ucb_value)
        return action
    
    def update_action_value(self, action, reward):
        self.action_count[action]+=1
        n=self.action_count[action]
        current_value=self.action_values[action]
        new_value=current_value+(1/n)*(current_value-reward)
        self.action_values[action]=new_value


if __name__=="__main__":
    num_actions=5 # arms of the bandit
    agent=UCB(num_actions)
    num_episodes=1000
    num_steps_per_episode=100

    env=np.random.randn(num_actions)
    agent=UCB(num_actions)

    cumulative_reward=0

    for i in range(0,1000):
        chosen_action=agent.choose_action()
        reward=np.random.normal(env[chosen_action],1.0)
        cumulative_reward+=reward
        agent.update_action_value(chosen_action,reward)
    avg_cumulative_reward=cumulative_reward/(num_episodes*num_steps_per_episode)
    print("Average cumuluative reward is: ", avg_cumulative_reward)