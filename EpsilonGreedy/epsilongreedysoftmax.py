
import numpy as np
class EpsilonGreedy:
    def EpsilonGreedy(self,epsilon,num_actions):
        self.epsilon=epsilon
        self.num_actions=num_actions
        self.action_count=[0]*num_actions
        self.action_values=[0.0]*num_actions

    def softmax_prob(self):
        exp_val=np.exp(self.action_values)
        probabilities=exp_val/np.sum(exp_val)
        return probabilities

    def choose_action(self):
        #two things will happen here, exploitation and exploration
        #exploration part
        if np.random.random()<self.epsilon:
            return np.random.randint(0,self.num_actions-1)
        else:
            probabilities=self.softmax_prob()
            return np.random.choice(self.num_actions,p=probabilities)

    def update_action_value(self, action, reward):
        self.action_count[action]+=1
        n=self.action_count[action]
        current_value=self.action_values[action]
        new_value=current_value+(1/n)*(current_value-reward)
        self.action_values[action]=new_value


if __name__=="__main__":
    epsilon=0.1
    num_actions=5 # arms of the bandit
    agent=EpsilonGreedy(epsilon,num_actions)

    for i in range(0,1000):
        chosen_action=agent.choose_action()
        reward=np.random.random()
        agent.update_action_value(chosen_action,reward)