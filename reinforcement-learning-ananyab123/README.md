[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/KB1woPx3)
# README

Tell us about your implementation!
My implementation simulated Blackjack such that value iteration updates state values until reaching convergence so it could find the optimal policy. My implementation of Q-learning learned action values through the simulated gameplay which required the use of random exploration. All of the adjusted implementations for value iteration and Q-learning were tested through the Monte Carlo simulation to estimate the win rate and the reward (which was around 43%). 

Answer the conceptual questions!

Part 1:

1. What are the probabilities that the player will win/draw/lose if the player stands on a 19 and the dealer's cards total **16**, but the dealer does not have an ace?
*Recall:* The dealer must draw another card from the deck at this point. The dealer draws until their total is at least 17.

if 2 then 18 so win
if 3 then 19 so draw
if 4 then 20 so lose
if 5 then 21 so lose
if 6-9 then bust so win
if 10 (prob is 4/13 for 10 val) then bust so win
if A (counts as 1) then 17 so win 

ANSWER
P (Win) = (1+1+1+1+4+1)/13 = 10/13
P (Draw) = 1/13
P (Lose) = 2/13

Use your answer to problem 1 and Bellman's equation to solve the next two problems:

2. What are the probabilities that the player will win/draw/lose if the player stands on a 19 and the dealer's cards total **15**, but the dealer does not have an ace?

one draw but Ace leads to Dealer = 16 so based on before
if 2, 3, 7, 8, 9, 10 then win so 9/13
if 4, then draw so 1/13
if 5, 6 then lose so 2/13

for ace (1/13) use answer to 1 probablities (win = 10/13, draw = 1/13, lose = 2/13)

ANSWER
P (Win) = (9/13) + (1/13) * (10/13) = 0.752
P (Draw) = (1/13) + (1/13) * (1/13) = 0.083
P (Lose) = (2/13) + (1/13) * (2/13) = 0.166

3. What are the probabilities that the player will win/draw/lose if the player stands on a 19 and the dealer's cards total **14**, but the dealer does not have an ace?

2 would lead to Dealer = 16, Ace would lead to Dealer = 15

if 3, 4, 8, 9, 10 then win so 8/13
if 5, then draw so 1/13
if 6, 7 then lose so 2/13

for 2: (1/13) * ((10/13), (1/13), (2/13))
for Ace: (1/13) * (0.752, 0.083, 0.166)

ANSWER
P (Win) = (8/13) + (1/13) * (10/13) + (1/13) * (0.752) = 0.732
P (Draw) = (1/13) + (1/13) * (1/13) + (1/13) * (0.083) = 0.089
P (Lose) = (2/13) + (1/13) * (2/13) + (1/13) * (0.166) = 0.178

Recall the definition of the **value function** at a state $s$, given a policy $\pi$: it is the expected value of the long-term rewards of following policy $\pi$ starting at $s$. The **optimal value function** is the expected value of the long-term rewards of following an optimal policy. 

4. Interpret the optimal value function for Blackjack in terms of these win/draw/lose probabilities.
The optimal value function is V(s) = P (win | s) - P (lose | s) such that a win is +1, draw is 0, loss is -1, and gamma = 1, so draw doesn't really contribute to the function. 


Part 5:
Blackjack in Casinos:
How would you modify our Blackjack MDP to capture all of this new complexity? In your README, describe the changes you would make to the states, actions, reward function, and transition function to model the Blackjack game played at casinos. For the transition function, you should discuss how the successor states might change given any modifications you make to your states, but you can leave out discussing transition probabilities. What effect would your proposed changes to the Blackjack MDP have on the algorithms you implemented in this notebook (value iteration and Q-learning)? 

To modify the Blackjack MDP to capture the new complexity, for the state, I would now include new information about the player bet size, doubling or splitting ability, and the hand played after a split. Then, I would expand and adjust actions to include double and split as actions, and change the reward function to handle betting and the special payouts because normally a win would have +1 times the bet, and blackjack is +1.5 times the bet, while losses are -bet. However, now with doubling and splitting, the rewards and risks are multiplied. Finally, I'd modify the transition function because in terms of successor states, doubling would lead to a terminal state, splitting causes subgames and a state branching into two new hands, and the active hand and remaining hands must also be accounted for. My proposed changes would make value iteration less effective because of the need to evaluate every single state, meanwhile Q-learning would be better because it can still learn albeit with more episodes for convergence. 


Card Counting:
1. In your README, describe how you'd modify your MDP to model this card counting strategy. You can leave out discussion of transition probabilities in this question.

I'd modify my MDP to model this card counting strategy by introducing a count variable to track the high to low cards ratio to add card counting, which would adjust the state. Then, the high cards would end up decreasing count by 1, low cards increase it by 1, and the count would update after every draw. Then, the successor states would begin to differ given that every new card draw adjusts the count. 

2. Does this problem still satisfy the Markov property? Do the probabilities of the next state depend only on the current state?

This problem still satisfies the Markov property and the probabilities of the next state depend only on the current state once the count is part of the state.

3. Given the exact probabilities of the transition function will be hard to calculate (e.g., what is the probability of drawing a 4 when the count is +2), which of the algorithms you implemented in this notebook would be easier to implement, Value Iteration of Q-Learning and why?

Given the exact probabilities of the transition function being hard to calculate, Q-learning should be easier to implement because it learns from experience and doesn't need transition probabilities which value iteration needs (this is difficult because it's hard to calculate the exact probability of the next card). 

4. If you were able to find the optimal policy for the state representation using card counting and the full state representation of the previous question, how would you expect the quality of these policies to compare. If we ran `monte_carlo_simulation` on each policy, which would perform better?

I would expect the quality of these policies to compare such that the full state representation policy would be better because of its use of information, but the count policy would be much simpler and learnable. Thus, if we ran the `monte_carlo_simulation` on each policy, the full state one would probably perform better, though the count one would still be similar in performance. 


Hours taken: 15

Collaborators: none

Known Bugs: none

AI Use Description: Used Google AI to understand Q Learning

You must acknowledge use here and submit transcript if AI was used for portions of the assignment

Q-learning is a model-free, off-policy reinforcement learning algorithm where an agent learns to make optimal decisions through trial-and-error, guided by a system of rewards. The goal is to learn a mapping of state-action pairs to their expected future rewards, known as Q-values, which are stored in a Q-table. Explanation of Q-Learning The core idea is that the agent interacts with an environment, observes its state, takes an action, and receives a reward. The agent's objective is to find an optimal policy (strategy) that maximizes the cumulative long-term reward. This is achieved by iteratively updating the Q-table using the Bellman equation. Key Components Agent: The entity that acts and operates within an environment.Environment: The external system with which the agent interacts.State (s): The agent's current situation or position in the environment.Action (a): The operation performed by the agent in a given state.Reward (r): Feedback (positive or negative) from the environment after an action is taken.Q-Table: A matrix where rows represent states and columns represent actions, with each cell containing the Q-value for that state-action pair.Q-Value (Q(s, a)): The estimated expected future reward for taking action a in state s.Hyperparameters:Learning Rate (\(\alpha \)): Controls how much new information overrides old information (typically between 0 and 1).Discount Factor (\(\gamma \)): Determines the importance of future rewards versus immediate rewards (between 0 and 1).Exploration Rate (\(\epsilon \)): Manages the balance between exploring new actions and exploiting known good actions (epsilon-greedy policy). The Q-Learning Update Rule (Bellman Equation) The Q-table is updated using the following formula after each action: \(Q(s,a)=Q(s,a)+\alpha [r+\gamma \cdot \max _{a^{\prime }}Q(s^{\prime },a^{\prime })-Q(s,a)]\) \(Q(s,a)\): The current Q-value for the state-action pair.\(\alpha \): The learning rate.\(r\): The immediate reward received.\(\gamma \): The discount factor.\(\max _{a^{\prime }}Q(s^{\prime },a^{\prime })\): The maximum Q-value for all possible actions \(a^{\prime }\) in the new state \(s^{\prime }\). This represents the expected optimal future reward from the next state onwards. Implementation Steps (Python Example) Implementing Q-learning involves defining the environment, setting hyperparameters, initializing the Q-table, and running the training loop. The numpy library is essential for handling the Q-table. 