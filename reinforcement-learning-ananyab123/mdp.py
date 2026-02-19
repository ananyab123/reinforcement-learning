import gymnasium as gym
from typing import DefaultDict, List, Tuple, Dict, Optional, Union
from collections import defaultdict

class MDP(gym.Wrapper):
    """
    Abstract base class for Markov Decision Processes using Gymnasium wrapper.
    
    This class provides a standard interface for MDP environments, extending
    the Gymnasium Wrapper to add MDP-specific functionality.
    """
    def __init__(self, env: gym.Env) -> None:
        """
        Initialize the MDP wrapper.
        
        Args:
            env: Gymnasium environment to wrap
        """
        super().__init__(env)
        self.env = env

    def reset(self) -> Optional[Tuple]:
        """
        Reset the environment to an initial state.
        
        Returns:
            Initial state of the environment
        """
        pass

    def step(self, action: int) -> Tuple:
        """
        Take an action in the environment and return the result.
        
        Args:
            action: Action to take in the current state
            
        Returns:
            Tuple containing next state, reward, and done flag
        """
        pass

    def get_transition_prob(self, state: Tuple, action: int) -> Dict:
        """
        Get transition probabilities for all possible next states.
        
        Args:
            state: Current state
            action: Action to take
            
        Returns:
            Dictionary mapping next states to their transition probabilities
        """
        pass

    def is_terminal(self, state: Tuple) -> bool:
        """
        Check if a state is terminal (episode has ended).
        
        Args:
            state: State to check
            
        Returns:
            True if state is terminal, False otherwise
        """
        pass

    def get_reward(self, state: Tuple) -> Union[int, float]:
        """
        Get the reward for being in a state.
        
        Args:
            state: State to evaluate
            
        Returns:
            Reward value (typically 0 for non-terminal states)
        """
        pass

    def display_state(self) -> None:
        """
        Display the current state of the environment.
        
        This method should provide a human-readable representation
        of the current state.
        """
        pass


BlackjackState = Tuple[int, int, int]
BlackjackAction = int
BlackjackPolicy = Dict[BlackjackState, BlackjackAction]

class Blackjack(MDP):
    """
    Blackjack MDP implementation using Gymnasium environment.
    
    This class models the game of Blackjack as a Markov Decision Process,
    where states are represented as (player_total, dealer_card, usable_ace)
    and actions are hit (1) or stand (0).
    """
    def __init__(self, env: gym.Env) -> None:
        """
        Initialize the Blackjack MDP.
        
        Args:
            env: Gymnasium Blackjack environment
        """
        super().__init__(env)
        self.env = env
        self.current_state: Optional[BlackjackState] = self.reset()

    def get_all_states(self) -> List[BlackjackState]:
        """
        Generate all possible states in the Blackjack state space.
        
        Returns:
            List of all possible BlackjackState tuples:
            - player total: 4-21 (3 is impossible with standard rules)
            - dealer card: 1-10 (Ace represented as 1)
            - usable ace: 0 or 1 (boolean flag)
        """
        states = [(player, dealer, ace) for player in range(4, 22) for dealer in range(1, 11) for ace in range(2)]
        return states

    def get_all_actions(self) -> Tuple[int, int]:
        """
        Get all possible actions in Blackjack.
        
        Returns:
            Tuple of actions: (0=stand, 1=hit)
        """
        return (0, 1)

    def reset(self) -> BlackjackState:
        """
        Resets the MDP to a starting state. Sets the current state to this state.

        Returns:
            self.current_state: The current state after resetting the environment.
        """
        self.current_state, _ = self.env.reset() # Ignores the info dict from Gymnasium's reset
        return self.current_state

    def step(self, action: BlackjackAction) -> Tuple[BlackjackState, float, bool]:
        """
        Performs an action which transitions from one state to another.

        Parameters:
            action (BlackjackAction): The action taken in the environment.

        Returns:
            next_state (BlackjackState): The state of the environment after taking the action.
            reward (float): Reward received after taking the action.
            done (bool): Indicates if the game is over.
        """
        next_state, reward, done, _, _ = self.env.step(action) # Ignores the truncated and info dict from Gymnasium's step
        self.current_state = next_state if not done else None
        return next_state, reward, done

    def get_transition_prob(self, state: BlackjackState, action: BlackjackAction) -> Dict[BlackjackState, float]:
        """
        Calculates every path and gets the cumulative probability of reaching next_state from state with action.

        Parameters:
            state (BlackjackState): The current state of the environment.
            action (BlackjackState): The action selected by the agent.
        
        Returns:
            prob (float): The probability of reaching next_state from state using action. 
        """
        def get_hit_transition_prob(state: BlackjackState) -> Dict[BlackjackState, float]:
            """
            Calculate all possible successor states when the player hits (draws a card).
            
            This function determines the probabilities of all possible next states
            when the player draws one additional card. It handles the special case
            where a usable ace can be converted from 11 to 1 to avoid busting.
            
            Card probabilities (assuming infinite deck):
            - Cards 2-9: 1/13 each (4 cards per value)
            - Card 10: 4/13 (10, J, Q, K all count as 10)
            - Ace: 1/13 (1 card, counts as 11 when drawn)
            
            Args:
                state: Current BlackjackState (player_total, dealer_card, usable_ace)
            
            Returns:
                Dictionary mapping possible next states to their probabilities
            """
            cards = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # 11 represents Ace (initially counted as 11)
            probs = [1/13] * 8 + [4/13, 1/13]  # 10-value cards have 4x probability
            successor_states: DefaultDict[BlackjackState, float] = defaultdict(lambda: 0.0)
            
            for c, p in zip(cards, probs):
                new_total = state[0] + c
                
                # Handle usable ace conversion if player would bust
                if new_total > 21 and state[2] == 1:  # Would bust but have usable ace
                    new_state = (new_total - 10, state[1], 0)  # Convert ace 11→1, mark as no longer usable
                elif c == 11 and state[0] + 11 <= 21:  # Drawing an ace that can be used as 11
                    new_state = (new_total, state[1], 1)  # Mark ace as usable
                else:
                    new_state = (new_total, state[1], state[2])  # Normal addition, preserve ace status
                    
                successor_states[new_state] += p
            
            return dict(successor_states)

        def get_stay_transition_prob(state: BlackjackState) -> Dict[BlackjackState, float]:
            """
            Calculate all possible successor states when the player stays (dealer's turn).
            
            This function recursively calculates the probabilities of all possible end states
            when the dealer draws cards until reaching a score of 17 or higher.
            
            The dealer follows a fixed strategy:
            - Must hit if total < 17
            - Must stand if total >= 17
            - Does not consider usable aces optimally (simplified rule)
            
            Args:
                state: Current BlackjackState (player_sum, dealer_sum, usable_ace)
                       Note: usable_ace refers to player's ace, not dealer's
            
            Returns:
                Dictionary mapping possible terminal states to their probabilities
            """
            if state[1] > 16:  # Dealer stands with probability 1.0
                return {state: 1.0}
            
            cards = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # Card values dealer can draw
            probs = [1/13] * 8 + [4/13, 1/13]  # Corresponding probabilities
            successor_states: DefaultDict[BlackjackState, float] = defaultdict(lambda: 0.0)
            
            for c, p in zip(cards, probs):
                # Calculate dealer's new total (simplified: no dealer ace optimization)
                new_dealer_total = state[1] + c
                next_state = (state[0], new_dealer_total, state[2])  # Player ace status unchanged
                
                if new_dealer_total > 16:  # Dealer must stand after this card
                    successor_states[next_state] += p
                else:  # Dealer must continue hitting
                    downstream_states = get_stay_transition_prob(next_state)
                    for s, p_downstream in downstream_states.items():
                        successor_states[s] += p * p_downstream  # Chain rule of probability
            
            return dict(successor_states)
    
        if action == 0:
            transition_dict = get_stay_transition_prob(state)
        elif action == 1:
            transition_dict = get_hit_transition_prob(state)
        else:
            raise ValueError("Invalid action. Must be 0 (Stay) or 1 (Hit).")
        return transition_dict

    def display_state(self) -> None:
        """
        Displays the current state of the game.
        """
        if self.current_state is not None:
            player_total, dealer_card, usable_ace = self.current_state
            ace_status = 'with usable ace' if usable_ace else 'without usable ace'
            print(f"Player's Total: {player_total} {ace_status}, Dealer's Card: {dealer_card}")
        else:
            print("Game has ended or not started.")

    def run_manual_round(self) -> None:
        """
        Plays a manual round to choose actions yourself.
        """
        try:
            while True:
                self.display_state()
                action = input("Choose action (0: Stick, 1: Hit): ").strip()
                if action not in ['0', '1']:
                    print("Invalid action. Please enter 0 or 1.")
                    continue
                _, reward, done = self.step(int(action))
                if done:
                    # self.display_state()
                    print(f"Game over. Reward: {reward}")
                    break
        except KeyboardInterrupt:
            print("Game interrupted.")

    def is_terminal(self, state: BlackjackState) -> bool:
        """
        Determine if a given state is terminal (game ends).
        
        A state is terminal when:
        - Player busts (total > 21), or
        - Dealer has 17 or more (dealer must stand)
        
        Args:
            state: BlackjackState tuple to evaluate
            
        Returns:
            True if the game is over, False otherwise
        """
        player_total, dealer_sum, _ = state
        done = player_total > 21 or dealer_sum > 16  # player busted or dealer has 17 or over (automatically ends)
        return done

    def get_reward(self, state: BlackjackState) -> int:
        """
        Calculate the reward for reaching a given state.
        
        Reward structure:
        - +1: Player wins (dealer busts or player has higher total ≤ 21)
        - -1: Player loses (player busts or dealer has higher total ≤ 21)
        -  0: Tie (both have same total ≤ 21)
        
        Args:
            state: BlackjackState tuple to evaluate
            
        Returns:
            Integer reward: +1 (win), -1 (loss), or 0 (tie)
        """
        player_sum, dealer_sum, _ = state
        if player_sum > 21:  # player busts
            return -1
        elif dealer_sum > 21:  # dealer busts
            return 1
        else:
            if player_sum < dealer_sum:  # dealer has higher score
                return -1
            elif player_sum > dealer_sum:  # player has higher score
                return 1
            else:  # dealer and player tie
                return 0