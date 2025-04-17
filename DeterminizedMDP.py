from MDP import MDP
from collections import defaultdict
from GridWorldClass import *
from Utils import ValueIteration, vectorized_value_iteration


class DeterminizedMDP(MDP):
    def __init__(self, mdp, policy=None):
        self.mdp = mdp
        self.state_space = mdp.get_state_space()
        self.original_actions = mdp.get_actions()
        self.actions = []
        self.transition_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.policy = policy
        if self.policy is None:
            self.create_determinized_transition_matrix()
        else:
            self.create_determinized_transition_matrix_with_policy()
        self.init_state = self.get_state_hash(mdp.get_init_state())
        self.reward_func = mdp.reward_func
        self.original_reward_func = mdp.reward_func
        self.goal_states = [self.get_state_hash(state) for state in mdp.get_goal_states()]
        self.discount = mdp.discount
        self.bottleneck_states = []
    def create_determinized_transition_matrix(self):
        all_states = self.state_space
        action_prefix = 'act_'
        act_count = 0
        for state in all_states:
            current_act_cnt = 0
            state_hash = self.get_state_hash(state)
            for act in self.original_actions:
                for next_state in all_states:
                    next_state_hash = self.get_state_hash(next_state)
                    if self.mdp.get_transition_probability(state, act, next_state) > 0:
                        self.transition_dict[state_hash][action_prefix + str(current_act_cnt)][next_state_hash] = 1
                        current_act_cnt += 1
            if current_act_cnt > act_count:
                act_count = current_act_cnt
        self.actions = [action_prefix + str(i) for i in range(act_count)]

    def create_determinized_transition_matrix_with_policy(self):
        all_states = self.state_space
        action_prefix = 'act_'
        act_count = 0
        for state in all_states:
            current_act_cnt = 0
            state_hash = self.get_state_hash(state)
            act = self.policy[state_hash]
            for next_state in all_states:
                next_state_hash = self.get_state_hash(next_state)
                if self.mdp.get_transition_probability(state, act, next_state) > 0:
                    self.transition_dict[state_hash][action_prefix + str(current_act_cnt)][next_state_hash] = 1
                    current_act_cnt += 1
            if current_act_cnt > act_count:
                act_count = current_act_cnt
        self.actions = [action_prefix + str(i) for i in range(act_count)]
    def get_state_hash(self, state):
        """Get a unique hash for a state.
        
        Args:
            state: The state to hash. Can be a string or any other type.
            
        Returns:
            str: A string representation of the state that can be used as a hash.
        """
        if isinstance(state, str):
            return state
        return str(state)
    def get_actions(self):
        return self.actions

    def get_transition_probability(self, state, action, state_prime):
        state_hash = self.get_state_hash(state)
        state_prime_hash = self.get_state_hash(state_prime)
        #print(state_hash, action, state_prime_hash, self.transition_dict[state_hash][action][state_prime_hash])
        return self.transition_dict[state_hash][action][state_prime_hash]

    def get_init_state(self):
        return self.init_state

    def get_reward(self, state, action, next_state):
        return self.reward_func(state, action, next_state)

    def get_state_space(self):
        return self.state_space

    def bottleneck_reward(self, state, action, next_state):
        if self.get_state_hash(next_state) == self.get_state_hash(self.bottleneck_state):
            return -1000
        if self.get_state_hash(next_state) in self.goal_states and self.get_state_hash(state) not in self.goal_states:
            return 1
        return 0

    def reward_function_for_avoiding_all_bottleneck(self, state, action, next_state):
        return -1* self.bottleneck_MDP.get_reward(state, action, next_state)

    def reward_function_for_goingthrough_all_bottleneck(self, state, action, next_state):
        #print(state[0], self.bottleneck_states)
        #print(state[0] in self.bottleneck_states)

        #rew = self.bottleneck_MDP.get_reward(state, action, next_state)
        if next_state[0] in self.bottleneck_states:
            return 1000
        return 0


# if __name__ == "__main__":
#     grid = GridWorld(start=(0,0), goal=(4, 4), divide_rooms=True)
#     grid.visualize()
#     det_mdp = DeterminizedMDP(grid)
#     print(det_mdp.get_actions())
#     # print(det_mdp.get_transition_probability(((0,0),), 'act_0', ((0,0),)))
#     # print(det_mdp.get_reward(((0,0),), 'act_0', ((0,0),)))
#     print(det_mdp.get_init_state())
#     init_state_hash = det_mdp.get_state_hash(det_mdp.get_init_state())
#     det_mdp.reward_func = det_mdp.bottleneck_reward
#     bottleneck_list = []
#     for state in det_mdp.get_state_space():
#         det_mdp.bottleneck_state = det_mdp.get_state_hash(state)
#         V = ValueIteration(det_mdp)
#         if V[init_state_hash] <= 0:
#             bottleneck_list.append(state)
#     print(bottleneck_list)

def identify_bottlenecks(M):
    det_mdp = DeterminizedMDP(M)
    init_state_hash = det_mdp.get_state_hash(det_mdp.get_init_state())
    det_mdp.reward_func = det_mdp.bottleneck_reward
    bottlenecks = []

    for state in det_mdp.get_state_space():
        det_mdp.bottleneck_state = det_mdp.get_state_hash(state)
        V = vectorized_value_iteration(det_mdp)
        if V[init_state_hash] <= 0:
            bottlenecks.append(tuple(state))  # Convert state to tuple
            print(f"Identified bottleneck state: {tuple(state)}")
        #else:
        #    print(f"Non-bottleneck state: {tuple(state)}", V[init_state_hash])

    return bottlenecks


if __name__ == "__main__":
    grid = GridWorld(start=(0,0), goal=(4, 4), divide_rooms=True)
    visualize_grid(grid)
    det_mdp = DeterminizedMDP(grid)
    print("Actions:", det_mdp.get_actions())
    print("Initial state:", det_mdp.get_init_state())

    # identify bottleneck states
    det_mdp.reward_func = det_mdp.bottleneck_reward
    bottleneck_list = []
    for state in det_mdp.get_state_space():
        det_mdp.bottleneck_state = det_mdp.get_state_hash(state)
        V = vectorized_value_iteration(det_mdp)
        if V[det_mdp.get_init_state()] <= 0:
            bottleneck_list.append(state)
    print("Bottleneck states:", bottleneck_list)

    grid2 = GridWorld(size=8, start=(0,0), goal=(7,7), divide_rooms=True)
    visualize_grid(grid2)
    det_mdp2 = DeterminizedMDP(grid2)
    print("Actions:", det_mdp2.get_actions())
    print("Initial state:", det_mdp2.get_init_state())

    # identify bottleneck states for larger grid
    det_mdp2.reward_func = det_mdp2.bottleneck_reward
    bottleneck_list2 = []
    for state in det_mdp2.get_state_space():
        det_mdp2.bottleneck_state = det_mdp2.get_state_hash(state)
        V = vectorized_value_iteration(det_mdp2)
        if V[det_mdp2.get_init_state()] <= 0:
            bottleneck_list2.append(state)
    print("Bottleneck states:", bottleneck_list2)

    # verify determinization
    original_transition = grid.get_transition_probability(((0,0),), 'up', ((0,0),))
    det_transition = det_mdp.get_transition_probability(((0,0),), 'act_0', ((0,0),))
    print(f"Original transition probability: {original_transition}")
    print(f"Determinized transition probability: {det_transition}")