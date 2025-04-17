from Utils import powerset, ValueIteration, vectorized_value_iteration, get_policy
from DeterminizedMDP import DeterminizedMDP, identify_bottlenecks
import random
from Utils import robust_vectorized_value_iteration
import inspect

# global counters for achievability checks
achievability_check_count_pruning = 0
achievability_check_count_no_pruning = 0

class BottleneckMDP(object):
    def __init__(self, mdp, bottlenecks):
        self.original_mdp = mdp
        self.bottlenecks = bottlenecks
        self.create_state_space()
        self.actions = mdp.get_actions()
        self.init_state = (mdp.get_init_state(),set())
        self.discount = mdp.discount

    def create_state_space(self):
        possible_bottleneck_sets = powerset(self.bottlenecks)
        self.state_space = []
        for I in possible_bottleneck_sets:
            #if len(I)>0:
            #    print(list(I[0]))
            #    print(type(I))
            #I_set = set(I)
            for J in self.original_mdp.get_state_space():
                #print(tuple(J))
                #print(type(J))
                #exit(0)
                self.state_space.append((J, set(I)))
        #exit(0)

    def get_state_space(self):
        return self.state_space

    def get_actions(self):
        return self.actions

    def get_transition_probability(self, state_tuple, action, next_state_tuple):
        state, subgoals = state_tuple
        next_state, next_subgoals = next_state_tuple

        current_transition_prob = self.original_mdp.get_transition_probability(state, action, next_state)
        if current_transition_prob == 0:
            return 0
        # You can't add more than one subgoal at a time
        if len(next_subgoals - subgoals) > 1:
            return 0
        # You can't ever remove a subgoal
        if len(subgoals - next_subgoals) > 0:
            return 0

        added_subgoal = list(next_subgoals - subgoals)
        if len(added_subgoal) > 0 and added_subgoal[0] != tuple(next_state):
            return 0

        if tuple(next_state) in self.bottlenecks and tuple(next_state) not in subgoals:
            return 0

        # # Add absorbing states for goal states
        # if self.original_mdp.check_goal_reached(state[0]):
        #     if state == next_state:
        #         return 1
        #     else:
        #         return 0

        return current_transition_prob

    def get_init_state(self):
        return self.init_state

    def get_state_hash(self, state):
        return str(state)

    def get_goal_states(self):
        return self.original_mdp.get_goal_states()

    def get_reward(self, state_tuple, action, next_state_tuple):
        next_state, next_subgoals = next_state_tuple
        if not self.original_mdp.check_goal_reached(next_state[0]):
            return 0

        if len(next_subgoals) == len(self.bottlenecks):
            return 1000
        #print("Missed bottleneck", state_tuple, next_state_tuple)
        return -1000

def check_achievability(I_prime, M_R):
    """Check if a set of states is achievable in the given MDP."""
    global achievability_check_count_pruning
    global achievability_check_count_no_pruning
    
    # Identify which function called check_achievability
    caller = inspect.currentframe().f_back.f_code.co_name
    if caller == 'find_maximally_achievable_subsets':
        achievability_check_count_pruning += 1
    elif caller == 'find_maximally_achievable_subsets_no_pruning':
        achievability_check_count_no_pruning += 1

    # Original check_achievability implementation
    M_R.reward_func = None
    det_mdp = DeterminizedMDP(M_R)
    det_mdp.reward_func = det_mdp.reward_function_for_goingthrough_all_bottleneck
    det_mdp.bottleneck_states = I_prime

    V_det = vectorized_value_iteration(det_mdp)
    initial_state_hash = det_mdp.get_state_hash(det_mdp.get_init_state())

    if V_det[initial_state_hash] <= (len(I_prime)-1)*1000:
        return False

    M = BottleneckMDP(M_R, I_prime)
    V = vectorized_value_iteration(M)
    policy = get_policy(M, V)

    M.reward_func = None
    det_mdp_for_policy = DeterminizedMDP(M, policy)
    det_mdp_for_policy.bottleneck_MDP = M
    det_mdp_for_policy.reward_func = det_mdp_for_policy.reward_function_for_avoiding_all_bottleneck

    V = robust_vectorized_value_iteration(det_mdp_for_policy)
    initial_state_hash = det_mdp_for_policy.get_state_hash(det_mdp_for_policy.get_init_state())

    return V[initial_state_hash] <= 0

if __name__ == "__main__":
    # from GridWorldClass import generate_and_visualize_gridworld
    # M_R = generate_and_visualize_gridworld(size=5, start=(0,0), goal=(4,4), obstacles_percent=0.1, divide_rooms=True, model_type="Robot Model")
    # bottleneck_states = identify_bottlenecks(M_R)
    # M = BottleneckMDP(M_R, bottleneck_states)
    # V = vectorized_value_iteration(M)
    # #for s in M.state_space:
    # #    print(s, V[M.get_state_hash(s)])
    # #exit(0)
    # policy = get_policy(M, V)
    #
    # M.reward_func = None
    # # Determinized for the policy
    # det_mdp_for_policy = DeterminizedMDP(M, policy)
    # det_mdp_for_policy.bottleneck_MDP = M
    #
    #
    # det_mdp_for_policy.reward_func = det_mdp_for_policy.reward_function_for_avoiding_all_bottleneck
    #
    # #print(det_mdp_for_policy.get_init_state())
    # V = vectorized_value_iteration(det_mdp_for_policy)
    # initial_state_hash = det_mdp_for_policy.get_state_hash(det_mdp_for_policy.get_init_state())
    #
    # if V[initial_state_hash] <=0:
    #     print("Achievable")
    #
    # #for s in det_mdp_for_policy.state_space:
    # #    print(s, V[M.get_state_hash(s)])
    # # exit(0)
    # #print(V[initial_state_hash])
    #
    # #print(V)




    from GridWorldClass import generate_and_visualize_gridworld, visualize_grids_with_bottlenecks
    from DeterminizedMDP import identify_bottlenecks

    # Generate Robot Model
    M_R = generate_and_visualize_gridworld(size=5, start=(0, 0), goal=(4, 4), obstacles_percent=0.1, divide_rooms=True,
                                           model_type="Robot Model", obstacle_seed=42)
    bottleneck_states_robot = identify_bottlenecks(M_R)
    # Filter out goal state from bottlenecks
    bottleneck_states_robot = [b for b in bottleneck_states_robot if b[0] != M_R.goal_pos]

    # Generate multiple Human Models
    num_human_models = 3
    human_models = []
    human_bottlenecks_list = []
    achievable_bottlenecks_list = []

    for i in range(num_human_models):
        M_H = generate_and_visualize_gridworld(size=5, start=(0, 0), goal=(4, 4), obstacles_percent=0.1,
                                               divide_rooms=True, model_type=f"Human Model {i + 1}",
                                               obstacle_seed=random.randint(1, 10000))
        bottleneck_states_human = identify_bottlenecks(M_H)
        # Filter out goal state from bottlenecks
        bottleneck_states_human = [b for b in bottleneck_states_human if b[0] != M_H.goal_pos]
        achievable_human_bottlenecks = check_achievability(M_R, bottleneck_states_robot,
                                                                      bottleneck_states_human)

        human_models.append(M_H)
        human_bottlenecks_list.append(bottleneck_states_human)
        achievable_bottlenecks_list.append(achievable_human_bottlenecks)

    # Visualize the comparison
    visualize_grids_with_bottlenecks(M_R, human_models, bottleneck_states_robot, human_bottlenecks_list,
                                     achievable_bottlenecks_list)

    print("\nRobot's bottleneck states:", bottleneck_states_robot)
    for i, (bottlenecks, achievable) in enumerate(zip(human_bottlenecks_list, achievable_bottlenecks_list)):
        print(f"\nHuman Model {i + 1}:")
        print("Bottleneck states:", bottlenecks)
        print("Achievable bottlenecks:", achievable)
        print("Unachievable bottlenecks:", [b for b in bottlenecks if b not in achievable])

    M = BottleneckMDP(M_R, achievable)
    V = vectorized_value_iteration(M)
    # for s in M.state_space:
    #    print(s, V[M.get_state_hash(s)])
    # exit(0)
    policy = get_policy(M, V)

    M.reward_func = None
    # Determinized for the policy
    det_mdp_for_policy = DeterminizedMDP(M, policy)
    det_mdp_for_policy.bottleneck_MDP = M

    det_mdp_for_policy.reward_func = det_mdp_for_policy.reward_function_for_avoiding_all_bottleneck

    V = vectorized_value_iteration(det_mdp_for_policy)
    initial_state_hash = det_mdp_for_policy.get_state_hash(det_mdp_for_policy.get_init_state())
    # print(policy)
    if V[initial_state_hash] <= 0:
        print("Achievable")


