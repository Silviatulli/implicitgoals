from MDP import MDP
from GridWorldClass import GridWorld, visualize_grid
from Utils import ValueIteration, vectorized_value_iteration, get_policy, get_robust_policy, robust_vectorized_value_iteration
from BottleneckCheckMDP import BottleneckMDP
from collections import defaultdict
import numpy as np
from DeterminizedMDP import DeterminizedMDP
from collections import deque
from typing import Set, Tuple, List, FrozenSet
import functools
import time
import inspect  # Add this at the top of the file
from concurrent.futures import ProcessPoolExecutor

# Global counters for achievability checks
achievability_check_count_pruning = 0
achievability_check_count_no_pruning = 0

def optimized_find_maximally_achievable_subsets(M_R: GridWorld, M_H_list: List[GridWorld]) -> Tuple[Set[FrozenSet], Set[Tuple]]:
    # Identify all bottleneck states
    B = set()
    for M in M_H_list:
        B.update(tuple(b) for b in identify_bottlenecks(M))
    
    print(f"Total bottleneck states: {len(B)}")

    # Convert B to a list and create a bit mask for efficient subset generation
    B_list = list(B)
    n = len(B_list)

    @functools.lru_cache(maxsize=None)
    def check_achievability_cached(subset):
        return check_achievability(subset, M_R)

    def generate_subsets(index, current_subset):
        if index == n:
            if check_achievability_cached(frozenset(current_subset)):
                for i in range(len(current_subset), n):
                    new_subset = current_subset + (B_list[i],)
                    if not check_achievability_cached(frozenset(new_subset)):
                        yield frozenset(current_subset)
                        return
                yield frozenset(current_subset)
            return

        # Don't include this element
        yield from generate_subsets(index + 1, current_subset)

        # Include this element
        yield from generate_subsets(index + 1, current_subset + (B_list[index],))

    I = set(generate_subsets(0, ()))

    return I, B

def find_maximally_achievable_subsets(M_R, M_H_list):
    B = set()  # Set of all bottleneck states
    for M in M_H_list:
        bottlenecks = identify_bottlenecks(M)
        B.update(tuple(b) for b in bottlenecks)  # Convert each bottleneck to a tuple
    
    print(f"Total bottleneck states: {len(B)}")
    print("Bottleneck states:", B)

    I = set()  # Set of maximal achievable subgoal sets
    fringe = [frozenset(B)]  # Start with the set of all bottleneck states
    
    while fringe:
        I_prime = fringe.pop(0)
        print(f"\nChecking subset of size {len(I_prime)}: {I_prime}")
        if check_achievability(I_prime, M_R):
            print("Subset is achievable")
            # Check if I_prime is maximal
            if all(not check_achievability(I_prime | {s}, M_R) for s in B - I_prime):
                I.add(I_prime)
                print(f"Added maximal achievable subset: {I_prime}")
            else:
                print("Subset is not maximal")
        else:
            print("Subset is not achievable")
            for s in I_prime:
                new_subset = frozenset(I_prime - {s})
                if new_subset not in fringe and not any(new_subset.issubset(i) for i in I):
                    fringe.append(new_subset)
                    print(f"Added new subset to fringe: {new_subset}")
    
    return I, B

def identify_bottlenecks(M):
    det_mdp = DeterminizedMDP(M)
    init_state_hash = det_mdp.get_state_hash(det_mdp.get_init_state())
    det_mdp.reward_func = det_mdp.bottleneck_reward
    bottlenecks = []
    
    for state in det_mdp.get_state_space():
        if state != M.get_init_state() and state != M.get_goal_states()[0]:
            det_mdp.bottleneck_state = det_mdp.get_state_hash(state)
            V = robust_vectorized_value_iteration(det_mdp)
            if V[init_state_hash] <= 0:
                bottlenecks.append(tuple(state))  # Convert state to tuple
                print(f"Identified bottleneck state: {tuple(state)}")
    
    return bottlenecks

# def check_achievability(I_prime, M_R):
#     print("Making the object")
#     start_time = time.time()
#     print("Time to make the bottleneck object", time.time() - start_time)
#     M_R.reward_func = None
#     start_time = time.time()
#     det_mdp = DeterminizedMDP(M_R)
#     print("Time to make the determinized object", time.time() - start_time)

#     det_mdp.reward_func = det_mdp.reward_function_for_goingthrough_all_bottleneck
#     det_mdp.bottleneck_states = I_prime
#     print(I_prime)

#     start_time = time.time()
#     V_det = vectorized_value_iteration(det_mdp)
#     print("Time to run the value iteration", time.time() - start_time)
#     initial_state_hash = det_mdp.get_state_hash(det_mdp.get_init_state())
#     # print(policy)
#     #is_achievable = False
#     if V_det[initial_state_hash] <= (len(I_prime)-1)*1000:
#         print("Rejecting it because all of the bottlenecks are not achieved")
#         return False
#     M = BottleneckMDP(M_R, I_prime)
#     V = vectorized_value_iteration(M)
#     # for s in M.state_space:
#     #    print(s, V[M.get_state_hash(s)])
#     # exit(0)
#     policy = get_policy(M, V)

#     M.reward_func = None
#     # Determinized for the policy
#     det_mdp_for_policy = DeterminizedMDP(M, policy)
#     det_mdp_for_policy.bottleneck_MDP = M

#     det_mdp_for_policy.reward_func = det_mdp_for_policy.reward_function_for_avoiding_all_bottleneck

#     V = robust_vectorized_value_iteration(det_mdp_for_policy)
#     initial_state_hash = det_mdp_for_policy.get_state_hash(det_mdp_for_policy.get_init_state())
#     # print(policy)
#     is_achievable = False
#     if V[initial_state_hash] <= 0:
#         is_achievable = True
#         #print("Achievable")
#     return is_achievable


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

def improved_find_maximally_achievable_subsets(M_R, M_H_list):
    B = set()
    for M in M_H_list:
        B.update(tuple(b) for b in identify_bottlenecks(M))
    
    # Start with smaller promising subsets instead of full set
    fringe = []
    # Try individual states first
    for b in B:
        if check_achievability({b}, M_R):
            fringe.append(frozenset({b}))
    
    I = set()
    while fringe:
        I_prime = fringe.pop(0)
        # Try to grow achievable sets
        candidates = B - I_prime
        for c in candidates:
            new_set = I_prime | {c}
            if check_achievability(new_set, M_R):
                fringe.append(new_set)
            else:
                if all(not check_achievability(I_prime | {s}, M_R) for s in B - I_prime):
                    I.add(I_prime)
    return I, B


def binary_search_maximal_subset(current_set, candidates, M_R):
    if not candidates:
        return current_set
    
    mid = len(candidates) // 2
    test_set = current_set | set(list(candidates)[:mid])
    
    if check_achievability(test_set, M_R):
        # Try including more states
        return binary_search_maximal_subset(test_set, set(list(candidates)[mid:]), M_R)
    else:
        # Try with fewer states
        return binary_search_maximal_subset(current_set, set(list(candidates)[:mid]), M_R)


def find_maximally_achievable_subsets_no_pruning(M_R, M_H_list):
    """Exhaustive search version without pruning optimization."""
    B = set()
    for M in M_H_list:
        bottlenecks = identify_bottlenecks(M)
        B.update(tuple(b) for b in bottlenecks)
    
    I = set()
    n = len(B)
    B_list = list(B)
    
    for i in range(2**n):
        subset = frozenset(B_list[j] for j in range(n) if (i & (1 << j)))
        if check_achievability(subset, M_R):
            is_maximal = True
            for additional_state in B - subset:
                larger_subset = subset | {additional_state}
                if check_achievability(larger_subset, M_R):
                    is_maximal = False
                    break
            if is_maximal:
                I.add(subset)
    
    return I, B


if __name__ == "__main__":
    from GridWorldClass import generate_and_visualize_gridworld

    # Generate robot model
    M_R = generate_and_visualize_gridworld(size=4, start=(0,0), goal=(3,3), obstacles_percent=0.1, divide_rooms=True, model_type="Robot Model")

    # Generate human models with different configurations
    M_H_list = []
    human_configs = [
        {"obstacles_percent": 0.1, "divide_rooms": True},
        {"obstacles_percent": 0.1, "divide_rooms": True},
        {"obstacles_percent": 0.1, "divide_rooms": False}
    ]


    for i, config in enumerate(human_configs):
        M_H = generate_and_visualize_gridworld(size=4, start=(0,0), goal=(3,3),
                                               obstacles_percent=config["obstacles_percent"],
                                               divide_rooms=config["divide_rooms"],
                                               model_type=f"Human Model {i+1}")
        if M_H:
            M_H_list.append(M_H)

    # Proceed with analysis only if we have valid models
    if M_R and M_H_list:
        print("\nRobot Model Initial State:", M_R.get_init_state())
        print("\nActions available in Human Model 1:", M_H_list[0].get_actions())

        I, B = optimized_find_maximally_achievable_subsets(M_R, M_H_list)
        print("\nMaximally achievable subsets of bottleneck states:")
        for subset in I:
            print(subset)
        print("\nAll bottleneck states:", B)
    else:
        print("Could not generate valid models for analysis.")
