from typing import List, Tuple, Any, Set
from itertools import combinations
from Utils import vectorized_value_iteration, get_policy, sparse_value_iteration, get_sparse_policy
import time
import random
import sys
import threading
import queue
import logging

class QueryMDP:
    def __init__(self, robot_mdp: Any, bottlenecks: List[Any], achievable_subsets: List[Any]):
        self.robot_mdp = robot_mdp
        self.achievable_subsets = [set(robot_mdp.get_state_hash(state) for state in achievable_subset)
                                   for achievable_subset in achievable_subsets]
        self.union_achievable_subsets = set().union(*self.achievable_subsets) if self.achievable_subsets else set()

        #print("Union Achievable Subsets:", self.union_achievable_subsets)

        self.bottleneck_hash = set(robot_mdp.get_state_hash(state) for state in bottlenecks)
        self.unachievable_bottlenecks = self.bottleneck_hash - self.union_achievable_subsets
        self.bottleneck_hash_map = {robot_mdp.get_state_hash(state): state for state in bottlenecks}

        self.state_space = []
        self.action_space = []
        
        self.create_state_space()
        #logging.info("State Space: %s", self.state_space)

        self.create_action_space()
        
        self.start_state = (frozenset(), frozenset())
        self.discount = robot_mdp.discount
        print_lock = threading.Lock()

        with print_lock:
            pass
            #logging.info("State Space: %s", self.state_space)

    def create_state_space(self):
        self.state_space = []
        bottleneck_list = list(self.bottleneck_hash)
        if not bottleneck_list:
            self.state_space = [(frozenset(), frozenset())]
            return
            
        for i in range(len(bottleneck_list) + 1):
            for subgoal_combo in combinations(bottleneck_list, i):
                subgoal_set = frozenset(subgoal_combo)
                remaining_bottlenecks = self.bottleneck_hash - set(subgoal_combo)
                
                for j in range(len(remaining_bottlenecks) + 1):
                    for non_subgoal_combo in combinations(remaining_bottlenecks, j):
                        non_subgoal_set = frozenset(non_subgoal_combo)
                        self.state_space.append((subgoal_set, non_subgoal_set))

    def create_action_space(self):
        self.action_space = ['Query_' + state for state in self.bottleneck_hash]

    def get_state_space(self):
        return self.state_space

    def get_actions(self, state=None):
        if state is None:
            return self.action_space
        else:
            subgoal, not_subgoal = map(frozenset, state)
            return ['Query_' + b for b in self.bottleneck_hash 
                   if b not in subgoal and b not in not_subgoal]

    def check_terminal_state(self, state: Tuple[Set[Any], Set[Any]]) -> bool:
        subgoal, not_subgoal = map(frozenset, state)
        
        # Check if we've found necessary unachievable bottlenecks
        necessary_unachievable = self.unachievable_bottlenecks.intersection(subgoal)
        if necessary_unachievable:
            return True
        
        # Check if we've classified all bottlenecks
        all_classified = subgoal | not_subgoal
        return all_classified == self.bottleneck_hash

    def get_transition_probability(self, state: Tuple[Set[Any], Set[Any]], action: str, 
                                 next_state: Tuple[Set[Any], Set[Any]]) -> float:
        subgoal, not_subgoal = map(frozenset, state)
        next_subgoal, next_non_subgoal = map(frozenset, next_state)
        query_state = action.split('_')[-1]

        if self.check_terminal_state(state):
            return 1.0 if state == next_state else 0.0

        if query_state in subgoal or query_state in not_subgoal:
            return 1.0 if state == next_state else 0.0

        possible_next_states = [
            (frozenset(subgoal | {query_state}), not_subgoal),
            (subgoal, frozenset(not_subgoal | {query_state}))
        ]

        return 0.5 if next_state in possible_next_states else 0.0

    def get_init_state(self):
        return self.start_state

    def get_state_hash(self, state: Tuple[Set[Any], Set[Any]]) -> str:
        subgoal, not_subgoal = map(frozenset, state)
        return f"{sorted(subgoal)}-{sorted(not_subgoal)}"

    def get_reward(self, state: Tuple[Set[Any], Set[Any]], action: str, 
                next_state: Tuple[Set[Any], Set[Any]]) -> float:
        subgoal, not_subgoal = map(frozenset, state)
        next_subgoal, next_non_subgoal = map(frozenset, next_state)
        
        # Reward for discovering new information
        if len(next_subgoal | next_non_subgoal) > len(subgoal | not_subgoal):
            return 10
            
        # Reward for reaching terminal state
        if self.check_terminal_state(next_state):
            return 100
            
        # Base query cost
        return -1

def simulate_policy_unachievable(query_mdp: QueryMDP, human_bottlenecks: List[Any], query_threshold: int = 1000) -> int:
    import random
    human_bottleneck_hash = frozenset(query_mdp.robot_mdp.get_state_hash(state) for state in human_bottlenecks)
    query_count = 0
    confirmed_subgoals = set()
    confirmed_non_subgoals = set()
    
    # First check all unachievable bottlenecks
    for unachievable in query_mdp.unachievable_bottlenecks:
        if query_count >= query_threshold:
            return query_threshold
            
        query_count += 1
        is_subgoal = unachievable in human_bottleneck_hash
        if is_subgoal:
            query_count += 1
            is_necessary = random.choice([True, False])
            if is_necessary:
                confirmed_subgoals.add(unachievable)
                return query_count
            confirmed_non_subgoals.add(unachievable)
    
    # Then use policy for achievable bottlenecks
    V = sparse_value_iteration(query_mdp)
    policy = get_sparse_policy(query_mdp, V)
    remaining_bottlenecks = query_mdp.bottleneck_hash - query_mdp.unachievable_bottlenecks - confirmed_subgoals - confirmed_non_subgoals
    
    while remaining_bottlenecks and query_count < query_threshold:
        current_state = (frozenset(confirmed_subgoals), frozenset(confirmed_non_subgoals))
        action = policy[query_mdp.get_state_hash(current_state)]
        query_state = action.split('_')[-1]
        
        if query_state not in remaining_bottlenecks:
            continue
            
        query_count += 1
        is_subgoal = query_state in human_bottleneck_hash
        
        if is_subgoal:
            confirmed_subgoals.add(query_state)
        else:
            confirmed_non_subgoals.add(query_state)
            
        remaining_bottlenecks.remove(query_state)
        if query_mdp.check_terminal_state((frozenset(confirmed_subgoals), frozenset(confirmed_non_subgoals))):
            return query_count
            
    return query_count

def simulate_policy_query_all(query_mdp: QueryMDP, human_bottlenecks: List[Any], query_threshold: int = 1000) -> int:
    import random
    human_bottleneck_hash = frozenset(query_mdp.robot_mdp.get_state_hash(state) for state in human_bottlenecks)
    query_count = 0
    
    # Must query EVERY bottleneck regardless of the result
    bottlenecks = list(query_mdp.bottleneck_hash)
    #print(f"Total bottlenecks to query: {len(bottlenecks)}")
    
    for bottleneck in bottlenecks:
        if query_count >= query_threshold:
            return query_threshold
            
        # Always query if it's a subgoal
        query_count += 1
        is_subgoal = bottleneck in human_bottleneck_hash
        
        # If unachievable and subgoal, must query necessity
        if is_subgoal and bottleneck in query_mdp.unachievable_bottlenecks:
            query_count += 1
            is_necessary = random.choice([True, False])
    
    return query_count  # Should equal total number of bottlenecks + necessity queries

def test_query_mdp(size=5, obstacles_percent=0.1):
    """Test function for QueryMDP."""
    from GridWorldClass import generate_and_visualize_gridworld
    
    print("generating test environment...")
    M_R = generate_and_visualize_gridworld(
        size=size,
        start=(0,0),
        goal=(size-1,size-1),
        obstacles_percent=obstacles_percent,
        divide_rooms=True,
        model_type="Robot Model"
    )

    test_case = {
        'bottlenecks': [((1, 0), (), ()), ((2, 2), (), ())],
        'achievable_subsets': [[((1, 0), (), ())]],
        'true_bottlenecks': [((1, 0), (), ())]
    }

    query_mdp = QueryMDP(
        robot_mdp=M_R,
        bottlenecks=test_case['bottlenecks'],
        achievable_subsets=test_case['achievable_subsets']
    )

    return simulate_policy_unachievable(query_mdp, test_case['true_bottlenecks'])

if __name__ == "__main__":
    test_query_mdp(size=5, obstacles_percent=0.1)
