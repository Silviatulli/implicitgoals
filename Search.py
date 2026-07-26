from queue import Queue

# def BFSearch(startState, goal_test, successor_generator):
#     fringe = Queue()
#     closed = set()
#     numberOfNodesExpanded = 0

#     fringe.put((startState,[]))

#     while not fringe.empty():
#         node = fringe.get()
#         goal_check = goal_test(node)

#         if goal_check:
#             return node

#         if frozenset(node[0]) not in closed:

#             closed.add(frozenset(node[0]))

#             successor_list = successor_generator(node)

#             numberOfNodesExpanded += 1

#             if not numberOfNodesExpanded % 100:
#                 print ("Number of Nodes Expanded =", numberOfNodesExpanded)

#             while successor_list:
#                 candidate_node = successor_list.pop()
#                 new_node = [candidate_node[0], node[1] + [candidate_node[1]]]
#                 fringe.put((new_node[0], new_node[1]))
#     return None



def BFSearch(startState, goal_test, successor_generator):
    fringe = Queue()
    closed = set()
    
    fringe.put((startState, []))
    # print(f"Starting search from {startState}")
    
    while not fringe.empty():
        state, path = fringe.get()
        # print(f"Examining state: {state}")
        
        if goal_test(state):
            # print(f"Goal reached at {state}")
            return path

        state_hash = hash(tuple(state))  # Create a hashable version of the state
        if state_hash not in closed:
            closed.add(state_hash)
            
            for next_state, action in successor_generator(state):
                # print(f"  Neighbor: {next_state}, Action: {action}")
                next_state_hash = hash(tuple(next_state))
                if next_state_hash not in closed:
                    new_path = path + [action]
                    fringe.put((next_state, new_path))
    
    return None  