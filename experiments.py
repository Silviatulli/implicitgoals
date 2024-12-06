from PuddleWorldClass import PuddleWorld
from RockWorldClass import RockWorld
from TaxiWorldClass import TaxiWorld
from MinigridWorldClass import UnlockEnv, UnlockPickupEnv
from GridWorldClass import generate_and_visualize_gridworld
from BottleneckCheckMDP import BottleneckMDP
from QueryMDP import QueryMDP, simulate_policy_unachievable, simulate_policy_query_all
from Utils import vectorized_value_iteration, get_policy, sparse_value_iteration, get_sparse_policy #, debug_value_iteration
from maximal_achievable_subsets import find_maximally_achievable_subsets, optimized_find_maximally_achievable_subsets, identify_bottlenecks
import numpy as np
import time
import random
from DeterminizedMDP import DeterminizedMDP
import pandas as pd


def visualize_taxiworld(taxi_world):
    print(f"Grid Size: {taxi_world.size}x{taxi_world.size}")
    print(f"Start: {taxi_world.start_pos}")
    print(f"Passengers: {taxi_world.passenger_locs}")
    print(f"Destinations: {taxi_world.destinations}")
    
    for i in range(taxi_world.size):
        for j in range(taxi_world.size):
            if (i, j) == taxi_world.start_pos:
                print("S", end=" ")
            elif (i, j) in taxi_world.passenger_locs:
                print(f"P{taxi_world.passenger_locs.index((i,j))+1}", end=" ")
            elif (i, j) in taxi_world.destinations:
                print(f"D{taxi_world.destinations.index((i,j))+1}", end=" ")
            elif taxi_world.map[i, j] == -1:
                print("█", end=" ")
            else:
                print(".", end=" ")
        print()


def generate_and_visualize_puddleworld(size, start, goal, obstacles_percent, puddle_percent, model_type, obstacle_seed):
    return PuddleWorld(size=size, start=start, goal=goal, 
                       obstacles_percent=obstacles_percent,
                       puddle_percent=puddle_percent,
                       obstacle_seed=obstacle_seed)

def generate_and_visualize_rockworld(size, start, goal, obstacles_percent, rock_percent, model_type, obstacle_seed):
    return RockWorld(size=size, start=start, goal=goal, 
                     obstacles_percent=obstacles_percent,
                     rock_percent=rock_percent,
                     obstacle_seed=obstacle_seed)


def generate_and_visualize_taxiworld(size, start, goal, obstacles_percent, model_type, obstacle_seed):
    # Generate random passenger locations and destinations
    num_passengers = 2  # You can adjust this as needed
    passenger_locs = [(random.randint(0, size-1), random.randint(0, size-1)) for _ in range(num_passengers)]
    destinations = [(random.randint(0, size-1), random.randint(0, size-1)) for _ in range(num_passengers)]
    
    # Create the TaxiWorld instance
    taxi_world = TaxiWorld(
        size=size,
        num_passengers=num_passengers,
        start=start,
        passenger_locs=passenger_locs,
        destinations=destinations,
        obstacles_percent=obstacles_percent,
        obstacle_seed=obstacle_seed
    )
    
    # visualize the world (optional)
    print(f"\n{model_type}:")
    visualize_taxiworld(taxi_world)
    
    return taxi_world

def generate_and_visualize_minigridworld(env_type, size, model_type):
    if env_type == 'unlock':
        return UnlockEnv(size=size)
    elif env_type == 'unlock_pickup':
        return UnlockPickupEnv(size=size)
    else:
        raise ValueError(f"Unknown Minigrid environment type: {env_type}")

def run_experiments(num_runs, num_models, grid_size, world_type, query_threshold):
    results = {
        "query_counts": [],
        "human_bottlenecks": [],
        "maximal_subset_times": [],
        "query_times": [],
        "query_mdp_state_space_sizes": [],
        "query_mdp_action_space_sizes": [],
        "initial_mdp_state_space_sizes": [],
        "initial_mdp_action_space_sizes": []
    }

    for run in range(num_runs):
        print(f"\nStarting run {run + 1}/{num_runs}")
        start_time_total = time.time()

        print("Generating robot model...")
        if world_type == 'taxi':
            M_R = generate_and_visualize_taxiworld(size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                                                   obstacles_percent=0.1, 
                                                   model_type="Robot Model", obstacle_seed=random.randint(1, 10000))

        if world_type == 'grid':
            M_R = generate_and_visualize_gridworld(size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                                                   obstacles_percent=0.1, divide_rooms=False, 
                                                   model_type="Robot Model", obstacle_seed=random.randint(1, 10000))
        elif world_type == 'four_rooms':
            M_R = generate_and_visualize_gridworld(size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                                                   obstacles_percent=0.1, divide_rooms=True, 
                                                   model_type="Robot Model", obstacle_seed=random.randint(1, 10000))
        elif world_type == 'puddle':
            M_R = generate_and_visualize_puddleworld(size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                                                     obstacles_percent=0.1, puddle_percent=0.1, 
                                                     model_type="Robot Model", obstacle_seed=random.randint(1, 10000))
        elif world_type == 'rock':
            M_R = generate_and_visualize_rockworld(size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                                                   obstacles_percent=0.1, rock_percent=0.1, 
                                                   model_type="Robot Model", obstacle_seed=random.randint(1, 10000))
        elif world_type == 'minigrid_unlock':
            M_R = generate_and_visualize_minigridworld('unlock', grid_size, "Robot Model")
        elif world_type == 'minigrid_unlock_pickup':
            M_R = generate_and_visualize_minigridworld('unlock_pickup', grid_size, "Robot Model")
        else:
            raise ValueError(f"Unknown world type: {world_type}")


        results["initial_mdp_state_space_sizes"].append(len(M_R.state_space))
        results["initial_mdp_action_space_sizes"].append(len(M_R.get_actions()))

        print("Generating human models...")
        M_H_list = []
        for i in range(num_models):
            if world_type == 'grid':
                M_H = generate_and_visualize_gridworld(size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                                                       obstacles_percent=0.1, divide_rooms=False, 
                                                       model_type=f"Human Model {i+1}", 
                                                       obstacle_seed=random.randint(1, 10000))
            elif world_type == 'four_rooms':
                M_H = generate_and_visualize_gridworld(size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                                                       obstacles_percent=0.1, divide_rooms=True, 
                                                       model_type=f"Human Model {i+1}", 
                                                       obstacle_seed=random.randint(1, 10000))
            elif world_type == 'puddle':
                M_H = generate_and_visualize_puddleworld(size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                                                         obstacles_percent=0.1, puddle_percent=0.1, 
                                                         model_type=f"Human Model {i+1}", 
                                                         obstacle_seed=random.randint(1, 10000))
            elif world_type == 'rock':
                M_H = generate_and_visualize_rockworld(size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                                                       obstacles_percent=0.1, rock_percent=0.1, 
                                                       model_type=f"Human Model {i+1}", 
                                                       obstacle_seed=random.randint(1, 10000))
            elif world_type == 'taxi':
                M_H = generate_and_visualize_taxiworld(
                    size=grid_size, 
                    start=(0,0), 
                    goal=(grid_size-1,grid_size-1),
                    obstacles_percent=0.1,
                    model_type=f"Human Model {i+1}",
                    obstacle_seed=random.randint(1, 10000)
                )
            elif world_type.startswith('minigrid'):
                M_H = generate_and_visualize_minigridworld(world_type.split('_')[1], grid_size, f"Human Model {i+1}")
            else:
                raise ValueError(f"Unknown world type: {world_type}")
            M_H_list.append(M_H)
            all_human_det_models = []
            for model in M_H_list:
                human_det_model = DeterminizedMDP(model)
                all_human_det_models.append(human_det_model)

        robot_det_model = DeterminizedMDP(M_R)

        # find maximally achievable subsets
        print("Finding maximally achievable subsets...")
        start_time_maximal = time.time()
        I, B = find_maximally_achievable_subsets(M_R, M_H_list)
        maximal_subset_time = time.time() - start_time_maximal
        results["maximal_subset_times"].append(maximal_subset_time)
        print(f"Maximally achievable subsets found in {maximal_subset_time:.2f} seconds")
        
        # record number of bottlenecks found
        num_bottlenecks = len(B)
        results["human_bottlenecks"].append(num_bottlenecks)
        print(f"Number of bottlenecks found: {num_bottlenecks}")

        # handle case with no bottlenecks
        if num_bottlenecks == 0:
            print("No bottlenecks found - recording zero queries needed")
            results["query_counts"].append(0)
            results["query_times"].append(0)
            results["query_mdp_state_space_sizes"].append(0)
            results["query_mdp_action_space_sizes"].append(0)
            print(f"Run {run + 1} completed in {time.time() - start_time_total:.2f} seconds")
            continue

        # only run query MDP if there are bottlenecks
        print("Running query MDP...")
        start_time = time.time()
        try:
            query_mdp = QueryMDP(M_R, list(B), list(I))
            print("Time taken to build query MDP: ", time.time() - start_time)
            simulate_start_time = time.time()
            query_count = simulate_policy_unachievable(query_mdp, list(B), query_threshold)
            print("Time taken to simulate policy: ", time.time() - simulate_start_time)
            query_time = time.time() - start_time
            
            results["query_times"].append(query_time)
            results["query_counts"].append(query_count)
            results["query_mdp_state_space_sizes"].append(len(query_mdp.state_space))
            results["query_mdp_action_space_sizes"].append(len(query_mdp.get_actions()))
            
            print(f"Query MDP completed in {query_time:.2f} seconds")
        except Exception as e:
            print(f"Error during query MDP: {e}")
            # Record zeros for this run if there's an error
            results["query_counts"].append(0)
            results["query_times"].append(0)
            results["query_mdp_state_space_sizes"].append(0)
            results["query_mdp_action_space_sizes"].append(0)

        print(f"Run {run + 1} completed in {time.time() - start_time_total:.2f} seconds")

    return results

def print_results(all_results, num_runs, num_models, grid_size):
    if not all_results:
        print("\nno results to display - all runs failed.")
        return
        
    print("\n# Experiment Config")
    print(f"- Num runs: {num_runs}")
    print(f"- Num human models: {num_models}")
    print(f"- Grid size: {grid_size}x{grid_size}")
    
    metrics = [
        'Query Count',
        'Human Bottlenecks',
        'Maximal Subset Time (s)',
        'Query Time (s)',
        'Initial MDP State Space Size',
        'Initial MDP Action Space Size',
        'Query MDP State Space Size',
        'Query MDP Action Space Size'
    ]
    
    results_dict = {}
    
    for world_type, results in all_results.items():
        world_results = {}
        for metric, values in results.items():
            if not values:  
                world_results[metric] = 'N/A'
                continue
            values_array = np.array(values)
            world_results[metric] = np.mean(values_array)
            
        results_dict[world_type.capitalize() + 'World'] = {
            'Query Count': world_results['query_counts'],
            'Human Bottlenecks': world_results['human_bottlenecks'],
            'Maximal Subset Time (s)': world_results['maximal_subset_times'],
            'Query Time (s)': world_results['query_times'],
            'Initial MDP State Space Size': world_results['initial_mdp_state_space_sizes'],
            'Initial MDP Action Space Size': world_results['initial_mdp_action_space_sizes'],
            'Query MDP State Space Size': world_results['query_mdp_state_space_sizes'],
            'Query MDP Action Space Size': world_results['query_mdp_action_space_sizes']
        }
    
    df = pd.DataFrame(results_dict)
    
    print("\n## Results Summary Table")
    #print(df.to_markdown())
    #print(df.to_string())
    print(df)

    print("\n## CSV Format")
    print(df.to_csv())

def run_all_experiments(num_runs, num_models, grid_size, query_threshold):
    """Run experiments for all world types and collect results"""
    #world_types = ['grid', 'four_rooms', 'puddle', 'rock']
    world_types = ['grid']
    all_results = {}
    
    for world_type in world_types:
        print(f"\nStarting experiments for {world_type.capitalize()}World...")
        results = run_experiments(num_runs, num_models, grid_size, world_type, query_threshold)
        all_results[world_type] = results
    
    return all_results

if __name__ == "__main__":
    # experiment parameters
    num_runs = 5
    num_models = 100
    grid_size = 8
    query_threshold = 1000
    
    # run experiments and collect all results
    all_results = run_all_experiments(num_runs, num_models, grid_size, query_threshold)
    
    # print synthetic results
    print_results(all_results, num_runs, num_models, grid_size)
