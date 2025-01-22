from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Value
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Set, FrozenSet
import time
import os
import signal
import sys
import random
import itertools
import inspect
import threading
import queue
import logging
import gc
import multiprocessing

from experiments import generate_and_visualize_gridworld, generate_and_visualize_puddleworld, generate_and_visualize_rockworld
from maximal_achievable_subsets import find_maximally_achievable_subsets, find_maximally_achievable_subsets_no_pruning, improved_find_maximally_achievable_subsets
from QueryMDP import QueryMDP, simulate_policy_unachievable, simulate_policy_query_all
from DeterminizedMDP import identify_bottlenecks

def get_safe_process_count():
    cpu_count = multiprocessing.cpu_count()
    return max(1, cpu_count // 2)

pruning_counter = Value('i', 0)
no_pruning_counter = Value('i', 0)
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        pass
        #print(*args, **kwargs)

def init_worker():
    global inspect
    import inspect

logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)


def generate_human_model(world_type, grid_size, obstacle_percent, puddle_percent, rock_percent, model_num, divide_rooms=False):
    """Helper function to generate appropriate human model based on world type"""
    M_H = None
    if world_type == 'grid' or world_type == 'four_rooms':
        M_H = generate_and_visualize_gridworld(
            size=grid_size,
            start=(0,0),
            goal=(grid_size-1,grid_size-1),
            obstacles_percent=obstacle_percent,
            divide_rooms=(world_type == 'four_rooms'),
            model_type=f"Human Model {model_num}",
            obstacle_seed=random.randint(1, 10000)
        )
    elif world_type == 'puddle':
        M_H = generate_and_visualize_puddleworld(
            size=grid_size,
            start=(0,0),
            goal=(grid_size-1,grid_size-1),
            obstacles_percent=obstacle_percent,
            puddle_percent=puddle_percent,
            model_type=f"Human Model {model_num}",
            obstacle_seed=random.randint(1, 10000)
        )
    elif world_type == 'rock':
        M_H = generate_and_visualize_rockworld(
            size=grid_size,
            start=(0,0),
            goal=(grid_size-1,grid_size-1),
            obstacles_percent=obstacle_percent,
            rock_percent=rock_percent,
            model_type=f"Human Model {model_num}",
            obstacle_seed=random.randint(1, 10000)
        )
    return M_H

def generate_robot_model(world_type, grid_size, obstacle_percent, puddle_percent, rock_percent, divide_rooms=False):
    """Helper function to generate appropriate robot model based on world type"""
    M_R = None
    if world_type == 'grid' or world_type == 'four_rooms':
        M_R = generate_and_visualize_gridworld(
            size=grid_size,
            start=(0,0),
            goal=(grid_size-1,grid_size-1),
            obstacles_percent=obstacle_percent,
            divide_rooms=(world_type == 'four_rooms'),
            model_type="Robot Model",
            obstacle_seed=random.randint(1, 10000)
        )
    elif world_type == 'puddle':
        M_R = generate_and_visualize_puddleworld(
            size=grid_size,
            start=(0,0),
            goal=(grid_size-1,grid_size-1),
            obstacles_percent=obstacle_percent,
            puddle_percent=puddle_percent,
            model_type="Robot Model",
            obstacle_seed=random.randint(1, 10000)
        )
    elif world_type == 'rock':
        M_R = generate_and_visualize_rockworld(
            size=grid_size,
            start=(0,0),
            goal=(grid_size-1,grid_size-1),
            obstacles_percent=obstacle_percent,
            rock_percent=rock_percent,
            model_type="Robot Model",
            obstacle_seed=random.randint(1, 10000)
        )
    return M_R

def run_single_experiment(params: Dict[str, Any]) -> Dict[str, Any]:
    trial_seed = params.get('seed', 0)
    np.random.seed(trial_seed)
    random.seed(trial_seed)
    
    try:
        # Extract parameters
        world_type = params['world_type']
        grid_size = params['grid_size']
        num_models = params['num_models']
        query_threshold = params['query_threshold']
        obstacle_percent = params['obstacle_percent']
        puddle_percent = params.get('puddle_percent', 0)
        rock_percent = params.get('rock_percent', 0)
        
        results = {
            "bottleneck_finding_times": [],
            "maximal_achievable_pruning_times": [],
            "maximal_achievable_no_pruning_times": [],
            "policy_computation_pruning_times": [],
            "policy_computation_no_pruning_times": [],
            "pruning": {"times": [], "checks": [], "subsets": []},
            "no_pruning": {"times": [], "checks": [], "subsets": []},
            "query_counts": [],
            "query_all_counts": [],
            "human_bottlenecks": [],
            "initial_mdp_state_space_sizes": [],
            "initial_mdp_action_space_sizes": [],
        }
        
        num_trials = 5
        for trial in range(num_trials):
            # Generate robot model
            start_time = time.time()
            M_R = generate_robot_model(
                world_type=world_type,
                grid_size=grid_size,
                obstacle_percent=obstacle_percent,
                puddle_percent=puddle_percent,
                rock_percent=rock_percent
            )
            
            if not M_R:
                logging.warning(f"Failed to generate robot model for trial {trial}")
                continue
                
            
            # Generate human models
            M_H_list = []
            for i in range(num_models):
                M_H = generate_human_model(
                    world_type=world_type,
                    grid_size=grid_size,
                    obstacle_percent=obstacle_percent,
                    puddle_percent=puddle_percent,
                    rock_percent=rock_percent,
                    model_num=i+1
                )
                if M_H:
                    M_H_list.append(M_H)
            
            
            if not M_H_list:
                logging.warning(f"No human models generated for trial {trial}")
                continue
            
            # First find bottlenecks separately
            start_time = time.time()
            B = set()
            for M in M_H_list:
                B.update(tuple(b) for b in identify_bottlenecks(M))
            bottleneck_finding_time = time.time() - start_time    
            results["bottleneck_finding_times"].append(bottleneck_finding_time)
            results["human_bottlenecks"].append(len(B))
            results["initial_mdp_state_space_sizes"].append(len(M_R.state_space))
            results["initial_mdp_action_space_sizes"].append(len(M_R.get_actions()))
            
            # Run pruning version
            with pruning_counter.get_lock():
                start_time = time.time()
                I_pruning, B = improved_find_maximally_achievable_subsets(M_R, M_H_list)
                maximal_achievable_pruning_time = time.time() - start_time
                check_count = pruning_counter.value
                pruning_counter.value = 0


                results["maximal_achievable_pruning_times"].append(maximal_achievable_pruning_time)
                results["pruning"]["times"].append(maximal_achievable_pruning_time)
                results["pruning"]["checks"].append(check_count)
                results["pruning"]["subsets"].append(len(I_pruning))
                results["human_bottlenecks"].append(len(B))
            
            # Run no-pruning version for small state spaces
            if len(B) <= 15:
                with no_pruning_counter.get_lock():
                    start_time = time.time()
                    I_no_pruning, _ = find_maximally_achievable_subsets_no_pruning(M_R, M_H_list)
                    maximal_achievable_no_pruning_time = time.time() - start_time
                    check_count = no_pruning_counter.value
                    no_pruning_counter.value = 0
                    
                    results["maximal_achievable_no_pruning_times"].append(maximal_achievable_no_pruning_time)
                    results["no_pruning"]["times"].append(maximal_achievable_no_pruning_time)
                    results["no_pruning"]["checks"].append(check_count)
                    results["no_pruning"]["subsets"].append(len(I_no_pruning))
            
            # Run query experiments if bottlenecks exist
            if len(B) > 0:
                query_mdp = QueryMDP(M_R, list(B), list(I_pruning))
                
                # Time policy computation with pruning
                print(f"Starting policy computation with pruning for {len(B)} bottlenecks")
                start_time = time.time()
                strategic_count = simulate_policy_unachievable(query_mdp, list(B), query_threshold)
                policy_pruning_time = time.time() - start_time
                print(f"Policy computation with pruning took {policy_pruning_time:.3f} seconds")
                print(f"Strategic count: {strategic_count}")
                results["policy_computation_pruning_times"].append(policy_pruning_time)
                results["query_counts"].append(strategic_count)
                
                # Time policy computation without pruning
                print(f"Starting policy computation without pruning")
                start_time = time.time()
                query_all_count = simulate_policy_query_all(query_mdp, list(B), query_threshold)
                policy_no_pruning_time = time.time() - start_time
                print(f"Policy computation without pruning took {policy_no_pruning_time:.3f} seconds")
                print(f"Query all count: {query_all_count}")
                results["policy_computation_no_pruning_times"].append(policy_no_pruning_time)
                results["query_all_counts"].append(query_all_count)
        
        return results
    except Exception as e:
        logging.error(f"Error in run_single_experiment: {str(e)}", exc_info=True)
        return None

def run_parallel_experiments_with_obstacles(num_runs: int, grid_sizes: list, 
                                         human_model_counts: list, world_types: list, 
                                         query_threshold: int, 
                                         obstacle_percentages: list, max_workers: int = None):
    all_environments_results = {}
    experiment_params = []
    
    # Reduce memory usage by processing in smaller batches
    batch_size = 10  # Adjust this based on your available memory
    
    # Create experiment parameters
    for grid_size in grid_sizes:
        for num_models in human_model_counts:
            for world_type in world_types:
                if world_type == 'four_rooms':
                    world_config = f"{world_type}_{grid_size}_{num_models}_models_0.0"
                    for _ in range(num_runs):
                        params = {
                            'world_type': world_type,
                            'grid_size': grid_size,
                            'num_models': num_models,
                            'query_threshold': query_threshold,
                            'obstacle_percent': 0.0,
                            'puddle_percent': 0.0,
                            'rock_percent': 0.0,
                            'seed': random.randint(1, 10000)
                        }
                        experiment_params.append((world_config, params))
                else:
                    for obstacle_percent in obstacle_percentages:
                        world_config = f"{world_type}_{grid_size}_{num_models}_models_{obstacle_percent}"
                        for _ in range(num_runs):
                            params = {
                                'world_type': world_type,
                                'grid_size': grid_size,
                                'num_models': num_models,
                                'query_threshold': query_threshold,
                                'obstacle_percent': obstacle_percent,
                                'puddle_percent': obstacle_percent,
                                'rock_percent': obstacle_percent,
                                'seed': random.randint(1, 10000)
                            }
                            experiment_params.append((world_config, params))
    
    # Process experiments in batches
    max_workers = min(max_workers or get_safe_process_count(), 4)  # Limit max workers
    
    for i in range(0, len(experiment_params), batch_size):
        batch = experiment_params[i:i + batch_size]
        logging.info(f"Processing batch {i//batch_size + 1}/{(len(experiment_params) + batch_size - 1)//batch_size}")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_single_experiment, params) for _, params in batch]
            
            # Collect results for this batch
            for j, future in enumerate(futures):
                try:
                    result = future.result()
                    if result:
                        world_config = batch[j][0]
                        if world_config not in all_environments_results:
                            all_environments_results[world_config] = {
                                "bottleneck_finding_times": [],
                                "maximal_achievable_pruning_times": [],
                                "maximal_achievable_no_pruning_times": [],
                                "policy_computation_pruning_times": [],
                                "policy_computation_no_pruning_times": [],
                                "pruning": {"times": [], "checks": [], "subsets": []},
                                "no_pruning": {"times": [], "checks": [], "subsets": []},
                                "query_counts": [],
                                "query_all_counts": [],
                                "human_bottlenecks": [],
                                "initial_mdp_state_space_sizes": [],
                                "initial_mdp_action_space_sizes": [],
                            }
                        
                        # Aggregate results
                        for key in result:
                            if isinstance(result[key], dict):
                                for subkey in result[key]:
                                    all_environments_results[world_config][key][subkey].extend(result[key][subkey])
                            else:
                                all_environments_results[world_config][key].extend(result[key])
                                
                except Exception as e:
                    logging.error(f"Error in experiment {i + j}: {str(e)}", exc_info=True)
                    continue
        
        # Force garbage collection between batches
        gc.collect()
    
    return all_environments_results

def create_combined_results_table(all_environments_results, output_file="experiment_results/combined_comparison.csv"):
    """Create a combined results table with additional metrics including obstacle percentage."""
    combined_data = {
        'Environment': [],
        'Grid Size': [],
        'Number of Human Models': [],
        'Obstacle Percentage': [],
        'Finding Bottlenecks Time (s)': [],
        'Finding Maximal Achievable With Pruning (s)': [],
        'Finding Maximal Achievable Without Pruning (s)': [],
        'Computing Policy With Pruning (s)': [],
        'Computing Policy Without Pruning (s)': [],
        'Total Runtime With Pruning (s)': [],
        'Total Runtime Without Pruning (s)': [],
        'Runtime Improvement (%)': [],
        'Human Bottlenecks': [],
        'Initial State Space': [],
        'Initial Actions': []
    }
    
    for env_type, results in all_environments_results.items():
        if not results["pruning"]["times"]:
            continue
            
        # Parse environment configuration
        env_parts = env_type.split('_')
        if "four_rooms" in env_type:
            grid_size = int(env_parts[2])  # Adjust index for "four_rooms" case
            num_models = int(env_parts[3])
        else:
            grid_size = int(env_parts[1])
            num_models = int(env_parts[2])
        obstacle_percent = float(env_parts[-1])
        
        # Add basic info
        combined_data['Environment'].append(env_type)
        combined_data['Grid Size'].append(grid_size)
        combined_data['Number of Human Models'].append(num_models)
        combined_data['Obstacle Percentage'].append(obstacle_percent)
        
        try:
            # Ensure all arrays have the same length by truncating to shortest
            bottleneck_times = np.array(results.get('bottleneck_finding_times', []))
            pruning_times = np.array(results["pruning"]["times"])
            policy_pruning_times = np.array(results.get('policy_computation_pruning_times', []))
            no_pruning_times = np.array(results["no_pruning"]["times"]) if results["no_pruning"]["times"] else []
            policy_no_pruning_times = np.array(results.get('policy_computation_no_pruning_times', []))
            
            # Find minimum length
            min_len = min(len(arr) for arr in [bottleneck_times, pruning_times, policy_pruning_times]
                         if len(arr) > 0)
            
            # Truncate arrays to minimum length
            bottleneck_times = bottleneck_times[:min_len]
            pruning_times = pruning_times[:min_len]
            policy_pruning_times = policy_pruning_times[:min_len]
            
            if len(no_pruning_times) > 0:
                no_pruning_times = no_pruning_times[:min_len]
            if len(policy_no_pruning_times) > 0:
                policy_no_pruning_times = policy_no_pruning_times[:min_len]
            
            # Calculate metrics with consistent array lengths
            combined_data['Finding Bottlenecks Time (s)'].append(
                f"{np.mean(bottleneck_times):.3f} ± {np.std(bottleneck_times):.3f}")
                
            combined_data['Finding Maximal Achievable With Pruning (s)'].append(
                f"{np.mean(pruning_times):.3f} ± {np.std(pruning_times):.3f}")
            
            if len(no_pruning_times) > 0:
                combined_data['Finding Maximal Achievable Without Pruning (s)'].append(
                    f"{np.mean(no_pruning_times):.3f} ± {np.std(no_pruning_times):.3f}")
            else:
                combined_data['Finding Maximal Achievable Without Pruning (s)'].append("N/A")
            
            if len(policy_pruning_times) > 0 and np.any(policy_pruning_times > 0):
                combined_data['Computing Policy With Pruning (s)'].append(
                    f"{np.mean(policy_pruning_times):.3f} ± {np.std(policy_pruning_times):.3f}")
            else:
                combined_data['Computing Policy With Pruning (s)'].append("N/A")

            if len(policy_no_pruning_times) > 0 and np.any(policy_no_pruning_times > 0):
                combined_data['Computing Policy Without Pruning (s)'].append(
                    f"{np.mean(policy_no_pruning_times):.3f} ± {np.std(policy_no_pruning_times):.3f}")
            else:
                combined_data['Computing Policy Without Pruning (s)'].append("N/A")

            
            # Calculate total runtimes with consistent lengths
            total_pruning = bottleneck_times + pruning_times + policy_pruning_times
            combined_data['Total Runtime With Pruning (s)'].append(
                f"{np.mean(total_pruning):.3f} ± {np.std(total_pruning):.3f}")
            
            if len(no_pruning_times) > 0 and len(policy_no_pruning_times) > 0:
                total_no_pruning = bottleneck_times + no_pruning_times + policy_no_pruning_times
                combined_data['Total Runtime Without Pruning (s)'].append(
                    f"{np.mean(total_no_pruning):.3f} ± {np.std(total_no_pruning):.3f}")
                
                # Calculate runtime improvement percentage
                improvement = ((np.mean(total_no_pruning) - np.mean(total_pruning)) / 
                             np.mean(total_no_pruning) * 100)
                improvement_std = np.std([(n - p)/n * 100 for n, p in zip(total_no_pruning, total_pruning)])
                combined_data['Runtime Improvement (%)'].append(
                    f"{improvement:.1f} ± {improvement_std:.1f}")
            else:
                combined_data['Total Runtime Without Pruning (s)'].append("N/A")
                combined_data['Runtime Improvement (%)'].append("N/A")
        
        except Exception as e:
            logging.error(f"Error processing results for {env_type}: {str(e)}", exc_info=True)
            continue
            
        # Add remaining metrics
        bottlenecks = np.array(results['human_bottlenecks'][:min_len])
        combined_data['Human Bottlenecks'].append(
            f"{np.mean(bottlenecks):.1f} ± {np.std(bottlenecks):.1f}")
        combined_data['Initial State Space'].append(
            f"{np.mean(results['initial_mdp_state_space_sizes'][:min_len]):.0f}")
        combined_data['Initial Actions'].append(
            f"{np.mean(results['initial_mdp_action_space_sizes'][:min_len]):.0f}")
    
    # Create DataFrame and save results
    df = pd.DataFrame(combined_data)
    df.to_csv(output_file, index=False)
    return df

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('experiment.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Updated experiment parameters
    num_runs = 3
    grid_sizes = [4]  
    human_model_counts = [5] 
    query_threshold = 1000
    world_types = ['grid', 'four_rooms', 'puddle', 'rock']
    obstacle_percentages = [0.1, 0.15] 
    max_workers = 2
    
    try:
        os.makedirs("experiment_results", exist_ok=True)
        results = run_parallel_experiments_with_obstacles(
            num_runs=num_runs,
            grid_sizes=grid_sizes,
            human_model_counts=human_model_counts,
            world_types=world_types,
            query_threshold=query_threshold,
            obstacle_percentages=obstacle_percentages,
            max_workers=max_workers
        )
        
        if results:
            combined_df = create_combined_results_table(results)
            print("Results saved successfully")
        else:
            logging.error("No results were generated from the experiments")
    except Exception as e:
        logging.error(f"Error during experiment execution: {str(e)}", exc_info=True)
