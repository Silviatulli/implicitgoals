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

# import environment classes
from experiments import generate_and_visualize_gridworld, generate_and_visualize_puddleworld, generate_and_visualize_rockworld
from ManiskillClass import VisualConstrainedManiSkillEnv
from GridWorldClass import GridWorld
from PuddleWorldClass import PuddleWorld
from RockWorldClass import RockWorld
from domain_randomization import DomainRandomizer

# import MDP and utility classes
from MDP import MDP
from DeterminizedMDP import DeterminizedMDP, identify_bottlenecks
from maximal_achievable_subsets import find_maximally_achievable_subsets, find_maximally_achievable_subsets_no_pruning, improved_find_maximally_achievable_subsets
from QueryMDP import QueryMDP, simulate_policy_unachievable, simulate_policy_query_all
from Utils import ValueIteration, get_policy

def get_safe_process_count():
    cpu_count = multiprocessing.cpu_count()
    return max(1, cpu_count // 2)

pruning_counter = Value('i', 0)
no_pruning_counter = Value('i', 0)
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

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

def create_empty_results_dict():
    """Create an empty results dictionary with the standard structure."""
    return {
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

def get_environment_config(world_type, grid_size, obstacle_percent, model_type, model_num=None):
    """Get the configuration dictionary for creating an environment."""
    base_config = {
        'size': grid_size,
        'start': (0, 0),
        'goal': (grid_size-1, grid_size-1),
        'obstacles_percent': obstacle_percent,
        'model_type': model_type,
        'obstacle_seed': random.randint(1, 10000)
    }
    
    if world_type == 'maniskill':
        return {
            'env_name': "LiftCube-v0",
            'num_bins': grid_size,
            'display_env': (model_num == 1) if model_type == "Human Model" else True
        }
    elif world_type == 'grid' or world_type == 'four_rooms':
        base_config['divide_rooms'] = (world_type == 'four_rooms')
        return base_config
    elif world_type == 'puddle':
        base_config['puddle_percent'] = obstacle_percent
        return base_config
    elif world_type == 'rock':
        base_config['rock_percent'] = obstacle_percent
        return base_config
    return base_config

def generate_model(world_type, grid_size, obstacle_percent, model_type, model_num=None):
    """Generate either a human or robot model based on the world type."""
    try:
        config = get_environment_config(world_type, grid_size, obstacle_percent, model_type, model_num)
        
        if world_type == 'maniskill':
            logging.info(f"Creating ManiSkill {model_type} {model_num if model_num else ''}")
            model = VisualConstrainedManiSkillEnv(**config)
            logging.info(f"Successfully created ManiSkill {model_type} {model_num if model_num else ''}")
        elif world_type == 'grid' or world_type == 'four_rooms':
            model = generate_and_visualize_gridworld(**config)
        elif world_type == 'puddle':
            model = generate_and_visualize_puddleworld(**config)
        elif world_type == 'rock':
            model = generate_and_visualize_rockworld(**config)
        else:
            logging.error(f"Unknown world type: {world_type}")
            return None
            
        return model
    except Exception as e:
        logging.error(f"Error creating {model_type} for {world_type}: {str(e)}", exc_info=True)
        return None

def get_randomized_human_params(world_type, base_params):
    """Generate randomized parameters for human models using DomainRandomizer."""
    return DomainRandomizer.get_randomized_human_params(world_type, base_params)

def generate_human_model(world_type, grid_size, obstacle_percent, puddle_percent, rock_percent, model_num, divide_rooms=False):
    """Helper function to generate appropriate human model based on world type with randomization"""
    base_params = {
        'world_type': world_type,
        'grid_size': grid_size,
        'obstacle_percent': obstacle_percent,
        'puddle_percent': puddle_percent,
        'rock_percent': rock_percent,
        'model_type': f"Human Model {model_num}",
        'model_num': model_num,
        'divide_rooms': divide_rooms
    }
    
    # Get randomized parameters
    randomized_params = get_randomized_human_params(world_type, base_params)
    
    return generate_model(**randomized_params)

def generate_robot_model(world_type, grid_size, obstacle_percent, puddle_percent, rock_percent, divide_rooms=False):
    """Helper function to generate robot model based on world type"""
    return generate_model(
        world_type=world_type,
        grid_size=grid_size,
        obstacle_percent=obstacle_percent,
        model_type="Robot Model"
    )

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
        
        logging.info(f"Starting experiment for {world_type} with grid_size={grid_size}, num_models={num_models}")
        
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
            logging.info(f"Starting trial {trial+1}/{num_trials} for {world_type}")
            
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
            
            logging.info(f"Successfully created {len(M_H_list)} human models for trial {trial}")
            
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
            
            logging.info(f"Found {len(B)} bottlenecks for trial {trial}")
            
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
            
            logging.info(f"Completed pruning version for trial {trial}")
            
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
                logging.info(f"Starting policy computation with pruning for {len(B)} bottlenecks")
                start_time = time.time()
                strategic_count = simulate_policy_unachievable(query_mdp, list(B), query_threshold)
                policy_pruning_time = time.time() - start_time
                logging.info(f"Policy computation with pruning took {policy_pruning_time:.3f} seconds")
                logging.info(f"Strategic count: {strategic_count}")
                results["policy_computation_pruning_times"].append(policy_pruning_time)
                results["query_counts"].append(strategic_count)
                
                # Time policy computation without pruning
                logging.info("Starting policy computation without pruning")
                start_time = time.time()
                query_all_count = simulate_policy_query_all(query_mdp, list(B), query_threshold)
                policy_no_pruning_time = time.time() - start_time
                logging.info(f"Policy computation without pruning took {policy_no_pruning_time:.3f} seconds")
                logging.info(f"Query all count: {query_all_count}")
                results["policy_computation_no_pruning_times"].append(policy_no_pruning_time)
                results["query_all_counts"].append(query_all_count)
        
        logging.info(f"Completed all trials for {world_type}")
        return results
    except Exception as e:
        logging.error(f"Error in run_single_experiment: {str(e)}", exc_info=True)
        return None

def run_parallel_experiments_with_obstacles(num_runs: int, grid_sizes: list, 
                                         human_model_counts: list, world_types: list, 
                                         query_threshold: int, 
                                         obstacle_percentages: dict, max_workers: int = None):
    """
    Run parallel experiments with different obstacle percentages for each environment type.
    
    Args:
        num_runs: Number of runs per configuration
        grid_sizes: List of grid sizes to test
        human_model_counts: List of human model counts to test
        world_types: List of world types to test
        query_threshold: Query threshold for experiments
        obstacle_percentages: Dictionary mapping world types to their obstacle percentages
            Example: {
                'maniskill': [0.0],
                'grid': [0.1, 0.2, 0.3],
                'puddle': [0.1, 0.2],
                'rock': [0.1, 0.2, 0.3],
                'four_rooms': [0.0]
            }
        max_workers: Maximum number of worker processes
    """
    all_environments_results = {}
    experiment_params = []
    
    # reduce memory usage by processing in smaller batches
    batch_size = 10  # adjust this based on your available memory
    
    # create experiment parameters
    for grid_size in grid_sizes:
        for num_models in human_model_counts:
            for world_type in world_types:
                # get obstacle percentages for this world type
                percentages = obstacle_percentages.get(world_type, [0.0])
                
                for obstacle_percent in percentages:
                    world_config = f"{world_type}_{grid_size}_{num_models}_models_{obstacle_percent}"
                    for _ in range(num_runs):
                        params = {
                            'world_type': world_type,
                            'grid_size': grid_size,
                            'num_models': num_models,
                            'query_threshold': query_threshold,
                            'obstacle_percent': obstacle_percent,
                            'puddle_percent': obstacle_percent if world_type == 'puddle' else 0.0,
                            'rock_percent': obstacle_percent if world_type == 'rock' else 0.0,
                            'seed': random.randint(1, 10000)
                        }
                        experiment_params.append((world_config, params))
    
    # process experiments in batches
    max_workers = min(max_workers or get_safe_process_count(), 4)  # limit max workers
    
    for i in range(0, len(experiment_params), batch_size):
        batch = experiment_params[i:i + batch_size]
        logging.info(f"Processing batch {i//batch_size + 1}/{(len(experiment_params) + batch_size - 1)//batch_size}")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_single_experiment, params) for _, params in batch]
            
            # collect results for this batch
            for j, future in enumerate(futures):
                try:
                    result = future.result()
                    if result:
                        world_config = batch[j][0]
                        if world_config not in all_environments_results:
                            all_environments_results[world_config] = create_empty_results_dict()
                        
                        # aggregate results
                        for key in result:
                            if isinstance(result[key], dict):
                                for subkey in result[key]:
                                    all_environments_results[world_config][key][subkey].extend(result[key][subkey])
                            else:
                                all_environments_results[world_config][key].extend(result[key])
                                
                except Exception as e:
                    logging.error(f"Error in experiment {i + j}: {str(e)}", exc_info=True)
                    continue
        
        # force garbage collection between batches
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
    
    logging.info(f"Processing results for environments: {list(all_environments_results.keys())}")
    
    for env_type, results in all_environments_results.items():
        logging.info(f"Processing environment: {env_type}")
        
        # skip only if there are no results at all
        if not any(results.values()):
            logging.warning(f"No results found for {env_type}, skipping")
            continue
            
        # parse environment configuration
        env_parts = env_type.split('_')
        if "four_rooms" in env_type:
            grid_size = int(env_parts[2])
            num_models = int(env_parts[3])
        else:
            grid_size = int(env_parts[1])
            num_models = int(env_parts[2])
        obstacle_percent = float(env_parts[-1])
        
        # add basic info
        combined_data['Environment'].append(env_type)
        combined_data['Grid Size'].append(grid_size)
        combined_data['Number of Human Models'].append(num_models)
        combined_data['Obstacle Percentage'].append(obstacle_percent)
        
        try:
            # get all arrays with their lengths
            arrays = {
                'bottleneck_times': np.array(results.get('bottleneck_finding_times', [])),
                'pruning_times': np.array(results.get("pruning", {}).get("times", [])),
                'policy_pruning_times': np.array(results.get('policy_computation_pruning_times', [])),
                'no_pruning_times': np.array(results.get("no_pruning", {}).get("times", [])),
                'policy_no_pruning_times': np.array(results.get('policy_computation_no_pruning_times', [])),
                'bottlenecks': np.array(results.get('human_bottlenecks', [])),
                'state_space': np.array(results.get('initial_mdp_state_space_sizes', [])),
                'action_space': np.array(results.get('initial_mdp_action_space_sizes', []))
            }
            
            # log array lengths for debugging
            for key, arr in arrays.items():
                logging.info(f"{env_type} - {key} length: {len(arr)}")
            
            # find minimum length among non-empty arrays
            non_empty_arrays = [arr for arr in arrays.values() if len(arr) > 0]
            if not non_empty_arrays:
                logging.warning(f"No valid data for {env_type}")
                continue
                
            min_len = min(len(arr) for arr in non_empty_arrays)
            logging.info(f"{env_type} - Using minimum length: {min_len}")
            
            # truncate all arrays to minimum length
            for key in arrays:
                if len(arrays[key]) > 0:
                    arrays[key] = arrays[key][:min_len]
                else:
                    # fill empty arrays with zeros or appropriate default values
                    if key in ['bottleneck_times', 'pruning_times', 'policy_pruning_times', 
                             'no_pruning_times', 'policy_no_pruning_times']:
                        arrays[key] = np.zeros(min_len)
                    elif key == 'bottlenecks':
                        arrays[key] = np.zeros(min_len, dtype=int)
                    else:
                        arrays[key] = np.zeros(min_len, dtype=int)
            
            # calculate metrics with consistent array lengths
            combined_data['Finding Bottlenecks Time (s)'].append(
                f"{np.mean(arrays['bottleneck_times']):.3f} ± {np.std(arrays['bottleneck_times']):.3f}")
                
            combined_data['Finding Maximal Achievable With Pruning (s)'].append(
                f"{np.mean(arrays['pruning_times']):.3f} ± {np.std(arrays['pruning_times']):.3f}")
            
            if len(arrays['no_pruning_times']) > 0:
                combined_data['Finding Maximal Achievable Without Pruning (s)'].append(
                    f"{np.mean(arrays['no_pruning_times']):.3f} ± {np.std(arrays['no_pruning_times']):.3f}")
            else:
                combined_data['Finding Maximal Achievable Without Pruning (s)'].append("N/A")
            
            if len(arrays['policy_pruning_times']) > 0:
                combined_data['Computing Policy With Pruning (s)'].append(
                    f"{np.mean(arrays['policy_pruning_times']):.3f} ± {np.std(arrays['policy_pruning_times']):.3f}")
            else:
                combined_data['Computing Policy With Pruning (s)'].append("N/A")

            if len(arrays['policy_no_pruning_times']) > 0:
                combined_data['Computing Policy Without Pruning (s)'].append(
                    f"{np.mean(arrays['policy_no_pruning_times']):.3f} ± {np.std(arrays['policy_no_pruning_times']):.3f}")
            else:
                combined_data['Computing Policy Without Pruning (s)'].append("N/A")
            
            # calculate total runtimes with consistent lengths
            total_pruning = arrays['bottleneck_times'] + arrays['pruning_times'] + arrays['policy_pruning_times']
            combined_data['Total Runtime With Pruning (s)'].append(
                f"{np.mean(total_pruning):.3f} ± {np.std(total_pruning):.3f}")
            
            if len(arrays['no_pruning_times']) > 0 and len(arrays['policy_no_pruning_times']) > 0:
                total_no_pruning = arrays['bottleneck_times'] + arrays['no_pruning_times'] + arrays['policy_no_pruning_times']
                combined_data['Total Runtime Without Pruning (s)'].append(
                    f"{np.mean(total_no_pruning):.3f} ± {np.std(total_no_pruning):.3f}")
                
                # calculate runtime improvement percentage
                improvement = ((np.mean(total_no_pruning) - np.mean(total_pruning)) / 
                             np.mean(total_no_pruning) * 100)
                improvement_std = np.std([(n - p)/n * 100 for n, p in zip(total_no_pruning, total_pruning)])
                combined_data['Runtime Improvement (%)'].append(
                    f"{improvement:.1f} ± {improvement_std:.1f}")
            else:
                combined_data['Total Runtime Without Pruning (s)'].append("N/A")
                combined_data['Runtime Improvement (%)'].append("N/A")
            
            # add remaining metrics
            combined_data['Human Bottlenecks'].append(
                f"{np.mean(arrays['bottlenecks']):.1f} ± {np.std(arrays['bottlenecks']):.1f}")
            combined_data['Initial State Space'].append(
                f"{np.mean(arrays['state_space']):.0f}")
            combined_data['Initial Actions'].append(
                f"{np.mean(arrays['action_space']):.0f}")
            
            logging.info(f"Successfully processed {env_type}")
        
        except Exception as e:
            logging.error(f"Error processing results for {env_type}: {str(e)}", exc_info=True)
            continue
    
    # create DataFrame and save results
    df = pd.DataFrame(combined_data)
    df.to_csv(output_file, index=False)
    logging.info(f"Saved results to {output_file}")
    logging.info(f"Environments included in results: {df['Environment'].tolist()}")
    return df

if __name__ == "__main__":
    # configure logging
    logging.basicConfig(
        level=logging.INFO,  # Changed to INFO to see more details
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('experiment.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # experiment parameters for different environments
    num_runs = 3
    grid_sizes = [5]  # 4 for grid worlds, 5 for ManiSkill discretization
    human_model_counts = [3]  # 3 for ManiSkill, 5 for grid worlds
    query_threshold = 1000
    world_types = ['maniskill', 'grid', 'puddle', 'rock', 'four_rooms']
    
    # different obstacle percentages for each environment type
    obstacle_percentages = {
        'maniskill': [0.0],  # No obstacles for ManiSkill
        'grid': [0.1, 0.2, 0.3],  # Test different obstacle densities
        'puddle': [0.1, 0.2],  # Fewer puddle percentages to test
        'rock': [0.1, 0.2, 0.3],  # Test different rock densities
        'four_rooms': [0.0]  # No obstacles for four rooms
    }
    
    max_workers = 2  # Increased workers since we'll have non-visualization tasks
    
    try:
        os.makedirs("experiment_results", exist_ok=True)
        logging.info("Starting parallel experiments")
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
            logging.info("Creating combined results table")
            combined_df = create_combined_results_table(results)
            print("Results saved successfully")
            logging.info(f"Available environments in results: {list(results.keys())}")
        else:
            logging.error("No results were generated from the experiments")
    except Exception as e:
        logging.error(f"Error during experiment execution: {str(e)}", exc_info=True)
