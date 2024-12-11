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
import sys
import threading
import queue
import logging
import gc
import multiprocessing

from experiments import generate_and_visualize_gridworld, generate_and_visualize_puddleworld, generate_and_visualize_rockworld
from maximal_achievable_subsets import find_maximally_achievable_subsets, find_maximally_achievable_subsets_no_pruning, improved_find_maximally_achievable_subsets
from QueryMDP import QueryMDP, simulate_policy_unachievable, simulate_policy_query_all


def get_safe_process_count():
    # Use a fraction of available CPUs to be safe with memory
    cpu_count = multiprocessing.cpu_count()
    return max(1, cpu_count // 2)  # Use half of available CPUs


pruning_counter = Value('i', 0)
no_pruning_counter = Value('i', 0)

# Create a thread-safe print lock
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

def init_worker():
    """Initialize worker process with required imports"""
    global inspect
    import inspect

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
def run_single_experiment(params: Dict[str, Any]):
    trial_seed = params.get('seed', 0)
    np.random.seed(trial_seed)
    random.seed(trial_seed)
    """Run a single experiment with multiple trials to capture variance"""
    try:
        #print(f"Starting experiment with parameters: {params['world_type']}, size={params['grid_size']}, obstacles={params['obstacle_percent']}")
        # Extract parameters
        world_type = params['world_type']
        grid_size = params['grid_size']
        num_models = params['num_models']
        query_threshold = params['query_threshold']
        obstacle_percent = params['obstacle_percent']
        puddle_percent = params.get('puddle_percent', 0)
        rock_percent = params.get('rock_percent', 0)
        
        # Initialize results dictionary to store all trials
        results = {
            "pruning": {
                "times": [],
                "checks": [],
                "subsets": []
            },
            "no_pruning": {
                "times": [],
                "checks": [],
                "subsets": []
            },
            "query_counts": [],
            "query_times": [],
            "query_all_counts": [],
            "query_all_times": [],
            "human_bottlenecks": [],
            "initial_mdp_state_space_sizes": [],
            "initial_mdp_action_space_sizes": [],
            "query_mdp_state_space_sizes": [],
            "query_mdp_action_space_sizes": []
        }
        gc.collect()
        # Run multiple trials
        num_trials = 5  # Number of trials for variance
        for trial in range(num_trials):
            # Generate robot model
            M_R = None
            if world_type == 'grid':
                M_R = generate_and_visualize_gridworld(
                    size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                    obstacles_percent=obstacle_percent, divide_rooms=False, 
                    model_type="Robot Model", obstacle_seed=random.randint(1, 10000)
                )
            elif world_type == 'four_rooms':
                M_R = generate_and_visualize_gridworld(
                    size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                    obstacles_percent=obstacle_percent, divide_rooms=True, 
                    model_type="Robot Model", obstacle_seed=random.randint(1, 10000)
                )
            elif world_type == 'puddle':
                M_R = generate_and_visualize_puddleworld(
                    size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                    obstacles_percent=obstacle_percent, puddle_percent=puddle_percent, 
                    model_type="Robot Model", obstacle_seed=random.randint(1, 10000)
                )
            elif world_type == 'rock':
                M_R = generate_and_visualize_rockworld(
                    size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                    obstacles_percent=obstacle_percent, rock_percent=rock_percent, 
                    model_type="Robot Model", obstacle_seed=random.randint(1, 10000)
                )
            
            if not M_R:
                continue
                
            # Store initial MDP sizes for this trial
            results["initial_mdp_state_space_sizes"].append(len(M_R.state_space))
            results["initial_mdp_action_space_sizes"].append(len(M_R.get_actions()))
            
            # Generate human models for this trial
            M_H_list = []
            for i in range(num_models):
                M_H = None
                if world_type == 'grid':
                    M_H = generate_and_visualize_gridworld(
                        size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                        obstacles_percent=obstacle_percent, divide_rooms=False, 
                        model_type=f"Human Model {i+1}", 
                        obstacle_seed=random.randint(1, 10000)
                    )
                elif world_type == 'four_rooms':
                    M_H = generate_and_visualize_gridworld(
                        size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                        obstacles_percent=obstacle_percent, divide_rooms=True, 
                        model_type=f"Human Model {i+1}", 
                        obstacle_seed=random.randint(1, 10000)
                    )
                elif world_type == 'puddle':
                    M_H = generate_and_visualize_puddleworld(
                        size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                        obstacles_percent=obstacle_percent, puddle_percent=puddle_percent, 
                        model_type=f"Human Model {i+1}", 
                        obstacle_seed=random.randint(1, 10000)
                    )
                elif world_type == 'rock':
                    M_H = generate_and_visualize_rockworld(
                        size=grid_size, start=(0,0), goal=(grid_size-1,grid_size-1), 
                        obstacles_percent=obstacle_percent, rock_percent=rock_percent, 
                        model_type=f"Human Model {i+1}", 
                        obstacle_seed=random.randint(1, 10000)
                    )
                if M_H:
                    M_H_list.append(M_H)

            if not M_H_list:  # Skip if no human models were created
                continue

            # Run pruning version for this trial
            with pruning_counter.get_lock():
                start_time = time.time()
                I_pruning, B = improved_find_maximally_achievable_subsets(M_R, M_H_list)
                pruning_time = time.time() - start_time
                check_count = pruning_counter.value
                pruning_counter.value = 0
            
            results["pruning"]["times"].append(pruning_time)
            results["pruning"]["checks"].append(check_count)
            results["pruning"]["subsets"].append(len(I_pruning))
            results["human_bottlenecks"].append(len(B))
            
            # Run no-pruning version if state space is small enough
            if len(B) <= 15:
                with no_pruning_counter.get_lock():
                    start_time = time.time()
                    I_no_pruning, _ = find_maximally_achievable_subsets_no_pruning(M_R, M_H_list)
                    no_pruning_time = time.time() - start_time
                    check_count = no_pruning_counter.value
                    no_pruning_counter.value = 0
                
                results["no_pruning"]["times"].append(no_pruning_time)
                results["no_pruning"]["checks"].append(check_count)
                results["no_pruning"]["subsets"].append(len(I_no_pruning))
            
            # Run query experiments if bottlenecks exist
            if len(B) > 0:
                query_mdp = QueryMDP(M_R, list(B), list(I_pruning))
                
                # Strategic querying
                start_time = time.time()
                strategic_count = simulate_policy_unachievable(query_mdp, list(B), query_threshold)
                strategic_time = time.time() - start_time
                results["query_counts"].append(strategic_count)
                results["query_times"].append(strategic_time)
                
                # Query-all approach
                start_time = time.time()
                query_all_count = simulate_policy_query_all(query_mdp, list(B), query_threshold)
                query_all_time = time.time() - start_time
                results["query_all_counts"].append(query_all_count)
                results["query_all_times"].append(query_all_time)
        
        return results
    except Exception as e:
        print(f"Error in run_single_experiment: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        gc.collect()
        return None

def run_parallel_experiments_with_obstacles(num_runs: int, num_models: int, grid_sizes: list, 
                                    world_types: list, query_threshold: int, 
                                    obstacle_percentages: list, max_workers: int = None):
    """Run experiments in parallel for different grid sizes and obstacle percentages."""
    all_environments_results = {}
    experiment_params = []
    experiment_counter = 0
    # Create experiment parameters
    experiment_params = []
    for grid_size in grid_sizes:
        for world_type in world_types:
            if world_type == 'four_rooms':
                # Four rooms environment uses fixed 0% obstacles
                world_config = f"{world_type}_{grid_size}_0.0"
                for _ in range(num_runs):
                    params = {
                        'world_type': world_type,
                        'grid_size': grid_size,
                        'num_models': num_models,
                        'query_threshold': query_threshold,
                        'obstacle_percent': 0.0,
                        'puddle_percent': 0.0,
                        'rock_percent': 0.0
                    }
                    experiment_params.append((world_config, params))
            else:
                # Other environments test different obstacle percentages
                for obstacle_percent in obstacle_percentages:
                    world_config = f"{world_type}_{grid_size}_{obstacle_percent}"
                    for _ in range(num_runs):
                        params = {
                            'world_type': world_type,
                            'grid_size': grid_size,
                            'num_models': num_models,
                            'query_threshold': query_threshold,
                            'obstacle_percent': obstacle_percent,
                            'puddle_percent': obstacle_percent,
                            'rock_percent': obstacle_percent
                        }
                        experiment_params.append((world_config, params))

 
    # Run experiments in parallel
    # When creating your process pool:
    with ProcessPoolExecutor(max_workers=get_safe_process_count()) as executor:
        futures = [executor.submit(run_single_experiment, params) for _, params in experiment_params]
        
        # Collect results
        for i, future in enumerate(futures):
            result = future.result()
            try:
                result = future.result()
                if result:
                    world_config = experiment_params[i][0]
                    if world_config not in all_environments_results:
                        all_environments_results[world_config] = {
                            "pruning": {"times": [], "checks": [], "subsets": []},
                            "no_pruning": {"times": [], "checks": [], "subsets": []},
                            "query_counts": [],
                            "query_times": [],
                            "query_all_counts": [],
                            "query_all_times": [],
                            "human_bottlenecks": [],
                            "initial_mdp_state_space_sizes": [],
                            "initial_mdp_action_space_sizes": [],
                            "query_mdp_state_space_sizes": [],
                            "query_mdp_action_space_sizes": []
                        }
                    
                    # Aggregate results
                    for key in result:
                        if isinstance(result[key], dict):
                            for subkey in result[key]:
                                all_environments_results[world_config][key][subkey].extend(result[key][subkey])
                        else:
                            all_environments_results[world_config][key].extend(result[key])
            except Exception as e:
                print(f"Error in experiment {i}: {str(e)}")
                continue
    
    return all_environments_results

def create_combined_results_table(all_environments_results, output_file="experiment_results/combined_comparison.csv"):
    """Create a combined results table from all experiments."""
    if not all_environments_results:
        print("No results to process!")
        return pd.DataFrame()

    combined_data = {
        'Environment': [],
        'Pruning Time (s)': [],
        'Pruning Speedup': [],
        'Strategic Query Count': [],
        'Query-All Count': [],
        'Query Reduction (%)': [],
        'Human Bottlenecks': [],
        'Initial State Space': [],
        'Initial Actions': []
    }
    
    for env_type, results in all_environments_results.items():
        # Skip if no results for this environment
        if not results["pruning"]["times"]:
            continue
            
        combined_data['Environment'].append(env_type)
        
        # Pruning metrics with ± std
        pruning_times = np.array(results["pruning"]["times"])
        pruning_mean = np.mean(pruning_times)
        pruning_std = np.std(pruning_times)
        combined_data['Pruning Time (s)'].append(f"{pruning_mean:.3f} ± {pruning_std:.3f}")
        
        # No pruning metrics
        no_pruning_times = np.array(results["no_pruning"]["times"]) if results["no_pruning"]["times"] else []
        if len(no_pruning_times) > 0:
            speedup = np.mean(no_pruning_times) / pruning_mean
            speedup_std = np.std([n/p for n,p in zip(no_pruning_times, pruning_times)])
            combined_data['Pruning Speedup'].append(f"{speedup:.2f}x ± {speedup_std:.2f}")
        else:
            combined_data['Pruning Speedup'].append("N/A")
        
        # Query metrics
        strategic_queries = np.array(results.get('query_counts', []))
        query_all_counts = np.array(results.get('query_all_counts', []))
        
        if len(strategic_queries) > 0 and len(query_all_counts) > 0:
            # Strategic queries
            strategic_mean = np.mean(strategic_queries)
            strategic_std = np.std(strategic_queries)
            combined_data['Strategic Query Count'].append(
                f"{strategic_mean:.1f} ± {strategic_std:.1f}")
            
            # Query-all
            query_all_mean = np.mean(query_all_counts)
            query_all_std = np.std(query_all_counts)
            combined_data['Query-All Count'].append(
                f"{query_all_mean:.1f} ± {query_all_std:.1f}")
            
            # Reduction percentage
            reductions = [(1 - s/q)*100 for s, q in zip(strategic_queries, query_all_counts)]
            reduction_mean = np.mean(reductions)
            reduction_std = np.std(reductions)
            combined_data['Query Reduction (%)'].append(f"{reduction_mean:.1f} ± {reduction_std:.1f}")
        else:
            combined_data['Strategic Query Count'].append("N/A")
            combined_data['Query-All Count'].append("N/A")
            combined_data['Query Reduction (%)'].append("N/A")
        
        # MDP metrics
        bottlenecks = np.array(results['human_bottlenecks'])
        bottleneck_mean = np.mean(bottlenecks)
        bottleneck_std = np.std(bottlenecks)
        combined_data['Human Bottlenecks'].append(f"{bottleneck_mean:.1f} ± {bottleneck_std:.1f}")
        
        # State space and actions (no std dev needed)
        combined_data['Initial State Space'].append(f"{np.mean(results['initial_mdp_state_space_sizes']):.0f}")
        combined_data['Initial Actions'].append(f"{np.mean(results['initial_mdp_action_space_sizes']):.0f}")
    
    # Create DataFrame and save results
    df = pd.DataFrame(combined_data)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    # Format the display output
    print("\nResults Summary:")
    print("=" * 120)  # Increased width for better readability with ± values
    
    # Select columns for display
    display_columns = [
        'Environment', 
        'Pruning Time (s)',
        'Strategic Query Count',
        'Query-All Count',
        'Query Reduction (%)',
        'Human Bottlenecks'
    ]
    
    if not df.empty:
        print(df[display_columns].to_string(index=False))
    else:
        print("No results to display!")
    print("\nResults saved to:", output_file)
    
    return df

if __name__ == "__main__":
    # Define executor at module level
    executor = None
    
    def signal_handler(signum, frame):
        with print_lock:
            print("\nStopping all processes...")
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Set experiment parameters
    num_runs = 3
    num_models = 20
    grid_sizes = [4]
    query_threshold = 1000
    world_types = ['grid', 'four_rooms', 'puddle', 'rock']
    obstacle_percentages = [0.1]
    max_workers = 4
    
    try:
        # Create results directory
        os.makedirs("experiment_results", exist_ok=True)
        
        # Run parallel experiments with different grid sizes and obstacle percentages
        #print("Starting experiments...")
        results = run_parallel_experiments_with_obstacles(
            num_runs=num_runs,
            num_models=num_models,
            grid_sizes=grid_sizes,
            world_types=world_types,
            query_threshold=query_threshold,
            obstacle_percentages=obstacle_percentages,
            max_workers=max_workers
        )
        
        # Create and save combined results table
        if results:
            combined_df = create_combined_results_table(results)
        else:
            print("No results were generated from the experiments.")
    except Exception as e:
        print(f"Error during experiment execution: {str(e)}")
    finally:
        if executor:
            executor.shutdown(wait=False)
