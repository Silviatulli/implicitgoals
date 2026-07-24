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
import platform

from experiments import (
    generate_and_visualize_gridworld, 
    generate_and_visualize_puddleworld, 
    generate_and_visualize_rockworld,
    generate_and_visualize_taxiworld
)
from maximal_achievable_subsets import (
    find_maximally_achievable_subsets, 
    find_maximally_achievable_subsets_no_pruning, 
    improved_find_maximally_achievable_subsets
)
from QueryMDP import QueryMDP, simulate_policy_unachievable, simulate_policy_query_all
from DeterminizedMDP import identify_bottlenecks
from overcooked_env import (
    build_transition_matrix_nomove as overcooked_build_transition_matrix,
    extract_bottlenecks_nomove as overcooked_extract_bottlenecks,
    remove_toboggan_redundancies as overcooked_remove_toboggan_redundancies,
    find_maximally_achievable_subsets as overcooked_find_maximally_achievable_subsets,
    decode_subsets_to_2d_nomove as overcooked_decode_subsets_to_2d_nomove,
    solve_query_mdp_exact as overcooked_solve_query_mdp_exact,
    NUM_POT as OVERCOOKED_NUM_POT,
    SERVE_ACTION as OVERCOOKED_SERVE_ACTION,
    CLIENT_SERVED as OVERCOOKED_CLIENT_SERVED,
)

IS_MACOS = platform.system() == 'Darwin'

PYBULLET_AVAILABLE = False
try:
    from PyBulletMDPClass import PyBulletPickAndPlaceMDP
    PYBULLET_AVAILABLE = True
except ImportError:
    pass

def get_safe_process_count():
    cpu_count = multiprocessing.cpu_count()
    if IS_MACOS:
        return max(1, cpu_count // 3)
    else:
        return max(1, cpu_count // 2)

# Note: on platforms using the 'spawn' start method (macOS, Windows), each worker process
# re-imports this module and gets its own independent copy of these, not a value truly shared
# with the main process or other workers. Thus, if a worker increments these counters, it 
# won't be reflected in the main process. 
pruning_counter = Value('i', 0)
no_pruning_counter = Value('i', 0)
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

def init_worker():
    global inspect
    import inspect

# Minimal logging configuration
logging.basicConfig(level=logging.ERROR)

def generate_pybullet_model(grid_size, model_num, **kwargs):
    """Generate PyBullet model for parallel execution"""
    if not PYBULLET_AVAILABLE:
        return None
        
    try:
        env = PyBulletPickAndPlaceMDP(
            grid_resolution=min(grid_size, 4),
            use_gui=False,
            discrete_actions=True,
            max_steps=100
        )
        return env
    except Exception:
        return None

def generate_human_model(world_type, grid_size, obstacle_percent, puddle_percent, rock_percent, model_num, divide_rooms=False):
    M_H = None
    
    try:
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
        elif world_type == 'taxi':
            M_H = generate_and_visualize_taxiworld(
                size=grid_size,
                start=(0,0),
                goal=(grid_size-1,grid_size-1),
                obstacles_percent=obstacle_percent,
                model_type=f"Human Model {model_num}",
                obstacle_seed=random.randint(1, 10000)
            )
        elif world_type == 'pybullet':
            M_H = generate_pybullet_model(grid_size, model_num)
    except Exception:
        pass
        
    return M_H

def generate_robot_model(world_type, grid_size, obstacle_percent, puddle_percent, rock_percent, divide_rooms=False):
    M_R = None
    
    try:
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
        elif world_type == 'taxi':
            M_R = generate_and_visualize_taxiworld(
                size=grid_size,
                start=(0,0),
                goal=(grid_size-1,grid_size-1),
                obstacles_percent=obstacle_percent,
                model_type="Robot Model",
                obstacle_seed=random.randint(1, 10000)
            )
        elif world_type == 'pybullet':
            M_R = generate_pybullet_model(grid_size, "Robot")
    except Exception:
        pass
        
    return M_R

def cleanup_pybullet_models(models):
    for model in models:
        if hasattr(model, 'close'):
            try:
                model.close()
            except:
                pass 

def run_overcooked_experiment(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Overcooked (no-move) pipeline — deterministic, no grid/obstacles, so unlike
    the grid-family games, repeating trials only measures timing noise (mirrors
    notebook 9's N_REPEATS). There is no tractable "without pruning" alternative
    for this domain (solving the unfiltered 56-bottleneck Query MDP would need
    3^56 states), so the no-pruning columns are filled with the pruned numbers,
    matching notebook 9.
    """
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

    T_base, RECIPES, _ = overcooked_build_transition_matrix(verbose=False)
    T_R = T_base.copy()
    for r in RECIPES:
        T_R[r * OVERCOOKED_NUM_POT, OVERCOOKED_SERVE_ACTION] = OVERCOOKED_CLIENT_SERVED

    num_trials = 3 if IS_MACOS else 5

    for _ in range(num_trials):
        results["initial_mdp_state_space_sizes"].append(T_R.shape[0])
        results["initial_mdp_action_space_sizes"].append(T_R.shape[1])

        t0 = time.time()
        B = overcooked_extract_bottlenecks([T_R], verbose=False)
        B_filter = overcooked_remove_toboggan_redundancies(T_R, B)
        t1 = time.time()

        I = overcooked_find_maximally_achievable_subsets(B_filter, T_R)
        I_decoded = overcooked_decode_subsets_to_2d_nomove(I)
        t2 = time.time()

        overcooked_solve_query_mdp_exact(I_decoded)
        t3 = time.time()

        bottleneck_time = t1 - t0
        maximal_time    = t2 - t1
        policy_time     = t3 - t2

        results["bottleneck_finding_times"].append(bottleneck_time)
        results["maximal_achievable_pruning_times"].append(maximal_time)
        results["maximal_achievable_no_pruning_times"].append(maximal_time)
        results["pruning"]["times"].append(maximal_time)
        results["pruning"]["subsets"].append(len(I))
        results["no_pruning"]["times"].append(maximal_time)
        results["no_pruning"]["subsets"].append(len(I))
        results["policy_computation_pruning_times"].append(policy_time)
        results["policy_computation_no_pruning_times"].append(policy_time)
        results["human_bottlenecks"].append(len(B_filter))

        gc.collect()

    return results


def run_single_experiment(params: Dict[str, Any]) -> Dict[str, Any]:
    trial_seed = params.get('seed', 0)
    np.random.seed(trial_seed)
    random.seed(trial_seed)
    
    try:
        world_type = params['world_type']

        if world_type == 'overcooked':
            return run_overcooked_experiment(params)

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
        
        if world_type == 'pybullet':
            num_trials = 2 if IS_MACOS else 3
        else:
            num_trials = 3 if IS_MACOS else 5
        
        for trial in range(num_trials):
            M_R = generate_robot_model(
                world_type=world_type,
                grid_size=grid_size,
                obstacle_percent=obstacle_percent,
                puddle_percent=puddle_percent,
                rock_percent=rock_percent
            )
            
            if not M_R:
                continue
                
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
                if world_type == 'pybullet':
                    cleanup_pybullet_models([M_R])
                continue
            
            results["initial_mdp_state_space_sizes"].append(len(M_R.state_space))
            results["initial_mdp_action_space_sizes"].append(len(M_R.get_actions()))
            
            start_time = time.time()
            B = set()
            for M in M_H_list:
                B.update(tuple(b) for b in identify_bottlenecks(M))
            bottleneck_finding_time = time.time() - start_time    
            results["bottleneck_finding_times"].append(bottleneck_finding_time)
            results["human_bottlenecks"].append(len(B))
            
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
            
            if len(B) <= 12:
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
            
            if len(B) > 0:
                try:
                    query_mdp = QueryMDP(M_R, list(B), list(I_pruning))
                    
                    start_time = time.time()
                    strategic_count = simulate_policy_unachievable(query_mdp, list(B), query_threshold)
                    policy_pruning_time = time.time() - start_time
                    results["policy_computation_pruning_times"].append(policy_pruning_time)
                    results["query_counts"].append(strategic_count)
                    
                    start_time = time.time()
                    query_all_count = simulate_policy_query_all(query_mdp, list(B), query_threshold)
                    policy_no_pruning_time = time.time() - start_time
                    results["policy_computation_no_pruning_times"].append(policy_no_pruning_time)
                    results["query_all_counts"].append(query_all_count)
                    
                except Exception:
                    pass
            
            if world_type == 'pybullet':
                cleanup_pybullet_models([M_R] + M_H_list)
            
            gc.collect()
        
        return results
        
    except Exception as e:
        logging.error(f"Error in run_single_experiment ({world_type}): {str(e)}")
        return None

def get_available_world_types():
    """Get available world types based on platform and dependencies"""
    base_worlds = ['grid', 'puddle', 'rock', 'overcooked']

    if PYBULLET_AVAILABLE and False:
        base_worlds.append('pybullet')

    if not IS_MACOS:
        base_worlds.append('taxi')

    return base_worlds

def run_parallel_experiments_with_pybullet(num_runs: int, grid_sizes: list, 
                                         human_model_counts: list, 
                                         query_threshold: int, 
                                         obstacle_percentages: list, 
                                         max_workers: int = None):
    world_types = get_available_world_types()

    all_environments_results = {}
    experiment_params = []

    max_workers = max_workers or get_safe_process_count()
    batch_size = max(max_workers, 6 if PYBULLET_AVAILABLE else 10)
    
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
                        
                elif world_type == 'overcooked':
                    # No grid, no obstacles — grid_size/obstacle_percent don't apply.
                    world_config = f"{world_type}_{grid_size}_{num_models}_models_nomove"
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

                elif world_type == 'pybullet':
                    world_config = f"{world_type}_{grid_size}_{num_models}_models_physics"
                    pybullet_runs = max(1, num_runs // 2) if IS_MACOS else num_runs
                    for _ in range(pybullet_runs):
                        params = {
                            'world_type': world_type,
                            'grid_size': min(grid_size, 4),  
                            'num_models': min(num_models, 8),  
                            'query_threshold': query_threshold,
                            'obstacle_percent': 0.1,
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
    
    for i in range(0, len(experiment_params), batch_size):
        batch = experiment_params[i:i + batch_size]
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_single_experiment, params) for _, params in batch]
            
            for j, future in enumerate(futures):
                try:
                    result = future.result(timeout=300)
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
                                
                except Exception:
                    continue
        
        gc.collect()
    
    return all_environments_results

def create_enhanced_results_table(all_environments_results, output_file="experiment_results/pybullet_comparison.csv"):
    """Create enhanced results table including PyBullet results"""
    
    os.makedirs("experiment_results", exist_ok=True)
    
    combined_data = {
        'Environment': [],
        'Grid Size': [],
        'Number of Human Models': [],
        'Obstacle Percentage': [],
        'Environment Type': [], 
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
            
        env_parts = env_type.split('_')
        
        if "pybullet" in env_type:
            environment_name = "PyBullet"
            env_category = "3D Physics"
            grid_size = int(env_parts[1])
            num_models = int(env_parts[2])
            obstacle_percent = "Physics"
        elif "four_rooms" in env_type:
            environment_name = "Four Rooms"
            env_category = "2D Grid"
            grid_size = int(env_parts[2])
            num_models = int(env_parts[3])
            obstacle_percent = 0.0
        elif "overcooked" in env_type:
            environment_name = "Overcooked"
            env_category = "Recipe Game"
            grid_size = "n/a"
            num_models = int(env_parts[2])
            obstacle_percent = "N/A"
        else:
            environment_name = env_parts[0].capitalize()
            env_category = "2D Grid"
            grid_size = int(env_parts[1])
            num_models = int(env_parts[2])
            obstacle_percent = float(env_parts[-1])
        
        combined_data['Environment'].append(environment_name)
        combined_data['Environment Type'].append(env_category)
        combined_data['Grid Size'].append(grid_size)
        combined_data['Number of Human Models'].append(num_models)
        combined_data['Obstacle Percentage'].append(obstacle_percent)
        
        try:
            bottleneck_times = np.array(results.get('bottleneck_finding_times', []))
            pruning_times = np.array(results["pruning"]["times"])
            policy_pruning_times = np.array(results.get('policy_computation_pruning_times', []))
            no_pruning_times = np.array(results["no_pruning"]["times"]) if results["no_pruning"]["times"] else []
            policy_no_pruning_times = np.array(results.get('policy_computation_no_pruning_times', []))
            
            min_len = min(len(arr) for arr in [bottleneck_times, pruning_times, policy_pruning_times] if len(arr) > 0)
            
            if min_len == 0:
                continue
                
            bottleneck_times = bottleneck_times[:min_len]
            pruning_times = pruning_times[:min_len]
            policy_pruning_times = policy_pruning_times[:min_len]
            
            combined_data['Finding Bottlenecks Time (s)'].append(
                f"{np.mean(bottleneck_times):.3f} ± {np.std(bottleneck_times):.3f}")
            combined_data['Finding Maximal Achievable With Pruning (s)'].append(
                f"{np.mean(pruning_times):.3f} ± {np.std(pruning_times):.3f}")
            
            if len(no_pruning_times) > 0:
                no_pruning_times = no_pruning_times[:min_len]
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
                policy_no_pruning_times = policy_no_pruning_times[:min_len]
                combined_data['Computing Policy Without Pruning (s)'].append(
                    f"{np.mean(policy_no_pruning_times):.3f} ± {np.std(policy_no_pruning_times):.3f}")
            else:
                combined_data['Computing Policy Without Pruning (s)'].append("N/A")

            total_pruning = bottleneck_times + pruning_times + policy_pruning_times
            combined_data['Total Runtime With Pruning (s)'].append(
                f"{np.mean(total_pruning):.3f} ± {np.std(total_pruning):.3f}")
            
            if len(no_pruning_times) > 0 and len(policy_no_pruning_times) > 0:
                total_no_pruning = bottleneck_times + no_pruning_times + policy_no_pruning_times
                combined_data['Total Runtime Without Pruning (s)'].append(
                    f"{np.mean(total_no_pruning):.3f} ± {np.std(total_no_pruning):.3f}")
                
                improvement = ((np.mean(total_no_pruning) - np.mean(total_pruning)) / 
                             np.mean(total_no_pruning) * 100)
                improvement_std = np.std([(n - p)/n * 100 for n, p in zip(total_no_pruning, total_pruning)])
                combined_data['Runtime Improvement (%)'].append(
                    f"{improvement:.1f} ± {improvement_std:.1f}")
            else:
                combined_data['Total Runtime Without Pruning (s)'].append("N/A")
                combined_data['Runtime Improvement (%)'].append("N/A")
            
            bottlenecks = np.array(results['human_bottlenecks'][:min_len])
            combined_data['Human Bottlenecks'].append(
                f"{np.mean(bottlenecks):.1f} ± {np.std(bottlenecks):.1f}")
            combined_data['Initial State Space'].append(
                f"{np.mean(results['initial_mdp_state_space_sizes'][:min_len]):.0f}")
            combined_data['Initial Actions'].append(
                f"{np.mean(results['initial_mdp_action_space_sizes'][:min_len]):.0f}")
                
        except Exception as e:
            logging.error(f"Error processing results for {env_type}: {str(e)}")
            continue
    
    df = pd.DataFrame(combined_data)
    df.to_csv(output_file, index=False)
    
    pybullet_results = df[df['Environment Type'] == '3D Physics']
    if not pybullet_results.empty:
        pybullet_file = output_file.replace('.csv', '_pybullet_only.csv')
        pybullet_results.to_csv(pybullet_file, index=False)
    
    backup_file = output_file.replace('.csv', '_detailed_backup.csv')
    detailed_data = []
    for env_type, results in all_environments_results.items():
        for i in range(len(results.get('pruning', {}).get('times', []))):
            row = {'Environment': env_type}
            for key, values in results.items():
                if isinstance(values, dict):
                    for subkey, subvalues in values.items():
                        if i < len(subvalues):
                            row[f"{key}_{subkey}"] = subvalues[i]
                else:
                    if i < len(values):
                        row[key] = values[i]
            detailed_data.append(row)
    
    if detailed_data:
        detailed_df = pd.DataFrame(detailed_data)
        detailed_df.to_csv(backup_file, index=False)
    
    return df

def main():
    num_runs = 3
    grid_sizes = [4]
    human_model_counts = [3, 4]
    obstacle_percentages = [0.1, 0.15]
    max_workers = 10
    query_threshold = 1000
    

    
    try:
        # Create results directory
        os.makedirs("experiment_results", exist_ok=True)
        
        # Run experiments
        start_time = time.time()
        results = run_parallel_experiments_with_pybullet(
            num_runs=num_runs,
            grid_sizes=grid_sizes,
            human_model_counts=human_model_counts,
            query_threshold=query_threshold,
            obstacle_percentages=obstacle_percentages,
            max_workers=max_workers
        )
        total_time = time.time() - start_time
        
        if results:
            combined_df = create_enhanced_results_table(
                results, 
                "experiment_results/enhanced_pybullet_comparison.csv"
            )
            
            print(f"\nResults saved to experiment_results/")
            print(f"Environments tested: {len(results)}")
            print(f"Total runtime: {total_time/60:.1f} minutes")
            
            if not combined_df.empty:
                print(f"\nSummary:")
                for env_type in combined_df['Environment Type'].unique():
                    env_data = combined_df[combined_df['Environment Type'] == env_type]
                    print(f"  {env_type}: {len(env_data)} configurations")
            
            print("Experiment completed successfully!")
            
        else:
            print("No results were generated.")
            
    except Exception as e:
        print(f"Error during experiment execution: {str(e)}")
        if 'results' in locals() and results:
            try:
                create_enhanced_results_table(results, "experiment_results/partial_results.csv")
                print("Partial results saved to experiment_results/partial_results.csv")
            except:
                pass

if __name__ == "__main__":
    main()