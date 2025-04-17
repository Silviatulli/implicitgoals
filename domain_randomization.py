import numpy as np
from typing import Dict, Any
import random

class DomainRandomizer:
    """Class for generating randomized parameters for different environment types."""
    
    @staticmethod
    def get_randomized_human_params(world_type: str, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate randomized parameters for human models.
        
        Args:
            world_type: Type of environment ('maniskill', 'grid', etc.)
            base_params: Base parameters for the environment
            
        Returns:
            dict: Randomized parameters for the human model
        """
        randomized_params = base_params.copy()
        
        if world_type == 'maniskill':
            randomized_params.update(DomainRandomizer._randomize_maniskill_params())
        elif world_type == 'grid':
            randomized_params.update(DomainRandomizer._randomize_grid_params())
        elif world_type == 'puddle':
            randomized_params.update(DomainRandomizer._randomize_puddle_params())
        elif world_type == 'rock':
            randomized_params.update(DomainRandomizer._randomize_rock_params())
        
        return randomized_params
    
    @staticmethod
    def _randomize_maniskill_params() -> Dict[str, Any]:
        """Generate randomized parameters for ManiSkill environments."""
        return {
            'num_bins': np.random.randint(3, 7),  # Random discretization
            'grasp_threshold': np.random.uniform(0.1, 0.3),  # Random grasp threshold
            'collision_threshold': np.random.uniform(0.05, 0.15),  # Random collision threshold
            'action_noise': np.random.uniform(0.0, 0.1),  # Random action noise
            'visual_noise': np.random.uniform(0.0, 0.2),  # Random visual noise
            'dynamics_noise': np.random.uniform(0.0, 0.1),  # Random dynamics noise
        }
    
    @staticmethod
    def _randomize_grid_params() -> Dict[str, Any]:
        """Generate randomized parameters for grid worlds."""
        return {
            'obstacles_percent': np.random.uniform(0.05, 0.3),  # Random obstacle density
            'obstacle_seed': random.randint(1, 10000),  # Random obstacle layout
            'movement_noise': np.random.uniform(0.0, 0.2),  # Random movement noise
            'perception_noise': np.random.uniform(0.0, 0.1),  # Random perception noise
        }
    
    @staticmethod
    def _randomize_puddle_params() -> Dict[str, Any]:
        """Generate randomized parameters for puddle worlds."""
        return {
            'puddle_percent': np.random.uniform(0.05, 0.2),  # Random puddle density
            'puddle_cost': np.random.uniform(0.5, 2.0),  # Random puddle cost
            'obstacle_seed': random.randint(1, 10000),
            'puddle_noise': np.random.uniform(0.0, 0.15),  # Random puddle effect noise
        }
    
    @staticmethod
    def _randomize_rock_params() -> Dict[str, Any]:
        """Generate randomized parameters for rock worlds."""
        return {
            'rock_percent': np.random.uniform(0.05, 0.2),  # Random rock density
            'rock_cost': np.random.uniform(0.5, 2.0),  # Random rock cost
            'obstacle_seed': random.randint(1, 10000),
            'rock_noise': np.random.uniform(0.0, 0.15),  # Random rock effect noise
        }
    
    @staticmethod
    def add_action_noise(actions: list, noise_level: float) -> list:
        """Add noise to a list of actions.
        
        Args:
            actions: List of actions to add noise to
            noise_level: Level of noise to add
            
        Returns:
            list: Noisy actions
        """
        if noise_level <= 0:
            return actions
            
        noisy_actions = []
        for action in actions:
            if isinstance(action, (int, float)):
                noisy_action = action + np.random.normal(0, noise_level)
            else:
                noisy_action = action + np.random.normal(0, noise_level, len(action))
            noisy_actions.append(noisy_action)
        return noisy_actions
    
    @staticmethod
    def add_visual_noise(observation: np.ndarray, noise_level: float) -> np.ndarray:
        """Add noise to visual observations.
        
        Args:
            observation: Visual observation to add noise to
            noise_level: Level of noise to add
            
        Returns:
            np.ndarray: Noisy observation
        """
        if noise_level <= 0:
            return observation
            
        noise = np.random.normal(0, noise_level, observation.shape)
        return np.clip(observation + noise, 0, 1)
    
    @staticmethod
    def add_dynamics_noise(state: np.ndarray, noise_level: float) -> np.ndarray:
        """Add noise to state dynamics.
        
        Args:
            state: State to add noise to
            noise_level: Level of noise to add
            
        Returns:
            np.ndarray: Noisy state
        """
        if noise_level <= 0:
            return state
            
        noise = np.random.normal(0, noise_level, state.shape)
        return state + noise 