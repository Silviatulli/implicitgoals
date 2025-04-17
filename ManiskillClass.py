from MDP import MDP
import gymnasium as gym
import numpy as np
import time
import threading
import signal
import sys
import os
from DeterminizedMDP import DeterminizedMDP, identify_bottlenecks

try:
    import mani_skill2.envs
    import sapien.core as sapien
    MANISKILL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ManiSkill2 or Sapien not available: {e}")
    print("Please install them using: pip install mani-skill2")
    MANISKILL_AVAILABLE = False



# global renderer instance and lock
_shared_renderer = None # global renderer instance for mani-skill2
_renderer_lock = threading.Lock() # lock for renderer, meaning only one renderer can be used at a time

def cleanup_all_environments():
    """Clean up all running ManiSkill environments and their renderers."""
    global _shared_renderer
    try:
        with _renderer_lock:
            if _shared_renderer is not None:
                _shared_renderer = None
        # force garbage collection
        import gc
        gc.collect()
    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")

def signal_handler(signum, frame):
    """Handle Ctrl+C and other termination signals."""
    print("\nCleaning up environments...")
    cleanup_all_environments()
    sys.exit(0)

# register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class VisualConstrainedManiSkillEnv(MDP):
    def __init__(self, env_name="LiftCube-v0", num_bins=5, display_env=False, 
                 grasp_threshold=0.2, collision_threshold=0.1, action_noise=0.0):
        """Initialize the ManiSkill environment with randomized parameters.
        
        Args:
            env_name: Name of the ManiSkill environment
            num_bins: Number of bins for state discretization
            display_env: Whether to display the environment
            grasp_threshold: Threshold for successful grasping
            collision_threshold: Threshold for collision detection
            action_noise: Amount of noise to add to actions
        """
        if not MANISKILL_AVAILABLE:
            raise ImportError("ManiSkill2 is not available. Please install it using: pip install mani-skill2")
            
        super().__init__()
        global _shared_renderer
        
        self.env_name = env_name
        self.num_bins = num_bins
        self.display_env = display_env
        self.grasp_threshold = grasp_threshold
        self.collision_threshold = collision_threshold
        self.action_noise = action_noise
        
        # create environment with appropriate render mode
        render_mode = "human" if display_env else None
        self.env = gym.make(
            env_name,
            obs_mode="state",
            control_mode="pd_joint_vel",
            render_mode=render_mode
        )
        
        # if this is a display environment, manage the renderer with a lock
        if display_env:
            with _renderer_lock:
                if _shared_renderer is None:
                    _shared_renderer = self.env.unwrapped._renderer
                else:
                    # If renderer exists, use it for this environment
                    self.env.unwrapped._renderer = _shared_renderer
        
        # reset environment
        self.obs, _ = self.env.reset()
        
        # create state space and actions before visualization
        self._create_discrete_spaces()
        self.actions = self._create_discrete_actions()
        
        # setup visualization only for display environment
        if display_env:
            self._setup_visualization()
        
        # initialize state and goals
        self.init_state = self._discretize_state(self.obs)
        self.goal_states = self._create_goal_states()
        
        # set reward function
        self.reward_func = self.reward_func
        
        # Add noise to actions if specified
        if action_noise > 0:
            self._add_action_noise()

    def close(self):
        """Properly close the environment and clean up renderer."""
        global _shared_renderer
        if hasattr(self, 'env'):
            if self.display_env:
                with _renderer_lock:
                    if _shared_renderer is not None:
                        _shared_renderer = None
            self.env.close()

    def __del__(self):
        """Ensure environment is closed when object is destroyed."""
        self.close()

    def _check_collision(self, position, obstacle):
        """Check if position collides with obstacle."""
        pos = np.array(position)
        min_bound = obstacle['center'] - obstacle['size']/2
        max_bound = obstacle['center'] + obstacle['size']/2
        return np.all(pos >= min_bound) and np.all(pos <= max_bound)

    def _is_in_grasp_region(self, position):
        """Check if position is in the grasp region."""
        dist = np.linalg.norm(np.array(position) - self.grasp_region['center'])
        return dist <= self.grasp_region['radius']

    def _discretize_state(self, obs):
        """Convert continuous state to discrete state with phase information."""
        if isinstance(obs, dict):
            robot_state = obs['agent']['robot_state']
            ee_pos = robot_state[:3]  # End effector position
            gripper_state = robot_state[7] if len(robot_state) > 7 else 0
        else:
            ee_pos = obs[:3]
            gripper_state = obs[7] if len(obs) > 7 else 0
            
        # check conditions
        in_grasp_region = self._is_in_grasp_region(ee_pos)
        in_obstacle = any(self._check_collision(ee_pos, obs) for obs in self.obstacles)
        
        # determine phase
        if in_obstacle:
            return ('collision',)
        elif in_grasp_region and gripper_state > 0:
            if ee_pos[2] > self.grasp_region['center'][2]:
                return ('lifting',)
            return ('grasping',)
            
        # position-based state
        scaled = (np.array(ee_pos) + 1) * (self.num_bins / 2)
        discrete = np.clip(scaled.astype(int), 0, self.num_bins - 1)
        return tuple(discrete.tolist())

    def _create_discrete_spaces(self):
        """Create discretized state space."""
        self.state_space = []
        
        # add position-based states
        for i in range(self.num_bins):
            for j in range(self.num_bins):
                for k in range(self.num_bins):
                    pos = self._grid_to_continuous((i, j, k))
                    if not any(self._check_collision(pos, obs) for obs in self.obstacles):
                        self.state_space.append((i, j, k))
        
        # add phase states
        self.state_space.extend([
            ('grasping',),
            ('lifting',),
            ('collision',)
        ])

    def _grid_to_continuous(self, grid_pos):
        """Convert grid position to continuous space."""
        return np.array([x/self.num_bins * 2 - 1 for x in grid_pos])

    def _continuous_to_grid(self, cont_pos):
        """Convert continuous position to grid space."""
        scaled = (np.array(cont_pos) + 1) * (self.num_bins / 2)
        return tuple(np.clip(scaled.astype(int), 0, self.num_bins - 1))

    def _create_discrete_actions(self):
        """Create discrete action space."""
        return ['move_x+', 'move_x-', 'move_y+', 'move_y-', 'move_z+', 'move_z-', 'grasp']

    def _create_goal_states(self):
        """Define goal states."""
        return [(self.num_bins-1, self.num_bins-1, self.num_bins-1)]

    def _setup_visualization(self):
        """Setup visual markers for constraints."""
        try:
            scene = self.env.unwrapped._scene
            
            # create visual markers for obstacles
            for obs in self.obstacles:
                builder = scene.create_actor_builder()
                builder.add_box_visual(
                    half_size=obs['size'] / 2,
                    color=obs['color'],
                    pose=sapien.Pose(obs['center'])
                )
                builder.build_static(name='obstacle')
            
            # create visual marker for grasp region
            builder = scene.create_actor_builder()
            builder.add_sphere_visual(
                radius=self.grasp_region['radius'],
                color=np.array([0.0, 1.0, 0.0, 0.3]),
                pose=sapien.Pose(self.grasp_region['center'])
            )
            builder.build_static(name='grasp_region')
        except Exception as e:
            print(f"Warning: Could not setup visualization markers: {e}")

    def get_state_space(self):
        return self.state_space

    def get_actions(self):
        return self.actions

    def get_init_state(self):
        return self.init_state

    def get_state_hash(self, state):
        return str(state)

    def get_goal_states(self):
        return self.goal_states

    def reward_func(self, state, action, next_state):
        """Reward function for the MDP."""
        if next_state in self.goal_states:
            return 1000
        if next_state == ('collision',):
            return -1000
        if next_state == ('lifting',) and state != ('lifting',):
            return 100
        if next_state == ('grasping',) and state != ('grasping',):
            return 50
        return -1

    def get_transition_probability(self, state, action, next_state):
        """Get transition probability for state-action-next_state tuple."""
        # handle special states
        if state == ('collision',) or next_state == ('collision',):
            return 1.0 if state == next_state else 0.0
        
        # handle phase transitions
        if state == ('grasping',) and next_state == ('lifting',):
            return 1.0 if action == 'move_z+' else 0.0
        
        # handle normal movements
        if len(state) == 3 and len(next_state) == 3:
            current = np.array(state)
            target = np.array(next_state)
            diff = np.abs(current - target)
            
            if sum(diff) == 1:
                if action.startswith('move_'):
                    direction = action[5:]
                    axis = {'x+': 0, 'x-': 0, 'y+': 1, 'y-': 1, 'z+': 2, 'z-': 2}[direction]
                    sign = 1 if '+' in direction else -1
                    
                    # check if movement is valid
                    if diff[axis] == 1 and sign * (target[axis] - current[axis]) > 0:
                        next_pos = self._grid_to_continuous(target)
                        if not any(self._check_collision(next_pos, obs) for obs in self.obstacles):
                            return 1.0
        return 0.0

    def visualize_state(self, obs, bottlenecks):
        """Visualize current state with constraints."""
        try:
            if isinstance(obs, dict):
                ee_pos = obs['agent']['robot_state'][:3]
            else:
                ee_pos = obs[:3]
            
            current_state = self._discretize_state(obs)
            is_bottleneck = current_state in bottlenecks
            in_grasp = self._is_in_grasp_region(ee_pos)
            in_collision = any(self._check_collision(ee_pos, obs) for obs in self.obstacles)
            
            status = []
            if is_bottleneck:
                status.append("BOTTLENECK!")
            if in_grasp:
                status.append("GRASP REGION")
            if in_collision:
                status.append("COLLISION!")
            
            state_info = (f"\rState: {current_state} | "
                         f"Position: ({ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}) | "
                         f"{' | '.join(status if status else [''])}")
            print(state_info, end="")
            
        except Exception as e:
            print(f"\rVisualization error: {e}", end="")

    def check_goal_reached(self, state):
        """Check if the given state is a goal state.
        
        Args:
            state: The state to check
            
        Returns:
            bool: True if the state is a goal state, False otherwise
        """
        return state in self.goal_states

    def _add_action_noise(self):
        """Add noise to the action space."""
        noisy_actions = []
        for action in self.actions:
            # Add Gaussian noise to the action
            noisy_action = action + np.random.normal(0, self.action_noise, len(action))
            noisy_actions.append(noisy_action)
        self.actions = noisy_actions
    
    def _check_grasp(self, state):
        """Check if the object is grasped with the randomized threshold."""
        # Implementation depends on the specific environment
        # This is a placeholder for the actual grasp check
        return self._compute_grasp_quality(state) > self.grasp_threshold
    
    def _check_collision(self, state):
        """Check for collisions with the randomized threshold."""
        # Implementation depends on the specific environment
        # This is a placeholder for the actual collision check
        return self._compute_collision_risk(state) > self.collision_threshold

def test_determinization_and_bottlenecks():
    """Test process of determinization and bottleneck identification."""
    env = None
    try:
        # first create the base ManiSkill environment with discretization
        base_env = VisualConstrainedManiSkillEnv("LiftCube-v0")
        print("\nBase environment created with state space size:", len(base_env.get_state_space()))
        
        # create determinized version
        det_env = DeterminizedMDP(base_env)
        print("\nDeterminized environment created")
        print("Number of determinized actions:", len(det_env.get_actions()))
        
        # now identify bottlenecks using the determinized MDP
        print("\nIdentifying bottlenecks...")
        bottlenecks = identify_bottlenecks(det_env)
        
        print("\nFound bottleneck states:")
        for b in bottlenecks:
            if isinstance(b, tuple) and len(b) == 3:  # position state
                cont_pos = base_env._grid_to_continuous(b)
                print(f"- Grid: {b} → Position: ({cont_pos[0]:.2f}, {cont_pos[1]:.2f}, {cont_pos[2]:.2f})")
            else:  # phase state
                print(f"- Phase: {b}")
        
        # test transitions in determinized MDP
        print("\nTesting some transitions in determinized MDP:")
        init_state = det_env.get_init_state()
        for action in det_env.get_actions()[:3]:  # test first 3 actions
            for next_state in base_env.get_state_space()[:3]:  # test first 3 states
                prob = det_env.get_transition_probability(init_state, action, next_state)
                if prob > 0:
                    print(f"Transition: {init_state} --({action})--> {next_state} = {prob}")
        
        return det_env, bottlenecks
        
    except Exception as e:
        print(f"\nError in determinization test: {e}")
        import traceback
        traceback.print_exc()
        return None, []
    

def visualize_enhanced_environment():
    env = None
    try:
        # create environment and identify bottlenecks
        env = VisualConstrainedManiSkillEnv("LiftCube-v0")
        bottlenecks = identify_bottlenecks(env)
        
        print("\nConstraints and Bottlenecks:")
        print("- Red boxes: Obstacles (must avoid)")
        print("- Green sphere: Grasp region (must pass through)")
        print("\nBottleneck States:")
        for b in bottlenecks:
            if len(b) == 3:
                cont_pos = env._grid_to_continuous(b)
                print(f"- Grid: {b} → Position: ({cont_pos[0]:.2f}, {cont_pos[1]:.2f}, {cont_pos[2]:.2f})")
            else:
                print(f"- Phase: {b}")
        
        print("\nStarting visualization... (Press Ctrl+C to stop)")
        obs, _ = env.env.reset()
        
        step = 0
        while step < 200:
            action = env.env.action_space.sample()
            obs, reward, terminated, truncated, info = env.env.step(action)
            
            env.visualize_state(obs, bottlenecks)
            env.env.render()
            time.sleep(0.01)
            
            if terminated or truncated:
                obs, _ = env.env.reset()
            
            step += 1
            
    except Exception as e:
        print(f"\nError in visualization: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if env is not None:
            env.close()

if __name__ == "__main__":
    # first test determinization and identify bottlenecks
    det_env, bottlenecks = test_determinization_and_bottlenecks()
    
    if det_env is not None:
        print("\nStarting visualization with determinized transitions...")
        # now visualize with the identified bottlenecks
        try:
            env = VisualConstrainedManiSkillEnv("LiftCube-v0", display_env=True)
            obs, _ = env.env.reset()
            
            step = 0
            while step < 200:
                action = env.env.action_space.sample()
                obs, reward, terminated, truncated, info = env.env.step(action)
                
                # show both original and determinized state info
                if isinstance(obs, dict):
                    ee_pos = obs['agent']['robot_state'][:3]
                else:
                    ee_pos = obs[:3]
                
                current_state = env._discretize_state(obs)
                det_state = det_env.get_state_hash(current_state)
                is_bottleneck = current_state in bottlenecks
                
                # print status with both original and determinized information
                print(f"\rStep: {step:3d} | "
                      f"Original State: {current_state} | "
                      f"Determinized State: {det_state} | "
                      f"Position: ({ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}) | "
                      f"{'BOTTLENECK!' if is_bottleneck else ''}", end="")
                
                env.env.render()
                time.sleep(0.01)
                
                if terminated or truncated:
                    obs, _ = env.env.reset()
                
                step += 1
                
        except Exception as e:
            print(f"\nError in visualization: {e}")
        finally:
            env.close()
