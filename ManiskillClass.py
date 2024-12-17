import gymnasium as gym
import mani_skill2.envs
import numpy as np
import time
import sapien.core as sapien
from DeterminizedMDP import DeterminizedMDP, identify_bottlenecks

class VisualConstrainedManiSkillEnv:
    def __init__(self, env_name="LiftCube-v0", num_bins=5):
        # Create environment
        self.env = gym.make(
            env_name,
            obs_mode="state",
            control_mode="pd_joint_vel",
            render_mode="human"
        )
        self.num_bins = num_bins
        self.discount = 0.99
        
        # Define constraints
        self.cube_position = np.array([0.0, 0.0, 0.0])
        self.grasp_region = {
            'center': self.cube_position + np.array([0, 0, 0.15]),
            'radius': 0.1
        }
        self.obstacles = [
            {
                'center': np.array([-0.3, -0.3, 0.2]),
                'size': np.array([0.2, 0.2, 0.4]),
                'color': np.array([1.0, 0.0, 0.0, 0.5])
            },
            {
                'center': np.array([0.3, 0.3, 0.2]),
                'size': np.array([0.2, 0.2, 0.4]),
                'color': np.array([1.0, 0.0, 0.0, 0.5])
            }
        ]
        
        # Reset environment
        self.obs, _ = self.env.reset()
        
        # Create state space and actions before visualization
        self._create_discrete_spaces()
        self.actions = self._create_discrete_actions()
        
        # Setup visualization
        self._setup_visualization()
        
        # Initialize state and goals
        self.init_state = self._discretize_state(self.obs)
        self.goal_states = self._create_goal_states()

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
            
        # Check conditions
        in_grasp_region = self._is_in_grasp_region(ee_pos)
        in_obstacle = any(self._check_collision(ee_pos, obs) for obs in self.obstacles)
        
        # Determine phase
        if in_obstacle:
            return ('collision',)
        elif in_grasp_region and gripper_state > 0:
            if ee_pos[2] > self.grasp_region['center'][2]:
                return ('lifting',)
            return ('grasping',)
            
        # Position-based state
        scaled = (np.array(ee_pos) + 1) * (self.num_bins / 2)
        discrete = np.clip(scaled.astype(int), 0, self.num_bins - 1)
        return tuple(discrete.tolist())

    def _create_discrete_spaces(self):
        """Create discretized state space."""
        self.state_space = []
        
        # Add position-based states
        for i in range(self.num_bins):
            for j in range(self.num_bins):
                for k in range(self.num_bins):
                    pos = self._grid_to_continuous((i, j, k))
                    if not any(self._check_collision(pos, obs) for obs in self.obstacles):
                        self.state_space.append((i, j, k))
        
        # Add phase states
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
            
            # Create visual markers for obstacles
            for obs in self.obstacles:
                builder = scene.create_actor_builder()
                builder.add_box_visual(
                    half_size=obs['size'] / 2,
                    color=obs['color'],
                    pose=sapien.Pose(obs['center'])
                )
                builder.build_static(name='obstacle')
            
            # Create visual marker for grasp region
            builder = scene.create_actor_builder()
            builder.add_sphere_visual(
                radius=self.grasp_region['radius'],
                color=np.array([0.0, 1.0, 0.0, 0.3]),
                pose=sapien.Pose(self.grasp_region['center'])
            )
            builder.build_static(name='grasp_region')
        except Exception as e:
            print(f"Warning: Could not setup visualization markers: {e}")

    # Required MDP interface methods
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
        if next_state in self.goal_states:
            return 1000
        if next_state == ('collision',):
            return -1000
        return -1

    def get_transition_probability(self, state, action, next_state):
        # Handle special states
        if state == ('collision',) or next_state == ('collision',):
            return 0.0
        
        # Handle phase transitions
        if state == ('grasping',) and next_state == ('lifting',) and action == 'move_z+':
            return 1.0
        
        # Handle normal movements
        if len(state) == 3 and len(next_state) == 3:
            current = np.array(state)
            target = np.array(next_state)
            diff = np.abs(current - target)
            
            if sum(diff) == 1:
                if action.startswith('move_'):
                    direction = action[5:]
                    axis = {'x+': 0, 'x-': 0, 'y+': 1, 'y-': 1, 'z+': 2, 'z-': 2}[direction]
                    sign = 1 if '+' in direction else -1
                    
                    # Check if movement is valid
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

def test_determinization_and_bottlenecks():
    """Test process of determinization and bottleneck identification."""
    try:
        # First create the base ManiSkill environment with discretization
        base_env = VisualConstrainedManiSkillEnv("LiftCube-v0")
        print("\nBase environment created with state space size:", len(base_env.get_state_space()))
        
        # Create determinized version
        det_env = DeterminizedMDP(base_env)
        print("\nDeterminized environment created")
        print("Number of determinized actions:", len(det_env.get_actions()))
        
        # Now identify bottlenecks using the determinized MDP
        print("\nIdentifying bottlenecks...")
        bottlenecks = identify_bottlenecks(det_env)
        
        print("\nFound bottleneck states:")
        for b in bottlenecks:
            if isinstance(b, tuple) and len(b) == 3:  # Position state
                cont_pos = base_env._grid_to_continuous(b)
                print(f"- Grid: {b} → Position: ({cont_pos[0]:.2f}, {cont_pos[1]:.2f}, {cont_pos[2]:.2f})")
            else:  # Phase state
                print(f"- Phase: {b}")
        
        # Test transitions in determinized MDP
        print("\nTesting some transitions in determinized MDP:")
        init_state = det_env.get_init_state()
        for action in det_env.get_actions()[:3]:  # Test first 3 actions
            for next_state in base_env.get_state_space()[:3]:  # Test first 3 states
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
        # Create environment and identify bottlenecks
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
        if env is not None and hasattr(env, 'env'):
            env.env.close()

if __name__ == "__main__":
    # First test determinization and identify bottlenecks
    det_env, bottlenecks = test_determinization_and_bottlenecks()
    
    if det_env is not None:
        print("\nStarting visualization with determinized transitions...")
        # Now visualize with the identified bottlenecks
        try:
            env = VisualConstrainedManiSkillEnv("LiftCube-v0")
            obs, _ = env.env.reset()
            
            step = 0
            while step < 200:
                action = env.env.action_space.sample()
                obs, reward, terminated, truncated, info = env.env.step(action)
                
                # Show both original and determinized state info
                if isinstance(obs, dict):
                    ee_pos = obs['agent']['robot_state'][:3]
                else:
                    ee_pos = obs[:3]
                
                current_state = env._discretize_state(obs)
                det_state = det_env.get_state_hash(current_state)
                is_bottleneck = current_state in bottlenecks
                
                # Print status with both original and determinized information
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
            env.env.close()
