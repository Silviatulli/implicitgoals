import pybullet as p
import pybullet_data
import numpy as np
import time
from typing import List, Tuple, Dict, Any
from MDP import MDP

class PyBulletPickAndPlaceMDP(MDP):
    """
    A PyBullet-based pick-and-place MDP for bottleneck analysis
    """
    
    def __init__(self, 
                 grid_resolution: int = 5,
                 use_gui: bool = False,
                 discrete_actions: bool = True,
                 max_steps: int = 100):
        """
        Initialize PyBullet Pick-and-Place Environment
        
        Args:
            grid_resolution: Discretization resolution for state space
            use_gui: Whether to show PyBullet GUI (set False for faster execution)
            discrete_actions: Whether to use discrete action space
            max_steps: Maximum episode length
        """
        super().__init__()
        
        self.grid_resolution = grid_resolution
        self.use_gui = use_gui
        self.discrete_actions = discrete_actions
        self.max_steps = max_steps
        self.step_count = 0
        
        self.physics_client = None
        self._connect_pybullet()
        
        self._setup_environment()
        
        self.discount = 0.99
        self.current_state = None
        
        self._create_discrete_spaces()
        
        self._define_constraints()
        
    def _connect_pybullet(self):
        """Connect to PyBullet physics engine"""
        try:
            if self.use_gui:
                self.physics_client = p.connect(p.GUI)
            else:
                self.physics_client = p.connect(p.DIRECT)
            
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
            print("✅ PyBullet connected successfully")
            
        except Exception as e:
            self.physics_client = p.connect(p.DIRECT)
    
    def _setup_environment(self):
        """Setup the PyBullet environment with robot and objects"""
        try:
            self.plane_id = p.loadURDF("plane.urdf")
            
            # Load robot (simple cube robot for pick-and-place)
            self.robot_id = p.loadURDF("r2d2.urdf", [0, 0, 0.1])
            
            # Load objects to manipulate
            self.cube_id = p.loadURDF("cube_small.urdf", [0.5, 0.5, 0.05])
            
            # Add some obstacles
            self.obstacle_ids = []
            obstacle_positions = [[0.3, 0.0, 0.05], [-0.3, 0.3, 0.05]]
            for pos in obstacle_positions:
                obs_id = p.loadURDF("cube_small.urdf", pos)
                self.obstacle_ids.append(obs_id)
                # Make obstacles red
                p.changeVisualShape(obs_id, -1, rgbaColor=[1, 0, 0, 1])
            
            # Set goal area (green cube)
            self.goal_id = p.loadURDF("cube_small.urdf", [0.8, 0.8, 0.05])
            p.changeVisualShape(self.goal_id, -1, rgbaColor=[0, 1, 0, 1])
            
        except Exception as e:
            print(f"⚠️  Warning: Could not load all URDF files: {e}")
            print("Creating minimal environment...")
            self._create_minimal_environment()
    
    def _create_minimal_environment(self):
        """Create a minimal environment if URDF loading fails"""
        # Create simple geometric shapes
        self.robot_id = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1]),
            basePosition=[0, 0, 0.1]
        )
        
        self.cube_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05]),
            basePosition=[0.5, 0.5, 0.05]
        )
    
    def _create_discrete_spaces(self):
        """Create discrete state and action spaces"""
        # Discrete positions on a grid
        self.positions = []
        for i in range(self.grid_resolution):
            for j in range(self.grid_resolution):
                x = (i / (self.grid_resolution - 1)) * 2.0 - 1.0  # Range [-1, 1]
                y = (j / (self.grid_resolution - 1)) * 2.0 - 1.0  # Range [-1, 1]
                self.positions.append((x, y))
        
        # State space: (robot_pos, cube_pos, is_holding)
        self.state_space = []
        for robot_pos in range(len(self.positions)):
            for cube_pos in range(len(self.positions)):
                for is_holding in [False, True]:
                    self.state_space.append((robot_pos, cube_pos, is_holding))
        
        # Action space
        if self.discrete_actions:
            self.actions = [
                'move_north', 'move_south', 'move_east', 'move_west',
                'pick_up', 'drop'
            ]
        else:
            self.actions = ['continuous_action']  # For continuous control
        
        # Initial state
        self.init_state = (0, len(self.positions)//4, False)  # Robot at origin, cube at quarter position
        self.current_state = self.init_state
    
    def _define_constraints(self):
        """Define goal regions and constraints for bottleneck analysis"""
        # Goal region (top-right corner)
        self.goal_region = list(range(len(self.positions) - self.grid_resolution, len(self.positions)))
        
        # Obstacle regions (for bottleneck analysis)
        center_idx = len(self.positions) // 2
        self.obstacle_region = [center_idx - 1, center_idx, center_idx + 1]
        
        # Bottleneck states (states that must be passed through)
        self.bottleneck_states = []
        for robot_pos in range(len(self.positions) // 2, len(self.positions) // 2 + 2):
            for cube_pos in range(len(self.positions)):
                for is_holding in [False, True]:
                    self.bottleneck_states.append((robot_pos, cube_pos, is_holding))
    
    # MDP
    def get_state_space(self) -> List[Tuple]:
        """Return the discrete state space"""
        return self.state_space
    
    def get_actions(self) -> List[str]:
        """Return available actions"""
        return self.actions
    
    def get_init_state(self) -> Tuple:
        """Return initial state"""
        return self.init_state
    
    def get_state_hash(self, state: Tuple) -> str:
        """Convert state to hashable string"""
        return str(state)
    
    def get_goal_states(self) -> List[Tuple]:
        """Return goal states (cube in goal region, robot can be anywhere)"""
        goal_states = []
        for robot_pos in range(len(self.positions)):
            for cube_pos in self.goal_region:
                goal_states.append((robot_pos, cube_pos, False))  # Cube dropped in goal
        return goal_states
    
    def get_transition_probability(self, state: Tuple, action: str, next_state: Tuple) -> float:
        """
        Get transition probability for discrete MDP
        Simplified deterministic transitions for bottleneck analysis
        """
        robot_pos, cube_pos, is_holding = state
        next_robot_pos, next_cube_pos, next_is_holding = next_state
        
        # Movement actions
        if action in ['move_north', 'move_south', 'move_east', 'move_west']:
            # Calculate intended new robot position
            intended_robot_pos = self._get_intended_position(robot_pos, action)
            
            # Check if transition matches
            if (next_robot_pos == intended_robot_pos and 
                next_cube_pos == cube_pos and 
                next_is_holding == is_holding):
                return 0.9  # Successful move
            elif (next_robot_pos == robot_pos and 
                  next_cube_pos == cube_pos and 
                  next_is_holding == is_holding):
                return 0.1  # Failed move (stayed in place)
            else:
                return 0.0
        
        # Pick up action
        elif action == 'pick_up':
            if (robot_pos == cube_pos and not is_holding and 
                next_robot_pos == robot_pos and next_cube_pos == cube_pos and next_is_holding):
                return 1.0
            elif state == next_state:  # No change if conditions not met
                return 1.0
            else:
                return 0.0
        
        # Drop action
        elif action == 'drop':
            if (is_holding and next_robot_pos == robot_pos and 
                next_cube_pos == robot_pos and not next_is_holding):
                return 1.0
            elif state == next_state:  # No change if not holding
                return 1.0
            else:
                return 0.0
        
        return 0.0
    
    def _get_intended_position(self, current_pos: int, action: str) -> int:
        """Calculate intended position after movement action"""
        current_x, current_y = self.positions[current_pos]
        
        if action == 'move_north':
            new_y = min(current_y + 2.0/(self.grid_resolution-1), 1.0)
        elif action == 'move_south':
            new_y = max(current_y - 2.0/(self.grid_resolution-1), -1.0)
        elif action == 'move_east':
            new_x = min(current_x + 2.0/(self.grid_resolution-1), 1.0)
        elif action == 'move_west':
            new_x = max(current_x - 2.0/(self.grid_resolution-1), -1.0)
        else:
            return current_pos
        
        # Find closest position in grid
        new_pos = current_pos
        min_dist = float('inf')
        for i, (x, y) in enumerate(self.positions):
            dist = (x - current_x)**2 + (y - current_y)**2
            if dist < min_dist:
                min_dist = dist
                new_pos = i
        
        return new_pos
    
    def get_reward(self, state: Tuple, action: str, next_state: Tuple) -> float:
        """Define reward function for pick-and-place task"""
        robot_pos, cube_pos, is_holding = state
        next_robot_pos, next_cube_pos, next_is_holding = next_state
        
        # Large reward for reaching goal
        if next_state in self.get_goal_states():
            return 100.0
        
        # Small reward for picking up cube
        if action == 'pick_up' and not is_holding and next_is_holding:
            return 10.0
        
        # Penalty for being in obstacle region
        if next_robot_pos in self.obstacle_region:
            return -10.0
        
        # Small penalty for each step (encourage efficiency)
        return -1.0
    
    def reset(self) -> Tuple:
        """Reset environment to initial state"""
        self.step_count = 0
        self.current_state = self.init_state
        
        # Reset PyBullet simulation
        if self.physics_client is not None:
            try:
                p.resetSimulation()
                self._setup_environment()
            except:
                pass  # Ignore reset errors
        
        return self.current_state
    
    def step(self, action: str) -> Tuple[Tuple, float, bool]:
        """Execute action and return next state, reward, done"""
        self.step_count += 1
        
        # Get next state probabilistically
        possible_next_states = []
        probabilities = []
        
        for next_state in self.state_space:
            prob = self.get_transition_probability(self.current_state, action, next_state)
            if prob > 0:
                possible_next_states.append(next_state)
                probabilities.append(prob)
        
        if possible_next_states:
            # Choose next state based on probabilities
            next_state = np.random.choice(
                len(possible_next_states), 
                p=np.array(probabilities) / sum(probabilities)
            )
            next_state = possible_next_states[next_state]
        else:
            next_state = self.current_state
        
        reward = self.get_reward(self.current_state, action, next_state)
        done = (next_state in self.get_goal_states() or 
                self.step_count >= self.max_steps)
        
        self.current_state = next_state
        return next_state, reward, done
    
    def render(self):
        """Render current state (if GUI is enabled)"""
        if self.use_gui and self.physics_client is not None:
            try:
                # Update robot position based on discrete state
                robot_pos_idx, cube_pos_idx, is_holding = self.current_state
                robot_x, robot_y = self.positions[robot_pos_idx]
                cube_x, cube_y = self.positions[cube_pos_idx]
                
                # Move robot
                p.resetBasePositionAndOrientation(
                    self.robot_id, 
                    [robot_x, robot_y, 0.1], 
                    [0, 0, 0, 1]
                )
                
                # Move cube (if not being held)
                if not is_holding:
                    p.resetBasePositionAndOrientation(
                        self.cube_id, 
                        [cube_x, cube_y, 0.05], 
                        [0, 0, 0, 1]
                    )
                else:
                    # Cube follows robot when held
                    p.resetBasePositionAndOrientation(
                        self.cube_id, 
                        [robot_x, robot_y, 0.2], 
                        [0, 0, 0, 1]
                    )
                
                p.stepSimulation()
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Render error: {e}")
    
    def close(self):
        """Close PyBullet connection"""
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except:
                pass
    
    def __del__(self):
        """Destructor to ensure PyBullet cleanup"""
        self.close()

# Test function
def test_pybullet_mdp():
    """Test the PyBullet MDP implementation"""
    env = PyBulletPickAndPlaceMDP(grid_resolution=3, use_gui=False)
    
    state = env.get_init_state()
    action = 'move_north'
    
    for next_state in env.get_state_space()[:5]: 
        prob = env.get_transition_probability(state, action, next_state)
        if prob > 0:
            print(f"  → {next_state}: {prob:.2f}")
    
    env.reset()
    for step in range(5):
        action = np.random.choice(env.get_actions())
        next_state, reward, done = env.step(action)
        print(f"  Step {step}: {action} → {next_state}, reward: {reward:.1f}")
        if done:
            break
    
    env.close()

if __name__ == "__main__":
    test_pybullet_mdp()