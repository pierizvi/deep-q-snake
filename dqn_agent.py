import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# Import from your existing files
try:
    from snake_game import Direction, Point
except ImportError:
    # Fallback definitions if import fails
    from enum import Enum
    from collections import namedtuple
    
    class Direction(Enum):
        RIGHT = 1
        LEFT = 2
        UP = 3
        DOWN = 4
    
    Point = namedtuple('Point', 'x, y')

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size // 2)
        self.linear4 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = F.relu(self.linear2(x))
        x = self.dropout(x)
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './models'
        import os
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # Convert to numpy arrays first for efficiency
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        action = np.array(action, dtype=np.int64)
        reward = np.array(reward, dtype=np.float32)
        
        # Now convert to tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # Predicted Q values with current state
        pred = self.model(state)
        target = pred.clone()
        
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.95  # increased discount rate for longer-term planning
        self.memory = deque(maxlen=100_000)
        
        # We'll determine the input size dynamically
        self.model = None
        self.trainer = None
        self._model_initialized = False

    def _initialize_model(self, state_size):
        """Initialize model with correct input size"""
        if not self._model_initialized:
            print(f"Initializing neural network with {state_size} input features")
            self.model = Linear_QNet(state_size, 512, 3)
            self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)
            self._model_initialized = True

    def get_state(self, game):
        head = game.snake[0]
        
        # Basic directional points
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        # Current direction
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Enhanced obstacle detection - check multiple distances
        obstacle_positions = [obs['pos'] for obs in game.obstacles] if hasattr(game, 'obstacles') else []
        
        # Immediate danger (1 block away)
        obstacle_immediate_l = any(obs.x == head.x - 20 and obs.y == head.y for obs in obstacle_positions)
        obstacle_immediate_r = any(obs.x == head.x + 20 and obs.y == head.y for obs in obstacle_positions)
        obstacle_immediate_u = any(obs.x == head.x and obs.y == head.y - 20 for obs in obstacle_positions)
        obstacle_immediate_d = any(obs.x == head.x and obs.y == head.y + 20 for obs in obstacle_positions)
        
        # Near danger (2 blocks away)
        obstacle_near_l = any(obs.x == head.x - 40 and obs.y == head.y for obs in obstacle_positions)
        obstacle_near_r = any(obs.x == head.x + 40 and obs.y == head.y for obs in obstacle_positions)
        obstacle_near_u = any(obs.x == head.x and obs.y == head.y - 40 for obs in obstacle_positions)
        obstacle_near_d = any(obs.x == head.x and obs.y == head.y + 40 for obs in obstacle_positions)
        
        # Diagonal obstacle detection
        obstacle_diag_ul = any(obs.x == head.x - 20 and obs.y == head.y - 20 for obs in obstacle_positions)
        obstacle_diag_ur = any(obs.x == head.x + 20 and obs.y == head.y - 20 for obs in obstacle_positions)
        obstacle_diag_dl = any(obs.x == head.x - 20 and obs.y == head.y + 20 for obs in obstacle_positions)
        obstacle_diag_dr = any(obs.x == head.x + 20 and obs.y == head.y + 20 for obs in obstacle_positions)
        
        # Obstacle movement direction analysis
        obstacles_moving_towards = 0
        obstacles_moving_away = 0
        
        if hasattr(game, 'obstacles'):
            for obstacle in game.obstacles:
                obs_dir = obstacle['dir']
                obs_pos = obstacle['pos']
                
                # Determine if obstacle is moving towards snake head
                if obs_dir == Direction.RIGHT and obs_pos.x < head.x and obs_pos.y == head.y:
                    obstacles_moving_towards += 1
                elif obs_dir == Direction.LEFT and obs_pos.x > head.x and obs_pos.y == head.y:
                    obstacles_moving_towards += 1
                elif obs_dir == Direction.DOWN and obs_pos.y < head.y and obs_pos.x == head.x:
                    obstacles_moving_towards += 1
                elif obs_dir == Direction.UP and obs_pos.y > head.y and obs_pos.x == head.x:
                    obstacles_moving_towards += 1
                else:
                    obstacles_moving_away += 1
        
        # Normalize obstacle movement data
        total_obstacles = len(obstacle_positions)
        obstacles_moving_towards_ratio = obstacles_moving_towards / max(1, total_obstacles)
        obstacles_moving_away_ratio = obstacles_moving_away / max(1, total_obstacles)
        
        # Calculate distances and directions
        food_distance_x = (game.food.x - head.x) / game.w
        food_distance_y = (game.food.y - head.y) / game.h
        
        # Wall distances
        wall_dist_l = head.x / game.w
        wall_dist_r = (game.w - head.x) / game.w
        wall_dist_u = head.y / game.h
        wall_dist_d = (game.h - head.y) / game.h
        
        # Body collision detection
        body_left = any(part.x == head.x - 20 and part.y == head.y for part in game.snake[1:])
        body_right = any(part.x == head.x + 20 and part.y == head.y for part in game.snake[1:])
        body_up = any(part.x == head.x and part.y == head.y - 20 for part in game.snake[1:])
        body_down = any(part.x == head.x and part.y == head.y + 20 for part in game.snake[1:])
        
        # Enhanced obstacle distance calculations
        if obstacle_positions:
            # Closest obstacle distance
            closest_obstacle_dist = min([abs(head.x - obs.x) + abs(head.y - obs.y) for obs in obstacle_positions])
            closest_obstacle_dist = closest_obstacle_dist / (game.w + game.h)
            
            # Average obstacle distance
            avg_obstacle_dist = sum([abs(head.x - obs.x) + abs(head.y - obs.y) for obs in obstacle_positions]) / len(obstacle_positions)
            avg_obstacle_dist = avg_obstacle_dist / (game.w + game.h)
            
            # Count obstacles in each quadrant relative to head
            obstacles_quadrant_1 = sum(1 for obs in obstacle_positions if obs.x > head.x and obs.y < head.y)  # Upper right
            obstacles_quadrant_2 = sum(1 for obs in obstacle_positions if obs.x < head.x and obs.y < head.y)  # Upper left
            obstacles_quadrant_3 = sum(1 for obs in obstacle_positions if obs.x < head.x and obs.y > head.y)  # Lower left
            obstacles_quadrant_4 = sum(1 for obs in obstacle_positions if obs.x > head.x and obs.y > head.y)  # Lower right
        else:
            closest_obstacle_dist = 1.0
            avg_obstacle_dist = 1.0
            obstacles_quadrant_1 = obstacles_quadrant_2 = obstacles_quadrant_3 = obstacles_quadrant_4 = 0
        
        # Normalize quadrant counts
        obstacles_quadrant_1 /= max(1, total_obstacles)
        obstacles_quadrant_2 /= max(1, total_obstacles)
        obstacles_quadrant_3 /= max(1, total_obstacles)
        obstacles_quadrant_4 /= max(1, total_obstacles)
        
        # Enhanced danger detection
        danger_straight = (dir_r and game.is_collision(point_r)) or \
                         (dir_l and game.is_collision(point_l)) or \
                         (dir_u and game.is_collision(point_u)) or \
                         (dir_d and game.is_collision(point_d))

        danger_right = (dir_u and game.is_collision(point_r)) or \
                      (dir_d and game.is_collision(point_l)) or \
                      (dir_l and game.is_collision(point_u)) or \
                      (dir_r and game.is_collision(point_d))

        danger_left = (dir_d and game.is_collision(point_r)) or \
                     (dir_u and game.is_collision(point_l)) or \
                     (dir_r and game.is_collision(point_u)) or \
                     (dir_l and game.is_collision(point_d))

        state = [
            # Immediate danger detection (3)
            danger_straight,
            danger_right,
            danger_left,
            
            # Current direction (4)
            dir_l, dir_r, dir_u, dir_d,
            
            # Food location relative to head (4)
            game.food.x < head.x,  # food left
            game.food.x > head.x,  # food right
            game.food.y < head.y,  # food up
            game.food.y > head.y,  # food down
            
            # Distance information (2)
            food_distance_x,
            food_distance_y,
            
            # Wall distances (4)
            wall_dist_l, wall_dist_r, wall_dist_u, wall_dist_d,
            
            # Body collision detection (4)
            body_left, body_right, body_up, body_down,
            
            # Immediate obstacle detection (4)
            obstacle_immediate_l, obstacle_immediate_r, obstacle_immediate_u, obstacle_immediate_d,
            
            # Near obstacle detection (4)
            obstacle_near_l, obstacle_near_r, obstacle_near_u, obstacle_near_d,
            
            # Diagonal obstacle detection (4)
            obstacle_diag_ul, obstacle_diag_ur, obstacle_diag_dl, obstacle_diag_dr,
            
            # Obstacle movement analysis (2)
            obstacles_moving_towards_ratio,
            obstacles_moving_away_ratio,
            
            # Obstacle distance metrics (2)
            closest_obstacle_dist,
            avg_obstacle_dist,
            
            # Obstacle distribution in quadrants (4)
            obstacles_quadrant_1, obstacles_quadrant_2, obstacles_quadrant_3, obstacles_quadrant_4,
            
            # Snake length (1)
            len(game.snake) / 100.0
        ]

        # Initialize model on first call
        if not self._model_initialized:
            self._initialize_model(len(state))

        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > 1000:
            mini_sample = random.sample(self.memory, 1000)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        final_move = [0, 0, 0]
        
        # Check if we're in inference mode (watching trained model)
        if hasattr(self, '_inference_mode') and self._inference_mode:
            # PURE EXPLOITATION - NO randomness at all
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        else:
            # Training mode - use epsilon-greedy
            self.epsilon = max(0, 80 - self.n_games)
            
            if random.randint(0, 100) < self.epsilon:
                move = random.randint(0, 2)
                final_move[move] = 1
            else:
                state0 = torch.tensor(state, dtype=torch.float)
                prediction = self.model(state0)
                move = torch.argmax(prediction).item()
                final_move[move] = 1

        return final_move

    def set_inference_mode(self):
        """Set agent to inference mode (pure exploitation, no exploration)"""
        self._inference_mode = True
        self.epsilon = 0
        print("Agent set to PURE EXPLOITATION mode (epsilon locked at 0)")
        
    def set_training_mode(self):
        """Set agent back to training mode"""
        self._inference_mode = False
        print("Agent set to TRAINING mode (epsilon-greedy)")
        
    def is_exploiting(self):
        """Check if agent is in pure exploitation mode"""
        return hasattr(self, '_inference_mode') and self._inference_mode