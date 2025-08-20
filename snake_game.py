import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple
import sys

pygame.init()
font = pygame.font.Font(None, 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# RGB Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
ORANGE = (255, 165, 0)
DARK_ORANGE = (200, 100, 0)
YELLOW = (255, 255, 0)

BLOCK_SIZE = 20
SPEED = 15  # Slightly increased for better training observation
NUM_OBSTACLES = 6  # Increased number of obstacles
OBSTACLE_SPEED = 4  # Move obstacles every N frames
MIN_OBSTACLE_DISTANCE = 3  # Minimum blocks between obstacles and snake at start

class SnakeGameAI:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake AI with Enhanced Moving Obstacles')
        self.clock = pygame.time.Clock()
        self.reset()
        
    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self.frame_iteration = 0
        
        # Initialize moving obstacles with enhanced logic
        self.obstacles = []
        self._init_obstacles()
        self._place_food()
        
    def _init_obstacles(self):
        """Initialize moving obstacles with better distribution and movement patterns"""
        self.obstacles = []
        grid_width = self.w // BLOCK_SIZE
        grid_height = self.h // BLOCK_SIZE
        
        for i in range(NUM_OBSTACLES):
            attempts = 0
            while attempts < 200:  # More attempts for better placement
                # Create more diverse spawn positions
                if i % 3 == 0:  # Some obstacles near edges
                    x = random.choice([1, 2, grid_width-3, grid_width-2]) * BLOCK_SIZE
                    y = random.randint(2, grid_height-3) * BLOCK_SIZE
                elif i % 3 == 1:  # Some obstacles in middle areas
                    x = random.randint(grid_width//4, 3*grid_width//4) * BLOCK_SIZE
                    y = random.randint(2, grid_height-3) * BLOCK_SIZE
                else:  # Random placement
                    x = random.randint(2, grid_width-3) * BLOCK_SIZE
                    y = random.randint(2, grid_height-3) * BLOCK_SIZE
                
                pos = Point(x, y)
                
                # Check distance from snake
                min_distance = BLOCK_SIZE * MIN_OBSTACLE_DISTANCE
                safe_from_snake = True
                for snake_part in self.snake:
                    distance = abs(pos.x - snake_part.x) + abs(pos.y - snake_part.y)
                    if distance < min_distance:
                        safe_from_snake = False
                        break
                
                # Check distance from other obstacles (prevent clustering)
                safe_from_obstacles = True
                for existing_obstacle in self.obstacles:
                    distance = abs(pos.x - existing_obstacle['pos'].x) + abs(pos.y - existing_obstacle['pos'].y)
                    if distance < BLOCK_SIZE * 2:  # At least 2 blocks apart
                        safe_from_obstacles = False
                        break
                
                if safe_from_snake and safe_from_obstacles:
                    # Create obstacle with enhanced movement pattern
                    direction = random.choice(list(Direction))
                    movement_pattern = random.choice(['linear', 'bouncing', 'wandering'])
                    speed_multiplier = random.uniform(0.8, 1.2)  # Slight speed variation
                    
                    obstacle = {
                        'pos': pos,
                        'dir': direction,
                        'pattern': movement_pattern,
                        'speed_multiplier': speed_multiplier,
                        'direction_timer': random.randint(10, 30),  # How long to move in current direction
                        'original_direction_timer': random.randint(10, 30)
                    }
                    self.obstacles.append(obstacle)
                    break
                attempts += 1
                    
    def _move_obstacles(self):
        """Enhanced obstacle movement with different patterns"""
        for obstacle in self.obstacles:
            obstacle['direction_timer'] -= 1
            
            # Different movement patterns
            if obstacle['pattern'] == 'linear':
                # Move in straight lines, change direction at walls or randomly
                if obstacle['direction_timer'] <= 0 or random.random() < 0.02:
                    obstacle['dir'] = random.choice(list(Direction))
                    obstacle['direction_timer'] = random.randint(15, 40)
                    
            elif obstacle['pattern'] == 'bouncing':
                # Bounce off walls more predictably
                if obstacle['direction_timer'] <= 0:
                    # Random direction change
                    obstacle['dir'] = random.choice(list(Direction))
                    obstacle['direction_timer'] = random.randint(20, 50)
                    
            elif obstacle['pattern'] == 'wandering':
                # More frequent direction changes
                if obstacle['direction_timer'] <= 0 or random.random() < 0.08:
                    # Prefer perpendicular directions for more interesting movement
                    current_dir = obstacle['dir']
                    if current_dir in [Direction.LEFT, Direction.RIGHT]:
                        new_directions = [Direction.UP, Direction.DOWN, current_dir]
                    else:
                        new_directions = [Direction.LEFT, Direction.RIGHT, current_dir]
                    obstacle['dir'] = random.choice(new_directions)
                    obstacle['direction_timer'] = random.randint(8, 25)
            
            # Move obstacle in current direction
            x, y = obstacle['pos'].x, obstacle['pos'].y
            
            if obstacle['dir'] == Direction.RIGHT:
                x += BLOCK_SIZE
            elif obstacle['dir'] == Direction.LEFT:
                x -= BLOCK_SIZE
            elif obstacle['dir'] == Direction.DOWN:
                y += BLOCK_SIZE
            elif obstacle['dir'] == Direction.UP:
                y -= BLOCK_SIZE
            
            # Enhanced wall bouncing logic
            direction_changed = False
            if x < 0:
                x = 0
                obstacle['dir'] = Direction.RIGHT
                direction_changed = True
            elif x >= self.w:
                x = self.w - BLOCK_SIZE
                obstacle['dir'] = Direction.LEFT
                direction_changed = True
            
            if y < 0:
                y = 0
                obstacle['dir'] = Direction.DOWN
                direction_changed = True
            elif y >= self.h:
                y = self.h - BLOCK_SIZE
                obstacle['dir'] = Direction.UP
                direction_changed = True
            
            if direction_changed:
                obstacle['direction_timer'] = obstacle['original_direction_timer']
            
            obstacle['pos'] = Point(x, y)
        
    def _place_food(self):
        """Enhanced food placement that avoids obstacles and considers safe paths"""
        max_attempts = 100
        for attempt in range(max_attempts):
            x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
            self.food = Point(x, y)
            
            # Check if food overlaps with snake
            if self.food in self.snake:
                continue
                
            # Check if food overlaps with obstacles
            obstacle_positions = [obs['pos'] for obs in self.obstacles]
            if self.food in obstacle_positions:
                continue
                
            # Check if food is too close to obstacles (leave some space)
            too_close = False
            for obs_pos in obstacle_positions:
                distance = abs(self.food.x - obs_pos.x) + abs(self.food.y - obs_pos.y)
                if distance < BLOCK_SIZE * 1.5:  # At least 1.5 blocks away
                    too_close = True
                    break
            
            if not too_close:
                break
        
    def _get_safe_path_reward(self):
        """Calculate reward based on available safe paths around the snake"""
        head = self.head
        safe_directions = 0
        
        # Check all 4 directions for safety
        directions_to_check = [
            Point(head.x + BLOCK_SIZE, head.y),  # Right
            Point(head.x - BLOCK_SIZE, head.y),  # Left
            Point(head.x, head.y + BLOCK_SIZE),  # Down
            Point(head.x, head.y - BLOCK_SIZE)   # Up
        ]
        
        for point in directions_to_check:
            if not self.is_collision(point):
                safe_directions += 1
        
        # Reward having more escape routes
        return safe_directions * 0.1
        
    def play_step(self, action):
        self.frame_iteration += 1
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # Move obstacles with enhanced timing
        if self.frame_iteration % OBSTACLE_SPEED == 0:
            self._move_obstacles()
        
        # Store previous distance to food for reward calculation
        prev_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        
        # Move snake
        self._move(action)
        self.snake.insert(0, self.head)
        
        # Check if game over
        reward = 0
        game_over = False
        
        # Enhanced timeout that scales with obstacles and snake length
        timeout = 150 * len(self.snake) + 300 + (NUM_OBSTACLES * 50)
        
        if self.is_collision() or self.frame_iteration > timeout:
            game_over = True
            reward = -15  # Stronger penalty for collision/timeout
            return reward, game_over, self.score
        
        # Calculate new distance to food
        new_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        
        # Enhanced obstacle avoidance rewards
        obstacle_reward = 0
        min_obstacle_distance = float('inf')
        
        for obstacle in self.obstacles:
            dist_to_obstacle = abs(self.head.x - obstacle['pos'].x) + abs(self.head.y - obstacle['pos'].y)
            min_obstacle_distance = min(min_obstacle_distance, dist_to_obstacle)
            
            if dist_to_obstacle <= BLOCK_SIZE:  # Very close
                obstacle_reward -= 0.5
            elif dist_to_obstacle <= BLOCK_SIZE * 2:  # Close
                obstacle_reward -= 0.2
            elif dist_to_obstacle <= BLOCK_SIZE * 3:  # Moderately close
                obstacle_reward -= 0.1
        
        # Reward for maintaining safe distance from all obstacles
        if min_obstacle_distance > BLOCK_SIZE * 3:
            obstacle_reward += 0.2
        
        # Add safe path reward
        safe_path_reward = self._get_safe_path_reward()
        
        # Check if food eaten
        if self.head == self.food:
            self.score += 1
            reward = 15  # Strong reward for eating food
            self._place_food()
        else:
            self.snake.pop()
            
            # Movement rewards
            if new_distance < prev_distance:
                reward = 1.2  # Reward for moving closer to food
            else:
                reward = -0.15  # Small penalty for moving away
            
            # Add obstacle and safety rewards
            reward += obstacle_reward + safe_path_reward
        
        # Update display
        self._update_ui()
        self.clock.tick(SPEED)
        
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
            
        # Check boundary collision
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
            
        # Check self collision
        if pt in self.snake[1:]:
            return True
            
        # Check obstacle collision
        obstacle_positions = [obs['pos'] for obs in self.obstacles]
        if pt in obstacle_positions:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        # Draw snake with enhanced visuals
        for i, pt in enumerate(self.snake):
            if i == 0:  # Head
                pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, WHITE, pygame.Rect(pt.x+2, pt.y+2, 16, 16))
            else:  # Body
                pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        
        # Draw food with pulsing effect
        food_color = RED if (self.frame_iteration // 10) % 2 == 0 else (255, 100, 100)
        pygame.draw.rect(self.display, food_color, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # Draw moving obstacles with enhanced visuals
        for i, obstacle in enumerate(self.obstacles):
            # Different colors based on movement pattern
            if obstacle['pattern'] == 'linear':
                color = ORANGE
            elif obstacle['pattern'] == 'bouncing':
                color = DARK_ORANGE
            else:  # wandering
                color = YELLOW
                
            pygame.draw.rect(self.display, color, pygame.Rect(obstacle['pos'].x, obstacle['pos'].y, BLOCK_SIZE, BLOCK_SIZE))
            
            # Add directional indicator
            center_x = obstacle['pos'].x + BLOCK_SIZE // 2
            center_y = obstacle['pos'].y + BLOCK_SIZE // 2
            
            # Draw small arrow indicating direction
            if obstacle['dir'] == Direction.RIGHT:
                pygame.draw.polygon(self.display, BLACK, [(center_x-3, center_y-3), (center_x+3, center_y), (center_x-3, center_y+3)])
            elif obstacle['dir'] == Direction.LEFT:
                pygame.draw.polygon(self.display, BLACK, [(center_x+3, center_y-3), (center_x-3, center_y), (center_x+3, center_y+3)])
            elif obstacle['dir'] == Direction.UP:
                pygame.draw.polygon(self.display, BLACK, [(center_x-3, center_y+3), (center_x, center_y-3), (center_x+3, center_y+3)])
            elif obstacle['dir'] == Direction.DOWN:
                pygame.draw.polygon(self.display, BLACK, [(center_x-3, center_y-3), (center_x, center_y+3), (center_x+3, center_y-3)])
        
        # Enhanced UI information
        text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])
        
        obstacle_text = font.render(f"Obstacles: {NUM_OBSTACLES}", True, WHITE)
        self.display.blit(obstacle_text, [0, 25])
        
        pygame.display.flip()
        
    def _move(self, action):
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn
            
        self.direction = new_dir
        
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)