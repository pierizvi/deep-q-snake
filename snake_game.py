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

BLOCK_SIZE = 20
NUM_OBSTACLES = 6  # Start with 6 obstacles
OBSTACLE_SPEED = 4  # Move obstacles every 4 frames

class SnakeGameAI:
    
    def __init__(self, w=640, h=480, speed=20):
        self.w = w
        self.h = h
        self.speed = speed  # Adjustable speed
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption(f'Snake AI - Speed: {speed} FPS')
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
        self._place_food()
        self.frame_iteration = 0
        
        # Initialize moving obstacles
        self.obstacles = []
        self._init_obstacles()
        
    def set_speed(self, new_speed):
        """Change game speed dynamically"""
        self.speed = new_speed
        pygame.display.set_caption(f'Snake AI - Speed: {new_speed} FPS')
        
    def _init_obstacles(self):
        """Initialize moving obstacles at safe distances from snake"""
        self.obstacles = []
        for i in range(NUM_OBSTACLES):
            attempts = 0
            while attempts < 100:  # Prevent infinite loop
                # Spawn obstacles away from center where snake starts
                x = random.randint(1, (self.w//BLOCK_SIZE)-2) * BLOCK_SIZE
                y = random.randint(1, (self.h//BLOCK_SIZE)-2) * BLOCK_SIZE
                pos = Point(x, y)
                
                # Make sure obstacle is far from snake (at least 4 blocks away)
                min_distance = BLOCK_SIZE * 4
                safe = True
                for snake_part in self.snake:
                    distance = abs(pos.x - snake_part.x) + abs(pos.y - snake_part.y)
                    if distance < min_distance:
                        safe = False
                        break
                
                if safe:
                    # Random direction for obstacle
                    direction = random.choice(list(Direction))
                    self.obstacles.append({'pos': pos, 'dir': direction})
                    break
                attempts += 1

    def _add_new_obstacle(self):
        """Add one new obstacle when snake eats food"""
        attempts = 0
        while attempts < 50:
            # Random placement
            x = random.randint(1, (self.w//BLOCK_SIZE)-2) * BLOCK_SIZE
            y = random.randint(1, (self.h//BLOCK_SIZE)-2) * BLOCK_SIZE
            pos = Point(x, y)
            
            # Make sure it's not too close to snake
            safe = True
            for snake_part in self.snake:
                if abs(pos.x - snake_part.x) + abs(pos.y - snake_part.y) < BLOCK_SIZE * 3:
                    safe = False
                    break
            
            # Make sure it's not on food
            if abs(pos.x - self.food.x) + abs(pos.y - self.food.y) < BLOCK_SIZE * 2:
                safe = False
            
            if safe:
                direction = random.choice(list(Direction))
                new_obstacle = {'pos': pos, 'dir': direction}
                self.obstacles.append(new_obstacle)
                print(f"🟠 Added obstacle! Total: {len(self.obstacles)}")
                break
            attempts += 1
                    
    def _move_obstacles(self):
        """Move obstacles randomly"""
        for obstacle in self.obstacles:
            # Randomly change direction less frequently
            if random.random() < 0.05:  # 5% chance to change direction
                obstacle['dir'] = random.choice(list(Direction))
            
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
            
            # Bounce off walls and stay within bounds
            if x < 0:
                x = 0
                obstacle['dir'] = Direction.RIGHT
            elif x >= self.w:
                x = self.w - BLOCK_SIZE
                obstacle['dir'] = Direction.LEFT
            
            if y < 0:
                y = 0
                obstacle['dir'] = Direction.DOWN
            elif y >= self.h:
                y = self.h - BLOCK_SIZE
                obstacle['dir'] = Direction.UP
            
            obstacle['pos'] = Point(x, y)
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        
        # Make sure food doesn't spawn on snake or obstacles
        obstacle_positions = [obs['pos'] for obs in self.obstacles] if hasattr(self, 'obstacles') else []
        if self.food in self.snake or self.food in obstacle_positions:
            self._place_food()
        
    def play_step(self, action):
        self.frame_iteration += 1
        
        # Handle speed adjustment keys
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.set_speed(min(self.speed + 5, 60))  # Max 60 FPS
                elif event.key == pygame.K_MINUS:
                    self.set_speed(max(self.speed - 5, 5))   # Min 5 FPS
        
        # Move obstacles every 4 frames
        if self.frame_iteration % OBSTACLE_SPEED == 0:
            self._move_obstacles()
        
        # Store previous distance to food for reward calculation
        prev_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        # Give AI more time to find food
        if self.is_collision() or self.frame_iteration > 100*len(self.snake) + 500:
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        # Calculate new distance to food
        new_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        
        # Check if too close to obstacles (penalty)
        obstacle_penalty = 0
        for obstacle in self.obstacles:
            dist_to_obstacle = abs(self.head.x - obstacle['pos'].x) + abs(self.head.y - obstacle['pos'].y)
            if dist_to_obstacle <= BLOCK_SIZE:  # Very close
                obstacle_penalty = -0.2
                break
            
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10  # Big reward for eating food
            
            # ADD NEW OBSTACLE WHEN FOOD IS EATEN
            self._add_new_obstacle()
            
            self._place_food()
        else:
            self.snake.pop()
            # Reward system for getting closer to food
            if new_distance < prev_distance:
                reward = 1  # Small reward for moving closer
            else:
                reward = -0.1  # Small penalty for moving away
            
            # Add obstacle penalty
            reward += obstacle_penalty
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(self.speed)  # Use adjustable speed
        # 6. return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        # hits obstacles
        obstacle_positions = [obs['pos'] for obs in self.obstacles]
        if pt in obstacle_positions:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        # Draw snake
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        
        # Highlight snake head
        pygame.draw.rect(self.display, GREEN, pygame.Rect(self.snake[0].x+2, self.snake[0].y+2, 16, 16))
        
        # Draw food    
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # Draw moving obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(self.display, ORANGE, pygame.Rect(obstacle['pos'].x, obstacle['pos'].y, BLOCK_SIZE, BLOCK_SIZE))
            # Add a small black center to make obstacles more visible
            pygame.draw.rect(self.display, BLACK, pygame.Rect(obstacle['pos'].x+6, obstacle['pos'].y+6, 8, 8))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        
        # SHOW CURRENT OBSTACLE COUNT
        obstacle_text = font.render(f"Obstacles: {len(self.obstacles)}", True, WHITE)
        self.display.blit(obstacle_text, [0, 25])
        
        # Show speed controls
        speed_text = font.render(f"Speed: {self.speed} FPS (+/- to adjust)", True, WHITE)
        self.display.blit(speed_text, [0, 50])
        
        pygame.display.flip()
        
    def _move(self, action):
        # [straight, right, left]
        
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d
            
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