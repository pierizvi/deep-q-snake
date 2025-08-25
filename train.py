import os
from dqn_agent import Agent
from snake_game import SnakeGameAI, Direction, Point

def train():
    """Clean training without checkpoints - focused on performance"""
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    
    print("Starting Enhanced AI Training with Dynamic Obstacles...")
    print("Goal: Start with 6 obstacles, +1 per food eaten")
    print("Training progress will be shown in console")
    print("Press Ctrl+C to stop training and save model")
    print("-" * 60)
    
    # Training statistics
    collision_with_obstacles = 0
    collision_with_walls = 0
    collision_with_self = 0
    food_collected = 0
    max_obstacles_reached = 6
    
    try:
        while True:
            # Get old state
            state_old = agent.get_state(game)

            # Get move
            final_move = agent.get_action(state_old)

            # Perform move and get new state
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            # Train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # Remember
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                # Track max obstacles reached
                current_obstacles = len(game.obstacles) if hasattr(game, 'obstacles') else 6
                max_obstacles_reached = max(max_obstacles_reached, current_obstacles)
                
                # Analyze cause of game over for statistics
                head = game.head
                if head.x < 0 or head.x >= game.w or head.y < 0 or head.y >= game.h:
                    collision_with_walls += 1
                elif head in game.snake[1:]:
                    collision_with_self += 1
                else:
                    # Check if collision was with obstacle
                    if hasattr(game, 'obstacles'):
                        obstacle_positions = [obs['pos'] for obs in game.obstacles]
                        if head in obstacle_positions:
                            collision_with_obstacles += 1
                
                # Count food collected this game
                food_collected += score
                
                # Train long memory and reset
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save()
                    print(f"NEW RECORD! Game {agent.n_games}, Score: {score}, Max Obstacles: {current_obstacles}")

                # Calculate statistics
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)

                # Print progress every game
                epsilon = max(0, 80 - agent.n_games)
                survival_rate = (agent.n_games - collision_with_obstacles) / agent.n_games * 100
                
                print(f"Game {agent.n_games:4d} | Score: {score:2d} | Record: {record:2d} | "
                      f"Avg: {mean_score:.1f} | Max Obs: {current_obstacles:2d} | "
                      f"Exploration: {epsilon:.0f}% | Survival: {survival_rate:.1f}%")
                
                # Detailed progress every 50 games
                if agent.n_games % 50 == 0:
                    recent_avg = sum(plot_scores[-50:]) / min(50, len(plot_scores))
                    obstacle_collision_rate = collision_with_obstacles / agent.n_games * 100
                    wall_collision_rate = collision_with_walls / agent.n_games * 100
                    self_collision_rate = collision_with_self / agent.n_games * 100
                    avg_food_per_game = food_collected / agent.n_games
                    
                    print(f"\nProgress Report - Game {agent.n_games}")
                    print(f"   Recent 50 games average: {recent_avg:.2f}")
                    print(f"   Overall average: {mean_score:.2f}")
                    print(f"   Best score: {record}")
                    print(f"   Max obstacles reached: {max_obstacles_reached}")
                    print(f"   Collision Analysis:")
                    print(f"      Obstacles: {obstacle_collision_rate:.1f}%")
                    print(f"      Walls: {wall_collision_rate:.1f}%")
                    print(f"      Self: {self_collision_rate:.1f}%")
                    print(f"   Average food per game: {avg_food_per_game:.2f}")
                    print(f"   Current exploration rate: {epsilon:.0f}%")
                    print(f"   Dynamic challenge working: 6→{max_obstacles_reached} obstacles")
                    print("-" * 60)
                
                # Milestone messages without checkpoints
                if agent.n_games % 100 == 0:
                    print(f"Milestone: {agent.n_games} games completed!")
                
                # Auto-save at major milestones
                if agent.n_games % 500 == 0:
                    milestone_name = f'model_{agent.n_games}_games.pth'
                    agent.model.save(milestone_name)
                    print(f"Auto-saved milestone: {milestone_name}")
                    
    except KeyboardInterrupt:
        print(f"\nTraining stopped by user")
        print(f"Final Statistics:")
        print(f"   Games played: {agent.n_games}")
        print(f"   Best score: {record}")
        print(f"   Final average: {mean_score:.2f}")
        print(f"   Max obstacles reached: {max_obstacles_reached}")
        print(f"   Obstacle collisions: {collision_with_obstacles} ({collision_with_obstacles/agent.n_games*100:.1f}%)")
        print(f"   Wall collisions: {collision_with_walls} ({collision_with_walls/agent.n_games*100:.1f}%)")
        print(f"   Self collisions: {collision_with_self} ({collision_with_self/agent.n_games*100:.1f}%)")
        print(f"   Total food collected: {food_collected}")
        print(f"   Challenge progression: 6→{max_obstacles_reached} obstacles")
        agent.model.save()
        print("Final model saved!")

def train_intensive():
    """Intensive training for better performance"""
    from dqn_agent import Agent
    from snake_game import SnakeGameAI
    
    plot_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    
    print("Starting INTENSIVE Training (No console spam)")
    print("Target: Train until very high performance")
    print("Progress will be shown every 100 games")
    print("-" * 50)
    
    try:
        while True:
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old)
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save()
                    print(f"NEW RECORD! Game {agent.n_games}, Score: {score}")

                plot_scores.append(score)
                total_score += score
                
                # Show progress every 100 games only
                if agent.n_games % 100 == 0:
                    mean_score = total_score / agent.n_games
                    recent_avg = sum(plot_scores[-100:]) / min(100, len(plot_scores))
                    epsilon = max(0, 80 - agent.n_games)
                    
                    print(f"Games: {agent.n_games:4d} | Record: {record:2d} | "
                          f"Recent100Avg: {recent_avg:.2f} | Overall: {mean_score:.2f} | "
                          f"Exploration: {epsilon:.0f}%")
                    
    except KeyboardInterrupt:
        print(f"\nIntensive Training Complete!")
        print(f"   Games: {agent.n_games} | Record: {record}")
        agent.model.save()
        print("Model saved!")

if __name__ == '__main__':
    print("Choose training mode:")
    print("1. Standard training (detailed progress)")
    print("2. Intensive training (minimal output)")
    choice = input("Enter choice (1 or 2): ")
    
    if choice == '2':
        train_intensive()
    else:
        train()