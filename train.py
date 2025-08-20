import os
import time

def train():
    """Enhanced training for obstacle avoidance"""
    # Import here to avoid circular imports
    from dqn_agent import Agent
    from snake_game import SnakeGameAI, Direction, Point
    
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    
    print("🤖 Starting Enhanced AI Training with Moving Obstacles...")
    print("📊 Training progress will be shown in console")
    print("ℹ️  Press Ctrl+C to stop training and save model")
    print("🎯 Goal: Learn to avoid moving obstacles while collecting food")
    print("-" * 60)
    
    # Training statistics
    collision_with_obstacles = 0
    collision_with_walls = 0
    collision_with_self = 0
    food_collected = 0
    
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
                    print(f"🎉 NEW RECORD! Game {agent.n_games}, Score: {score}")

                # Calculate statistics
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)

                # Print progress every game with enhanced statistics
                epsilon = max(5, 80 - agent.n_games)
                survival_rate = (agent.n_games - collision_with_obstacles) / agent.n_games * 100
                
                print(f"Game {agent.n_games:4d} | Score: {score:2d} | Record: {record:2d} | "
                      f"Avg: {mean_score:.1f} | Exploration: {epsilon:.0f}% | "
                      f"Obstacle Avoidance: {survival_rate:.1f}%")
                
                # Detailed progress every 25 games
                if agent.n_games % 25 == 0:
                    recent_avg = sum(plot_scores[-25:]) / min(25, len(plot_scores))
                    obstacle_collision_rate = collision_with_obstacles / agent.n_games * 100
                    wall_collision_rate = collision_with_walls / agent.n_games * 100
                    self_collision_rate = collision_with_self / agent.n_games * 100
                    avg_food_per_game = food_collected / agent.n_games
                    
                    print(f"\n📈 Progress Report - Game {agent.n_games}")
                    print(f"   Recent 25 games average: {recent_avg:.2f}")
                    print(f"   Overall average: {mean_score:.2f}")
                    print(f"   Best score: {record}")
                    print(f"   📊 Collision Analysis:")
                    print(f"      🟠 Obstacles: {obstacle_collision_rate:.1f}%")
                    print(f"      🔴 Walls: {wall_collision_rate:.1f}%")
                    print(f"      🔵 Self: {self_collision_rate:.1f}%")
                    print(f"   🍎 Average food per game: {avg_food_per_game:.2f}")
                    print(f"   🧠 Current exploration rate: {epsilon:.0f}%")
                    print("-" * 60)
                
                # Save checkpoint every 100 games
                if agent.n_games % 100 == 0:
                    checkpoint_name = f'checkpoint_{agent.n_games}.pth'
                    agent.model.save(checkpoint_name)
                    print(f"💾 Checkpoint saved: {checkpoint_name}")
                    
    except KeyboardInterrupt:
        print(f"\nℹ️  Training stopped by user")
        print(f"📊 Final Statistics:")
        print(f"   Games played: {agent.n_games}")
        print(f"   Best score: {record}")
        print(f"   Final average: {mean_score:.2f}")
        print(f"   🟠 Obstacle collisions: {collision_with_obstacles} ({collision_with_obstacles/agent.n_games*100:.1f}%)")
        print(f"   🔴 Wall collisions: {collision_with_walls} ({collision_with_walls/agent.n_games*100:.1f}%)")
        print(f"   🔵 Self collisions: {collision_with_self} ({collision_with_self/agent.n_games*100:.1f}%)")
        print(f"   🍎 Total food collected: {food_collected}")
        agent.model.save()
        print("💾 Model saved!")

def train_with_adaptive_obstacles():
    """Training with gradually increasing obstacle difficulty"""
    from dqn_agent import Agent
    from snake_game import SnakeGameAI, Direction, Point
    
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    
    print("🚀 Starting Adaptive Obstacle Training...")
    print("📈 Obstacles will increase in difficulty as AI improves")
    print("-" * 60)
    
    try:
        while True:
            # Adaptive difficulty: increase obstacles as AI improves
            if agent.n_games > 0 and agent.n_games % 200 == 0:
                current_avg = sum(plot_scores[-50:]) / min(50, len(plot_scores)) if plot_scores else 0
                if current_avg > 5:  # If AI is performing well
                    # Increase obstacle challenge
                    if hasattr(game, 'obstacles') and len(game.obstacles) < 10:
                        print(f"🎯 Increasing difficulty: Adding more obstacles at game {agent.n_games}")
                        new_obstacle = {
                            'pos': Point(200, 200),
                            'dir': Direction.RIGHT,
                            'pattern': 'wandering',
                            'speed_multiplier': 1.0,
                            'direction_timer': 20,
                            'original_direction_timer': 20
                        }
                        game.obstacles.append(new_obstacle)
            
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
                    print(f"🎉 NEW RECORD! Game {agent.n_games}, Score: {score}")

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)

                epsilon = max(5, 80 - agent.n_games)
                num_obstacles = len(game.obstacles) if hasattr(game, 'obstacles') else 0
                
                print(f"Game {agent.n_games:4d} | Score: {score:2d} | Record: {record:2d} | "
                      f"Avg: {mean_score:.1f} | Obstacles: {num_obstacles} | Exploration: {epsilon:.0f}%")
                
                if agent.n_games % 50 == 0:
                    recent_avg = sum(plot_scores[-50:]) / min(50, len(plot_scores))
                    print(f"📊 Recent 50 games average: {recent_avg:.2f} | Current obstacles: {num_obstacles}")
                    
    except KeyboardInterrupt:
        print(f"\n📊 Adaptive training completed!")
        print(f"   Games played: {agent.n_games}")
        print(f"   Best score: {record}")
        print(f"   Final obstacle count: {len(game.obstacles) if hasattr(game, 'obstacles') else 0}")
        agent.model.save()
        print("💾 Model saved!")

def train_with_plot():
    """Train with live plotting (may cause issues on some systems)"""
    try:
        import matplotlib.pyplot as plt
        from dqn_agent import Agent
        from snake_game import SnakeGameAI
        
        def plot(scores, mean_scores):
            plt.clf()
            plt.title('AI Training Progress')
            plt.xlabel('Number of Games')
            plt.ylabel('Score')
            plt.plot(scores, label='Score', color='blue', alpha=0.6)
            plt.plot(mean_scores, label='Average Score', color='red', linewidth=2)
            plt.ylim(ymin=0)
            plt.legend()
            
            if len(scores) > 0:
                plt.text(len(scores)-1, scores[-1], str(scores[-1]))
            if len(mean_scores) > 0:
                plt.text(len(mean_scores)-1, mean_scores[-1], f'{mean_scores[-1]:.1f}')
            
            plt.show(block=False)
            plt.pause(0.01)
        
        # Similar training loop but with plotting
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = 0
        agent = Agent()
        game = SnakeGameAI()
        
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

                print('Game', agent.n_games, 'Score', score, 'Record:', record)

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                
                # Plot every 10 games to reduce conflicts
                if agent.n_games % 10 == 0:
                    plot(plot_scores, plot_mean_scores)
                    
    except ImportError:
        print("Matplotlib not available, falling back to console-only training")
        train()
    except Exception as e:
        print(f"Plotting failed ({e}), switching to console-only mode")
        train()

if __name__ == '__main__':
    print("Choose training mode:")
    print("1. Standard training")
    print("2. Adaptive obstacle training")
    choice = input("Enter choice (1 or 2): ")
    
    if choice == '2':
        train_with_adaptive_obstacles()
    else:
        train()