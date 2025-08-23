import pygame
from snake_game import SnakeGameAI, Direction
from dqn_agent import Agent
import numpy as np

def play_human():
    """Play Snake manually with arrow keys - enhanced with obstacle information"""
    print("Select game speed:")
    print("1. Slow (10 FPS) - Easy to see")
    print("2. Medium (20 FPS) - Balanced")  
    print("3. Fast (30 FPS) - Challenge")
    print("4. Very Fast (40 FPS) - Expert")
    
    try:
        choice = int(input("Choose speed (1-4): "))
        speeds = {1: 10, 2: 20, 3: 30, 4: 40}
        selected_speed = speeds.get(choice, 20)
    except:
        selected_speed = 20
        
    game = SnakeGameAI(speed=selected_speed)
    
    print("🎮 Human Controls:")
    print("   ⬅️ Left Arrow: Turn Left")
    print("   ➡️ Right Arrow: Turn Right")
    print("   + Key: Increase speed")
    print("   - Key: Decrease speed")
    print("   🟠 Orange blocks: Moving obstacles (avoid them!)")
    print("   🍎 Red block: Food (collect it!)")
    print("   🎯 CHALLENGE: Start with 6 obstacles, +1 per food eaten!")
    print("   Press ESC to quit")
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    quit()
                
        # Human controls
        keys = pygame.key.get_pressed()
        action = [1, 0, 0]  # Default: straight
        
        if keys[pygame.K_LEFT]:
            action = [0, 0, 1]  # Left
        elif keys[pygame.K_RIGHT]:
            action = [0, 1, 0]  # Right
            
        reward, game_over, score = game.play_step(action)
        
        if game_over:
            obstacle_count = len(game.obstacles) if hasattr(game, 'obstacles') else 0
            print(f'🎯 Game Over! Final Score: {score} | Max Obstacles Reached: {obstacle_count}')
            
            # Analyze cause of death
            head = game.head
            if head.x < 0 or head.x >= game.w or head.y < 0 or head.y >= game.h:
                print("💥 Collision cause: Hit wall")
            elif head in game.snake[1:]:
                print("💥 Collision cause: Hit yourself")
            else:
                obstacle_positions = [obs['pos'] for obs in game.obstacles]
                if head in obstacle_positions:
                    print("💥 Collision cause: Hit moving obstacle")
                    
            if score > 0:
                print(f"🏆 Impressive! You survived {score} food collections with increasing obstacles!")
            
            play_again = input("Play again? (y/n): ").lower()
            if play_again == 'y':
                game.reset()
            else:
                break

def play_ai():
    """Watch trained AI play Snake with enhanced debugging"""
    print("Select viewing speed:")
    print("1. Slow (10 FPS) - Easy to watch")
    print("2. Medium (20 FPS) - Balanced")  
    print("3. Fast (30 FPS) - Quick training")
    print("4. Very Fast (40 FPS) - Maximum speed")
    
    try:
        choice = int(input("Choose speed (1-4): "))
        speeds = {1: 10, 2: 20, 3: 30, 4: 40}
        selected_speed = speeds.get(choice, 20)
    except:
        selected_speed = 20
    
    agent = Agent()
    game = SnakeGameAI(speed=selected_speed)
    
    # Initialize the model first by getting a state
    dummy_state = agent.get_state(game)
    
    # Set epsilon to 0 for pure exploitation
    agent.epsilon = 0
    agent.n_games = 1000  # Ensure epsilon calculation gives 0
    
    # Load trained model
    try:
        import torch
        agent.model.load_state_dict(torch.load('./models/model.pth'))
        print("✅ Loaded trained model!")
        print(f"🧠 Agent epsilon: {agent.epsilon}")
        print(f"🎮 Agent games trained: {agent.n_games}")
        print(f"🎬 Viewing at {selected_speed} FPS (press +/- to adjust)")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Make sure you have trained a model first (option 2 or 3)")
        return
    game_count = 0
    total_score = 0
    obstacle_collisions = 0
    successful_games = 0
    
    print("\n🤖 AI Playing Snake with Dynamic Moving Obstacles")
    print("🎯 Challenge: Start with 6 obstacles, +1 per food eaten!")
    print("📊 Statistics will be shown every 10 games")
    print("⏹️ Press Ctrl+C to stop watching")
    print("-" * 50)
    
    try:
        while True:
            state = agent.get_state(game)
            action = agent.get_action(state)
            reward, game_over, score = game.play_step(action)
            
            if game_over:
                game_count += 1
                total_score += score
                
                # Analyze performance
                head = game.head
                if score > 0:
                    successful_games += 1
                    
                # Check if it was an obstacle collision
                obstacle_positions = [obs['pos'] for obs in game.obstacles]
                if head in obstacle_positions:
                    obstacle_collisions += 1
                
                max_obstacles = len(game.obstacles) if hasattr(game, 'obstacles') else 0
                print(f'🎯 AI Game {game_count}: Score {score} | Max Obstacles: {max_obstacles}')
                
                # Detailed statistics every 10 games
                if game_count % 10 == 0:
                    avg_score = total_score / game_count
                    obstacle_collision_rate = (obstacle_collisions / game_count) * 100
                    success_rate = (successful_games / game_count) * 100
                    
                    print(f"\n📈 Statistics after {game_count} games:")
                    print(f"   Average Score: {avg_score:.2f}")
                    print(f"   Success Rate (Score > 0): {success_rate:.1f}%")
                    print(f"   Obstacle Collision Rate: {obstacle_collision_rate:.1f}%")
                    print(f"   Total Food Collected: {total_score}")
                    
                    # Show AI's current understanding
                    print(f"   🧠 AI State Analysis:")
                    print(f"      Current epsilon: {agent.epsilon}")
                    print(f"      State vector size: {len(state)}")
                    print(f"      Action taken: {['Straight', 'Right', 'Left'][action.index(1)]}")
                    print("-" * 50)
                
                game.reset()
                
    except KeyboardInterrupt:
        print(f"\n📊 Final AI Performance Summary:")
        print(f"   Games played: {game_count}")
        print(f"   Average score: {total_score/max(1,game_count):.2f}")
        print(f"   Obstacle avoidance: {((game_count-obstacle_collisions)/max(1,game_count)*100):.1f}%")
        print(f"   Success rate: {(successful_games/max(1,game_count)*100):.1f}%")

def compare_models():
    """Compare performance of different saved models"""
    import os
    models_dir = './models'
    
    if not os.path.exists(models_dir):
        print("❌ No models directory found. Train a model first!")
        return
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    
    if not model_files:
        print("❌ No model files found. Train a model first!")
        return
    
    # Sort by modification time (newest first)
    model_files_with_time = []
    for f in model_files:
        full_path = os.path.join(models_dir, f)
        mod_time = os.path.getmtime(full_path)
        model_files_with_time.append((f, mod_time))
    
    model_files_with_time.sort(key=lambda x: x[1], reverse=True)
    
    print("🔍 Available models (sorted by newest first):")
    for i, (model_file, _) in enumerate(model_files_with_time):
        print(f"   {i+1}. {model_file}")
    
    try:
        choice = int(input("Select model to test (number): ")) - 1
        if choice < 0 or choice >= len(model_files_with_time):
            print("❌ Invalid choice")
            return
        
        selected_model = model_files_with_time[choice][0]
        print(f"🧪 Testing model: {selected_model}")
        
        # Test the selected model
        agent = Agent()
        game = SnakeGameAI()
        
        # Initialize the model first
        dummy_state = agent.get_state(game)
        agent.set_inference_mode()  # No exploration for testing
        
        import torch
        agent.model.load_state_dict(torch.load(os.path.join(models_dir, selected_model)))
        
        # Run quick performance test
        scores = []
        obstacle_hits = 0
        max_obstacles_reached = 6
        
        print(f"🧪 Running 20 test games with {selected_model}...")
        
        for test_game in range(20):  # Test 20 games
            max_obstacles_this_game = 6
            while True:
                state = agent.get_state(game)
                action = agent.get_action(state)
                reward, done, score = game.play_step(action)
                
                # Track max obstacles reached this game
                current_obstacles = len(game.obstacles) if hasattr(game, 'obstacles') else 6
                max_obstacles_this_game = max(max_obstacles_this_game, current_obstacles)
                
                if done:
                    scores.append(score)
                    max_obstacles_reached = max(max_obstacles_reached, max_obstacles_this_game)
                    
                    # Check if obstacle collision
                    head = game.head
                    obstacle_positions = [obs['pos'] for obs in game.obstacles]
                    if head in obstacle_positions:
                        obstacle_hits += 1
                    
                    print(f"Test game {test_game + 1}: Score {score}, Max obstacles: {max_obstacles_this_game}")
                    game.reset()
                    break
        
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
        obstacle_avoidance = ((20 - obstacle_hits) / 20) * 100
        success_rate = (len([s for s in scores if s > 0]) / 20) * 100
        
        print(f"\n📊 Model Performance Analysis ({selected_model}):")
        print(f"   Average Score: {avg_score:.2f}")
        print(f"   Best Score: {max_score}")
        print(f"   Worst Score: {min_score}")
        print(f"   Success Rate (Score > 0): {success_rate:.1f}%")
        print(f"   Obstacle Avoidance: {obstacle_avoidance:.1f}%")
        print(f"   Max Obstacles Reached: {max_obstacles_reached}")
        print(f"   Score Distribution: {scores}")
        
        # Performance rating
        if avg_score >= 10:
            rating = "🏆 EXCELLENT"
        elif avg_score >= 5:
            rating = "🥈 GOOD"
        elif avg_score >= 2:
            rating = "🥉 DECENT"
        else:
            rating = "📈 NEEDS IMPROVEMENT"
            
        print(f"   Overall Rating: {rating}")
        
    except ValueError:
        print("❌ Please enter a valid number")
    except Exception as e:
        print(f"❌ Error testing model: {e}")

def menu():
    """Enhanced main menu"""
    print("\n🐍 Snake AI Project with Enhanced Moving Obstacles 🤖")
    print("=" * 55)
    print("1. 🎮 Play Snake manually (with moving obstacles)")
    print("2. 🤖 Train AI (console only - recommended)")
    print("3. 📈 Train AI with adaptive obstacles")
    print("4. 👀 Watch AI play (load existing model)")
    print("5. 🔍 Compare/Test different models")
    print("6. ℹ️  Show training tips")
    print("7. 🚪 Exit")
    
    choice = input("\nSelect option (1-7): ")
    
    if choice == '1':
        play_human()
    elif choice == '2':
        from train import train
        train()
    elif choice == '3':
        from train import train_with_adaptive_obstacles
        train_with_adaptive_obstacles()
    elif choice == '4':
        play_ai()
    elif choice == '5':
        compare_models()
    elif choice == '6':
        show_training_tips()
    elif choice == '7':
        print("Goodbye! 👋")
        exit()
    else:
        print("❌ Invalid choice!")
        menu()

def show_training_tips():
    """Show training tips and information"""
    print("\n📚 Training Tips for Dynamic Obstacle Avoidance:")
    print("=" * 55)
    print("🎯 Goal: Train AI to collect food while avoiding increasing obstacles")
    print("🔥 Challenge: Each food eaten adds 1 more moving obstacle!")
    print()
    print("📈 Training Progression:")
    print("   • Early games (0-100): Learning basic movement with 2 obstacles")
    print("   • Mid training (100-500): Adapting to 3-6 obstacles")
    print("   • Advanced (500+): Mastering complex scenarios with 7+ obstacles")
    print()
    print("📊 Key Metrics to Watch:")
    print("   • Obstacle collision rate should decrease over time")
    print("   • Average score should gradually increase despite difficulty")
    print("   • AI should learn to survive longer with more obstacles")
    print()
    print("🧠 Neural Network Features:")
    print("   • Dynamic input size (adapts to state complexity)")
    print("   • Multi-distance obstacle detection")
    print("   • Movement pattern analysis")
    print("   • Adaptive exploration (slower decay for increasing difficulty)")
    print()
    print("⚙️ Dynamic Difficulty System:")
    print("   • Starts with 2 obstacles")
    print("   • +1 obstacle per food eaten")
    print("   • Rewards scale with difficulty (more obstacles = bigger rewards)")
    print("   • Timeout scales with obstacle count")
    print()
    print("💡 Pro Tips:")
    print("   • Train for 1000+ games due to increasing complexity")
    print("   • Watch for AI learning to balance risk vs reward")
    print("   • Higher scores become exponentially more impressive")
    print("   • AI must develop long-term survival strategies")
    
    input("\nPress Enter to return to menu...")

if __name__ == '__main__':
    menu()