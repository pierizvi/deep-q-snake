import pygame
from snake_game import SnakeGameAI, Direction
from dqn_agent import Agent
import numpy as np

def play_human():
    """Play Snake manually with arrow keys"""
    game = SnakeGameAI()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
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
            print(f'Game Over! Final Score: {score}')
            break

def play_ai():
    """Watch trained AI play Snake"""
    agent = Agent()
    # CRITICAL: Set epsilon to 0 for pure exploitation
    agent.epsilon = 0
    agent.n_games = 1000  # Ensure epsilon calculation gives 0
    
    # Load trained model
    try:
        import torch
        agent.model.load_state_dict(torch.load('./models/model.pth'))
        print("Loaded trained model!")
        print(f"Agent epsilon: {agent.epsilon}")
        print(f"Agent n_games: {agent.n_games}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    game = SnakeGameAI()
    game_count = 0
    
    while True:
        state = agent.get_state(game)
        action = agent.get_action(state)
        reward, game_over, score = game.play_step(action)
        
        if game_over:
            game_count += 1
            print(f'AI Score: {score}')
            
            # Debug info every 10 games
            if game_count % 10 == 0:
                print(f"--- Debug Info (Game {game_count}) ---")
                print(f"Current epsilon: {agent.epsilon}")
                print(f"Sample state: {state[:5]}...")  # First 5 state values
                print(f"Action taken: {action}")
                print("-" * 30)
            
            game.reset()

def menu():
    """Main menu"""
    print("\n🐍 Snake AI Project 🤖")
    print("=" * 30)
    print("1. Play Snake manually")
    print("2. Train AI (console only - recommended)")
    print("3. Train AI (with live plots - may cause issues)")
    print("4. Watch AI play")
    print("5. Exit")
    
    choice = input("\nSelect option (1-5): ")
    
    if choice == '1':
        play_human()
    elif choice == '2':
        from train import train
        train()
    elif choice == '3':
        from train import train_with_plot
        train_with_plot()
    elif choice == '4':
        import torch
        play_ai()
    elif choice == '5':
        print("Goodbye! 👋")
        exit()
    else:
        print("Invalid choice!")
        menu()

if __name__ == '__main__':
    menu()