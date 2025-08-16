import os
from dqn_agent import Agent
from snake_game import SnakeGameAI, Direction, Point

def train():
    """Train the AI to play Snake"""
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    
    print("🤖 Starting AI Training...")
    print("📊 Training progress will be shown in console")
    print("⏹️  Press Ctrl+C to stop training and save model")
    print("-" * 50)
    
    try:
        while True:
            # get old state
            state_old = agent.get_state(game)

            # get move
            final_move = agent.get_action(state_old)

            # perform move and get new state
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                # train long memory
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save()
                    print(f"🎉 NEW RECORD! Game {agent.n_games}, Score: {score}")

                # Calculate average score
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)

                # Print progress every game
                epsilon = max(0, 80 - agent.n_games)
                print(f"Game {agent.n_games:4d} | Score: {score:2d} | Record: {record:2d} | Avg: {mean_score:.1f} | Exploration: {epsilon:.0f}%")
                
                # Show detailed progress every 50 games
                if agent.n_games % 50 == 0:
                    recent_avg = sum(plot_scores[-50:]) / min(50, len(plot_scores))
                    print(f"📈 Progress Update - Game {agent.n_games}")
                    print(f"   Recent 50 games average: {recent_avg:.2f}")
                    print(f"   Overall average: {mean_score:.2f}")
                    print(f"   Best score: {record}")
                    print("-" * 50)
                    
    except KeyboardInterrupt:
        print(f"\n⏹️  Training stopped by user")
        print(f"📊 Final Stats:")
        print(f"   Games played: {agent.n_games}")
        print(f"   Best score: {record}")
        print(f"   Final average: {mean_score:.2f}")
        agent.model.save()
        print("💾 Model saved!")
        
def train_with_plot():
    """Train with live plotting (may cause issues on some systems)"""
    try:
        import matplotlib.pyplot as plt
        
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
    train()