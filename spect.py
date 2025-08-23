import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os

def explore_model_weights(model_path='./models/model.pth'):
    """Explore and visualize the weights in your trained model"""
    
    print("🔍 Loading and Exploring Model Weights...")
    print("=" * 60)
    
    try:
        # Load the model
        model_state = torch.load(model_path, map_location='cpu')
        
        print(f"📁 Model file: {model_path}")
        print(f"🏗️  Layers found: {len(model_state)} layers")
        print()
        
        # Analyze each layer
        for layer_name, weights in model_state.items():
            print(f"🧠 Layer: {layer_name}")
            print(f"   Shape: {weights.shape}")
            print(f"   Type: {'Weight Matrix' if 'weight' in layer_name else 'Bias Vector'}")
            print(f"   Parameters: {weights.numel():,}")
            
            # Statistical analysis
            weights_flat = weights.flatten()
            print(f"   📊 Statistics:")
            print(f"      Mean: {weights_flat.mean():.6f}")
            print(f"      Std:  {weights_flat.std():.6f}")
            print(f"      Min:  {weights_flat.min():.6f}")
            print(f"      Max:  {weights_flat.max():.6f}")
            
            # Show some actual weight values
            print(f"   🔢 Sample weights: {weights_flat[:10].tolist()}")
            print()
        
        return model_state
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def print_specific_weights(model_state, layer_name, num_weights=20):
    """Print specific weights from a layer"""
    
    if layer_name not in model_state:
        print(f"❌ Layer '{layer_name}' not found!")
        print(f"Available layers: {list(model_state.keys())}")
        return
    
    weights = model_state[layer_name]
    weights_flat = weights.flatten()
    
    print(f"🔍 Detailed View: {layer_name}")
    print("=" * 50)
    print(f"Shape: {weights.shape}")
    print(f"Total parameters: {weights.numel():,}")
    print()
    
    # Print first N weights with their positions
    print(f"📋 First {num_weights} weights:")
    for i in range(min(num_weights, len(weights_flat))):
        weight_val = weights_flat[i].item()
        print(f"   Weight[{i:3d}]: {weight_val:8.6f}")
    
    if len(weights_flat) > num_weights:
        print(f"   ... and {len(weights_flat) - num_weights:,} more weights")
    
    print()
    
    # If it's a 2D weight matrix, show some structure
    if len(weights.shape) == 2:
        print("🔗 Connection Strength Examples:")
        print("   (How much input neuron X influences output neuron Y)")
        rows, cols = weights.shape
        
        # Show connections from first few input neurons
        for i in range(min(5, rows)):
            for j in range(min(5, cols)):
                strength = weights[i, j].item()
                connection_type = "Strong +" if strength > 0.5 else "Strong -" if strength < -0.5 else "Weak"
                print(f"   Input[{i}] → Output[{j}]: {strength:7.4f} ({connection_type})")
            if j >= 4 and cols > 5:
                print(f"   Input[{i}] → ... ({cols-5} more connections)")
            print()

def analyze_learning_patterns(model_state):
    """Analyze what patterns the AI learned"""
    
    print("🧠 Learning Pattern Analysis")
    print("=" * 40)
    
    # Analyze input layer weights (most interpretable)
    input_weights = model_state['linear1.weight']  # 512 x 42
    
    print(f"📊 Input Layer Analysis (42 game features → 512 hidden neurons)")
    print()
    
    # Each row represents how one hidden neuron responds to all 42 input features
    # Each column represents how all hidden neurons respond to one input feature
    
    # Feature importance: which input features have the strongest overall connections
    feature_importance = torch.abs(input_weights).mean(dim=0)  # Average across all hidden neurons
    
    print("🎯 Most Important Game Features (by weight strength):")
    feature_names = [
        "Danger straight", "Danger right", "Danger left",
        "Facing left", "Facing right", "Facing up", "Facing down",
        "Food left", "Food right", "Food up", "Food down",
        "Food distance X", "Food distance Y",
        "Wall dist left", "Wall dist right", "Wall dist up", "Wall dist down",
        "Body left", "Body right", "Body up", "Body down",
        "Obstacle immediate left", "Obstacle immediate right", 
        "Obstacle immediate up", "Obstacle immediate down",
        "Obstacle near left", "Obstacle near right", 
        "Obstacle near up", "Obstacle near down",
        "Obstacle diagonal UL", "Obstacle diagonal UR",
        "Obstacle diagonal DL", "Obstacle diagonal DR",
        "Obstacles moving toward", "Obstacles moving away",
        "Closest obstacle dist", "Average obstacle dist",
        "Obstacles quadrant 1", "Obstacles quadrant 2",
        "Obstacles quadrant 3", "Obstacles quadrant 4",
        "Snake length"
    ]
    
    # Sort features by importance
    sorted_indices = torch.argsort(feature_importance, descending=True)
    
    for i, idx in enumerate(sorted_indices[:10]):  # Top 10 most important
        feature_name = feature_names[idx] if idx < len(feature_names) else f"Feature {idx}"
        importance = feature_importance[idx].item()
        print(f"   {i+1:2d}. {feature_name:25}: {importance:.4f}")
    
    print()
    
    # Hidden neuron specialization
    print("🔍 Hidden Neuron Specializations:")
    print("   (What each hidden neuron learned to detect)")
    
    for neuron_idx in range(min(5, input_weights.shape[0])):  # First 5 hidden neurons
        neuron_weights = input_weights[neuron_idx]
        
        # Find strongest positive and negative connections
        pos_idx = torch.argmax(neuron_weights)
        neg_idx = torch.argmin(neuron_weights)
        
        pos_feature = feature_names[pos_idx] if pos_idx < len(feature_names) else f"Feature {pos_idx}"
        neg_feature = feature_names[neg_idx] if neg_idx < len(feature_names) else f"Feature {neg_idx}"
        
        print(f"   Neuron {neuron_idx:3d}: Activates for '{pos_feature}' (+{neuron_weights[pos_idx]:.3f})")
        print(f"              Suppressed by '{neg_feature}' ({neuron_weights[neg_idx]:.3f})")
        print()

def visualize_weights(model_state, save_plots=True):
    """Create visualizations of the weight distributions"""
    
    print("📊 Creating Weight Visualizations...")
    
    # Set up the plot style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Snake AI Neural Network Weight Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Weight distribution histogram
    all_weights = []
    layer_names = []
    
    for name, weights in model_state.items():
        if 'weight' in name:
            all_weights.extend(weights.flatten().tolist())
            layer_names.append(name)
    
    axes[0, 0].hist(all_weights, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Overall Weight Distribution')
    axes[0, 0].set_xlabel('Weight Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Weight distribution by layer
    layer_data = []
    layer_labels = []
    
    for name, weights in model_state.items():
        if 'weight' in name:
            layer_data.append(weights.flatten().numpy())
            layer_labels.append(name.replace('.weight', ''))
    
    axes[0, 1].boxplot(layer_data, labels=layer_labels)
    axes[0, 1].set_title('Weight Distribution by Layer')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Weight Value')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Input layer heatmap (first 20x20 section)
    input_weights = model_state['linear1.weight']
    subset = input_weights[:20, :20].numpy()  # First 20x20 section
    
    im = axes[1, 0].imshow(subset, cmap='RdBu_r', aspect='auto')
    axes[1, 0].set_title('Input Layer Weights Heatmap (20x20 sample)')
    axes[1, 0].set_xlabel('Input Features')
    axes[1, 0].set_ylabel('Hidden Neurons')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Plot 4: Weight magnitude by layer
    layer_magnitudes = []
    layer_names_clean = []
    
    for name, weights in model_state.items():
        if 'weight' in name:
            avg_magnitude = torch.abs(weights).mean().item()
            layer_magnitudes.append(avg_magnitude)
            layer_names_clean.append(name.replace('.weight', ''))
    
    bars = axes[1, 1].bar(layer_names_clean, layer_magnitudes, color=['red', 'green', 'blue', 'orange'][:len(layer_magnitudes)])
    axes[1, 1].set_title('Average Weight Magnitude by Layer')
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Average |Weight|')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mag in zip(bars, layer_magnitudes):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{mag:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('snake_ai_weights_analysis.png', dpi=300, bbox_inches='tight')
        print("💾 Visualization saved as 'snake_ai_weights_analysis.png'")
    
    plt.show()

def save_weights_to_file(model_state, filename='model_weights.txt'):
    """Save all weights to a text file for detailed examination"""
    
    print(f"💾 Saving all weights to {filename}...")
    
    with open(filename, 'w') as f:
        f.write("SNAKE AI NEURAL NETWORK WEIGHTS\n")
        f.write("=" * 50 + "\n\n")
        
        total_params = 0
        
        for layer_name, weights in model_state.items():
            f.write(f"LAYER: {layer_name}\n")
            f.write(f"Shape: {weights.shape}\n")
            f.write(f"Parameters: {weights.numel():,}\n")
            f.write("-" * 30 + "\n")
            
            # Write all weights
            weights_flat = weights.flatten()
            for i, weight in enumerate(weights_flat):
                f.write(f"{i:6d}: {weight.item():12.8f}\n")
            
            f.write("\n" + "=" * 50 + "\n\n")
            total_params += weights.numel()
        
        f.write(f"TOTAL PARAMETERS: {total_params:,}\n")
    
    print(f"✅ Complete weight dump saved to {filename}")
    print(f"📊 File contains {total_params:,} individual weight values")

def main():
    """Main function to explore your model weights"""
    
    print("🧠 Snake AI Weight Explorer")
    print("=" * 40)
    
    # Check available models
    models_dir = './models'
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        if model_files:
            print("📁 Available models:")
            for i, model_file in enumerate(model_files, 1):
                print(f"   {i}. {model_file}")
            
            try:
                choice = int(input(f"\nSelect model (1-{len(model_files)}): ")) - 1
                if 0 <= choice < len(model_files):
                    model_path = os.path.join(models_dir, model_files[choice])
                else:
                    model_path = './models/model.pth'
            except:
                model_path = './models/model.pth'
        else:
            model_path = './models/model.pth'
    else:
        model_path = './models/model.pth'
    
    # Load and explore the model
    model_state = explore_model_weights(model_path)
    
    if model_state is None:
        return
    
    print("\n" + "="*60)
    
    # Interactive menu
    while True:
        print("\n🔍 What would you like to explore?")
        print("1. 📋 Print specific layer weights")
        print("2. 🧠 Analyze learning patterns")
        print("3. 📊 Create visualizations")
        print("4. 💾 Save all weights to file")
        print("5. 🚪 Exit")
        
        choice = input("\nSelect option (1-5): ")
        
        if choice == '1':
            print(f"\nAvailable layers: {list(model_state.keys())}")
            layer_name = input("Enter layer name (e.g., 'linear1.weight'): ")
            num_weights = int(input("How many weights to show? (default 20): ") or "20")
            print_specific_weights(model_state, layer_name, num_weights)
            
        elif choice == '2':
            analyze_learning_patterns(model_state)
            
        elif choice == '3':
            try:
                visualize_weights(model_state)
            except Exception as e:
                print(f"❌ Visualization failed: {e}")
                print("💡 Make sure matplotlib is installed: pip install matplotlib seaborn")
                
        elif choice == '4':
            filename = input("Enter filename (default: model_weights.txt): ") or "model_weights.txt"
            save_weights_to_file(model_state, filename)
            
        elif choice == '5':
            print("👋 Happy exploring!")
            break
            
        else:
            print("❌ Invalid choice!")

if __name__ == "__main__":
    main()