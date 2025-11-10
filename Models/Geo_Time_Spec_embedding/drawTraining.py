import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot_training_history(history_path):
    """
    Load and visualize training history data.

    Args:
        history_path (str): Path to the training history .npy file.
    """
    # --- 1. Load data ---
    if not os.path.exists(history_path):
        print(f"Error: File '{history_path}' does not exist.")
        return

    try:
        history = np.load(history_path, allow_pickle=True).item()
    except Exception as e:
        print(f"Failed to load file: {e}")
        return

    # --- 2. Set plot style ---
    plt.style.use('_classic_test_patch')  # Use classic style
    plt.rcParams['figure.dpi'] = 120  # Figure resolution

    # --- 3. Create figure and subplots ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Model Training Process Visualization', fontsize=16, y=0.98)

    # Get training epochs
    epochs = range(1, len(history['train_total_loss']) + 1)

    # --- 4. Plot loss curves ---
    ax1.plot(epochs, history['train_total_loss'], 
             label='Training Total Loss', linewidth=2.0, marker='o', markersize=4)
    ax1.plot(epochs, history['val_total_loss'], 
             label='Validation Total Loss', linewidth=2.0, linestyle='--', marker='s', markersize=4)
    
    ax1.set_ylabel('Loss Value', fontsize=12)
    ax1.set_title('Training vs Validation Loss', fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_yscale('log')  # Use log scale for clearer loss trend

    # --- 5. Plot Lon/Lat MAE curves ---
    ax2.plot(epochs, history['train_lonlat_mae'], 
             label='Training Lon/Lat MAE', linewidth=2.0, marker='o', markersize=4)
    ax2.plot(epochs, history['val_lonlat_mae'], 
             label='Validation Lon/Lat MAE', linewidth=2.0, linestyle='--', marker='s', markersize=4)
    
    ax2.set_xlabel('Training Epochs', fontsize=12)
    ax2.set_ylabel('Lon/Lat Mean Absolute Error (°)', fontsize=12)
    ax2.set_title('Geographic Prediction Accuracy', fontsize=14)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Annotate minimum validation MAE
    min_mae = np.min(history['val_lonlat_mae'])
    min_epoch = np.argmin(history['val_lonlat_mae']) + 1
    ax2.annotate(f'Min: {min_mae:.4f}°\nEpoch {min_epoch}',
                 xy=(min_epoch, min_mae),
                 xytext=(min_epoch, min_mae * 1.1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10,
                 ha='center')

    # --- 6. Adjust layout and save/display ---
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Reserve space for main title
    
    # Save plot
    save_dir = os.path.dirname(history_path)
    save_path = os.path.join(save_dir, 'training_history_plot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    
    # Display plot
    plt.show()

if __name__ == '__main__':
    # Create command line argument parser
    parser = argparse.ArgumentParser(description='Visualize training history data.')
    parser.add_argument('history_file', default='./clip_geo_models_20251110_093258/training_history.npy', 
                        type=str, help='Path to the training history .npy file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call visualization function
    plot_training_history(args.history_file)