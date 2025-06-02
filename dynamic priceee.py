import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
import logging
import os
from datetime import datetime, timedelta
import torch.nn as nn
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from typing import Dict, Tuple
import traceback

# 1. Environment Implementation
class DairyPricingEnv(gym.Env):
    def __init__(self, data_path='/Users/AryanaGhugharawala1/Desktop/dqn2/dairy.csv'):
        super().__init__()
        
        try:
            # Load data
            self.data = pd.read_csv(data_path)
            
            # Define action space: [-10%, -5%, 0%, +5%, +10%]
            self.action_space = spaces.Discrete(5)
            self.price_adjustments = [-0.10, -0.05, 0.0, 0.05, 0.10]
            
            # Preprocess data and set up state space
            self._preprocess_data()
            
            # Initialize step counter
            self.current_step = 0
            
            print("Environment initialized successfully!")
            
        except Exception as e:
            print(f"Error in initialization: {str(e)}")
            traceback.print_exc()
            raise

    def _preprocess_data(self):
        """Preprocess the data and set up the state space."""
        try:
            # Create a copy for preprocessing
            self.processed_data = self.data.copy()
            
            # Convert date columns to datetime
            date_columns = ['Production Date', 'Expiration Date', 'Date']
            for col in date_columns:
                self.processed_data[col] = pd.to_datetime(self.processed_data[col])
            
            # Calculate days-based features
            current_date = pd.Timestamp.now()
            self.processed_data['DaysTillExpiry'] = (
                self.processed_data['Expiration Date'] - current_date
            ).dt.days
            
            self.processed_data['DaysSinceProduction'] = (
                current_date - self.processed_data['Production Date']
            ).dt.days
            
            # Normalize numerical features
            numerical_features = [
                'Quantity in Stock (liters/kg)',
                'Price per Unit',
                'Quantity Sold (liters/kg)',
                'Minimum Stock Threshold (liters/kg)',
                'DaysTillExpiry',
                'DaysSinceProduction',
                'Shelf Life (days)'
            ]
            
            # Store scaling parameters
            self.feature_means = {}
            self.feature_stds = {}
            
            # Normalize each numerical feature
            for feature in numerical_features:
                if feature in self.processed_data.columns:
                    mean = self.processed_data[feature].mean()
                    std = self.processed_data[feature].std()
                    if std == 0:
                        std = 1.0
                    self.feature_means[feature] = mean
                    self.feature_stds[feature] = std
                    self.processed_data[feature] = (self.processed_data[feature] - mean) / std
            
            # One-hot encode Storage Condition
            storage_dummies = pd.get_dummies(
                self.processed_data['Storage Condition'],
                prefix='Storage',
                dummy_na=True
            )
            self.processed_data = pd.concat([self.processed_data, storage_dummies], axis=1)
            
            # Define state columns
            self.state_columns = (
                numerical_features + 
                [col for col in storage_dummies.columns]
            )
            
            # Set up observation space
            self.n_features = len(self.state_columns)
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.n_features,),
                dtype=np.float32
            )
            
            print(f"Data preprocessed successfully! State space dimension: {self.n_features}")
            
        except Exception as e:
            print(f"Error in _preprocess_data: {str(e)}")
            traceback.print_exc()
            raise

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        try:
            self.current_step = 0
            
            # Get initial state
            initial_state = self.processed_data[self.state_columns].iloc[0].values.astype(np.float32)
            
            return initial_state, {}
            
        except Exception as e:
            print(f"Error in reset: {str(e)}")
            traceback.print_exc()
            raise

    def step(self, action):
        """Execute one step in the environment."""
        try:
            current_data = self.data.iloc[self.current_step]
            
            # Calculate baseline scenario
            baseline_sales = current_data['Quantity Sold (liters/kg)']
            available_stock = current_data['Quantity in Stock (liters/kg)']
            
            # Apply price adjustment and calculate actual sales
            price_adjustment = self.price_adjustments[action]
            current_price = current_data['Price per Unit']
            new_price = current_price * (1 + price_adjustment)
            
            # Calculate demand with price elasticity
            price_elasticity = -1.5
            demand_change = price_elasticity * price_adjustment
            expected_demand = baseline_sales * (1 + demand_change)
            
            # Add some randomness
            noise = np.random.normal(1, 0.1)
            actual_demand = expected_demand * noise
            
            # Actual sales can't exceed stock
            actual_sales = min(actual_demand, available_stock)
            
            # Calculate reward
            reward = self._calculate_reward(current_data, action, actual_sales, new_price)
            
            # Update step
            self.current_step += 1
            done = self.current_step >= len(self.data) - 1
            
            # Get next state
            if not done:
                next_state = self.processed_data[self.state_columns].iloc[self.current_step].values
            else:
                next_state = self.processed_data[self.state_columns].iloc[0].values
            
            next_state = next_state.astype(np.float32)
            
            # Enhanced info dictionary
            info = {
                'sales': float(actual_sales),
                'baseline_sales': float(baseline_sales),
                'available_stock': float(available_stock),
                'revenue': float(actual_sales * new_price),
                'waste': float(max(0, available_stock - actual_sales)),
                'baseline_waste': float(max(0, available_stock - baseline_sales)),
                'price_adjustment': float(price_adjustment)
            }
            
            return next_state, float(reward), done, False, info
            
        except Exception as e:
            print(f"Error in step: {str(e)}")
            traceback.print_exc()
            raise

    def _calculate_reward(self, state_data, action, actual_sales, new_price):
        """Calculate the reward based on multiple objectives."""
        try:
            # Revenue component
            revenue = actual_sales * new_price
            baseline_revenue = state_data['Quantity Sold (liters/kg)'] * state_data['Price per Unit']
            revenue_reward = (revenue - baseline_revenue) / (baseline_revenue + 1e-6)
            
            # Waste prevention component
            available_stock = state_data['Quantity in Stock (liters/kg)']
            waste = max(0, available_stock - actual_sales)
            max_possible_waste = available_stock
            waste_prevention = 1.0 - (waste / (max_possible_waste + 1e-6))
            
            # Price stability component
            price_change = abs(self.price_adjustments[action])
            price_stability = 1.0 - price_change
            
            # Stock management component
            min_threshold = state_data['Minimum Stock Threshold (liters/kg)']
            stock_penalty = -1.0 if (available_stock - actual_sales) < min_threshold else 0.0
            
            # Combine rewards
            reward = (
                0.4 * revenue_reward +
                0.3 * waste_prevention +
                0.2 * price_stability +
                0.1 * stock_penalty
            )
            
            # Clip reward
            reward = np.clip(reward, -1.0, 1.0)
            
            return float(reward)
            
        except Exception as e:
            print(f"Error in _calculate_reward: {str(e)}")
            traceback.print_exc()
            return 0.0

# 2. Training Setup and Execution
def setup_training():
    """Setup training with DQN parameters."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environments
    env = DairyPricingEnv()
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.,
        clip_reward=10.,
        gamma=0.99,
        epsilon=1e-08
    )
    
    # Create validation environment
    eval_env = DairyPricingEnv()
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.,
        clip_reward=10.,
        gamma=0.99,
        epsilon=1e-08
    )
    
    # Setup neural network architecture
    policy_kwargs = dict(
        net_arch=[64, 64],  # Two hidden layers
        activation_fn=nn.ReLU
    )
    
    # Create DQN model with optimized hyperparameters
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=os.path.join(output_dir, "tensorboard")
    )
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(output_dir, "best_model"),
        log_path=os.path.join(output_dir, "eval_results"),
        eval_freq=1000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=os.path.join(output_dir, "checkpoints"),
        name_prefix="dqn_model"
    )
    
    return model, env, eval_env, eval_callback, checkpoint_callback, output_dir

# 3. Train the Model
def train_model(model, env, eval_callback, checkpoint_callback, output_dir):
    total_timesteps = 200000
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True
        )
        
        # Save the final model
        model.save(os.path.join(output_dir, 'final_model'))
        env.save(os.path.join(output_dir, 'vec_normalize.pkl'))
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        raise

# 4. Evaluate the Model
def evaluate_model(model_path, output_dir):
    """Evaluate the trained model and collect metrics with actual waste reduction tracking."""
    try:
        # Load the trained model
        model = DQN.load(model_path)
        
        # Create and wrap the environment
        eval_env = DairyPricingEnv()
        eval_env = DummyVecEnv([lambda: eval_env])
        
        # Initialize metrics
        waste_reduction_history = []
        baseline_wastes = []
        actual_wastes = []
        
        # Run evaluation episodes
        n_eval_episodes = 10
        
        for episode in range(n_eval_episodes):
            obs = eval_env.reset()
            episode_baseline_waste = 0
            episode_actual_waste = 0
            done = False
            
            while not done:
                # Get action from model (returns numpy array)
                action, _states = model.predict(obs, deterministic=True)
                
                # Step environment (handle vectorized output)
                obs, rewards, dones, infos = eval_env.step(action)
                
                # Extract info from vectorized environment
                info = infos[0]  # Get info from first (and only) environment
                done = dones[0]  # Get done flag from first environment
                
                # Calculate waste for this step
                current_stock = info.get('available_stock', 0)
                baseline_sales = info.get('baseline_sales', 0)
                actual_sales = info.get('sales', 0)
                
                step_baseline_waste = max(0, current_stock - baseline_sales)
                step_actual_waste = max(0, current_stock - actual_sales)
                
                episode_baseline_waste += step_baseline_waste
                episode_actual_waste += step_actual_waste
            
            # Calculate waste reduction for this episode
            if episode_baseline_waste > 0:
                waste_reduction = ((episode_baseline_waste - episode_actual_waste) / episode_baseline_waste * 100)
            else:
                waste_reduction = 0
                
            waste_reduction_history.append(waste_reduction)
            baseline_wastes.append(episode_baseline_waste)
            actual_wastes.append(episode_actual_waste)
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Plot waste reduction progression
        plt.subplot(2, 1, 1)
        plt.plot(waste_reduction_history, 'b-', label='Waste Reduction')
        plt.axhline(y=np.mean(waste_reduction_history), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(waste_reduction_history):.2f}%')
        plt.title('Waste Reduction Progress')
        plt.xlabel('Episode')
        plt.ylabel('Waste Reduction (%)')
        plt.legend()
        plt.grid(True)
        
        # Plot actual vs baseline waste
        plt.subplot(2, 1, 2)
        plt.plot(baseline_wastes, 'r-', label='Baseline Waste')
        plt.plot(actual_wastes, 'g-', label='Actual Waste')
        plt.title('Waste Comparison')
        plt.xlabel('Episode')
        plt.ylabel('Waste Amount')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'waste_reduction_analysis.png'))
        plt.close()
        
        # Calculate summary metrics
        metrics = {
            'mean_waste_reduction': np.mean(waste_reduction_history),
            'std_waste_reduction': np.std(waste_reduction_history),
            'final_waste_reduction': waste_reduction_history[-1],
            'mean_baseline_waste': np.mean(baseline_wastes),
            'mean_actual_waste': np.mean(actual_wastes)
        }
        
        # Save metrics
        metrics_path = os.path.join(output_dir, 'waste_metrics.txt')
        with open(metrics_path, 'w') as f:
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.2f}\n")
        
        return metrics
        
    except Exception as e:
        print(f"Error in evaluate_model: {str(e)}")
        traceback.print_exc()
        raise

# 5. Visualize Results
def visualize_results(episode_rewards, episode_lengths, waste_values, 
                     stockout_values, revenue_values, waste_prevention_values,
                     revenue_improvement):
    try:
        plt.figure(figsize=(15, 12))
        
        # Plot rewards
        plt.subplot(3, 2, 1)
        plt.plot(episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # Plot waste and prevention
        plt.subplot(3, 2, 2)
        plt.plot(waste_values, label='Waste')
        plt.plot(waste_prevention_values, label='Prevention %')
        plt.title('Waste and Prevention')
        plt.xlabel('Episode')
        plt.ylabel('Amount')
        plt.legend()
        
        # Plot stockouts
        plt.subplot(3, 2, 3)
        plt.plot(stockout_values)
        plt.title('Stockouts per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Stockouts')
        
        # Plot revenue
        plt.subplot(3, 2, 4)
        plt.plot(revenue_values)
        plt.title('Revenue per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Revenue')
        
        # Plot revenue improvement
        plt.subplot(3, 2, 5)
        plt.plot(revenue_improvement)
        plt.title('Revenue Improvement %')
        plt.xlabel('Episode')
        plt.ylabel('Improvement %')
        
        # Plot waste prevention trend
        plt.subplot(3, 2, 6)
        z = np.polyfit(range(len(waste_prevention_values)), waste_prevention_values, 1)
        p = np.poly1d(z)
        plt.plot(waste_prevention_values, label='Actual')
        plt.plot(range(len(waste_prevention_values)), p(range(len(waste_prevention_values))), 
                'r--', label='Trend')
        plt.title('Waste Prevention Trend')
        plt.xlabel('Episode')
        plt.ylabel('Prevention %')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_results.png')
        plt.show()
        
        # Save summary statistics
        summary = {
            'mean_reward': np.mean(episode_rewards),
            'mean_waste_prevention': np.mean(waste_prevention_values),
            'mean_revenue_improvement': np.mean(revenue_improvement),
            'final_waste_prevention': waste_prevention_values[-1],
            'trend_slope': z[0]  # Positive slope indicates improving prevention
        }
        
        with open('performance_summary.txt', 'w') as f:
            f.write("=== Performance Summary ===\n")
            f.write(f"Mean Reward: {summary['mean_reward']:.2f}\n")
            f.write(f"Mean Waste Prevention: {summary['mean_waste_prevention']:.2f}%\n")
            f.write(f"Mean Revenue Improvement: {summary['mean_revenue_improvement']:.2f}%\n")
            f.write(f"Final Waste Prevention: {summary['final_waste_prevention']:.2f}%\n")
            f.write(f"Waste Prevention Trend: {'Improving' if z[0] > 0 else 'Declining'}\n")
    
    except Exception as e:
        print(f"An error occurred during visualization: {str(e)}")
        traceback.print_exc()
        raise

def main():
    """Main execution with DQN training."""
    env = None
    eval_env = None
    
    try:
        print("Initializing training setup...")
        model, env, eval_env, eval_callback, checkpoint_callback, output_dir = setup_training()
        
        print("Starting DQN training...")
        total_timesteps = 150000
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True
        )
        
        print("Saving model...")
        model_path = os.path.join(output_dir, "final_model")
        model.save(model_path)
        env.save(os.path.join(output_dir, "vec_normalize.pkl"))
        
        print("\nEvaluating on validation set...")
        eval_metrics = evaluate_model(model_path, output_dir)
        
        print("\nTraining completed successfully!")
        print("\nValidation Metrics:")
        for metric, value in eval_metrics.items():
            print(f"{metric}: {value:.2f}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()
    
    finally:
        print("Cleaning up...")
        if env is not None:
            env.close()
        if eval_env is not None:
            eval_env.close()

if __name__ == "__main__":
    main()