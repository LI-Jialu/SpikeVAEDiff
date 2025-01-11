import sys
import numpy as np
import sklearn.linear_model as skl
import argparse
import pickle
import os

# Argument Parser
parser = argparse.ArgumentParser(description="Spike-CLIP Regression")
parser.add_argument("-spike", "--spike_path", help="Path to spike data CSV", default="./data/design_array_VISp.csv")
parser.add_argument("-map", "--map_path", help="Path to image-spike map CSV", default="./data/stimuli_index_image_index.csv")
parser.add_argument("-train_clip", "--train_clip_path", help="Path to training CLIP latents npy file", default="./data/extracted_features/nsd_cliptext_train.npy")
parser.add_argument("-test_clip", "--test_clip_path", help="Path to testing CLIP latents npy file", default="./data/extracted_features/nsd_cliptext_test.npy")
parser.add_argument("-out", "--output_path", help="Output path for predicted latents", default="./data/predicted_features/nsd_cliptext_predtest.npy")
parser.add_argument("-reg_weights", "--regression_weights_path", help="Path to save regression weights", default="./data/regression_weights/cliptext_regression_weights.pkl")
parser.add_argument("-alpha", "--ridge_alpha", help="Ridge regularization parameter", default=100000, type=float)
args = parser.parse_args()

# Load Spike Data
print("Loading spike data...")
spike_data = np.loadtxt(args.spike_path, delimiter=",", skiprows=1)
print(f"Spike data shape: {spike_data.shape}")

# Load Image-Spike Map
print("Loading image-spike map...")
map_data = np.loadtxt(args.map_path, delimiter=",", skiprows=1, dtype=int)
spike_to_image = map_data[:, -1]  # Use the last column for image indices
print(f"Image-spike map shape: {map_data.shape}, unique images: {len(np.unique(spike_to_image))}")

# Load Training and Testing CLIP Latents
print("Loading CLIP latents...")
train_clip = np.load(args.train_clip_path)
test_clip = np.load(args.test_clip_path)
num_train, num_embed, num_dim = train_clip.shape
num_test = test_clip.shape[0]
print(f"Train CLIP shape: {train_clip.shape}, Test CLIP shape: {test_clip.shape}")

# Align Spike Data
print("Aligning spike data with images...")
image_indices = map_data[:, -1]  # Image indices corresponding to spike data
train_spike = spike_data[:num_train]  # Align train spikes
test_spike = spike_data[num_train:num_train + num_test]  # Align test spikes

print(f"Train spike shape: {train_spike.shape}, Test spike shape: {test_spike.shape}")

# Preprocessing Spike Data
print("Preprocessing spike data...")
norm_mean_train = np.mean(train_spike, axis=0)
norm_scale_train = np.std(train_spike, axis=0, ddof=1)
train_spike = (train_spike - norm_mean_train) / norm_scale_train
test_spike = (test_spike - norm_mean_train) / norm_scale_train

print(f"Train spike mean: {np.mean(train_spike):.6f}, std: {np.std(train_spike):.6f}")
print(f"Test spike mean: {np.mean(test_spike):.6f}, std: {np.std(test_spike):.6f}")

# Ridge Regression for CLIP Latents
print("Training Ridge Regression for CLIP latents...")
pred_clip = np.zeros_like(test_clip)
reg_w = np.zeros((num_embed, num_dim, train_spike.shape[1]))  # Store weights
reg_b = np.zeros((num_embed, num_dim))  # Store biases

for i in range(num_embed):
    for j in range(num_dim):
        reg = skl.Ridge(alpha=args.ridge_alpha, max_iter=50000, fit_intercept=True)
        reg.fit(train_spike, train_clip[:, i, j])  # Train on the train_clip subset
        reg_w[i, j] = reg.coef_  # Save weights
        reg_b[i, j] = reg.intercept_  # Save biases
        pred_test_latent = reg.predict(test_spike)  # Predict on the test_spike subset
        # Normalize prediction to match training latent distribution
        std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent)) / np.std(pred_test_latent)
        pred_clip[:, i, j] = std_norm_test_latent * np.std(train_clip[:, i, j]) + np.mean(train_clip[:, i, j])
    print(f"Embedding {i+1}/{num_embed} complete.")

# Save Predicted Latents
print("Saving predicted latents...")
np.save(args.output_path, pred_clip)
print(f"Predicted latents saved to {args.output_path}")

# Save Regression Weights and Biases
print("Saving regression weights and biases...")
datadict = {
    'weight': reg_w,
    'bias': reg_b
}
with open(args.regression_weights_path, "wb") as f:
    pickle.dump(datadict, f)
print(f"Regression weights and biases saved to {args.regression_weights_path}")
