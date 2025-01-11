import sys
import numpy as np
import pandas as pd
import sklearn.linear_model as skl
import pickle
import argparse
import os

# Argument Parser
parser = argparse.ArgumentParser(description="Spike Regression for VAE Latents")
parser.add_argument("-spike", "--spike_path", help="Path to spike data CSV", default="./data/design_array_VISp.csv")
parser.add_argument("-map", "--map_path", help="Path to image-spike map CSV", default="./data/stimuli_index_image_index.csv")
parser.add_argument("-img", "--image_path", help="Path to image latents", default="./data/extracted_features/nsd_vdvae_features_31l.npz")
args = parser.parse_args()

# Paths
spike_path = args.spike_path
map_path = args.map_path
image_path = args.image_path

print(f"Using spike data path: {spike_path}")
print(f"Using mapping data path: {map_path}")
print(f"Using image latents path: {image_path}")

# Load spike data
spike_data = pd.read_csv(spike_path, header=0)  # No header in the file
design_array = spike_data.values  # Convert to NumPy array
print(f"[INFO] Spike data loaded. Shape: {design_array.shape}")

# Load image-to-spike map
mapping = pd.read_csv(map_path)
print(f"[INFO] Mapping data loaded. Shape: {mapping.shape}")
print(f"[INFO] Mapping columns: {mapping.columns.tolist()}")

# Load image latents
image_latents = np.load(image_path)
train_latents = image_latents['train_latents']  # Training image latents
print(f"[INFO] Image latents loaded. Shape: {train_latents.shape}")

# Expand images to match spike data
image_indices = mapping['frame'].astype(int).values  # Image indices (frame column in mapping)
spike_to_image_indices = mapping['index'].values  # Spike indices
expanded_images = train_latents[image_indices]  # Repeat images to match spikes
print(f"[INFO] Expanded image latents to match spikes. Shape: {expanded_images.shape}")

# Aggregate spike data by image index
print("[INFO] Aggregating spike data by image index...")
train_spike = np.zeros((train_latents.shape[0], design_array.shape[1]))
image_spike_counts = np.zeros(train_latents.shape[0])  # Count occurrences for each image index
for i, img_idx in enumerate(image_indices):
    train_spike[img_idx] += design_array[i]
    image_spike_counts[img_idx] += 1

# Average aggregated spike data
train_spike = train_spike / np.maximum(image_spike_counts[:, None], 1)
print(f"[DEBUG] Train spike shape after aggregation: {train_spike.shape}")

# Check shapes of train_spike and train_latents
print(f"[DEBUG] Train spike shape: {train_spike.shape}")
print(f"[DEBUG] Train latents shape: {train_latents.shape}")

# Normalize spike data
norm_mean_train = np.mean(train_spike, axis=0)
norm_scale_train = np.std(train_spike, axis=0, ddof=1)
train_spike = (train_spike - norm_mean_train) / norm_scale_train
print(f"[INFO] Normalized train spike data. Mean: {np.mean(train_spike)}, Std: {np.std(train_spike)}")

# Ensure consistent samples
if train_spike.shape[0] != train_latents.shape[0]:
    print("[ERROR] Sample size mismatch between train_spike and train_latents!")
    print(f"train_spike samples: {train_spike.shape[0]}, train_latents samples: {train_latents.shape[0]}")
    sys.exit(1)

# Regression
print("[INFO] Training regression model...")
try:
    reg = skl.Ridge(alpha=100000, max_iter=50000, fit_intercept=True)
    reg.fit(train_spike, train_latents)  # Regression training
    predicted_latents = reg.predict(train_spike)
    print(f"[INFO] Regression score: {reg.score(train_spike, train_latents)}")
except ValueError as e:
    print(f"[ERROR] Regression failed: {e}")
    print(f"[DEBUG] Spike shape: {train_spike.shape}, Latents shape: {train_latents.shape}")
    sys.exit(1)

# Save results
output_dir = "./data/predicted_features/"
os.makedirs(output_dir, exist_ok=True)

np.savez(
    os.path.join(output_dir, "vdvae_predicted_latents.npz"),
    predicted_latents=predicted_latents
)

datadict = {
    'weight': reg.coef_,
    'bias': reg.intercept_
}

with open(os.path.join(output_dir, "vdvae_regression_weights.pkl"), "wb") as f:
    pickle.dump(datadict, f)

print("[INFO] Regression complete and results saved.")
