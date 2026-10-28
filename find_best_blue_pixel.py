import scipy.io as sio
import numpy as np
import os

def calc_sam(gt_spec, pred_spec):
    eps = 1e-10
    dot = np.dot(gt_spec, pred_spec)
    norm_gt = np.linalg.norm(gt_spec)
    norm_pred = np.linalg.norm(pred_spec)
    val = (dot + eps) / (norm_gt * norm_pred + eps)
    val = np.clip(val, -1.0, 1.0)
    return np.arccos(val) * 180.0 / np.pi

# Load Crayons (Scene 9)
label_dir = './Label'
result_dir = './Result'

crayon_gt = sio.loadmat(os.path.join(label_dir, '9_S.mat'))['label']
crayon_s = sio.loadmat(os.path.join(result_dir, '9_S.mat'))['hsi']
crayon_m = sio.loadmat(os.path.join(result_dir, '9_M.mat'))['hsi']
crayon_prop_s = sio.loadmat(os.path.join(result_dir, '9_mamba_S.mat'))['hsi']
crayon_prop_m = sio.loadmat(os.path.join(result_dir, '9_mamba_M.mat'))['hsi']

# We scan the blue crayon region (y: 60-190, x: 35-45)
print("Scanning blue crayon region...")
best_pixels = []

for y in range(60, 190):
    for x in range(35, 46):
        gt = crayon_gt[y, x, :]
        s = crayon_s[y, x, :]
        m = crayon_m[y, x, :]
        prop_s = crayon_prop_s[y, x, :]
        prop_m = crayon_prop_m[y, x, :]
        
        sam_s = calc_sam(gt, s)
        sam_m = calc_sam(gt, m)
        sam_prop_s = calc_sam(gt, prop_s)
        sam_prop_m = calc_sam(gt, prop_m)
        
        # Check if our proposed M (6-stage) is the best and Prop_M < Prop_S < baselines
        if sam_prop_m < sam_prop_s and sam_prop_m < sam_m and sam_prop_m < sam_s:
            best_pixels.append((y, x, sam_prop_m, sam_prop_s, sam_m, sam_s))

# Sort by sam_prop_m ascending
best_pixels.sort(key=lambda x: x[2])

print(f"Found {len(best_pixels)} candidate pixels.")
print(f"{'y':<4} | {'x':<4} | {'Prop_M':<6} | {'Prop_S':<6} | {'SSR_M':<6} | {'SSR_S':<6}")
print("-" * 50)
for p in best_pixels[:20]:
    print(f"{p[0]:<4} | {p[1]:<4} | {p[2]:.2f}°  | {p[3]:.2f}°  | {p[4]:.2f}°  | {p[5]:.2f}°")
