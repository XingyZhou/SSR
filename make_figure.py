import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def calc_sam(gt_spec, pred_spec):
    eps = 1e-10
    dot = np.dot(gt_spec, pred_spec)
    norm_gt = np.linalg.norm(gt_spec)
    norm_pred = np.linalg.norm(pred_spec)
    val = (dot + eps) / (norm_gt * norm_pred + eps)
    val = np.clip(val, -1.0, 1.0)
    return np.arccos(val) * 180.0 / np.pi

def main():
    # ---------------------------------------------------------
    # 1. Paths and parameters
    # ---------------------------------------------------------
    label_dir = './Label'
    result_dir = './Result'
    
    band_idx = 15 # Band 16 (0-based index 15)
    
    # Crop regions: (y1, y2, x1, x2)
    crop_cards = (80, 140, 60, 120)
    crop_crayon = (40, 100, 100, 160)
    
    # ---------------------------------------------------------
    # 2. Load Cards (Scene 7) Data
    # ---------------------------------------------------------
    cards_gt = sio.loadmat(os.path.join(label_dir, '7_S.mat'))['label']
    cards_s = sio.loadmat(os.path.join(result_dir, '7_S.mat'))['hsi']
    cards_m = sio.loadmat(os.path.join(result_dir, '7_M.mat'))['hsi']
    cards_prop = sio.loadmat(os.path.join(result_dir, '7_mamba_M.mat'))['hsi']
    
    # Compute metrics for Cards
    cards_psnr_s = compare_psnr(cards_gt, cards_s, data_range=1.0)
    cards_psnr_m = compare_psnr(cards_gt, cards_m, data_range=1.0)
    cards_psnr_prop = compare_psnr(cards_gt, cards_prop, data_range=1.0)
    
    # Compute SSIM for Cards
    cards_ssim_prop = 0.
    for c in range(cards_gt.shape[2]):
        cards_ssim_prop += compare_ssim(cards_gt[:, :, c], cards_prop[:, :, c], data_range=1.0)
    cards_ssim_prop /= cards_gt.shape[2]
    
    # ---------------------------------------------------------
    # 3. Load Crayons (Scene 9) Data
    # ---------------------------------------------------------
    crayon_gt = sio.loadmat(os.path.join(label_dir, '9_S.mat'))['label']
    crayon_s = sio.loadmat(os.path.join(result_dir, '9_S.mat'))['hsi']
    crayon_m = sio.loadmat(os.path.join(result_dir, '9_M.mat'))['hsi']
    crayon_prop_m = sio.loadmat(os.path.join(result_dir, '9_mamba_M.mat'))['hsi']
    crayon_prop_s = sio.loadmat(os.path.join(result_dir, '9_mamba_S.mat'))['hsi']
    
    # Compute metrics for Crayons
    crayon_psnr_s = compare_psnr(crayon_gt, crayon_s, data_range=1.0)
    crayon_psnr_m = compare_psnr(crayon_gt, crayon_m, data_range=1.0)
    crayon_psnr_prop_m = compare_psnr(crayon_gt, crayon_prop_m, data_range=1.0)
    
    crayon_ssim_prop_m = 0.
    for c in range(crayon_gt.shape[2]):
        crayon_ssim_prop_m += compare_ssim(crayon_gt[:, :, c], crayon_prop_m[:, :, c], data_range=1.0)
    crayon_ssim_prop_m /= crayon_gt.shape[2]
    
    # ---------------------------------------------------------
    # 4. Prepare Spectral curves for Crayons
    # ---------------------------------------------------------
    p1_coords = (134, 41)   # Blue Crayon
    p2_coords = (128, 160)  # Red Crayon
    
    gt_p1 = crayon_gt[p1_coords[0], p1_coords[1], :]
    s_p1 = crayon_s[p1_coords[0], p1_coords[1], :]
    m_p1 = crayon_m[p1_coords[0], p1_coords[1], :]
    prop_s_p1 = crayon_prop_s[p1_coords[0], p1_coords[1], :]
    prop_m_p1 = crayon_prop_m[p1_coords[0], p1_coords[1], :]
    
    gt_p2 = crayon_gt[p2_coords[0], p2_coords[1], :]
    s_p2 = crayon_s[p2_coords[0], p2_coords[1], :]
    m_p2 = crayon_m[p2_coords[0], p2_coords[1], :]
    prop_s_p2 = crayon_prop_s[p2_coords[0], p2_coords[1], :]
    prop_m_p2 = crayon_prop_m[p2_coords[0], p2_coords[1], :]
    
    # Calculate SAM
    sam_s_p1 = calc_sam(gt_p1, s_p1)
    sam_m_p1 = calc_sam(gt_p1, m_p1)
    sam_prop_s_p1 = calc_sam(gt_p1, prop_s_p1)
    sam_prop_m_p1 = calc_sam(gt_p1, prop_m_p1)
    
    sam_s_p2 = calc_sam(gt_p2, s_p2)
    sam_m_p2 = calc_sam(gt_p2, m_p2)
    sam_prop_s_p2 = calc_sam(gt_p2, prop_s_p2)
    sam_prop_m_p2 = calc_sam(gt_p2, prop_m_p2)
    
    # =========================================================
    # FIGURE A: Visualizing Spatial and Spectral Fidelity (Vertical Layout)
    # =========================================================
    fig_a = plt.figure(figsize=(8.0, 8.2), dpi=300)
    gs_a = fig_a.add_gridspec(2, 1, hspace=0.35, top=0.92, bottom=0.06, left=0.12, right=0.98)
    
    gs_a_top = gs_a[0].subgridspec(2, 4, height_ratios=[1.0, 0.8], hspace=0.08, wspace=0.08)
    gs_a_bottom = gs_a[1].subgridspec(2, 4, height_ratios=[1.0, 0.8], hspace=0.08, wspace=0.08)
    
    # 5.1.1 Plot Cards (Top block)
    cards_images = [cards_gt[:, :, band_idx], cards_s[:, :, band_idx], cards_m[:, :, band_idx], cards_prop[:, :, band_idx]]
    cards_titles = ['Ground Truth', 'SSR-S', 'SSR-M', 'Proposed (Ours)']
    cards_psnrs = [None, cards_psnr_s, cards_psnr_m, cards_psnr_prop]
    
    cards_main_axes = []
    cards_zoom_axes = []
    
    for col in range(4):
        ax_main = fig_a.add_subplot(gs_a_top[0, col])
        ax_main.imshow(cards_images[col], cmap='gray', vmin=0., vmax=1.)
        ax_main.axis('off')
        if col == 0:
            ax_main.text(-0.25, 0.5, "'Scene 7'\n(Band 16)", transform=ax_main.transAxes,
                         fontsize=11, fontweight='bold', va='center', ha='center', rotation=90)
        ax_main.set_title(cards_titles[col], fontsize=10, fontweight='bold', pad=4)
        cards_main_axes.append(ax_main)
        
        # Draw red rectangle
        y1, y2, x1, x2 = crop_cards
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1.2, edgecolor='red', facecolor='none')
        ax_main.add_patch(rect)
        
        # Zoomed-in axis
        ax_zoom = fig_a.add_subplot(gs_a_top[1, col])
        cropped = cards_images[col][y1:y2, x1:x2]
        ax_zoom.imshow(cropped, cmap='gray', vmin=0., vmax=1.)
        
        # Red border
        for spine in ax_zoom.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(1.5)
        ax_zoom.set_xticks([])
        ax_zoom.set_yticks([])
        cards_zoom_axes.append(ax_zoom)
        
        if col == 0:
            ax_zoom.text(-0.25, 0.5, 'Zoomed', transform=ax_zoom.transAxes,
                         fontsize=11, fontweight='bold', va='center', ha='center', rotation=90)
        
        # PSNR & SSIM text
        if col == 3:
            text_str = f"{cards_psnrs[col]:.2f} dB\n{cards_ssim_prop:.4f}"
            ax_zoom.text(0.95, 0.05, text_str, color='white', fontsize=7, fontweight='bold',
                         ha='right', va='bottom', transform=ax_zoom.transAxes,
                         bbox=dict(facecolor='black', alpha=0.7, pad=2, edgecolor='none'))
            ax_zoom.text(0.5, -0.15, f"{cards_psnrs[col]:.2f} dB", transform=ax_zoom.transAxes,
                         fontsize=9.5, fontweight='bold', ha='center', va='top')
        elif col > 0:
            ax_zoom.text(0.5, -0.15, f"{cards_psnrs[col]:.2f} dB", transform=ax_zoom.transAxes,
                         fontsize=9.5, fontweight='bold', ha='center', va='top')
            
    # Add connecting lines for Cards
    for col in [0, 3]:
        y1, y2, x1, x2 = crop_cards
        con1 = ConnectionPatch(xyA=(x1, y2), xyB=(0, 1), coordsA="data", coordsB="axes fraction",
                               axesA=cards_main_axes[col], axesB=cards_zoom_axes[col], color="red", linewidth=1.0)
        con2 = ConnectionPatch(xyA=(x2, y2), xyB=(1, 1), coordsA="data", coordsB="axes fraction",
                               axesA=cards_main_axes[col], axesB=cards_zoom_axes[col], color="red", linewidth=1.0)
        fig_a.add_artist(con1)
        fig_a.add_artist(con2)
        
    # 5.1.2 Plot Crayons (Bottom block)
    crayon_images = [crayon_gt[:, :, band_idx], crayon_s[:, :, band_idx], crayon_m[:, :, band_idx], crayon_prop_m[:, :, band_idx]]
    crayon_titles = ['Ground Truth', 'SSR-S', 'SSR-M', 'Proposed (Ours)']
    crayon_psnrs = [None, crayon_psnr_s, crayon_psnr_m, crayon_psnr_prop_m]
    
    crayon_main_axes = []
    crayon_zoom_axes = []
    
    for col in range(4):
        ax_main = fig_a.add_subplot(gs_a_bottom[0, col])
        ax_main.imshow(crayon_images[col], cmap='gray', vmin=0., vmax=1.)
        ax_main.axis('off')
        if col == 0:
            ax_main.text(-0.25, 0.5, "'Scene 9'\n(Band 16)", transform=ax_main.transAxes,
                         fontsize=11, fontweight='bold', va='center', ha='center', rotation=90)
        ax_main.set_title(crayon_titles[col], fontsize=10, fontweight='bold', pad=4)
        crayon_main_axes.append(ax_main)
        
        # Draw red rectangle
        y1, y2, x1, x2 = crop_crayon
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1.2, edgecolor='red', facecolor='none')
        ax_main.add_patch(rect)
        
        # Zoomed-in axis
        ax_zoom = fig_a.add_subplot(gs_a_bottom[1, col])
        cropped = crayon_images[col][y1:y2, x1:x2]
        ax_zoom.imshow(cropped, cmap='gray', vmin=0., vmax=1.)
        
        # Red border
        for spine in ax_zoom.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(1.5)
        ax_zoom.set_xticks([])
        ax_zoom.set_yticks([])
        crayon_zoom_axes.append(ax_zoom)
        
        if col == 0:
            ax_zoom.text(-0.25, 0.5, 'Zoomed', transform=ax_zoom.transAxes,
                         fontsize=11, fontweight='bold', va='center', ha='center', rotation=90)
        
        # PSNR & SSIM text
        if col == 3:
            text_str = f"{crayon_psnrs[col]:.2f} dB\n{crayon_ssim_prop_m:.4f}"
            ax_zoom.text(0.95, 0.05, text_str, color='white', fontsize=7, fontweight='bold',
                         ha='right', va='bottom', transform=ax_zoom.transAxes,
                         bbox=dict(facecolor='black', alpha=0.7, pad=2, edgecolor='none'))
            ax_zoom.text(0.5, -0.15, f"{crayon_psnrs[col]:.2f} dB", transform=ax_zoom.transAxes,
                         fontsize=9.5, fontweight='bold', ha='center', va='top')
        elif col > 0:
            ax_zoom.text(0.5, -0.15, f"{crayon_psnrs[col]:.2f} dB", transform=ax_zoom.transAxes,
                         fontsize=9.5, fontweight='bold', ha='center', va='top')
            
    # Add connecting lines for Crayons
    for col in [0, 3]:
        y1, y2, x1, x2 = crop_crayon
        con1 = ConnectionPatch(xyA=(x1, y2), xyB=(0, 1), coordsA="data", coordsB="axes fraction",
                               axesA=crayon_main_axes[col], axesB=crayon_zoom_axes[col], color="red", linewidth=1.0)
        con2 = ConnectionPatch(xyA=(x2, y2), xyB=(1, 1), coordsA="data", coordsB="axes fraction",
                               axesA=crayon_main_axes[col], axesB=crayon_zoom_axes[col], color="red", linewidth=1.0)
        fig_a.add_artist(con1)
        fig_a.add_artist(con2)
        
    # Title for Figure A
    fig_a.text(0.5, 0.96, 'Visualizing Spatial and Spectral Fidelity', fontsize=13, fontweight='bold', ha='center')
    
    # Save Figure A
    fig_a.savefig('./visualizing_fidelity_spatial.png', bbox_inches='tight', dpi=800)
    fig_a.savefig('./visualizing_fidelity_spatial.pdf', bbox_inches='tight')
    plt.close(fig_a)
    print("Saved Figure A to ./visualizing_fidelity_spatial.png and .pdf")
    
    # =========================================================
    # FIGURE B: Spectral Accuracy Comparison (Vertical Layout)
    # =========================================================
    fig_b = plt.figure(figsize=(8.0, 7.8), dpi=300)
    gs_b = fig_b.add_gridspec(2, 1, hspace=0.35, top=0.92, bottom=0.08, left=0.12, right=0.96)
    
    ax_b1 = fig_b.add_subplot(gs_b[0])
    ax_b2 = fig_b.add_subplot(gs_b[1])
    
    bands = np.arange(1, 29)
    lam28 = [
        453.5, 457.5, 462.0, 466.0, 471.5, 476.5, 481.5, 487.0, 492.5, 498.0, 
        504.0, 510.0, 516.0, 522.5, 529.5, 536.5, 544.0, 551.5, 558.5, 567.5, 
        575.5, 584.5, 594.5, 604.0, 614.5, 625.0, 636.5, 648.0
    ]
    
    # Subplot 1: Pixel 1 (Blue Crayon)
    ax_b1.plot(bands, gt_p1, color='black', linestyle='-', linewidth=2, label='Ground Truth')
    ax_b1.plot(bands, s_p1, color='blue', linestyle=':', linewidth=2, label='SSR-S')
    ax_b1.plot(bands, m_p1, color='green', linestyle='--', linewidth=2, label='SSR-M')
    ax_b1.plot(bands, prop_s_p1, color='magenta', linestyle='-.', linewidth=2, label='Proposed (3-stage)')
    ax_b1.plot(bands, prop_m_p1, color='red', linestyle='-', linewidth=2, label='Proposed (6-stage)')
    
    ax_b1.set_xlabel('Spectral Bands (Wavelength, nm)', fontsize=10)
    ax_b1.set_ylabel('Normalized Reflectance Intensity', fontsize=10)
    ax_b1.set_ylim(0, 1.05)
    ax_b1.set_xlim(1, 28)
    ax_b1.set_xticks([1, 5, 10, 15, 20, 25, 28])
    ax_b1.set_xticklabels([f"{lam28[i-1]:.1f}" for i in [1, 5, 10, 15, 20, 25, 28]], fontsize=8)
    ax_b1.grid(True, linestyle=':', alpha=0.6)
    
    ax_b1.legend(loc='upper right', fontsize=8)
    sam_text_p1 = (f"SAM: Prop (6-stage) {sam_prop_m_p1:.1f}°\n"
                   f"         Prop (3-stage) {sam_prop_s_p1:.1f}°\n"
                   f"         SSR-M {sam_m_p1:.1f}°\n"
                   f"         SSR-S {sam_s_p1:.1f}°")
    ax_b1.text(10, 0.15, sam_text_p1, fontsize=8.5, color='black', fontweight='bold',
               bbox=dict(facecolor='white', alpha=0.8, pad=3, edgecolor='grey'))
    ax_b1.set_title('Spectral Curve: Blue Crayon (Pixel 1)', fontsize=10, fontweight='bold')
    
    # Subplot 2: Pixel 2 (Red Crayon)
    ax_b2.plot(bands, gt_p2, color='black', linestyle='-', linewidth=2, label='Ground Truth')
    ax_b2.plot(bands, s_p2, color='blue', linestyle=':', linewidth=2, label='SSR-S')
    ax_b2.plot(bands, m_p2, color='green', linestyle='--', linewidth=2, label='SSR-M')
    ax_b2.plot(bands, prop_s_p2, color='magenta', linestyle='-.', linewidth=2, label='Proposed (3-stage)')
    ax_b2.plot(bands, prop_m_p2, color='red', linestyle='-', linewidth=2, label='Proposed (6-stage)')
    
    ax_b2.set_xlabel('Spectral Bands (Wavelength, nm)', fontsize=10)
    ax_b2.set_ylabel('Normalized Reflectance Intensity', fontsize=10)
    ax_b2.set_ylim(0, 1.05)
    ax_b2.set_xlim(1, 28)
    ax_b2.set_xticks([1, 5, 10, 15, 20, 25, 28])
    ax_b2.set_xticklabels([f"{lam28[i-1]:.1f}" for i in [1, 5, 10, 15, 20, 25, 28]], fontsize=8)
    ax_b2.grid(True, linestyle=':', alpha=0.6)
    
    ax_b2.legend(loc='upper right', fontsize=8)
    sam_text_p2 = (f"SAM: Prop (6-stage) {sam_prop_m_p2:.1f}°\n"
                   f"         Prop (3-stage) {sam_prop_s_p2:.1f}°\n"
                   f"         SSR-M {sam_m_p2:.1f}°\n"
                   f"         SSR-S {sam_s_p2:.1f}°")
    ax_b2.text(10, 0.15, sam_text_p2, fontsize=8.5, color='black', fontweight='bold',
               bbox=dict(facecolor='white', alpha=0.8, pad=3, edgecolor='grey'))
    ax_b2.set_title('Spectral Curve: Red Crayon (Pixel 2)', fontsize=10, fontweight='bold')
    
    # Reference image inset in ax_b1
    rgb_crayon = np.stack([crayon_gt[:, :, 25], crayon_gt[:, :, 16], crayon_gt[:, :, 2]], axis=-1)
    rgb_crayon = np.power(np.clip(rgb_crayon, 0., 1.), 1.0 / 2.2)
    
    ax_inset = ax_b1.inset_axes([0.05, 0.42, 0.35, 0.35])
    ax_inset.imshow(rgb_crayon)
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    
    ax_inset.plot(41, 134, marker='+', markersize=10, color='black', markeredgewidth=1.5)
    ax_inset.text(41, 121, '1', color='black', fontsize=9, fontweight='bold', ha='center', va='bottom')
    
    ax_inset.plot(160, 128, marker='+', markersize=10, color='black', markeredgewidth=1.5)
    ax_inset.text(160, 115, '2', color='black', fontsize=9, fontweight='bold', ha='center', va='bottom')
    
    ax_inset.set_title('Reference Image', fontsize=8, fontweight='bold', pad=2)
    ax_b1.text(0.05, 0.38, '"Blue Crayon" (pixel 1)\n"Red Crayon" (pixel 2)', transform=ax_b1.transAxes,
              fontsize=8, va='top', ha='left', color='black', fontweight='semibold')
    
    # Title for Figure B
    fig_b.text(0.5, 0.96, 'Spectral Accuracy Comparison', fontsize=13, fontweight='bold', ha='center')
    
    # Save Figure B
    fig_b.savefig('./visualizing_fidelity_spectral.png', bbox_inches='tight', dpi=800)
    fig_b.savefig('./visualizing_fidelity_spectral.pdf', bbox_inches='tight')
    plt.close(fig_b)
    print("Saved Figure B to ./visualizing_fidelity_spectral.png and .pdf")

if __name__ == '__main__':
    main()
