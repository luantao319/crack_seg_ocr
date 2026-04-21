import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.fftpack import fft2, fftshift
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# ====================== 1. Global Configuration =======================
# Path settings
generic_dir = r'C:\Users\H3C\Desktop\coco\coco500'  # Generic images directory (RGB)
pavement_dir = r'C:\Users\H3C\Desktop\coco\crack500'  # Pavement images directory (RGB)
save_root = r'./'  # Output directory for plots
os.makedirs(save_root, exist_ok=True)

# Data settings
image_formats = ['.jpg', '.png', '.jpeg']
sample_size = 100  # Max samples per class
random_seed = 42
np.random.seed(random_seed)

# Visualization settings
gen_color = '#2E86AB'  # Color for generic images (blue)
pav_color = '#E63946'  # Color for pavement images (red)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # English font
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# ====================== 2. Core Functions =======================
def load_images_from_dir(dir_path, max_count=sample_size):
    """Load images from directory, return grayscale and RGB image lists"""
    gray_imgs = []
    rgb_imgs = []
    file_list = [f for f in os.listdir(dir_path) if any(f.lower().endswith(fmt) for fmt in image_formats)]
    
    # Random sampling to balance data
    if len(file_list) > max_count:
        file_list = np.random.choice(file_list, max_count, replace=False)
    
    for filename in file_list:
        img_path = os.path.join(dir_path, filename)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"⚠️  Skip corrupted image: {img_path}")
            continue
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        rgb_imgs.append(img_rgb)
        gray_imgs.append(img_gray)
    
    print(f"✅ Loaded {dir_path}: {len(rgb_imgs)} RGB images | {len(gray_imgs)} grayscale images")
    return gray_imgs, rgb_imgs

def extract_rgb_pixels(rgb_imgs):
    """Extract all RGB channel pixels from images (for histogram plotting)"""
    r_pixels = []
    g_pixels = []
    b_pixels = []
    
    for img in rgb_imgs:
        r_pixels.extend(img[..., 0].flatten())  # R channel
        g_pixels.extend(img[..., 1].flatten())  # G channel
        b_pixels.extend(img[..., 2].flatten())  # B channel
    
    return np.array(r_pixels), np.array(g_pixels), np.array(b_pixels)

def extract_comprehensive_features(gray_imgs, rgb_imgs):
    """Extract multi-dimensional features for t-SNE (texture+color+edge+freq)"""
    # 1. Texture features (GLCM + LBP)
    texture_feats = {
        'glcm_contrast': [], 'glcm_correlation': [], 'glcm_energy': [], 'glcm_homogeneity': [],
        'lbp_smooth': [], 'lbp_entropy': []
    }
    gray_levels = 8
    lbp_radius = 3
    lbp_neighbors = 8
    
    for img in gray_imgs:
        # GLCM
        img_compressed = img // (256 // gray_levels)
        glcm = graycomatrix(img_compressed, distances=[1], angles=[0], levels=gray_levels, symmetric=True, normed=True)
        texture_feats['glcm_contrast'].append(graycoprops(glcm, 'contrast')[0, 0])
        texture_feats['glcm_correlation'].append(graycoprops(glcm, 'correlation')[0, 0])
        texture_feats['glcm_energy'].append(graycoprops(glcm, 'energy')[0, 0])
        texture_feats['glcm_homogeneity'].append(graycoprops(glcm, 'homogeneity')[0, 0])
        
        # LBP
        lbp = local_binary_pattern(img, lbp_neighbors, lbp_radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=int(lbp.max()+1), density=True)
        texture_feats['lbp_smooth'].append(lbp_hist[0])
        lbp_hist_valid = lbp_hist[lbp_hist > 0]
        texture_feats['lbp_entropy'].append(-np.sum(lbp_hist_valid * np.log2(lbp_hist_valid)))
    
    # 2. Color features (full distribution for t-SNE)
    color_feats = {
        'r_mean': [], 'g_mean': [], 'b_mean': [],
        'r_var': [], 'g_var': [], 'b_var': []
    }
    for img in rgb_imgs:
        for i, ch in enumerate(['r', 'g', 'b']):
            ch_vals = img[..., i].flatten()
            color_feats[f'{ch}_mean'].append(np.mean(ch_vals))
            color_feats[f'{ch}_var'].append(np.var(ch_vals))
    
    # 3. Edge features (use cv2.Canny to avoid skimage dependency issues)
    edge_feats = {'edge_density': []}
    for img in gray_imgs:
        edges = cv2.Canny(img, threshold1=100, threshold2=200)  # cv2 built-in Canny
        edge_density = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
        edge_feats['edge_density'].append(edge_density)
    
    # 4. Frequency features (FFT)
    freq_feats = {'low_freq_ratio': []}
    freq_low_ratio = 0.2
    for img in gray_imgs:
        fft_shifted = fftshift(fft2(img))
        power_spectrum = np.abs(fft_shifted)**2
        h, w = power_spectrum.shape
        center_h, center_w = h//2, w//2
        radius = int(min(center_h, center_w) * freq_low_ratio)
        y, x = np.ogrid[:h, :w]
        mask = (x - center_w)**2 + (y - center_h)**2 <= radius**2
        low_freq_energy = np.sum(power_spectrum[mask])
        total_energy = np.sum(power_spectrum)
        freq_feats['low_freq_ratio'].append(low_freq_energy / (total_energy + 1e-6))
    
    # Convert all to numpy arrays
    for feats in [texture_feats, color_feats, edge_feats, freq_feats]:
        for key in feats:
            feats[key] = np.array(feats[key])
    
    # Combine all features into a single matrix
    all_feats_list = []
    for i in range(len(gray_imgs)):
        sample_feats = []
        for feats_dict in [texture_feats, color_feats, edge_feats, freq_feats]:
            sample_feats.extend([feats_dict[key][i] for key in feats_dict])
        all_feats_list.append(sample_feats)
    
    return np.array(all_feats_list)

# ====================== 3. Data Loading & Feature Extraction =======================
# Load images
generic_gray, generic_rgb = load_images_from_dir(generic_dir)
pavement_gray, pavement_rgb = load_images_from_dir(pavement_dir)

# Extract RGB pixels for histogram (merge all samples)
print("\n📊 Extracting RGB pixels for histogram...")
generic_r, generic_g, generic_b = extract_rgb_pixels(generic_rgb)
pavement_r, pavement_g, pavement_b = extract_rgb_pixels(pavement_rgb)
print(f"✅ RGB pixel extraction completed:")
print(f"   - Generic images: {len(generic_r):,} R pixels | {len(generic_g):,} G pixels | {len(generic_b):,} B pixels")
print(f"   - Pavement images: {len(pavement_r):,} R pixels | {len(pavement_g):,} G pixels | {len(pavement_b):,} B pixels")

# Extract comprehensive features for t-SNE
print("\n📊 Extracting comprehensive features for t-SNE...")
generic_tsne_feats = extract_comprehensive_features(generic_gray, generic_rgb)
pavement_tsne_feats = extract_comprehensive_features(pavement_gray, pavement_rgb)

# Combine data for t-SNE
X = np.vstack([generic_tsne_feats, pavement_tsne_feats])
y = np.hstack([np.zeros(len(generic_tsne_feats)), np.ones(len(pavement_tsne_feats))])  # 0=generic, 1=pavement
print(f"✅ Feature extraction completed: {len(X)} samples | {X.shape[1]} features")

# ====================== 4. Visualization (2 Plots: RGB Histogram + t-SNE) =======================
print("\n🎨 Generating visualization plots...")

# ---------------------- Plot 1: RGB Channel Histogram (3 Subplots) ----------------------
fig1, axes = plt.subplots(1, 3, figsize=(15, 5))
fig1.suptitle('RGB Channel Pixel Distribution: Generic vs Pavement Images', fontsize=16, fontweight='bold')
bins = 50  # Number of bins for histogram
alpha = 0.7  # Transparency for overlapping

# Subplot 1: R Channel
ax1 = axes[0]
ax1.hist(generic_r, bins=bins, alpha=alpha, label='Generic', color=gen_color, density=True, edgecolor='black', linewidth=0.3)
ax1.hist(pavement_r, bins=bins, alpha=alpha, label='Pavement', color=pav_color, density=True, edgecolor='black', linewidth=0.3)
ax1.set_xlabel('Red Channel Pixel Value (0-255)', fontsize=11)
ax1.set_ylabel('Density', fontsize=11)
ax1.set_title('Red Channel', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)
ax1.set_xlim(0, 255)

# Subplot 2: G Channel
ax2 = axes[1]
ax2.hist(generic_g, bins=bins, alpha=alpha, label='Generic', color=gen_color, density=True, edgecolor='black', linewidth=0.3)
ax2.hist(pavement_g, bins=bins, alpha=alpha, label='Pavement', color=pav_color, density=True, edgecolor='black', linewidth=0.3)
ax2.set_xlabel('Green Channel Pixel Value (0-255)', fontsize=11)
ax2.set_ylabel('Density', fontsize=11)
ax2.set_title('Green Channel', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)
ax2.set_xlim(0, 255)

# Subplot 3: B Channel
ax3 = axes[2]
ax3.hist(generic_b, bins=bins, alpha=alpha, label='Generic', color=gen_color, density=True, edgecolor='black', linewidth=0.3)
ax3.hist(pavement_b, bins=bins, alpha=alpha, label='Pavement', color=pav_color, density=True, edgecolor='black', linewidth=0.3)
ax3.set_xlabel('Blue Channel Pixel Value (0-255)', fontsize=11)
ax3.set_ylabel('Density', fontsize=11)
ax3.set_title('Blue Channel', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3)
ax3.set_xlim(0, 255)

# Adjust layout
plt.tight_layout()
# Save plot 1
rgb_plot_path = os.path.join(save_root, 'RGB_Channel_Histogram.png')
plt.savefig(rgb_plot_path, dpi=300, bbox_inches='tight')
print(f"✅ RGB histogram plot saved: {rgb_plot_path}")

# ---------------------- Plot 2: t-SNE Dimensionality Reduction Plot ----------------------
fig2, ax = plt.subplots(1, 1, figsize=(10, 8))

# Standardize features (critical for t-SNE)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply t-SNE
perplexity = min(30, len(X)//5)  # Adaptive perplexity
tsne = TSNE(n_components=2, random_state=random_seed, perplexity=perplexity)
X_tsne = tsne.fit_transform(X_scaled)

# Calculate silhouette score (clustering quality)
from sklearn.metrics import silhouette_score
sil_score = silhouette_score(X_scaled, y) if len(X) >= 2 else 0.0

# Plot t-SNE results
scatter1 = ax.scatter(X_tsne[y==0, 0], X_tsne[y==0, 1], 
                      alpha=0.7, label=f'Generic',
                      color=gen_color, s=60, edgecolors='black', linewidth=0.5)
scatter2 = ax.scatter(X_tsne[y==1, 0], X_tsne[y==1, 1], 
                      alpha=0.7, label=f'Pavement',
                      color=pav_color, s=60, edgecolors='black', linewidth=0.5)

# Customize plot
ax.set_title(f't-SNE Feature Clustering (Silhouette Score: {sil_score:.3f})', fontsize=14, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

# Save plot 2
tsne_plot_path = os.path.join(save_root, 'tSNE_Feature_Clustering.png')
plt.tight_layout()
plt.savefig(tsne_plot_path, dpi=300, bbox_inches='tight')
print(f"✅ t-SNE clustering plot saved: {tsne_plot_path}")

# ====================== 5. Summary =======================
print("\n" + "="*60)
print("Analysis Summary")
print("="*60)
print(f"📊 Generated 2 key plots:")
print(f"   1. RGB Channel Histogram: {rgb_plot_path}")
print(f"   2. t-SNE Feature Clustering: {tsne_plot_path}")
print(f"\n🎯 Key RGB Distribution Insights (Typical Patterns):")
print(f"   - Generic Images: Wider pixel value distribution (brighter, more color diversity)")
print(f"   - Pavement Images: Peaked distribution around lower values (darker, less color variation)")
print(f"\n🎯 Clustering Quality:")
print(f"   - Silhouette Score > 0.5: Excellent separation between image types")
print(f"   - Silhouette Score 0.2-0.5: Moderate separation")
print(f"   - Silhouette Score < 0.2: Poor separation")

plt.show()