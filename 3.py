# ========== 1.  Basic setup ==========
import os, cv2, numpy as np, matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# --- paths (modify if needed) ---
generic_dir   = r'C:\Users\H3C\Desktop\coco\coco500'      # Generic RGB images
pavement_dir  = r'C:\Users\H3C\Desktop\coco\crack500'     # pavement RGB images
save_path     = r'C:\Users\H3C\Desktop\RGB_vs_Pavement_tSNE.png'

SAMPLE        = 100         # images per class
IMG_TYPES     = ('.jpg','.jpeg','.png')

# --- small helpers ---
def list_imgs(dir_):
    if not os.path.exists(dir_):
        print(f"Warning: Directory {dir_} does not exist")
        return []
    return [os.path.join(dir_, f) for f in os.listdir(dir_)
            if f.lower().endswith(IMG_TYPES)][:SAMPLE]

def read_rgb(path):
    im = cv2.imread(path)
    if im is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # 保持为uint8格式(0-255)

# ========== 2.  Build a simple 512-D colour feature ==========
# 8×8×8 = 512-bin RGB histogram (coarse but fast)
def colour_vector(rgb):
    return cv2.calcHist([rgb], [0,1,2], None, [8,8,8], [0,256,0,256,0,256]).flatten()

# ========== 3.  Collect features & labels ==========
generic_paths = list_imgs(generic_dir)
pavement_paths = list_imgs(pavement_dir)

# Check if we have data
if len(generic_paths) == 0 and len(pavement_paths) == 0:
    print("Error: No images found in either directory")
    exit()

paths = generic_paths + pavement_paths
labels = ['Generic']*len(generic_paths) + ['Pavement']*len(pavement_paths)

print('Extracting colour histograms …')
try:
    X = np.vstack([colour_vector(read_rgb(p)) for p in paths])
except Exception as e:
    print(f"Error extracting features: {e}")
    exit()

# ========== 4.  t-SNE (2-D) ==========
X = StandardScaler().fit_transform(X)
print('Running t-SNE …')
emb = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X)

# ========== 5.  Plot ==========
plt.figure(figsize=(5,4))
for cls, col in zip(['Generic','Pavement'], ['#2E86AB','#E63946']):
    mask = np.array(labels)==cls
    plt.scatter(emb[mask,0], emb[mask,1], s=35, label=cls, c=col, alpha=.7)

plt.legend()
plt.title('t-SNE on 512-D RGB histogram')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.tight_layout()

# 确保保存路径的目录存在
save_dir = os.path.dirname(save_path)
if save_dir and not os.path.exists(save_dir):
    os.makedirs(save_dir)

plt.savefig(save_path, dpi=300)
plt.show()
print('Figure saved →', save_path)