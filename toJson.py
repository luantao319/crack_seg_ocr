# gen_tasks.py
import os, json, cv2, numpy as np

IMAGE_DIR  = "/home/peipengfei/data/data/label-studio-cn/label-studio-chinese/data/defect-detection/crack500/picture"
MASK_DIR   = "/home/peipengfei/data/data/label-studio-cn/label-studio-chinese/data/defect-detection/crack500/mask"
URL_PREFIX = "/data/local-files/?d=defect-detection/crack500/picture"

def mask2rle(mask):
    """二值 mask → RLE（Label Studio 兼容）"""
    pixels = (mask > 127).astype(np.uint8).flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs   = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(map(str, runs))

tasks = []
for img in sorted(os.listdir(IMAGE_DIR)):
    base, _ = os.path.splitext(img)
    mask_path = os.path.join(MASK_DIR, base + '.png')
    if not os.path.exists(mask_path):
        continue

    mask = (cv2.imread(mask_path, 0) > 0) * 255   # 强制 0/255 二值
    rle  = mask2rle(mask)

    task = {
        "data": {"image": f"{URL_PREFIX}/{img}"},
        "annotations": [{
            "result": [{
                "id": "mask1",
                "type": "brushlabels",
                "from_name": "tag",
                "to_name": "image",
                "original_width": mask.shape[1],
                "original_height": mask.shape[0],
                "image_rotation": 0,
                "value": {
                    "format": "rle",
                    "rle": rle,
                    "brushlabels": ["crack"]
                }
            }],
            "score": 1.0,
            "completed_by": {
                    "first_name": "",
                    "last_name": "",
                    "avatar": None,
                    "email": "294881866@qq.com",
                    "initials": "29"
                },
            "ground_truth": False
        }]
    }
    tasks.append(task)

with open("tasks.json", "w") as f:
    json.dump(tasks, f, indent=2)

print("tasks.json 生成完成，共", len(tasks), "条任务")