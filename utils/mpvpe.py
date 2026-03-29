import os
import glob
import numpy as np
from tqdm import tqdm

def load_vertices_from_obj(file_path):
    """读取 obj 文件中的顶点坐标（忽略法线、面等信息）"""
    vertices = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):  # 顶点行以 v 开头
                parts = line.strip().split()
                vertex = list(map(float, parts[1:4]))  # x, y, z
                vertices.append(vertex)
    return np.array(vertices)  # [N, 3]

def compute_mpvpe(folder_gt, folder_pred):
    gt_files = sorted(glob.glob(os.path.join(folder_gt, "*.obj")))
    pred_files = sorted(glob.glob(os.path.join(folder_pred, "*.obj")))

    assert len(gt_files) == len(pred_files), "Ground truth and prediction folder must contain the same number of files."

    mpvpe_list = []

    for gt_path, pred_path in tqdm(zip(gt_files, pred_files), total=len(gt_files), desc="Computing MPVPE"):
        gt_vertices = load_vertices_from_obj(gt_path)
        pred_vertices = load_vertices_from_obj(pred_path)

        assert gt_vertices.shape == pred_vertices.shape, f"Shape mismatch in {gt_path} and {pred_path}"

        error = np.linalg.norm(gt_vertices - pred_vertices, axis=1)  # [N]
        mpvpe = np.mean(error)
        mpvpe_list.append(mpvpe)

    mean_mpvpe = np.mean(mpvpe_list)
    return mean_mpvpe


if __name__ == "__main__":
    folder_gt = "/home/huangx/TriplaneGaussian_online/mesh_gt"       # 替换为你的 ground truth mesh 文件夹路径
    folder_pred = "/home/huangx/TriplaneGaussian_online/mesh_handy"   # 替换为你的 prediction mesh 文件夹路径

    mpvpe = compute_mpvpe(folder_gt, folder_pred)
    print(f"MPVPE: {mpvpe:.4f} units")
