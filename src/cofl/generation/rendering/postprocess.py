"""Optional historical mask cleanup; never applied automatically by renderers.

Adapted from the local upstream ScanNet/postprocess_masks.py.
Upstream SHA-256: e39265825c11ad13876970a9d72715ef573576cb01762426988930b3599ed627

This optional cleanup applies per-label morphology before hole filling, unlike
the inline ScanNet renderer cleanup. Its historical default hole limit is 500
pixels; the inline renderer uses 1000. The caller owns any file writes.
"""

# Copyright 2017
# Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, Matthias Niessner
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, label as connected_components


def fill_unlabeled_nearest(mask, unlabeled_val=-1):
    """
    使用最近邻方法填充未标注区域
    将每个未标注像素赋值为最近的已标注像素的label
    """
    unlabeled_mask = (mask == unlabeled_val)

    if not np.any(unlabeled_mask):
        return mask  # 没有未标注区域

    if np.all(unlabeled_mask):
        return mask  # 全是未标注，无法填充

    # 计算到最近已标注像素的距离和索引
    labeled_mask = ~unlabeled_mask

    # distance_transform_edt返回每个点到最近0的距离
    # 我们需要找到每个未标注点最近的已标注点
    dist, indices = distance_transform_edt(unlabeled_mask, return_indices=True)

    # 使用最近邻的索引来填充
    filled_mask = mask.copy()
    filled_mask[unlabeled_mask] = mask[indices[0][unlabeled_mask], indices[1][unlabeled_mask]]

    return filled_mask


def fill_small_holes(mask, unlabeled_val=-1, max_hole_size=500):
    """
    只填充小的未标注孔洞（被同一label包围的区域）
    保留大的背景区域
    """
    filled_mask = mask.copy()
    unlabeled_mask = (mask == unlabeled_val)

    if not np.any(unlabeled_mask):
        return mask

    # 找到所有连通的未标注区域
    labeled_regions, num_regions = connected_components(unlabeled_mask)

    for region_id in range(1, num_regions + 1):
        region_mask = (labeled_regions == region_id)
        region_size = np.sum(region_mask)

        # 只处理小孔洞
        if region_size > max_hole_size:
            continue

        # 找到这个孔洞周围的label
        # 膨胀这个区域1像素，找到边界上的label
        dilated = ndimage.binary_dilation(region_mask, iterations=1)
        boundary = dilated & ~region_mask & ~unlabeled_mask

        if not np.any(boundary):
            continue

        # 获取边界上的label
        boundary_labels = mask[boundary]
        unique_labels, counts = np.unique(boundary_labels, return_counts=True)

        # 排除未标注
        valid_idx = unique_labels != unlabeled_val
        if not np.any(valid_idx):
            continue

        unique_labels = unique_labels[valid_idx]
        counts = counts[valid_idx]

        # 使用出现最多的label填充
        majority_label = unique_labels[np.argmax(counts)]
        filled_mask[region_mask] = majority_label

    return filled_mask


def morphological_close(mask, unlabeled_val=-1, kernel_size=3):
    """
    对每个label类别单独进行形态学闭运算
    填充label内部的小孔洞
    """
    filled_mask = mask.copy()
    unique_labels = np.unique(mask)

    struct = ndimage.generate_binary_structure(2, 1)  # 4-连通

    for label_id in unique_labels:
        if label_id == unlabeled_val:
            continue

        # 创建该label的二值mask
        binary_mask = (mask == label_id)

        # 闭运算：先膨胀后腐蚀，填充小孔洞
        closed = ndimage.binary_closing(binary_mask, structure=struct, iterations=kernel_size)

        # 只填充新增的区域（原来是unlabeled的）
        new_pixels = closed & ~binary_mask & (mask == unlabeled_val)
        filled_mask[new_pixels] = label_id

    return filled_mask


def process_mask(mask, method='all', max_hole_size=500, preserve_background=True):
    """
    处理单个mask

    Args:
        mask: 输入mask
        method: 'holes'=只填小孔, 'nearest'=最近邻填充, 'all'=组合方法
        max_hole_size: 最大孔洞大小（像素）
        preserve_background: 是否保留大的背景区域（图像边缘的未标注区域）
    """
    if method not in {"holes", "nearest", "all"}:
        raise ValueError(f"Unknown mask cleanup method: {method!r}")
    result = mask.copy()

    if preserve_background:
        # 识别图像边缘的大背景区域
        h, w = mask.shape
        edge_mask = np.zeros_like(mask, dtype=bool)
        edge_mask[0, :] = True
        edge_mask[-1, :] = True
        edge_mask[:, 0] = True
        edge_mask[:, -1] = True

        # 找到与边缘连通的未标注区域
        unlabeled = (mask == -1)
        labeled_regions, num_regions = connected_components(unlabeled)

        background_regions = set()
        for region_id in range(1, num_regions + 1):
            region_mask = (labeled_regions == region_id)
            if np.any(region_mask & edge_mask):
                # 这个区域与边缘相连，是背景
                background_regions.add(region_id)

        # 创建背景mask
        background_mask = np.zeros_like(mask, dtype=bool)
        for region_id in background_regions:
            background_mask |= (labeled_regions == region_id)

    if method in ['holes', 'all']:
        # 1. 形态学闭运算
        result = morphological_close(result, kernel_size=2)

        # 2. 填充小孔洞
        result = fill_small_holes(result, max_hole_size=max_hole_size)

    if method in ['nearest', 'all']:
        # 3. 最近邻填充剩余未标注区域
        result = fill_unlabeled_nearest(result)

    if preserve_background:
        # 恢复背景区域
        result[background_mask] = -1

    return result

