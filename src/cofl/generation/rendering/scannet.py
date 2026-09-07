"""ScanNet scene rendering for CoFL source assets.

Adapted from the local upstream ScanNet/render_scannet_scenes.py.
Upstream SHA-256: 0964b0ea45ed4fce1c5676580731b0ca41b938bc07ebc9d697ab28e150d63272

The caller owns output staging and seeds NumPy before rendering.
The historical camera, RGB, semantic rasterization and zoom rules are preserved.
"""

# Copyright 2017
# Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, Matthias Niessner
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import json
import os

import numpy as np
from PIL import Image

from .cameras import apply_camera, resolve_render_cameras
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, label as connected_components

width, height = 640, 480

# ========= 坐标系 / 相机方向设定 =========
# ScanNet mesh是Z-up的
# Open3D的set_front: front向量指向相机背后（与视线方向相反）
# 所以要从上往下看，front应该是+Z（相机在上方，背对+Z方向，看向-Z）
world_up = np.array([0.0, 0.0, 1.0])  # ScanNet是Z-up
_tmp = np.array([1.0, 0.0, 0.0])
h1 = np.cross(world_up, _tmp)
h1 /= np.linalg.norm(h1)
h2 = np.cross(world_up, h1)
h2 /= np.linalg.norm(h2)


def fill_unlabeled_nearest(mask, unlabeled_val=-1):
    """使用最近邻方法填充未标注区域"""
    unlabeled_mask = (mask == unlabeled_val)

    if not np.any(unlabeled_mask):
        return mask
    if np.all(unlabeled_mask):
        return mask

    dist, indices = distance_transform_edt(unlabeled_mask, return_indices=True)
    filled_mask = mask.copy()
    filled_mask[unlabeled_mask] = mask[indices[0][unlabeled_mask], indices[1][unlabeled_mask]]

    return filled_mask


def fill_small_holes(mask, unlabeled_val=-1, max_hole_size=1000):
    """只填充小的未标注孔洞（被标注区域包围的小区域）"""
    filled_mask = mask.copy()
    unlabeled_mask = (mask == unlabeled_val)

    if not np.any(unlabeled_mask):
        return mask

    labeled_regions, num_regions = connected_components(unlabeled_mask)

    for region_id in range(1, num_regions + 1):
        region_mask = (labeled_regions == region_id)
        region_size = np.sum(region_mask)

        if region_size > max_hole_size:
            continue

        # 膨胀找边界上的label
        dilated = ndimage.binary_dilation(region_mask, iterations=1)
        boundary = dilated & ~region_mask & ~unlabeled_mask

        if not np.any(boundary):
            continue

        boundary_labels = mask[boundary]
        unique_labels, counts = np.unique(boundary_labels, return_counts=True)

        valid_idx = unique_labels != unlabeled_val
        if not np.any(valid_idx):
            continue

        unique_labels = unique_labels[valid_idx]
        counts = counts[valid_idx]

        # majority voting
        majority_label = unique_labels[np.argmax(counts)]
        filled_mask[region_mask] = majority_label

    return filled_mask


def postprocess_mask(mask, max_hole_size=1000, preserve_background=True):
    """
    后处理mask：填充未标注区域

    Args:
        mask: 输入mask，-1表示未标注
        max_hole_size: 小孔洞最大尺寸
        preserve_background: 是否保留与图像边缘连通的背景
    """
    result = mask.copy()

    if preserve_background:
        # 找到与边缘连通的未标注区域（真正的背景）
        h, w = mask.shape
        edge_mask = np.zeros_like(mask, dtype=bool)
        edge_mask[0, :] = True
        edge_mask[-1, :] = True
        edge_mask[:, 0] = True
        edge_mask[:, -1] = True

        unlabeled = (mask == -1)
        labeled_regions, num_regions = connected_components(unlabeled)

        background_mask = np.zeros_like(mask, dtype=bool)
        for region_id in range(1, num_regions + 1):
            region_mask = (labeled_regions == region_id)
            if np.any(region_mask & edge_mask):
                background_mask |= region_mask

    # 1. 填充小孔洞
    result = fill_small_holes(result, max_hole_size=max_hole_size)

    # 2. 最近邻填充剩余
    result = fill_unlabeled_nearest(result)

    if preserve_background:
        # 恢复背景
        result[background_mask] = -1

    return result


def find_all_scans(root_dir):
    """扫描ROOT目录下所有可用的scan_id"""
    scan_ids = []

    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"ScanNet root directory not found: {root_dir}")

    for item in sorted(os.listdir(root_dir)):
        scan_path = os.path.join(root_dir, item)
        if not os.path.isdir(scan_path):
            continue

        # ScanNet场景ID格式: scene0000_00
        if not item.startswith("scene"):
            continue

        # 检查必要文件是否存在
        mesh_file = os.path.join(scan_path, f"{item}_vh_clean_2.ply")
        agg_file = os.path.join(scan_path, f"{item}.aggregation.json")
        seg_file = os.path.join(scan_path, f"{item}_vh_clean_2.0.010000.segs.json")

        if os.path.exists(mesh_file):
            # aggregation和segs文件是可选的（test集可能没有）
            scan_ids.append(item)

    return scan_ids


def compute_zoom_from_extent(extent, strength=3.5, min_zoom=0.25, max_zoom=1.3):
    """根据房间尺寸估算zoom"""
    diag = float(np.linalg.norm(extent)) + 1e-6
    if diag < 3.0:
        return 1.1
    zoom_raw = strength / diag
    return float(np.clip(zoom_raw, min_zoom, max_zoom))


class ScanNetRenderer:
    """单个ScanNet场景的渲染器"""

    def __init__(
        self, scan_id, root_dir, output_dir, *, cameras=None,
        cameras_from=None, bundled_cameras=False,
    ):
        self.scan_id = scan_id
        self.root_dir = root_dir
        self.cameras = resolve_render_cameras(
            cameras, cameras_from, bundled_cameras,
            dataset="scannet", scan_id=scan_id, regions=["region0"],
        )
        self.scan_dir = os.path.join(root_dir, scan_id)
        self.out_dir = os.fspath(output_dir)
        os.makedirs(self.out_dir, exist_ok=True)

        # 文件路径
        self.mesh_file = os.path.join(self.scan_dir, f"{scan_id}_vh_clean_2.ply")
        self.agg_file = os.path.join(self.scan_dir, f"{scan_id}.aggregation.json")
        self.seg_file = os.path.join(self.scan_dir, f"{scan_id}_vh_clean_2.0.010000.segs.json")
        self.info_file = os.path.join(self.scan_dir, f"{scan_id}.txt")

        # 全局label映射（整个scan共享）
        self.global_label2id = {}
        self.global_id2label = []

        # annotations结构
        self.annotations = {
            "scan_id": scan_id,
            "dataset": "ScanNet",
            "label_map": {},
            "views": []
        }

        # 颜色映射
        self.id2color = {}
        self._rng = np.random.default_rng(42)

        # 轴对齐变换矩阵
        self.axis_alignment = None

    def get_label_id(self, label: str) -> int:
        """获取或创建global label ID"""
        if label not in self.global_label2id:
            self.global_label2id[label] = len(self.global_id2label)
            self.global_id2label.append(label)
        return self.global_label2id[label]

    def get_color_for_label_id(self, label_id: int):
        """获取label的可视化颜色"""
        if label_id in self.id2color:
            return self.id2color[label_id]
        c = self._rng.integers(low=64, high=255, size=3, dtype=np.uint8)
        self.id2color[label_id] = (int(c[0]), int(c[1]), int(c[2]))
        return self.id2color[label_id]

    def colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        """将mask转换为彩色可视化"""
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        unique_ids = np.unique(mask)
        for lid in unique_ids:
            if lid < 0:
                continue
            color = self.get_color_for_label_id(int(lid))
            rgb[mask == lid] = color
        return rgb

    def load_axis_alignment(self):
        """Load scene alignment; malformed or missing input is an error."""
        info_dict = {}
        with open(self.info_file, encoding="utf-8") as handle:
            for line in handle:
                if " = " in line:
                    key, value = line.split(" = ", 1)
                    info_dict[key.strip()] = value.strip()
        if "axisAlignment" in info_dict:
            self.axis_alignment = np.fromstring(info_dict["axisAlignment"], sep=" ").reshape(4, 4)
        else:
            self.axis_alignment = np.eye(4)

    def load_semantics(self, num_verts: int):
        """
        载入场景的语义标注（vertex-level）

        ScanNet标注格式:
        - aggregation.json: segGroups包含每个物体的segments和label
        - segs.json: segIndices是每个顶点的segment ID
        """
        for path in (self.agg_file, self.seg_file):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Required ScanNet semantics not found: {path}")

        # 加载aggregation（物体标注）
        with open(self.agg_file, "r", encoding='utf-8') as f:
            aggregation = json.load(f)

        seg_groups = aggregation.get("segGroups", [])

        # segment_id -> label_id 映射
        segid_to_labelid = {}
        for group in seg_groups:
            label_str = group.get("label", "unknown")
            label_id = self.get_label_id(label_str)
            for seg_id in group.get("segments", []):
                segid_to_labelid[int(seg_id)] = label_id

        # 加载segmentation（顶点分割）
        with open(self.seg_file, "r", encoding='utf-8') as f:
            segmentation = json.load(f)

        seg_indices = np.array(segmentation["segIndices"], dtype=np.int32)

        if len(seg_indices) != num_verts:
            print(f"  [WARN] segIndices({len(seg_indices)}) != num_verts({num_verts})")
            # 截断或填充
            if len(seg_indices) > num_verts:
                seg_indices = seg_indices[:num_verts]
            else:
                seg_indices = np.pad(seg_indices, (0, num_verts - len(seg_indices)),
                                     constant_values=-1)

        # 为每个顶点分配label_id
        vertex_label_ids = np.full(num_verts, -1, dtype=np.int32)
        for vidx, seg_id in enumerate(seg_indices):
            vertex_label_ids[vidx] = segid_to_labelid.get(int(seg_id), -1)

        print(f"  Loaded semantics: {len(self.global_id2label)} unique labels")
        return vertex_label_ids

    def create_flat_semantic_mesh(self, mesh, vertex_label_ids):
        """
        创建flat-shaded语义mesh（避免顶点颜色插值）

        方法：将每个三角形的3个顶点复制出来，赋予相同的颜色
        这样渲染时每个三角形内部颜色一致，不会有插值问题
        """
        import open3d as o3d

        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        num_faces = len(triangles)

        # 预分配数组
        new_vertices = np.zeros((num_faces * 3, 3), dtype=np.float64)
        new_triangles = np.zeros((num_faces, 3), dtype=np.int32)
        new_colors = np.zeros((num_faces * 3, 3), dtype=np.float64)

        for fidx in range(num_faces):
            tri = triangles[fidx]
            v0, v1, v2 = tri

            # 使用三角形第一个顶点的label（或者可以用majority voting）
            label_id = vertex_label_ids[v0]

            # 复制3个顶点
            base_idx = fidx * 3
            new_vertices[base_idx:base_idx+3] = vertices[tri]
            new_triangles[fidx] = [base_idx, base_idx+1, base_idx+2]

            # 颜色编码：整个三角形统一颜色（flat shading）
            if label_id < 0:
                color = [0.0, 0.0, 0.0]
            else:
                encoded = min(label_id + 1, 255)
                color = [encoded / 255.0, 0.0, 0.0]

            new_colors[base_idx:base_idx+3] = color

        sem_mesh = o3d.geometry.TriangleMesh()
        sem_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
        sem_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
        sem_mesh.vertex_colors = o3d.utility.Vector3dVector(new_colors)
        return sem_mesh

    def decode_mask(self, rgb_image: np.ndarray) -> np.ndarray:
        """从R通道解码label_id"""
        if rgb_image.dtype == np.float32 or rgb_image.dtype == np.float64:
            rgb_scaled = rgb_image * 255.0
            rgb = np.where(rgb_scaled >= 254, 255, np.round(rgb_scaled)).astype(np.uint8)
        else:
            rgb = rgb_image.astype(np.uint8)

        h, w = rgb.shape[:2]

        # 识别背景：白色(255,255,255)表示空白区域
        is_background = np.all(rgb == 255, axis=2)

        r_channel = rgb[:, :, 0].astype(np.int32)

        # 解码：R=0或白色背景 → -1，R=N → label(N-1)
        mask = np.where(r_channel == 0, -1, r_channel - 1)
        mask = np.where(is_background, -1, mask)

        return mask.astype(np.int32)

    def _restore_camera(self, ctr, parameters):
        """Restore RGB/semantic camera state, verifying every replay restore."""
        if self.cameras is None:
            ctr.convert_from_pinhole_camera_parameters(parameters)
        else:
            apply_camera(ctr, {
                "intrinsic": parameters.intrinsic.intrinsic_matrix.tolist(),
                "extrinsic": parameters.extrinsic.tolist(),
                "width": width,
                "height": height,
            })

    def rasterize_semantic_mask(self, mesh, vertex_label_ids, vis, ctr, img_w, img_h):
        """使用Open3D渲染semantic mask（flat shading避免插值）+ 后处理填充未标注区域"""
        saved_params = ctr.convert_to_pinhole_camera_parameters()

        # 使用flat shading的mesh
        sem_mesh = self.create_flat_semantic_mesh(mesh, vertex_label_ids)

        vis.clear_geometries()
        vis.add_geometry(sem_mesh)
        self._restore_camera(ctr, saved_params)
        vis.poll_events()
        vis.update_renderer()

        img = vis.capture_screen_float_buffer(do_render=True)
        img_np = np.asarray(img)
        mask = self.decode_mask(img_np)

        # 后处理：填充未标注区域
        mask = postprocess_mask(mask, max_hole_size=1000, preserve_background=True)

        return mask

    def save_mask_with_metadata(self, mask, filepath, view_id):
        """
        保存.npy文件，并包含label映射元数据

        保存格式：
        {
            'mask': np.ndarray,  # [H, W] int32
            'label_map': dict,   # {label_id: category_name}
            'scan_id': str,
            'view_id': str
        }
        """
        # 提取当前mask中实际出现的labels
        unique_labels = np.unique(mask[mask >= 0])  # 排除背景-1

        # 构建label映射
        label_map = {
            int(lid): self.global_id2label[lid]
            for lid in unique_labels
            if lid < len(self.global_id2label)
        }

        # 保存为包含元数据的dict
        data = {
            'mask': mask,
            'label_map': label_map,
            'scan_id': self.scan_id,
            'view_id': view_id,
            'shape': mask.shape,
            'dtype': str(mask.dtype)
        }

        np.save(filepath, data, allow_pickle=True)

    def render_scene(self):
        """渲染单个场景的所有视角"""
        # 检查mesh文件
        if not os.path.exists(self.mesh_file):
            raise FileNotFoundError(f"Required ScanNet mesh not found: {self.mesh_file}")

        # 加载轴对齐矩阵
        self.load_axis_alignment()

        # 加载mesh
        print(f"  Loading mesh: {self.mesh_file}")
        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(self.mesh_file)
        if mesh.is_empty():
            raise ValueError(f"Empty ScanNet mesh: {self.mesh_file}")

        # 应用轴对齐变换
        if self.axis_alignment is not None:
            mesh.transform(self.axis_alignment)

        mesh.compute_vertex_normals()

        vertices = np.asarray(mesh.vertices)
        num_verts = len(vertices)
        print(f"  Mesh: {num_verts} vertices")

        # 加载语义标注
        vertex_label_ids = self.load_semantics(num_verts)

        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        try:
            created = vis.create_window(
                window_name=f"{self.scan_id}",
                width=width,
                height=height,
                visible=False,
            )
            if not created:
                raise RuntimeError("Open3D failed to create a rendering window; an OpenGL display is required")
            vis.add_geometry(mesh)
            vis.poll_events()
            vis.update_renderer()

            region_cameras = None if self.cameras is None else self.cameras["region0"]
            ctr = vis.get_view_control()
            if region_cameras is None:
                bbox = mesh.get_axis_aligned_bounding_box()
                center = bbox.get_center()
                extent = bbox.get_extent()
                zoom = compute_zoom_from_extent(extent)
                print(f"  Extent: {extent}, Zoom: {zoom:.2f}")

            views_data = []

            # ========= 顶视图 =========
            view_id = "top"
            if region_cameras is None:
                # Z-up: 从上往下看
                # Open3D set_front: front是相机背后的方向（与视线相反）
                # 要从上往下看（视线朝-Z），front应该是+Z
                front_top = np.array([0.0, 0.0, 1.0])   # 相机背后朝+Z，所以视线朝-Z（向下看）
                up_top = np.array([0.0, 1.0, 0.0])      # 图像上方向对应+Y

                ctr.set_lookat(center)
                ctr.set_front(front_top)
                ctr.set_up(up_top)
                ctr.set_zoom(zoom)
            else:
                apply_camera(ctr, region_cameras[view_id])
            vis.poll_events()
            vis.update_renderer()

            pinhole_top = ctr.convert_to_pinhole_camera_parameters()

            # 保存RGB
            img_rgb_name = f"{view_id}.png"
            out_rgb = os.path.join(self.out_dir, img_rgb_name)
            vis.capture_screen_image(out_rgb, do_render=True)

            # 保存mask
            mask_top = self.rasterize_semantic_mask(mesh, vertex_label_ids, vis, ctr, width, height)

            # 恢复原始mesh
            vis.clear_geometries()
            vis.add_geometry(mesh)
            self._restore_camera(ctr, pinhole_top)
            vis.poll_events()
            vis.update_renderer()

            mask_npy_name = f"{view_id}_mask.npy"
            mask_png_name = f"{view_id}_mask.png"

            self.save_mask_with_metadata(mask_top, os.path.join(self.out_dir, mask_npy_name), view_id)

            mask_vis = self.colorize_mask(mask_top)
            Image.fromarray(mask_vis).save(os.path.join(self.out_dir, mask_png_name))
            print(f"  Saved: {view_id} (RGB + mask)")

            views_data.append({
                "view_id": view_id,
                "image": img_rgb_name,
                "mask_npy": mask_npy_name,
                "mask_png": mask_png_name,
                "camera": {
                    "intrinsic": pinhole_top.intrinsic.intrinsic_matrix.tolist(),
                    "extrinsic": pinhole_top.extrinsic.tolist(),
                    "width": width,
                    "height": height,
                }
            })

            # ========= 8个斜俯视 =========
            if region_cameras is None:
                def norm(v):
                    return v / np.linalg.norm(v)

                # 水平方向
                dir_edge = [h1, -h1, h2, -h2]
                dir_corner = [
                    norm(h1 + h2), norm(h1 - h2),
                    norm(-h1 + h2), norm(-h1 - h2),
                ]
                dir_list = dir_edge + dir_corner
            else:
                dir_list = [None] * 8

            for k, dir_h in enumerate(dir_list):
                view_id = f"view{k}"
                if region_cameras is None:
                    pitch_deg = np.random.uniform(45.0, 80.0)
                    pitch = np.deg2rad(pitch_deg)

                    # 从水平方向+俯仰角计算视线方向
                    # 俯视：视线从水平方向向下倾斜
                    # front是相机背后方向（与视线相反），所以向上倾斜
                    # 视线 = cos(pitch)*dir_h - sin(pitch)*world_up  (向下看)
                    # front = -视线 = -cos(pitch)*dir_h + sin(pitch)*world_up (背后方向)
                    front = -np.cos(pitch) * dir_h + np.sin(pitch) * world_up
                    front /= np.linalg.norm(front)

                    up_vec = world_up

                    ctr.set_lookat(center)
                    ctr.set_front(front)
                    ctr.set_up(up_vec)
                    ctr.set_zoom(zoom)
                else:
                    apply_camera(ctr, region_cameras[view_id])
                vis.poll_events()
                vis.update_renderer()

                pinhole_view = ctr.convert_to_pinhole_camera_parameters()

                # 保存RGB
                img_rgb_name = f"{view_id}.png"
                out_rgb = os.path.join(self.out_dir, img_rgb_name)
                vis.capture_screen_image(out_rgb, do_render=True)

                # 保存mask
                mask_view = self.rasterize_semantic_mask(mesh, vertex_label_ids, vis, ctr, width, height)

                # 恢复原始mesh
                vis.clear_geometries()
                vis.add_geometry(mesh)
                self._restore_camera(ctr, pinhole_view)
                vis.poll_events()
                vis.update_renderer()

                mask_npy_name = f"{view_id}_mask.npy"
                mask_png_name = f"{view_id}_mask.png"

                self.save_mask_with_metadata(mask_view, os.path.join(self.out_dir, mask_npy_name), view_id)

                mask_vis = self.colorize_mask(mask_view)
                Image.fromarray(mask_vis).save(os.path.join(self.out_dir, mask_png_name))
                if region_cameras is None:
                    print(f"  Saved: {view_id} (pitch={pitch_deg:.1f}°)")
                else:
                    print(f"  Saved: {view_id} (camera replay)")

                views_data.append({
                    "view_id": view_id,
                    "image": img_rgb_name,
                    "mask_npy": mask_npy_name,
                    "mask_png": mask_png_name,
                    "camera": {
                        "intrinsic": pinhole_view.intrinsic.intrinsic_matrix.tolist(),
                        "extrinsic": pinhole_view.extrinsic.tolist(),
                        "width": width,
                        "height": height,
                    }
                })


            # 保存annotations
            self.annotations["views"] = views_data
            self.annotations["label_map"] = {
                str(i): label for i, label in enumerate(self.global_id2label)
            }
            self.annotations["axis_alignment"] = self.axis_alignment.tolist() if self.axis_alignment is not None else None

            json_path = os.path.join(self.out_dir, "annotations.json")
            with open(json_path, "w") as f:
                json.dump(self.annotations, f, indent=2)

        finally:
            vis.destroy_window()


def render_scan(
    scan_id, root_dir, output_dir, *, cameras=None, cameras_from=None, bundled_cameras=False,
):
    """Write a scan to its scene directory; the caller owns seeding and staging.

    Supply cameras records, cameras_from (a catalog file or annotations root),
    or bundled_cameras=True to replay an optional catalog in a local checkout.

    root_dir/scan_id must provide the cleaned mesh, aggregation and segmentation
    JSON files, and scene information text file used by ScanNetRenderer.
    """
    renderer = ScanNetRenderer(
        scan_id, root_dir, output_dir, cameras=cameras,
        cameras_from=cameras_from, bundled_cameras=bundled_cameras,
    )
    renderer.render_scene()
    return renderer.annotations
