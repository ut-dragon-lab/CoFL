"""Matterport3D region rendering for CoFL source assets.

Adapted from the local upstream Matterport3D/visualizer_v2.py.
Upstream SHA-256: ce7826435769ce601a1d66fbff6d5f8d59a803ed837197208321559e703bac66

The caller owns output staging and seeds NumPy before rendering.
The historical camera, RGB, semantic rasterization and zoom rules are preserved.
"""

import glob
import json
import os

import numpy as np
from PIL import Image

from .cameras import apply_camera, resolve_render_cameras

width, height = 640, 480

# ========= 坐标系 / 相机方向设定 =========
world_up = np.array([0.0, 0.0, -1.0])
_tmp = np.array([1.0, 0.0, 0.0])
h1 = np.cross(world_up, _tmp)
h1 /= np.linalg.norm(h1)
h2 = np.cross(world_up, h1)
h2 /= np.linalg.norm(h2)


def find_all_scans(root_dir):
    """扫描ROOT目录下所有可用的scan_id"""
    scan_ids = []
    for item in sorted(os.listdir(root_dir)):
        scan_path = os.path.join(root_dir, item)
        if os.path.isdir(scan_path):
            region_dir = os.path.join(scan_path, "region_segmentations", item, "region_segmentations")
            if os.path.exists(region_dir):
                region_plys = glob.glob(os.path.join(region_dir, "region*.ply"))
                if region_plys:
                    scan_ids.append(item)
    return scan_ids


def compute_zoom_from_extent(extent, strength=3.0, min_zoom=0.3, max_zoom=1.3):
    """根据房间尺寸估算zoom"""
    diag = float(np.linalg.norm(extent)) + 1e-6
    if diag < 3.0:
        return 1.1
    zoom_raw = strength / diag
    return float(np.clip(zoom_raw, min_zoom, max_zoom))


class ScanRenderer:
    """单个scan的渲染器"""

    def __init__(
        self, scan_id, root_dir, output_dir, *, cameras=None,
        cameras_from=None, bundled_cameras=False,
    ):
        self.scan_id = scan_id
        self.root_dir = root_dir
        self.region_dir = os.path.join(
            root_dir, scan_id, "region_segmentations", scan_id, "region_segmentations"
        )
        self.cameras = resolve_render_cameras(
            cameras, cameras_from, bundled_cameras, dataset="matterport", scan_id=scan_id,
            regions=[os.path.splitext(os.path.basename(path))[0]
                     for path in sorted(glob.glob(os.path.join(self.region_dir, "region*.ply")))],
        )
        self.out_dir = os.fspath(output_dir)
        os.makedirs(self.out_dir, exist_ok=True)

        # 全局label映射（整个scan共享）
        self.global_label2id = {}
        self.global_id2label = []

        # annotations结构
        self.annotations = {
            "scan_id": scan_id,
            "label_map": {},
            "regions": {}
        }

        # 颜色映射
        self.id2color = {}
        self._rng = np.random.default_rng(42)

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

    def load_region_semantics(self, region_name: str, vertices: np.ndarray, triangles: np.ndarray):
        """载入region的语义标注（face-level）"""
        semseg_path = os.path.join(self.region_dir, f"{region_name}.semseg.json")
        fsegs_path = os.path.join(self.region_dir, f"{region_name}.fsegs.json")

        num_faces = len(triangles)
        for path in (semseg_path, fsegs_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Required Matterport semantics not found: {path}")
        with open(semseg_path, "r", encoding="utf-8") as handle:
            semseg = json.load(handle)

        seg_groups = semseg.get("segGroups", [])

        segid_to_labelid = {}
        for g in seg_groups:
            label_str = g.get("label", "unknown")
            label_id = self.get_label_id(label_str)
            for seg_id in g.get("segments", []):
                segid_to_labelid[int(seg_id)] = label_id

        # 加载face->segment映射
        with open(fsegs_path, "r") as f:
            fsegs = json.load(f)
        seg_indices = np.array(fsegs["segIndices"], dtype=np.int32)

        if len(seg_indices) != num_faces:
            print(f"  [WARN] segIndices({len(seg_indices)}) != n_faces({num_faces}), truncating")
            seg_indices = seg_indices[:num_faces]

        face_label_ids = np.full(num_faces, -1, dtype=np.int32)
        for fidx, seg_id in enumerate(seg_indices):
            face_label_ids[fidx] = segid_to_labelid.get(int(seg_id), -1)

        print(f"  loaded semantics: {len(self.global_id2label)} total labels")
        return face_label_ids

    def create_flat_semantic_mesh(self, vertices, triangles, face_label_ids):
        """创建flat-shaded语义mesh（避免颜色插值）"""
        import open3d as o3d

        num_faces = len(triangles)

        # 预分配数组
        new_vertices = np.zeros((num_faces * 3, 3), dtype=np.float64)
        new_triangles = np.zeros((num_faces, 3), dtype=np.int32)
        new_colors = np.zeros((num_faces * 3, 3), dtype=np.float64)

        vert_idx = 0
        for fidx in range(num_faces):
            tri = triangles[fidx]
            label_id = face_label_ids[fidx]

            # 复制3个顶点
            new_vertices[vert_idx:vert_idx+3] = vertices[tri]
            new_triangles[fidx] = [vert_idx, vert_idx+1, vert_idx+2]

            # 颜色：整个三角形统一颜色（flat shading）
            if label_id < 0:
                color = [0.0, 0.0, 0.0]
            else:
                encoded = min(label_id + 1, 255)
                color = [encoded / 255.0, 0.0, 0.0]

            new_colors[vert_idx:vert_idx+3] = color
            vert_idx += 3

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
        mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
        mesh.vertex_colors = o3d.utility.Vector3dVector(new_colors)
        return mesh

    def decode_flat_mask(self, rgb_image: np.ndarray) -> np.ndarray:
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

    def rasterize_semantic_mask(self, vertices, triangles, face_label_ids, vis, ctr, img_w, img_h):
        """使用Open3D渲染semantic mask"""
        saved_params = ctr.convert_to_pinhole_camera_parameters()

        sem_mesh = self.create_flat_semantic_mesh(vertices, triangles, face_label_ids)

        vis.clear_geometries()
        vis.add_geometry(sem_mesh)
        self._restore_camera(ctr, saved_params)
        vis.poll_events()
        vis.update_renderer()

        img = vis.capture_screen_float_buffer(do_render=True)
        img_np = np.asarray(img)
        mask = self.decode_flat_mask(img_np)

        return mask

    def save_mask_with_metadata(self, mask, filepath, region_name, view_id):
        """
        保存.npy文件，并包含label映射元数据

        保存格式：
        {
            'mask': np.ndarray,  # [H, W] int32
            'label_map': dict,   # {label_id: category_name}
            'region_name': str,
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
            'region_name': region_name,
            'view_id': view_id,
            'shape': mask.shape,
            'dtype': str(mask.dtype)
        }

        np.save(filepath, data, allow_pickle=True)

    def render_one_region(self, ply_path: str):
        """渲染单个region的所有视角"""
        region_name = os.path.splitext(os.path.basename(ply_path))[0]

        # 为每个region创建独立文件夹
        region_out_dir = os.path.join(self.out_dir, region_name)

        print(f"\n=== {self.scan_id}/{region_name} ===")
        os.makedirs(region_out_dir, exist_ok=True)

        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(ply_path)
        if mesh.is_empty():
            raise ValueError(f"Empty Matterport mesh: {ply_path}")
        mesh.compute_vertex_normals()

        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles, dtype=np.int32)
        face_label_ids = self.load_region_semantics(region_name, vertices, triangles)

        vis = o3d.visualization.Visualizer()
        try:
            created = vis.create_window(
                window_name=f"{self.scan_id}_{region_name}",
                width=width,
                height=height,
                visible=False,
            )
            if not created:
                raise RuntimeError("Open3D failed to create a rendering window; an OpenGL display is required")
            vis.add_geometry(mesh)
            vis.poll_events()
            vis.update_renderer()

            region_cameras = None if self.cameras is None else self.cameras[region_name]
            ctr = vis.get_view_control()
            if region_cameras is None:
                bbox = mesh.get_axis_aligned_bounding_box()
                center = bbox.get_center()
                extent = bbox.get_extent()
                zoom = compute_zoom_from_extent(extent)
                print(f"  extent: {extent}, zoom: {zoom}")

            region_views = []

            # ========= 顶视图 =========
            view_id = "top"
            if region_cameras is None:
                # 从上往下看：相机应该朝向 -world_up (即[0,0,1]，朝向+Z方向，向上看mesh)
                front_top = -world_up    # 修正：原来是world_up导致从下往上看
                up_top = -h1

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
            out_rgb = os.path.join(region_out_dir, img_rgb_name)
            vis.capture_screen_image(out_rgb, do_render=True)

            # 保存mask
            mask_top = self.rasterize_semantic_mask(vertices, triangles, face_label_ids, vis, ctr, width, height)

            vis.clear_geometries()
            vis.add_geometry(mesh)
            self._restore_camera(ctr, pinhole_top)
            vis.poll_events()
            vis.update_renderer()

            mask_npy_name = f"{view_id}_mask.npy"
            mask_png_name = f"{view_id}_mask.png"

            # 使用新的保存函数（包含元数据）
            self.save_mask_with_metadata(mask_top, os.path.join(region_out_dir, mask_npy_name), region_name, view_id)

            mask_vis = self.colorize_mask(mask_top)
            Image.fromarray(mask_vis).save(os.path.join(region_out_dir, mask_png_name))
            print(f"  saved: {view_id} (RGB + mask)")

            region_views.append({
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

                    v_dir = np.cos(pitch) * dir_h + np.sin(pitch) * world_up
                    v_dir /= np.linalg.norm(v_dir)

                    front = -v_dir
                    up_vec = -world_up

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
                out_rgb = os.path.join(region_out_dir, img_rgb_name)
                vis.capture_screen_image(out_rgb, do_render=True)

                # 保存mask
                mask_view = self.rasterize_semantic_mask(vertices, triangles, face_label_ids, vis, ctr, width, height)

                vis.clear_geometries()
                vis.add_geometry(mesh)
                self._restore_camera(ctr, pinhole_view)
                vis.poll_events()
                vis.update_renderer()

                mask_npy_name = f"{view_id}_mask.npy"
                mask_png_name = f"{view_id}_mask.png"

                # 使用新的保存函数（包含元数据）
                self.save_mask_with_metadata(mask_view, os.path.join(region_out_dir, mask_npy_name), region_name, view_id)

                mask_vis = self.colorize_mask(mask_view)
                Image.fromarray(mask_vis).save(os.path.join(region_out_dir, mask_png_name))
                if region_cameras is None:
                    print(f"  saved: {view_id} (pitch={pitch_deg:.1f}°)")
                else:
                    print(f"  Saved: {view_id} (camera replay)")

                region_views.append({
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


            self.annotations["regions"][region_name] = {
                "output_dir": region_name,  # 相对于scan_id目录的路径
                "views": region_views
            }

        finally:
            vis.destroy_window()

    def render_all_regions(self):
        """Render every region and write scene annotations after all succeed."""
        region_plys = sorted(glob.glob(os.path.join(self.region_dir, "region*.ply")))
        if not region_plys:
            raise FileNotFoundError(f"No Matterport region meshes found: {self.region_dir}")
        for ply in region_plys:
            self.render_one_region(ply)
        self.annotations["label_map"] = {
            str(i): label for i, label in enumerate(self.global_id2label)
        }
        with open(os.path.join(self.out_dir, "annotations.json"), "w", encoding="utf-8") as handle:
            json.dump(self.annotations, handle, indent=2)


def render_scan(
    scan_id, root_dir, output_dir, *, cameras=None, cameras_from=None, bundled_cameras=False,
):
    """Write a scan to its scene directory; the caller owns seeding and staging.

    Supply cameras records, cameras_from (a catalog file or annotations root),
    or bundled_cameras=True to replay an optional catalog in a local checkout.

    Source regions live under root_dir/scan_id/region_segmentations/scan_id/
    region_segmentations. Each region needs .ply, .semseg.json and .fsegs.json.
    """
    renderer = ScanRenderer(
        scan_id, root_dir, output_dir, cameras=cameras,
        cameras_from=cameras_from, bundled_cameras=bundled_cameras,
    )
    renderer.render_all_regions()
    return renderer.annotations
