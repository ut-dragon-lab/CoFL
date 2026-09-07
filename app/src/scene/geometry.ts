import {
  Box3,
  Camera,
  Mesh,
  Object3D,
  Plane,
  Raycaster,
  Vector2,
  Vector3,
} from "three";
import type { Floor, Point2 } from "../api/client";

export type Point3 = [number, number, number];
export type SceneSource = {
  id: string;
  label: string;
  url?: string;
  up_axis?: "y" | "z";
  floors?: Floor[];
};

/** glTF is Y-up; explicitly declared Z-up assets are rotated at the scene root. */
export function clippingPlanes(floor: Floor, cutHeight: number) {
  return [
    new Plane(new Vector3(0, 1, 0), -floor.min_height),
    new Plane(new Vector3(0, -1, 0), cutHeight),
  ];
}

export function imagePoint(world: Point3, camera: Camera): Point2 | null {
  const cameraSpace = new Vector3(...world).applyMatrix4(
    camera.matrixWorldInverse,
  );
  if (cameraSpace.z >= 0) return null;
  const p = new Vector3(...world).project(camera);
  if (p.z < -1 || p.z > 1) return null;
  const xy: Point2 = [(p.x + 1) / 2, (1 - p.y) / 2];
  return xy.every((v) => v >= 0 && v <= 1) ? xy : null;
}

export function pointOnFloor(
  point: Point2,
  camera: Camera,
  elevation: number,
): Point3 | null {
  const ray = new Raycaster();
  ray.setFromCamera(new Vector2(point[0] * 2 - 1, 1 - point[1] * 2), camera);
  if (ray.ray.direction.y >= -1e-5) return null;
  const target = ray.ray.intersectPlane(
    new Plane(new Vector3(0, 1, 0), -elevation),
    new Vector3(),
  );
  return target ? target.toArray() : null;
}

/** Suggestions, not semantic segmentation: users can override height and cutaway. */
export function suggestFloors(scene: Object3D): {
  floors: Floor[];
  bounds: Box3;
} {
  scene.updateWorldMatrix(true, true);
  const bounds = new Box3().setFromObject(scene);
  const bins = new Map<number, number>();
  const a = new Vector3(),
    b = new Vector3(),
    c = new Vector3(),
    ab = new Vector3(),
    ac = new Vector3();
  scene.traverse((object) => {
    if (!(object instanceof Mesh)) return;
    const geometry = object.geometry,
      vertices = geometry.getAttribute("position");
    if (!vertices) return;
    const count = geometry.index?.count ?? vertices.count;
    // Bound processing on scan meshes while preserving area ranking.
    const step = Math.max(1, Math.ceil(count / 300_000)) * 3;
    for (let i = 0; i + 2 < count; i += step) {
      const index = (j: number) =>
        geometry.index ? geometry.index.getX(j) : j;
      a.fromBufferAttribute(vertices, index(i)).applyMatrix4(
        object.matrixWorld,
      );
      b.fromBufferAttribute(vertices, index(i + 1)).applyMatrix4(
        object.matrixWorld,
      );
      c.fromBufferAttribute(vertices, index(i + 2)).applyMatrix4(
        object.matrixWorld,
      );
      const normal = ab.subVectors(b, a).cross(ac.subVectors(c, a));
      const area = normal.length() / 2;
      if (area < 1e-8 || normal.y / (area * 2) < 0.96) continue;
      const y = Math.round((a.y + b.y + c.y) / 3 / 0.08) * 0.08;
      bins.set(y, (bins.get(y) ?? 0) + area);
    }
  });
  const ranked = [...bins.entries()].sort((a, b) => b[1] - a[1]);
  const elevations: number[] = [];
  const threshold = (ranked[0]?.[1] ?? 0) * 0.16;
  for (const [height, area] of ranked) {
    if (area < threshold || elevations.length >= 8) break;
    if (height > bounds.max.y - 0.3) continue;
    if (elevations.every((y) => Math.abs(y - height) > 1.5))
      elevations.push(height);
  }
  if (!elevations.length) elevations.push(bounds.min.y);
  elevations.sort((a, b) => a - b);
  return {
    bounds,
    floors: elevations.map((y, index) => ({
      id: `level-${index}`,
      label: `Level ${index + 1}`,
      elevation: y,
      min_height: y - 0.15,
      max_height: Math.max(
        y + 0.5,
        Math.min(elevations[index + 1] ?? bounds.max.y, y + 2.5) - 0.12,
      ),
    })),
  };
}
