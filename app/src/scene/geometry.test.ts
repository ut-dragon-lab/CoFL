import { describe, expect, test } from "vitest";
import {
  BoxGeometry,
  Group,
  Mesh,
  MeshBasicMaterial,
  PerspectiveCamera,
  Vector3,
} from "three";
import {
  clippingPlanes,
  imagePoint,
  pointOnFloor,
  suggestFloors,
} from "./geometry";

describe("floor-specific projection and cutaway", () => {
  const floor = {
    id: "upper",
    label: "Upper floor",
    elevation: 3.5,
    min_height: 3.25,
    max_height: 6,
  };
  test("both lower and upper structures are excluded while this floor survives", () => {
    const visible = (y: number) =>
      clippingPlanes(floor, 5.5).every(
        (p) => p.distanceToPoint(new Vector3(0, y, 0)) >= 0,
      );
    expect(visible(0)).toBe(false);
    expect(visible(3.5)).toBe(true);
    expect(visible(5)).toBe(true);
    expect(visible(6)).toBe(false);
  });
  test("image coordinates roundtrip onto the selected floor, including perspective", () => {
    const camera = new PerspectiveCamera(50, 1.6, 0.05, 100);
    camera.position.set(7, 13, 10);
    camera.lookAt(0, 3.5, 0);
    camera.updateMatrixWorld(true);
    const xy = imagePoint([1, 3.5, -1], camera)!;
    const actual = pointOnFloor(xy, camera, 3.5)!;
    actual.forEach((v, i) => expect(v).toBeCloseTo([1, 3.5, -1][i], 6));
    expect(pointOnFloor(xy, camera, 0)![1]).toBeCloseTo(0);
  });
  test("an upward ray and a point behind the camera cannot become a ground path", () => {
    const camera = new PerspectiveCamera(50, 1, 0.05, 100);
    camera.position.set(0, 2, 0);
    camera.lookAt(0, 3, -1);
    camera.updateMatrixWorld(true);
    expect(pointOnFloor([0.5, 0.5], camera, 0)).toBeNull();
    expect(imagePoint([0, 1, 3], camera)).toBeNull();
  });
  test.each(["y", "z"])(
    "detects separated floors in a declared %s-up scene",
    (axis) => {
      const group = new Group();
      for (const y of [0, 3.5]) {
        const mesh = new Mesh(
          new BoxGeometry(10, 0.2, 10),
          new MeshBasicMaterial(),
        );
        mesh.position.y = y - 0.1;
        group.add(mesh);
      }
      const roof = new Mesh(
        new BoxGeometry(10, 0.2, 10),
        new MeshBasicMaterial(),
      );
      roof.position.y = 6.7;
      group.add(roof);
      // A Z-up asset is brought into the viewer's Y-up world at its root.
      const root = new Group();
      if (axis === "z") {
        group.rotation.x = Math.PI / 2;
        root.rotation.x = -Math.PI / 2;
      }
      root.add(group);
      const floors = suggestFloors(root).floors;
      expect(floors).toHaveLength(2);
      expect(floors[0].elevation).toBeCloseTo(0);
      expect(floors[1].elevation).toBeCloseTo(3.52);
    },
  );
});
