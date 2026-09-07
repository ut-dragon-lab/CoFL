import {
  Component,
  forwardRef,
  Suspense,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
} from "react";
import type { ReactNode, Ref } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { Html, Line, OrbitControls, useGLTF } from "@react-three/drei";
import { Box3, Camera, Group, PerspectiveCamera, Vector3 } from "three";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import type { Floor, Point2, Prediction } from "../api/client";
import {
  clippingPlanes,
  imagePoint,
  pointOnFloor,
  suggestFloors,
} from "./geometry";
import type { Point3, SceneSource } from "./geometry";
import { SampleBuilding } from "./SampleBuilding";
import { registerHabitatBasis } from "./basis";
import { KTX2Loader } from "three-stdlib";

export interface Capture {
  image: string;
  observation_id: string;
  start: Point2;
}
export interface SceneHandle {
  capture(): Capture;
  topView(): void;
  fit(): void;
}
interface Props {
  source: SceneSource;
  floor: Floor | null;
  cutHeight: number;
  start: Point3 | null;
  prediction: Prediction | null;
  showField: boolean;
  onReady(floors: Floor[], bounds: [Point3, Point3]): void;
  onStart(point: Point3): void;
  onViewChange(): void;
}

class SceneError extends Component<
  { children: ReactNode },
  { error: string | null }
> {
  state = { error: null as string | null };
  static getDerivedStateFromError(error: Error) {
    return { error: error.message };
  }
  render() {
    return this.state.error ? (
      <div className="viewport-error" role="alert">
        <strong>Scene could not be opened</strong>
        <p>{this.state.error}</p>
        <p>Choose a self-contained GLB with embedded textures.</p>
      </div>
    ) : (
      this.props.children
    );
  }
}

function LoadedScene({ url }: { url: string }) {
  const { gl } = useThree();
  const ktx2 = useMemo(
    () =>
      new KTX2Loader().setTranscoderPath("/decoders/basis/").detectSupport(gl),
    [gl],
  );
  useEffect(
    () => () => {
      ktx2.dispose();
    },
    [ktx2],
  );
  const gltf = useGLTF(url, "/decoders/draco/gltf/", true, (loader) => {
    loader.setKTX2Loader(ktx2);
    registerHabitatBasis(loader);
  });
  const scene = useMemo(() => gltf.scene.clone(true), [gltf.scene]);
  return <primitive object={scene} dispose={null} />;
}

const SAMPLE_FLOORS: Floor[] = [0, 3.5].map((y, i) => ({
  id: `sample-${i}`,
  label: i ? "Upper floor" : "Ground floor",
  elevation: y,
  min_height: y - 0.25,
  max_height: y + 2.5,
}));

function World({
  source,
  floor,
  cutHeight,
  start,
  prediction,
  showField,
  onReady,
  onStart,
  onViewChange,
  handleRef,
}: Props & { handleRef: Ref<SceneHandle> }) {
  const { camera, gl, scene, invalidate } = useThree();
  const controls = useRef<OrbitControlsImpl>(null);
  const root = useRef<Group>(null);
  const overlays = useRef<Group>(null);
  const bounds = useRef(new Box3());
  const snapshot = useRef<{ camera: Camera; id: string } | null>(null);
  const pointerDown = useRef<[number, number]>([0, 0]);
  const floorY = floor?.elevation ?? 0;

  const fit = (top = false) => {
    if (bounds.current.isEmpty()) return;
    const center = bounds.current.getCenter(new Vector3());
    const size = bounds.current.getSize(new Vector3());
    center.y = floorY;
    const halfFov =
      camera instanceof PerspectiveCamera
        ? (camera.fov * Math.PI) / 360
        : Math.PI / 6;
    const aspect = camera instanceof PerspectiveCamera ? camera.aspect : 1;
    // Include visible wall height because the target lies on the selected floor.
    const distance =
      (Math.max(size.z, size.x / aspect, 3) / (2 * Math.tan(halfFov))) * 1.15 +
      Math.max(0, bounds.current.max.y - floorY);
    camera.position
      .copy(center)
      .add(
        new Vector3(
          top ? 0 : distance * 0.55,
          distance,
          top ? 0.001 : distance * 0.72,
        ),
      );
    camera.lookAt(center);
    controls.current?.target.copy(center);
    controls.current?.update();
    invalidate();
  };

  useEffect(() => {
    if (!root.current) return;
    const result = suggestFloors(root.current);
    bounds.current.copy(result.bounds);
    const floors = source.floors?.length
      ? source.floors
      : source.url
        ? result.floors
        : SAMPLE_FLOORS;
    onReady(floors, [result.bounds.min.toArray(), result.bounds.max.toArray()]);
    fit();
    // Source loading is the lifecycle boundary. User controls must not refit the camera.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source.id, source.url]);

  useEffect(() => {
    const planes = floor ? clippingPlanes(floor, cutHeight) : [];
    gl.clippingPlanes = planes;
    gl.localClippingEnabled = true;
    invalidate();
    return () => {
      gl.clippingPlanes = [];
    };
  }, [floor, cutHeight, gl, invalidate]);

  useEffect(() => {
    if (!controls.current) return;
    const delta = floorY - controls.current.target.y;
    camera.position.y += delta;
    controls.current.target.y = floorY;
    controls.current.update();
    invalidate();
  }, [floorY, camera, invalidate]);

  useImperativeHandle(handleRef, () => ({
    fit: () => fit(),
    topView: () => fit(true),
    capture() {
      if (!start)
        throw new Error(
          "Click an open area of the selected floor to choose a start.",
        );
      camera.updateMatrixWorld(true);
      const xy = imagePoint(start, camera);
      if (!xy)
        throw new Error(
          "The selected start is outside this view. Move the camera or choose another start.",
        );
      const original = overlays.current?.visible;
      try {
        if (overlays.current) overlays.current.visible = false;
        gl.render(scene, camera);
        const output = document.createElement("canvas");
        const scale = Math.min(
          1,
          1024 / Math.max(gl.domElement.width, gl.domElement.height),
        );
        output.width = Math.max(1, Math.round(gl.domElement.width * scale));
        output.height = Math.max(1, Math.round(gl.domElement.height * scale));
        output
          .getContext("2d")!
          .drawImage(gl.domElement, 0, 0, output.width, output.height);
        const id = crypto.randomUUID();
        snapshot.current = { camera: camera.clone(), id };
        return {
          image: output.toDataURL("image/png"),
          observation_id: id,
          start: xy,
        };
      } finally {
        if (overlays.current) overlays.current.visible = original ?? true;
        gl.render(scene, camera);
        invalidate();
      }
    },
  }));

  const result = useMemo(() => {
    const snap = snapshot.current;
    if (!prediction || !snap || snap.id !== prediction.observation_id)
      return { path: [] as Point3[], arrows: [] as Point3[][] };
    const lift = (p: number[]) =>
      pointOnFloor(p as Point2, snap.camera, floorY + 0.035);
    // Stop at unprojectable points; do not connect across a gap or horizon.
    const path: Point3[] = [];
    for (const p of prediction.trajectory) {
      const world = lift(p);
      if (
        !world ||
        !bounds.current
          .clone()
          .expandByScalar(0.5)
          .containsPoint(new Vector3(...world))
      )
        break;
      path.push(world);
    }
    const arrows: Point3[][] = [];
    if (showField)
      prediction.queries.forEach((q, index) => {
        const v = prediction.vectors[index],
          n = Math.hypot(...v);
        if (n < 1e-8) return;
        const a = lift(q),
          b = lift([q[0] + (v[0] / n) * 0.016, q[1] + (v[1] / n) * 0.016]);
        if (
          a &&
          b &&
          bounds.current.containsPoint(new Vector3(...a)) &&
          bounds.current.containsPoint(new Vector3(...b))
        )
          arrows.push([a, b]);
      });
    return { path, arrows };
  }, [prediction, showField, floorY]);

  return (
    <>
      <color attach="background" args={["#e8ebe7"]} />
      <ambientLight intensity={1.25} />
      <directionalLight position={[7, 18, 10]} intensity={2.2} />
      <directionalLight position={[-9, 8, -5]} intensity={0.55} />
      <group
        ref={root}
        rotation={source.up_axis === "z" ? [-Math.PI / 2, 0, 0] : [0, 0, 0]}
        onPointerDown={(e) => {
          pointerDown.current = [e.clientX, e.clientY];
        }}
        onPointerUp={(e) => {
          if (
            e.button !== 0 ||
            Math.hypot(
              e.clientX - pointerDown.current[0],
              e.clientY - pointerDown.current[1],
            ) > 5 ||
            !floor
          )
            return;
          const hit = e.intersections.find(
            (hit) =>
              hit.point.y >= floor.min_height && hit.point.y <= cutHeight,
          );
          if (!hit || Math.abs(hit.point.y - floorY) > 0.22) return;
          e.stopPropagation();
          onStart([hit.point.x, floorY + 0.015, hit.point.z]);
        }}
      >
        {source.url ? <LoadedScene url={source.url} /> : <SampleBuilding />}
      </group>
      <group ref={overlays}>
        {start && (
          <group position={start}>
            <mesh rotation={[-Math.PI / 2, 0, 0]}>
              <ringGeometry args={[0.12, 0.21, 40]} />
              <meshBasicMaterial color="#087f74" depthTest={false} />
            </mesh>
            <mesh position={[0, 0.06, 0]}>
              <sphereGeometry args={[0.075, 16, 12]} />
              <meshBasicMaterial color="#f3fffa" />
            </mesh>
          </group>
        )}
        {result.path.length > 1 && (
          <Line points={result.path} color="#087f74" lineWidth={4} />
        )}
        {result.arrows.map((points, i) => (
          <Line
            key={i}
            points={points}
            color="#397f77"
            transparent
            opacity={0.55}
            lineWidth={1.3}
          />
        ))}
      </group>
      <OrbitControls
        ref={controls}
        makeDefault
        enableDamping
        dampingFactor={0.12}
        minDistance={1}
        maxDistance={250}
        maxPolarAngle={Math.PI / 2 - 0.08}
        onChange={() => {
          onViewChange();
          invalidate();
        }}
      />
    </>
  );
}

export const SceneViewport = forwardRef<SceneHandle, Props>(
  function SceneViewport(props, ref) {
    return (
      <SceneError key={props.source.id}>
        <Canvas
          camera={{ position: [9, 14, 11], fov: 48, near: 0.05, far: 1500 }}
          dpr={[1, 1.5]}
          frameloop="demand"
          gl={{ antialias: true, preserveDrawingBuffer: true }}
        >
          <Suspense
            fallback={
              <Html center>
                <div className="scene-loading">Opening scene…</div>
              </Html>
            }
          >
            <World {...props} handleRef={ref} />
          </Suspense>
        </Canvas>
      </SceneError>
    );
  },
);
