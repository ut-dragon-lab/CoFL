/** Small built-in space for camera/cutaway exploration. No synthetic policy outputs. */
const levels = [0, 3.5];
export function SampleBuilding() {
  return (
    <group>
      {levels.map((y, level) => (
        <group key={y} position={[0, y, 0]}>
          <mesh position={[0, -0.1, 0]}>
            <boxGeometry args={[12, 0.2, 10]} />
            <meshStandardMaterial color={level ? "#d1c2ad" : "#e1d9c8"} />
          </mesh>
          <mesh position={[0, 3.1, 0]}>
            <boxGeometry args={[12, 0.18, 10]} />
            <meshStandardMaterial color="#e2ded4" />
          </mesh>
          {[
            [0, 1.5, -5, 12, 0.18],
            [0, 1.5, 5, 12, 0.18],
            [-6, 1.5, 0, 0.18, 10],
            [6, 1.5, 0, 0.18, 10],
            [0, 1.5, -2.3, 0.15, 5.3],
            [0, 1.5, 4, 0.15, 2],
          ].map((p, i) => (
            <mesh key={i} position={[p[0], p[1], p[2]]}>
              <boxGeometry args={[p[3], 3, p[4]]} />
              <meshStandardMaterial color="#dcd8ce" />
            </mesh>
          ))}
          <mesh position={[-3.7, 0.05, 0.4]}>
            <boxGeometry args={[3.7, 0.06, 4.4]} />
            <meshStandardMaterial color={level ? "#9ba8b2" : "#b6c6b8"} />
          </mesh>
          <mesh position={[-4.5, 0.43, -0.5]}>
            <boxGeometry args={[1.25, 0.8, 2.8]} />
            <meshStandardMaterial color="#637d7a" />
          </mesh>
          <mesh position={[-4.95, 0.82, -0.5]}>
            <boxGeometry args={[0.3, 0.75, 2.8]} />
            <meshStandardMaterial color="#4b6663" />
          </mesh>
          <mesh position={[-2.6, 0.36, -0.45]}>
            <boxGeometry args={[1.25, 0.65, 1.6]} />
            <meshStandardMaterial color="#a27551" />
          </mesh>
          <mesh position={[3, 0.8, -2.4]}>
            <boxGeometry args={[2.7, 0.14, 1.5]} />
            <meshStandardMaterial color="#b58a64" />
          </mesh>
          {[2, 4].flatMap((x) =>
            [-2.9, -1.9].map((z) => (
              <mesh key={`${x}-${z}`} position={[x, 0.4, z]}>
                <boxGeometry args={[0.1, 0.8, 0.1]} />
                <meshStandardMaterial color="#6f675c" />
              </mesh>
            )),
          )}
          {[1.3, 4.7].map((x) => (
            <mesh key={x} position={[x, 0.45, -2.4]}>
              <boxGeometry args={[0.65, 0.85, 0.75]} />
              <meshStandardMaterial color="#859087" />
            </mesh>
          ))}
          <mesh position={[4.7, 0.95, 3.8]}>
            <boxGeometry args={[1.8, 1.9, 0.5]} />
            <meshStandardMaterial color="#92765c" />
          </mesh>
          <mesh position={[-4.9, 0.3, 3.9]}>
            <cylinderGeometry args={[0.35, 0.26, 0.6, 16]} />
            <meshStandardMaterial color="#a36d52" />
          </mesh>
          <mesh position={[-4.9, 0.95, 3.9]}>
            <sphereGeometry args={[0.55, 16, 12]} />
            <meshStandardMaterial color="#63836a" />
          </mesh>
        </group>
      ))}
    </group>
  );
}
