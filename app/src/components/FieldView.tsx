import { useEffect, useId, useState } from "react";
import type { Point2 } from "../api/client";

interface Props {
  image?: string;
  queries?: number[][];
  vectors?: number[][];
  trajectory?: number[][] | null;
  reference?: number[][] | null;
  start?: Point2 | null;
  groundRadius?: number;
  showField?: boolean;
  color?: string;
  onStart?(point: Point2): void;
}

/** Image and sector charts deliberately keep their different coordinate systems. */
export function FieldView({
  image,
  queries = [],
  vectors = [],
  trajectory,
  reference,
  start,
  groundRadius,
  showField = true,
  color = "#0b8a7e",
  onStart,
}: Props) {
  const marker = useId().replaceAll(":", "");
  const [aspect, setAspect] = useState(1);
  useEffect(() => {
    if (!image) {
      setAspect(1);
      return;
    }
    let current = true;
    const source = new Image();
    source.onload = () => {
      if (current) setAspect(source.naturalWidth / source.naturalHeight);
    };
    source.src = image;
    return () => {
      current = false;
    };
  }, [image]);
  const project = (p: number[]) =>
    groundRadius
      ? [500 - (p[1] / groundRadius) * 450, 950 - (p[0] / groundRadius) * 900]
      : [p[0] * 1000, p[1] * 1000];
  const polyline = (points: number[][]) =>
    points.map((p) => project(p).join(",")).join(" ");
  return (
    <svg
      className="field-view"
      viewBox="0 0 1000 1000"
      preserveAspectRatio="none"
      style={{ aspectRatio: groundRadius ? 1 : aspect }}
      role="img"
      aria-label={
        groundRadius
          ? "Local sector field in metres"
          : "Image field and trajectory"
      }
      onClick={(event) => {
        if (!onStart) return;
        const matrix = event.currentTarget.getScreenCTM();
        if (!matrix) return;
        const p = new DOMPoint(event.clientX, event.clientY).matrixTransform(
          matrix.inverse(),
        );
        if (p.x < 0 || p.x > 1000 || p.y < 0 || p.y > 1000) return;
        onStart(
          groundRadius
            ? [
                ((950 - p.y) / 900) * groundRadius,
                ((500 - p.x) / 450) * groundRadius,
              ]
            : [p.x / 1000, p.y / 1000],
        );
      }}
    >
      <defs>
        <marker
          id={marker}
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="4"
          markerHeight="4"
          orient="auto-start-reverse"
        >
          <path d="M 0 0 L 10 5 L 0 10 z" fill={color} />
        </marker>
      </defs>
      <rect
        width="1000"
        height="1000"
        fill={groundRadius ? "#edf1eb" : "#e9ede8"}
      />
      {image && (
        <image
          href={image}
          width="1000"
          height="1000"
          preserveAspectRatio="none"
        />
      )}
      {groundRadius && (
        <g stroke="#c5d1c7" fill="none">
          {[0.25, 0.5, 0.75, 1].map((r) => (
            <ellipse key={r} cx="500" cy="950" rx={r * 450} ry={r * 900} />
          ))}
          <line x1="500" y1="950" x2="500" y2="30" />
          <text x="520" y="50" stroke="none" fill="#6e8076" fontSize="24">
            forward · {groundRadius} m
          </text>
          <text x="35" y="965" stroke="none" fill="#6e8076" fontSize="24">
            left
          </text>
        </g>
      )}
      {showField && (
        <g stroke={color} strokeWidth="2" opacity=".7">
          {queries.map((q, i) => {
            const v = vectors[i];
            if (!v) return null;
            const n = Math.hypot(...v);
            if (n < 1e-8) return null;
            const a = project(q),
              scale = groundRadius ? groundRadius * 0.022 : 0.022;
            const b = project([
              q[0] + (v[0] / n) * scale,
              q[1] + (v[1] / n) * scale,
            ]);
            return (
              <line
                key={i}
                x1={a[0]}
                y1={a[1]}
                x2={b[0]}
                y2={b[1]}
                markerEnd={`url(#${marker})`}
              />
            );
          })}
        </g>
      )}
      {reference && (
        <polyline
          points={polyline(reference)}
          fill="none"
          stroke="#db9558"
          strokeWidth="7"
          strokeDasharray="12 9"
        />
      )}
      {trajectory && (
        <>
          <polyline
            points={polyline(trajectory)}
            fill="none"
            stroke="white"
            strokeWidth="10"
            strokeLinejoin="round"
          />
          <polyline
            points={polyline(trajectory)}
            fill="none"
            stroke={color}
            strokeWidth="6"
            strokeLinejoin="round"
          />
        </>
      )}
      {start && (
        <circle
          cx={project(start)[0]}
          cy={project(start)[1]}
          r="10"
          fill={color}
          stroke="white"
          strokeWidth="4"
        />
      )}
    </svg>
  );
}
