import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Badge,
  Button,
  Callout,
  Flex,
  Select,
  Slider,
  Switch,
  Text,
  TextArea,
  TextField,
  Tooltip,
} from "@radix-ui/themes";
import {
  ArrowUpRight,
  Camera,
  Crosshair,
  Layers,
  LoaderCircle,
  Maximize,
  MousePointer2,
  Play,
  Upload,
} from "lucide-react";
import { predict } from "../api/client";
import type { Floor, Prediction } from "../api/client";
import { FieldView } from "../components/FieldView";
import { PredictionSummary } from "../components/PredictionSummary";
import { SceneViewport } from "../scene/SceneViewport";
import type { Capture, SceneHandle } from "../scene/SceneViewport";
import type { Point3, SceneSource } from "../scene/geometry";
import type { ModeProps } from "./types";

const DEMO: SceneSource = {
  id: "sample",
  label: "Courtyard House · sample space",
};

export function ScenePlayground({
  catalog,
  domain,
  mode,
  modelId,
  modelRevision,
}: ModeProps) {
  const [sourceId, setSourceId] = useState(catalog.scenes[0]?.id ?? "sample");
  const [local, setLocal] = useState<SceneSource | null>(null);
  const sources = useMemo(
    () => [DEMO, ...catalog.scenes, ...(local ? [local] : [])],
    [catalog.scenes, local],
  );
  const selectedSource = sources.find((s) => s.id === sourceId) ?? DEMO;
  const [axisOverrides, setAxisOverrides] = useState<Record<string, "y" | "z">>(
    {},
  );
  const upAxis = axisOverrides[sourceId] ?? selectedSource.up_axis ?? "y";
  const source: SceneSource = useMemo(
    () => ({
      ...selectedSource,
      id: `${selectedSource.id}:${upAxis}`,
      up_axis: upAxis,
    }),
    [selectedSource, upAxis],
  );
  const [floors, setFloors] = useState<Floor[]>([]);
  const [floorId, setFloorId] = useState("");
  const [floorOverride, setFloorOverride] = useState<number | null>(null);
  const baseFloor = floors.find((f) => f.id === floorId) ?? floors[0] ?? null;
  const floor = useMemo(
    () =>
      baseFloor && floorOverride !== null
        ? {
            ...baseFloor,
            elevation: floorOverride,
            min_height: floorOverride - 0.25,
            max_height: floorOverride + 2.5,
          }
        : baseFloor,
    [baseFloor, floorOverride],
  );
  const [cutHeight, setCutHeight] = useState(2.4);
  const [start, setStart] = useState<Point3 | null>(null);
  const [instruction, setInstruction] = useState(
    "Go around the table and stop in front of the sofa.",
  );
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [capture, setCapture] = useState<Capture | null>(null);
  const [showField, setShowField] = useState(true);
  const [auto, setAuto] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [viewRevision, setViewRevision] = useState(0);
  const generation = useRef(0);
  const controller = useRef<AbortController | null>(null);
  const viewport = useRef<SceneHandle>(null);
  const upload = useRef<HTMLInputElement>(null);

  const invalidate = useCallback(() => {
    generation.current += 1;
    controller.current?.abort();
    setBusy(false);
    setPrediction(null);
    setCapture(null);
    setViewRevision((v) => v + 1);
  }, []);
  const ready = useCallback((levels: Floor[]) => {
    setFloors(levels);
    setFloorId(levels[0]?.id ?? "");
  }, []);
  const chooseStart = useCallback(
    (point: Point3) => {
      invalidate();
      setStart(point);
      setError("");
    },
    [invalidate],
  );
  useEffect(() => {
    setStart(null);
    setFloorOverride(null);
    invalidate();
  }, [source.id, floorId, invalidate]);
  useEffect(() => {
    if (floor) {
      setCutHeight(floor.max_height);
      setStart(null);
      invalidate();
    }
  }, [floor, invalidate]);
  useEffect(() => {
    invalidate();
  }, [modelId, modelRevision, instruction, invalidate]);
  useEffect(() => () => controller.current?.abort(), []);
  useEffect(
    () => () => {
      if (local?.url) URL.revokeObjectURL(local.url);
    },
    [local],
  );

  const run = useCallback(async () => {
    if (!viewport.current || !modelId) return;
    const revision = ++generation.current;
    controller.current?.abort();
    const abort = new AbortController();
    controller.current = abort;
    setError("");
    setBusy(true);
    try {
      const observation = viewport.current.capture();
      setCapture(observation);
      const result = await predict(
        {
          mode_id: mode.id,
          domain_id: domain.id,
          model_id: modelId,
          ...observation,
          instruction,
          grid_size: 24,
          max_steps: 100,
          policy_dt: 0.01,
          split: "val",
        },
        abort.signal,
      );
      if (
        generation.current === revision &&
        result.observation_id === observation.observation_id
      )
        setPrediction(result);
    } catch (e) {
      if (generation.current === revision && !abort.signal.aborted)
        setError((e as Error).message);
    } finally {
      if (generation.current === revision) setBusy(false);
    }
  }, [modelId, instruction, mode.id, domain.id]);

  useEffect(() => {
    if (!auto || !start || !modelId) return;
    const timer = setTimeout(() => {
      void run();
    }, 650);
    return () => clearTimeout(timer);
  }, [auto, start, modelId, viewRevision, run]);

  return (
    <div className="workspace">
      <aside className="controls">
        <div className="control-section">
          <div className="section-label">
            <Layers size={15} />
            SPACE
          </div>
          <Select.Root value={sourceId} onValueChange={setSourceId}>
            <Select.Trigger className="full-width" aria-label="Scene" />
            <Select.Content>
              {sources.map((s) => (
                <Select.Item key={s.id} value={s.id}>
                  {s.label}
                </Select.Item>
              ))}
            </Select.Content>
          </Select.Root>
          {source.url && (
            <Flex justify="between" align="center" mt="3">
              <Text size="2">Up axis</Text>
              <Select.Root
                value={upAxis}
                onValueChange={(value: "y" | "z") =>
                  setAxisOverrides((current) => ({
                    ...current,
                    [sourceId]: value,
                  }))
                }
              >
                <Select.Trigger aria-label="Scene up axis" />
                <Select.Content>
                  <Select.Item value="y">Y · glTF</Select.Item>
                  <Select.Item value="z">Z · Matterport</Select.Item>
                </Select.Content>
              </Select.Root>
            </Flex>
          )}
          <input
            ref={upload}
            type="file"
            accept=".glb"
            hidden
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (!file) return;
              const id = `import-${crypto.randomUUID()}`;
              setLocal({
                id,
                label: file.name,
                url: URL.createObjectURL(file),
              });
              setSourceId(id);
            }}
          />
          <Button
            variant="ghost"
            size="1"
            mt="3"
            onClick={() => upload.current?.click()}
          >
            <Upload size={13} />
            Open a GLB file
          </Button>
        </div>
        <div className="control-section">
          <div className="section-label">FLOOR & CUTAWAY</div>
          <Select.Root value={floorId} onValueChange={setFloorId}>
            <Select.Trigger
              className="full-width"
              placeholder="Detecting floors…"
              aria-label="Floor"
            />
            <Select.Content>
              {floors.map((f) => (
                <Select.Item key={f.id} value={f.id}>
                  {f.label}
                </Select.Item>
              ))}
            </Select.Content>
          </Select.Root>
          <Flex justify="between" mt="4" mb="2">
            <Text size="2">Cut height</Text>
            <Text size="1" color="gray">
              {(cutHeight - (floor?.elevation ?? 0)).toFixed(1)} m above floor
            </Text>
          </Flex>
          <Slider
            aria-label="Cut height"
            min={(floor?.elevation ?? 0) + 0.3}
            max={(floor?.elevation ?? 0) + 4}
            step={0.05}
            value={[cutHeight]}
            onValueChange={([v]) => {
              setCutHeight(v);
              invalidate();
            }}
          />
          <details className="advanced">
            <summary>Adjust floor elevation</summary>
            <Text size="1" as="p" color="gray">
              Suggested levels can be adjusted for split-level spaces.
            </Text>
            <TextField.Root
              aria-label="Floor elevation"
              type="number"
              step="0.1"
              value={floor?.elevation ?? 0}
              onChange={(e) => {
                if (
                  e.target.value !== "" &&
                  Number.isFinite(Number(e.target.value))
                )
                  setFloorOverride(Number(e.target.value));
              }}
            />
          </details>
        </div>
        <div className="control-section">
          <div className="section-label">LANGUAGE INSTRUCTION</div>
          <TextArea
            aria-label="Language instruction"
            value={instruction}
            onChange={(e) => setInstruction(e.target.value)}
            rows={4}
            placeholder="Describe where to go…"
          />
          <Flex align="center" gap="2" mt="3">
            <span className={start ? "status-dot ready" : "status-dot"} />
            <Text size="2" color="gray">
              {start
                ? "Start selected on this floor"
                : "Click the floor to choose a start"}
            </Text>
          </Flex>
          <Button
            className="full-width"
            size="3"
            mt="4"
            disabled={!start || !modelId || !instruction.trim() || busy}
            onClick={() => void run()}
          >
            {busy ? (
              <LoaderCircle className="spin" size={16} />
            ) : (
              <Play size={16} />
            )}{" "}
            {busy ? "Running policy…" : "Confirm start & predict"}
          </Button>
          {!modelId && (
            <Text size="1" color="gray" as="p" mt="2">
              Explore freely. Use Load policy above to predict.
            </Text>
          )}
          <Flex justify="between" align="center" mt="3">
            <Text size="2">Update after moving</Text>
            <Switch
              aria-label="Update after moving"
              size="1"
              checked={auto}
              onCheckedChange={setAuto}
            />
          </Flex>
        </div>
        <PredictionSummary result={prediction} />
      </aside>
      <main className="scene-main">
        <div className="viewport-shell">
          <div className="viewport-top">
            <Badge color="gray" variant="solid">
              {source.url ? "GLB scene" : "Built-in sample space"}
            </Badge>
            <Badge color="teal" variant="soft">
              {floor?.label ?? "Loading"}
            </Badge>
          </div>
          <SceneViewport
            ref={viewport}
            source={source}
            floor={floor}
            cutHeight={cutHeight}
            start={start}
            prediction={prediction}
            showField={showField}
            onReady={ready}
            onStart={chooseStart}
            onViewChange={invalidate}
          />
          <div className="viewport-toolbar">
            <Tooltip content="Frame this floor">
              <Button
                aria-label="Frame this floor"
                variant="soft"
                color="gray"
                onClick={() => viewport.current?.fit()}
              >
                <Maximize size={16} />
              </Button>
            </Tooltip>
            <Tooltip content="Top view">
              <Button
                aria-label="Top view"
                variant="soft"
                color="gray"
                onClick={() => viewport.current?.topView()}
              >
                <Camera size={16} />
              </Button>
            </Tooltip>
            <Tooltip content="Clear start">
              <Button
                aria-label="Clear start"
                variant="soft"
                color="gray"
                onClick={() => {
                  setStart(null);
                  invalidate();
                }}
              >
                <Crosshair size={16} />
              </Button>
            </Tooltip>
          </div>
          <div className="viewport-bottom">
            <span>
              <MousePointer2 size={13} />
              Drag to orbit · right-drag to pan · scroll to zoom
            </span>
            <span>
              Click an open floor area to set the start
              <ArrowUpRight size={13} />
            </span>
          </div>
        </div>
        {error && (
          <Callout.Root color="red" role="alert">
            <Callout.Text>{error}</Callout.Text>
          </Callout.Root>
        )}
        <div className="observation-strip">
          <div>
            <div className="section-label">MODEL OBSERVATION</div>
            <Text size="2" color="gray">
              The clipped scene, captured from your chosen camera.
            </Text>
            <Flex gap="2" align="center" mt="3">
              <Switch
                size="1"
                aria-label="Show flow field"
                checked={showField}
                onCheckedChange={setShowField}
              />
              <Text size="2">Flow field</Text>
            </Flex>
          </div>
          {capture ? (
            <div className="observation-preview">
              <FieldView
                image={capture.image}
                queries={prediction?.queries}
                vectors={prediction?.vectors}
                trajectory={prediction?.trajectory}
                start={capture.start}
                showField={showField}
              />
            </div>
          ) : (
            <div className="capture-placeholder">
              <Camera size={24} />
              <Text size="1">Choose a start and predict</Text>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
