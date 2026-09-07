import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Badge,
  Button,
  Callout,
  Flex,
  Select,
  Switch,
  Text,
  TextArea,
  TextField,
} from "@radix-ui/themes";
import {
  ArrowLeft,
  ArrowRight,
  Database,
  FolderOpen,
  LoaderCircle,
  Play,
} from "lucide-react";
import { api, apiError, predict } from "../api/client";
import type { DatasetView, Point2, Prediction } from "../api/client";
import { FieldView } from "../components/FieldView";
import { PredictionSummary } from "../components/PredictionSummary";
import type { ModeProps } from "./types";

export function ValInspector({
  catalog,
  domain,
  mode,
  modelId,
  modelRevision,
  datasetId,
  onDatasetChange,
  onOpenResource,
}: ModeProps) {
  const split =
    catalog.datasets.find((d) => d.id === datasetId)?.split ?? "val";
  const [index, setIndex] = useState(0);
  const [instruction, setInstruction] = useState("");
  const [start, setStart] = useState<Point2 | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [showField, setShowField] = useState(true);
  const revision = useRef(0);
  const controller = useRef<AbortController | null>(null);
  const {
    data: sample,
    isFetching,
    error: loadError,
  } = useQuery({
    queryKey: ["sample", datasetId, index, split],
    enabled: !!datasetId,
    queryFn: async ({ signal }) => {
      const { data, error } = await api.GET(
        "/api/v1/datasets/{dataset_id}/samples/{index}",
        {
          params: { path: { dataset_id: datasetId, index }, query: { split } },
          signal,
        },
      );
      if (error || !data) throw apiError(error);
      return data as DatasetView;
    },
  });
  const invalidate = () => {
    revision.current++;
    controller.current?.abort();
    setBusy(false);
    setResult(null);
    setError("");
  };
  useEffect(() => {
    invalidate();
  }, [datasetId, index, split, modelId, modelRevision]);
  useEffect(() => {
    if (sample) {
      setInstruction(sample.instruction);
      setStart(
        sample.profile === "ground_sector_v1" ? null : (sample.start as Point2),
      );
      invalidate();
    }
  }, [sample]);
  useEffect(() => () => controller.current?.abort(), []);
  const groundRadius =
    sample?.profile === "ground_sector_v1"
      ? Number(sample.geometry.r_max_m)
      : undefined;
  const originalPrompt = instruction === sample?.instruction;
  const visibleStart: Point2 =
    start ?? result?.trajectory[0] ?? sample?.start ?? [0.5, 0.5];

  async function run() {
    if (!sample || !modelId) return;
    const current = ++revision.current;
    controller.current?.abort();
    const abort = new AbortController();
    controller.current = abort;
    setBusy(true);
    setError("");
    try {
      const prediction = await predict(
        {
          mode_id: mode.id,
          domain_id: domain.id,
          model_id: modelId,
          observation_id: `${sample.observation_id}-${current}`,
          instruction,
          start,
          dataset_id: datasetId,
          sample_index: index,
          split,
          grid_size: 24,
          max_steps: 100,
          policy_dt: 0.01,
        },
        abort.signal,
      );
      if (revision.current === current) setResult(prediction);
    } catch (e) {
      if (revision.current === current && !abort.signal.aborted)
        setError((e as Error).message);
    } finally {
      if (revision.current === current) setBusy(false);
    }
  }

  if (!catalog.datasets.length)
    return (
      <div className="empty-workspace">
        <Database size={36} />
        <h2>Your validation workspace</h2>
        <p>
          Open a native CoFL or CoFL-S dataset to inspect observations,
          reference fields and predictions.
        </p>
        <Button size="3" onClick={() => onOpenResource("dataset")}>
          <FolderOpen size={17} />
          Open dataset
        </Button>
        <Text color="gray" size="2">
          Select a dataset or collection folder and choose the split to inspect.
        </Text>
      </div>
    );

  return (
    <div className="workspace">
      <aside className="controls">
        <div className="control-section">
          <div className="section-label">VALIDATION DATASET</div>
          <Select.Root
            value={datasetId}
            onValueChange={(v) => {
              onDatasetChange(v);
              setIndex(0);
            }}
          >
            <Select.Trigger className="full-width" aria-label="Dataset" />
            <Select.Content>
              {catalog.datasets.map((d) => (
                <Select.Item key={d.id} value={d.id}>
                  {d.label}
                  {d.profile ? "" : " · verify"}
                </Select.Item>
              ))}
            </Select.Content>
          </Select.Root>
          <Button
            variant="ghost"
            size="1"
            mt="3"
            onClick={() => onOpenResource("dataset")}
          >
            <FolderOpen size={14} />
            Open dataset
          </Button>
          <Flex justify="between" mt="3" align="center">
            <Badge color="gray">{split}</Badge>
            <Text size="1" color="gray">
              {sample?.count ?? "—"} samples
            </Text>
          </Flex>
          <Flex gap="2" mt="3">
            <Button
              aria-label="Previous sample"
              variant="soft"
              color="gray"
              disabled={index === 0 || isFetching}
              onClick={() => setIndex((i) => i - 1)}
            >
              <ArrowLeft size={16} />
            </Button>
            <TextField.Root
              type="number"
              aria-label="Sample index"
              min={0}
              value={index}
              onChange={(e) => {
                const v = Number(e.target.value);
                if (
                  Number.isInteger(v) &&
                  v >= 0 &&
                  v < (sample?.count ?? Infinity)
                )
                  setIndex(v);
              }}
            />
            <Button
              aria-label="Next sample"
              variant="soft"
              color="gray"
              disabled={!sample || index >= sample.count - 1 || isFetching}
              onClick={() => setIndex((i) => i + 1)}
            >
              <ArrowRight size={16} />
            </Button>
          </Flex>
        </div>
        <div className="control-section">
          <div className="section-label">LANGUAGE INSTRUCTION</div>
          <TextArea
            aria-label="Validation instruction"
            rows={4}
            value={instruction}
            onChange={(e) => {
              setInstruction(e.target.value);
              invalidate();
            }}
          />
          {!originalPrompt && (
            <Text as="p" size="1" color="amber" mt="2">
              Edited instruction · reference scores are unavailable.
            </Text>
          )}
          <Flex mt="3" gap="2">
            <TextField.Root
              aria-label="Start first coordinate"
              type="number"
              step=".01"
              value={start?.[0] ?? ""}
              placeholder="Default"
              onChange={(e) => {
                setStart([Number(e.target.value), visibleStart[1]]);
                invalidate();
              }}
            />
            <TextField.Root
              aria-label="Start second coordinate"
              type="number"
              step=".01"
              value={start?.[1] ?? ""}
              placeholder="Default"
              onChange={(e) => {
                setStart([visibleStart[0], Number(e.target.value)]);
                invalidate();
              }}
            />
          </Flex>
          <Text size="1" color="gray" as="p" mt="1">
            {groundRadius
              ? "Start: forward / left, metres"
              : "Start: image x / y, normalized 0–1"}
          </Text>
          <Button
            className="full-width"
            size="3"
            mt="4"
            disabled={
              !modelId || !sample || busy || isFetching || !instruction.trim()
            }
            onClick={() => void run()}
          >
            {busy ? (
              <LoaderCircle className="spin" size={16} />
            ) : (
              <Play size={16} />
            )}{" "}
            {busy ? "Running policy…" : "Predict sample"}
          </Button>
          <Flex justify="between" mt="4" align="center">
            <Text size="2">Flow field</Text>
            <Switch
              size="1"
              checked={showField}
              aria-label="Show validation flow field"
              onCheckedChange={setShowField}
            />
          </Flex>
        </div>
        <PredictionSummary result={result} />
      </aside>
      <main className="validation-main">
        {(error || loadError) && (
          <Callout.Root color="red" role="alert">
            <Callout.Text>{error || loadError?.message}</Callout.Text>
          </Callout.Root>
        )}
        {isFetching && (
          <div className="loading-row">
            <LoaderCircle className="spin" size={18} />
            Loading sample…
          </div>
        )}
        {sample && (
          <>
            <Flex justify="between" align="center">
              <div>
                <Text size="1" color="gray">
                  SAMPLE {index + 1} / {sample.count}
                </Text>
                <Text as="p" size="2" className="sample-id">
                  {sample.sample_id}
                </Text>
              </div>
              <Badge color="teal">
                {sample.profile === "image_field_v1"
                  ? "CoFL · image field"
                  : "CoFL-S · sector field"}
              </Badge>
            </Flex>
            <div className="comparison-grid">
              <section className="comparison-panel">
                <div className="panel-heading">
                  <Text weight="medium" size="2">
                    Ground truth
                  </Text>
                  <Badge color="orange">Reference</Badge>
                </div>
                <FieldView
                  image={groundRadius ? undefined : sample.image}
                  queries={sample.queries}
                  vectors={sample.vectors}
                  trajectory={sample.trajectory}
                  start={visibleStart}
                  groundRadius={groundRadius}
                  showField={showField}
                  color="#b0763d"
                  onStart={(p) => {
                    setStart(p);
                    invalidate();
                  }}
                />
              </section>
              <section className="comparison-panel">
                <div className="panel-heading">
                  <Text weight="medium" size="2">
                    Policy prediction
                  </Text>
                  <Badge color={result ? "teal" : "gray"}>
                    {result ? "Predicted" : "Ready to query"}
                  </Badge>
                </div>
                <FieldView
                  image={groundRadius ? undefined : sample.image}
                  queries={result?.queries}
                  vectors={result?.vectors}
                  trajectory={result?.trajectory}
                  reference={originalPrompt ? sample.trajectory : undefined}
                  start={visibleStart}
                  groundRadius={groundRadius}
                  showField={showField}
                  onStart={(p) => {
                    setStart(p);
                    invalidate();
                  }}
                />
              </section>
            </div>
            <Flex justify="between" gap="3" wrap="wrap">
              <Text size="1" color="gray">
                {sample.preview_count} displayed / {sample.supervised_count}{" "}
                supervised cells · {sample.vector_unit}
              </Text>
              <Text size="1" color="gray">
                Click a field to change the start.
              </Text>
            </Flex>
            {groundRadius && (
              <div className="sector-observation">
                <img src={sample.image} alt="Egocentric RGB observation" />
                <div>
                  <div className="section-label">EGOCENTRIC OBSERVATION</div>
                  <Text size="2" color="gray">
                    The original RGB-D sample feeds CoFL-S. Sector coordinates
                    remain in the robot's local frame.
                  </Text>
                  {result?.actions && (
                    <Text as="p" size="2" mt="3">
                      Action logits ·{" "}
                      {result.actions
                        .map(
                          (v, i) =>
                            `${["STOP", "FORWARD", "LEFT", "RIGHT"][i]} ${v.toFixed(2)}`,
                        )
                        .join(" / ")}
                    </Text>
                  )}
                </div>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}
