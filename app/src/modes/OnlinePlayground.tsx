import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Badge,
  Button,
  Callout,
  Flex,
  Select,
  Text,
  TextArea,
  TextField,
} from "@radix-ui/themes";
import {
  ArrowLeft,
  ArrowRight,
  Download,
  FolderOpen,
  LoaderCircle,
  Pause,
  Play,
  RotateCcw,
  Send,
  Square,
} from "lucide-react";
import { api, apiError } from "../api/client";
import type { ModeProps } from "./types";
import { isActive, useOnlineSession } from "./online/useOnlineSession";
import { SessionView } from "./online/SessionView";

const PAGE_SIZE = 50;
interface Handoff {
  benchmarkId: string;
  sceneId: string;
  position: [number, number, number];
  yaw: number;
}
interface PoseDraft {
  key: string;
  position: [string, string, string];
  yaw: string;
}

export function OnlinePlayground({
  domain,
  mode,
  catalog,
  benchmarkId,
  modelId,
  onBenchmarkChange,
  onOpenResource,
  onNavigate,
}: ModeProps) {
  const interactive = mode.task === "interactive";
  const client = useQueryClient();
  const handoffKey = ["online-handoff", domain.id];
  const [handoff] = useState<Handoff | null>(() =>
    interactive ? (client.getQueryData(handoffKey) ?? null) : null,
  );
  const handoffStarted = useRef(false);
  const [sceneId, setSceneId] = useState(handoff?.sceneId ?? "");
  const [pose, setPose] = useState<PoseDraft | null>(() =>
    handoff
      ? {
          key: `${handoff.benchmarkId}:${handoff.sceneId}`,
          position: handoff.position.map(String) as PoseDraft["position"],
          yaw: String(handoff.yaw),
        }
      : null,
  );
  const [episodeSelection, setEpisodeSelection] = useState({
    benchmarkId,
    index: 0,
  });
  const index =
    episodeSelection.benchmarkId === benchmarkId ? episodeSelection.index : 0;
  const [commandText, setCommandText] = useState("");
  const [timeLimit, setTimeLimit] = useState(() =>
    Number(mode.default_parameters?.max_time_s ?? 120),
  );
  const online = useOnlineSession(domain.id);
  const busy = isActive(online.current);
  const current =
    online.current?.mode_id === mode.id ? online.current : undefined;
  const foreign = busy && !current ? online.current : undefined;
  const controllable =
    !!current && busy && !["starting", "stopping"].includes(current.status);

  useEffect(() => {
    if (handoff)
      client.removeQueries({ queryKey: ["online-handoff", domain.id] });
  }, [client, domain.id, handoff]);
  useEffect(() => {
    if (
      !handoff ||
      handoffStarted.current ||
      busy ||
      online.recovering ||
      !modelId
    )
      return;
    handoffStarted.current = true;
    online.start.mutate({
      benchmark_id: handoff.benchmarkId,
      model_id: modelId,
      mode: "interactive",
      scene_id: handoff.sceneId,
      start_position: handoff.position,
      start_yaw_deg: handoff.yaw,
    });
  }, [handoff, busy, modelId, online.recovering, online.start]);

  const benchmarks = useQuery({
    queryKey: ["online-benchmarks"],
    queryFn: async ({ signal }) => {
      const { data, error } = await api.GET(
        "/api/v1/domains/ego2d/benchmarks",
        { signal },
      );
      if (error || !data) throw apiError(error);
      return data;
    },
  });
  const benchmark = benchmarks.data?.find((item) => item.id === benchmarkId);
  const episodes = useQuery({
    queryKey: ["online-episodes", benchmarkId, Math.floor(index / PAGE_SIZE)],
    enabled: !!benchmarkId && !interactive,
    retry: false,
    queryFn: async ({ signal }) => {
      const { data, error } = await api.GET(
        "/api/v1/domains/ego2d/benchmarks/{benchmark_id}/episodes",
        {
          params: {
            path: { benchmark_id: benchmarkId },
            query: {
              offset: Math.floor(index / PAGE_SIZE) * PAGE_SIZE,
              limit: PAGE_SIZE,
            },
          },
          signal,
        },
      );
      if (error || !data) throw apiError(error);
      return data;
    },
  });
  const episode = episodes.data?.items.find((item) => item.index === index);
  const scenes = useQuery({
    queryKey: ["online-scenes", benchmarkId],
    enabled: !!benchmarkId && interactive,
    retry: false,
    queryFn: async ({ signal }) => {
      const { data, error } = await api.GET(
        "/api/v1/domains/ego2d/benchmarks/{benchmark_id}/scenes",
        { params: { path: { benchmark_id: benchmarkId } }, signal },
      );
      if (error || !data) throw apiError(error);
      return data;
    },
  });
  const scene =
    scenes.data?.items.find((item) => item.scene_id === sceneId) ??
    scenes.data?.items[0];
  const poseKey = `${benchmarkId}:${scene?.scene_id ?? ""}`;
  const draft =
    pose?.key === poseKey
      ? pose
      : {
          key: poseKey,
          position: (scene?.spawn_position?.map(String) ?? [
            "",
            "",
            "",
          ]) as PoseDraft["position"],
          yaw: String(scene?.spawn_yaw_deg ?? 0),
        };
  const autoPosition = draft.position.every((value) => !value.trim());
  const validPose =
    Number.isFinite(Number(draft.yaw)) &&
    !!draft.yaw.trim() &&
    (autoPosition ||
      draft.position.every(
        (value) => !!value.trim() && Number.isFinite(Number(value)),
      ));
  const position = autoPosition
    ? undefined
    : (draft.position.map(Number) as [number, number, number]);
  const controlsPending = online.action.isPending || online.reset.isPending;
  const error =
    online.error ?? benchmarks.error ?? episodes.error ?? scenes.error;
  const changeIndex = (value: number) => {
    if (
      Number.isInteger(value) &&
      value >= 0 &&
      value < (episodes.data?.total ?? Infinity)
    )
      setEpisodeSelection({ benchmarkId, index: value });
  };
  const start = () => {
    online.clearErrors();
    online.start.mutate({
      benchmark_id: benchmarkId,
      model_id: modelId,
      max_time_s: timeLimit,
      ...(interactive
        ? {
            mode: "interactive",
            scene_id: scene?.scene_id,
            start_position: position,
            start_yaw_deg: Number(draft.yaw),
            instruction: commandText.trim(),
          }
        : {
            mode: "benchmark",
            episode_index: index,
            instruction_mode: "oracle",
          }),
    });
  };
  const send = () => {
    if (!controllable || !commandText.trim() || online.command.isPending)
      return;
    const submitted = commandText.trim();
    online.clearErrors();
    online.command.mutate(submitted, {
      onSuccess: () =>
        setCommandText((value) => (value.trim() === submitted ? "" : value)),
    });
  };
  const explore = async () => {
    if (!current?.scene_id || !current.current_position) return;
    const seed: Handoff = {
      benchmarkId: current.benchmark_id,
      sceneId: current.scene_id,
      position: current.current_position,
      yaw: current.current_yaw_deg ?? 0,
    };
    try {
      if (busy) await online.action.mutateAsync("stop");
      client.setQueryData(handoffKey, seed);
      onBenchmarkChange(seed.benchmarkId);
      onNavigate("interactive");
    } catch {
      /* The shared mutation displays the server error. */
    }
  };
  const exportRun = () => {
    if (!current) return;
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(current, null, 2)], {
        type: "application/json",
      }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = `cofl-online-${current.id}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="workspace">
      <aside className="controls">
        <div className="control-section">
          <div className="section-label">
            {interactive ? "SCENE ENVIRONMENT" : "VLN BENCHMARK"}
          </div>
          <Select.Root
            value={benchmarkId}
            onValueChange={onBenchmarkChange}
            disabled={busy || online.start.isPending}
          >
            <Select.Trigger
              className="full-width"
              aria-label={interactive ? "Scene environment" : "VLN benchmark"}
              placeholder="Choose an environment"
            />
            <Select.Content>
              {catalog.benchmarks.map((item) => (
                <Select.Item key={item.id} value={item.id}>
                  {item.label}
                </Select.Item>
              ))}
            </Select.Content>
          </Select.Root>
          <Button
            variant="ghost"
            size="1"
            mt="3"
            disabled={busy}
            onClick={() => onOpenResource("benchmark")}
          >
            <FolderOpen size={14} />
            Open benchmark
          </Button>
          {interactive ? (
            <>
              <Text size="1" color="gray" as="div" mt="4" mb="2">
                Scene
              </Text>
              <Select.Root
                value={scene?.scene_id ?? ""}
                onValueChange={(value) => {
                  setSceneId(value);
                  setPose(null);
                }}
                disabled={busy || online.start.isPending}
              >
                <Select.Trigger
                  className="full-width"
                  aria-label="Navigation scene"
                  placeholder="Choose a scene"
                />
                <Select.Content>
                  {scenes.data?.items.map((item) => (
                    <Select.Item key={item.id} value={item.scene_id}>
                      {item.label}
                    </Select.Item>
                  ))}
                </Select.Content>
              </Select.Root>
              <details className="pose-settings">
                <summary>Starting position & heading</summary>
                <div className="pose-inputs">
                  {(["X", "Y", "Z"] as const).map((axis, i) => (
                    <label key={axis}>
                      <Text size="1" color="gray">
                        {axis} · m
                      </Text>
                      <TextField.Root
                        type="number"
                        step="any"
                        aria-label={`Start ${axis}`}
                        value={draft.position[i]}
                        placeholder="Auto"
                        onChange={(event) => {
                          const next = [
                            ...draft.position,
                          ] as PoseDraft["position"];
                          next[i] = event.target.value;
                          setPose({ ...draft, position: next });
                        }}
                      />
                    </label>
                  ))}
                </div>
                <label>
                  <Text size="1" color="gray" as="div" mt="2">
                    Heading · degrees
                  </Text>
                  <TextField.Root
                    type="number"
                    step="any"
                    aria-label="Start heading"
                    value={draft.yaw}
                    onChange={(event) =>
                      setPose({ ...draft, yaw: event.target.value })
                    }
                  />
                </label>
                <Text as="p" size="1" color="gray">
                  Y is height. Leave XYZ empty to use a navigable point.
                </Text>
                <Button size="1" variant="ghost" onClick={() => setPose(null)}>
                  Use scene start
                </Button>
              </details>
            </>
          ) : (
            <>
              <Flex justify="between" mt="4">
                <Badge color="gray">{benchmark?.split ?? "VLN"}</Badge>
                <Text size="1" color="gray">
                  {episodes.data?.total ?? 0} episodes
                </Text>
              </Flex>
              <Flex gap="2" mt="3">
                <Button
                  variant="soft"
                  color="gray"
                  aria-label="Previous episode"
                  disabled={busy || index === 0}
                  onClick={() => changeIndex(index - 1)}
                >
                  <ArrowLeft size={15} />
                </Button>
                <TextField.Root
                  aria-label="Episode index"
                  type="number"
                  value={index}
                  min={0}
                  disabled={busy}
                  onChange={(event) => changeIndex(Number(event.target.value))}
                />
                <Button
                  variant="soft"
                  color="gray"
                  aria-label="Next episode"
                  disabled={
                    busy || !episodes.data || index + 1 >= episodes.data.total
                  }
                  onClick={() => changeIndex(index + 1)}
                >
                  <ArrowRight size={15} />
                </Button>
              </Flex>
              {episode && (
                <Text as="p" size="1" color="gray" mt="3">
                  Episode {episode.episode_id} · {episode.scene_id}
                </Text>
              )}
            </>
          )}
          {benchmark && !benchmark.available && (
            <Text as="p" color="amber" size="1" mt="3">
              {benchmark.reason}
            </Text>
          )}
        </div>
        <div className="control-section">
          <div className="section-label">
            {interactive ? "YOUR INSTRUCTION" : "GLOBAL TASK"}
          </div>
          {interactive ? (
            <form
              onSubmit={(event) => {
                event.preventDefault();
                send();
              }}
            >
              <TextArea
                aria-label="Robot instruction"
                placeholder="Walk to the doorway, then turn left…"
                rows={4}
                value={commandText}
                maxLength={2048}
                onChange={(event) => setCommandText(event.target.value)}
                onKeyDown={(event) => {
                  if (
                    event.key === "Enter" &&
                    !event.shiftKey &&
                    !event.nativeEvent.isComposing
                  ) {
                    event.preventDefault();
                    send();
                  }
                }}
              />
              <Button
                type="submit"
                className="full-width"
                mt="3"
                disabled={
                  !controllable ||
                  !commandText.trim() ||
                  online.command.isPending
                }
              >
                <Send size={14} />
                Send command
              </Button>
              <Text as="p" size="1" color="gray">
                Enter to send · Shift + Enter for a new line.
              </Text>
            </form>
          ) : (
            <>
              <Text as="p" size="2" className="online-instruction">
                {episode?.instruction ??
                  "Choose an episode to inspect its task."}
              </Text>
              <Badge color="gray">Oracle sub-instructions</Badge>
            </>
          )}
          <Text size="1" color="gray" as="div" mt="4" mb="1">
            {interactive
              ? "Per-command simulation limit · seconds"
              : "Simulation time limit · seconds"}
          </Text>
          <TextField.Root
            type="number"
            min={0.01}
            step="any"
            max={600}
            aria-label="Simulation time limit"
            value={timeLimit}
            disabled={busy}
            onChange={(event) => setTimeLimit(Number(event.target.value))}
          />
          <Button
            className="full-width"
            size="3"
            mt="4"
            disabled={
              busy ||
              online.start.isPending ||
              online.recovering ||
              !modelId ||
              !benchmark?.available ||
              !(interactive ? scene && validPose : episode) ||
              !Number.isFinite(timeLimit) ||
              timeLimit <= 0 ||
              timeLimit > 600
            }
            onClick={start}
          >
            {online.start.isPending || current?.status === "starting" ? (
              <LoaderCircle className="spin" size={16} />
            ) : (
              <Play size={16} />
            )}
            {interactive ? "Open scene" : "Run episode"}
          </Button>
          {interactive && (
            <Flex className="session-actions" gap="2" mt="2">
              <Button
                className="full-width"
                variant="soft"
                color="gray"
                disabled={
                  !controllable ||
                  controlsPending ||
                  current?.status === "waiting"
                }
                onClick={() =>
                  online.action.mutate(
                    current?.status === "paused" ? "resume" : "pause",
                  )
                }
              >
                {current?.status === "paused" ? (
                  <Play size={14} />
                ) : (
                  <Pause size={14} />
                )}
                {current?.status === "paused" ? "Resume" : "Pause"}
              </Button>
              <Button
                className="full-width"
                variant="soft"
                color="gray"
                disabled={!controllable || controlsPending || !validPose}
                onClick={() =>
                  online.reset.mutate({
                    start_position: position,
                    start_yaw_deg: Number(draft.yaw),
                    instruction: "",
                  })
                }
              >
                <RotateCcw size={14} />
                Reset pose
              </Button>
            </Flex>
          )}
          <Button
            className="full-width"
            variant="soft"
            color="gray"
            mt="2"
            disabled={
              !current ||
              !busy ||
              controlsPending ||
              current.status === "stopping"
            }
            onClick={() => online.action.mutate("stop")}
          >
            <Square size={14} />
            {interactive ? "End session" : "Stop execution"}
          </Button>
        </div>
        {current && (
          <div className="prediction-summary">
            <Flex justify="between" align="center">
              <Text size="2" weight="medium">
                {interactive ? "Session status" : "Episode status"}
              </Text>
              <Badge
                color={
                  current.status === "failed" ? "red" : busy ? "teal" : "gray"
                }
              >
                {current.status}
              </Badge>
            </Flex>
            <div className="metrics-grid">
              <div>
                <strong>
                  {(
                    current.observation?.t_sim_s ??
                    current.steps?.at(-1)?.t_sim_s ??
                    0
                  ).toFixed(1)}
                  <small> s</small>
                </strong>
                <span>Simulation time</span>
              </div>
              <div>
                <strong>{current.total_steps}</strong>
                <span>Planner ticks</span>
              </div>
            </div>
            <Text as="p" size="1" color="gray">
              {catalog.models.find((item) => item.id === current.model_id)
                ?.label ?? current.model_id}
              <br />
              {current.reason}
            </Text>
            {!!current.command_state?.reset_error && (
              <Text as="p" color="red" size="1">
                {String(current.command_state.reset_error)}
              </Text>
            )}
            {!interactive && (
              <Button
                className="full-width"
                size="1"
                variant="soft"
                mb="3"
                disabled={
                  !current.scene_id ||
                  !current.current_position ||
                  online.action.isPending
                }
                onClick={() => void explore()}
              >
                Explore this scene
                <ArrowRight size={13} />
              </Button>
            )}
            <Button size="1" variant="soft" color="gray" onClick={exportRun}>
              <Download size={13} />
              Export run snapshot
            </Button>
          </div>
        )}
      </aside>
      <main className="online-main">
        {error && (
          <Callout.Root color="red" role="alert">
            <Callout.Text>{error.message}</Callout.Text>
          </Callout.Root>
        )}
        {foreign && (
          <Callout.Root color="amber">
            <Callout.Text>
              A{" "}
              {foreign.mode === "interactive"
                ? "Playground session"
                : "VLN Benchmark episode"}{" "}
              is active. Open it to continue, or end it to release the scene.
            </Callout.Text>
            <Flex gap="2" mt="2">
              <Button
                size="1"
                variant="soft"
                onClick={() =>
                  onNavigate(
                    foreign.mode === "interactive"
                      ? "interactive"
                      : "benchmark",
                  )
                }
              >
                Open active session
              </Button>
              <Button
                size="1"
                variant="soft"
                color="gray"
                disabled={
                  online.action.isPending || foreign.status === "stopping"
                }
                onClick={() => online.action.mutate("stop")}
              >
                End active session
              </Button>
            </Flex>
          </Callout.Root>
        )}
        <SessionView current={current} interactive={interactive} />
      </main>
    </div>
  );
}
