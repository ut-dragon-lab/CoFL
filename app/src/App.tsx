import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Badge, Button, Callout, Flex, Select, Text } from "@radix-ui/themes";
import {
  ArrowUpRight,
  BookOpen,
  Box,
  Database,
  FolderOpen,
  FlaskConical,
  Layers3,
  LoaderCircle,
  Navigation,
  MessagesSquare,
  Orbit,
  Settings2,
} from "lucide-react";
import { api, apiError } from "./api/client";
import type { Domain, RegisteredResource, ResourceKind } from "./api/client";
import { modeRenderers } from "./modes/registry";
import { ResourcePicker } from "./components/ResourcePicker";
import { DeviceControl } from "./components/DeviceControl";
import {
  ShutdownControl,
  ShutdownScreen,
  useStudioShutdown,
} from "./components/ShutdownControl";

interface WorkspaceSelection {
  task?: string;
  model?: string;
  dataset?: string;
  benchmark?: string;
  modelRevision?: number;
}
interface DomainResource {
  domain_id?: string | null;
  method?: string | null;
  profile?: string | null;
}
function compatible(resource: DomainResource, domain: Domain) {
  if (resource.domain_id && resource.domain_id !== domain.id) return false;
  if (resource.method && !domain.methods.includes(resource.method))
    return false;
  return !resource.profile || domain.profiles.includes(resource.profile);
}
function PerspectiveIcon({ id, size = 20 }: { id: string; size?: number }) {
  return id === "bev" ? (
    <Layers3 size={size} />
  ) : id === "ego2d" ? (
    <Navigation size={size} />
  ) : (
    <Orbit size={size} />
  );
}

export function App() {
  const shutdown = useStudioShutdown();
  return shutdown.state ? (
    <ShutdownScreen
      state={shutdown.state}
      onRetry={() => void shutdown.request()}
    />
  ) : (
    <StudioWorkspace onShutdown={() => void shutdown.request()} />
  );
}

function StudioWorkspace({ onShutdown }: { onShutdown: () => void }) {
  const [domainId, setDomainId] = useState<string | null>(null);
  const [selections, setSelections] = useState<
    Record<string, WorkspaceSelection>
  >({});
  const [resourceKind, setResourceKind] = useState<ResourceKind | null>(null);
  const [connecting, setConnecting] = useState(false);
  const [resourceError, setResourceError] = useState("");
  const queryClient = useQueryClient();
  const {
    data: catalog,
    error,
    isPending,
  } = useQuery({
    queryKey: ["catalog"],
    queryFn: async ({ signal }) => {
      const { data, error } = await api.GET("/api/v1/catalog", { signal });
      if (error || !data) throw apiError(error);
      return data;
    },
  });
  const domain = catalog?.domains.find((entry) => entry.id === domainId);
  const selection = domain ? (selections[domain.id] ?? {}) : {};
  const modes =
    catalog?.modes.filter((entry) => entry.domain_id === domain?.id) ?? [];
  const mode =
    modes.find((entry) => entry.task === selection.task) ??
    modes.find((entry) => entry.task === "interactive") ??
    modes.find((entry) => entry.task === "playground") ??
    modes[0];
  const View = mode && modeRenderers[mode.renderer];
  const scoped =
    catalog && domain
      ? {
          ...catalog,
          models: catalog.models.filter((entry) => compatible(entry, domain)),
          datasets: catalog.datasets.filter((entry) =>
            compatible(entry, domain),
          ),
          benchmarks: (domain.resource_kinds ?? []).includes("benchmark")
            ? catalog.benchmarks.filter((entry) => compatible(entry, domain))
            : [],
        }
      : undefined;
  const modelId =
    scoped?.models.find((entry) => entry.id === selection.model && entry.method)
      ?.id ??
    scoped?.models.find(
      (entry) => entry.domain_id === domain?.id && entry.method,
    )?.id ??
    "";
  const datasetId =
    scoped?.datasets.find(
      (entry) => entry.id === selection.dataset && entry.profile,
    )?.id ??
    scoped?.datasets.find(
      (entry) => entry.domain_id === domain?.id && entry.profile,
    )?.id ??
    "";
  const benchmarkId =
    scoped?.benchmarks.find((entry) => entry.id === selection.benchmark)?.id ??
    scoped?.benchmarks[0]?.id ??
    "";
  const datasetSplit =
    scoped?.datasets.find((entry) => entry.id === datasetId)?.split ?? "val";

  const updateSelection = (id: string, values: Partial<WorkspaceSelection>) => {
    setSelections((current) => ({
      ...current,
      [id]: { ...current[id], ...values },
    }));
  };
  const enterDomain = (id: string) => {
    setDomainId(id);
    setResourceError("");
  };
  const resourceLoaded = async (resource: RegisteredResource) => {
    await queryClient.invalidateQueries({ queryKey: ["catalog"] });
    const owner = resource.domain_id ?? domainId;
    if (!owner) return;
    setSelections((current) => ({
      ...current,
      [owner]: {
        ...current[owner],
        [resource.kind]: resource.resource_id,
        ...(resource.kind === "model"
          ? { modelRevision: (current[owner]?.modelRevision ?? 0) + 1 }
          : {
              task:
                resource.kind === "dataset"
                  ? "validation"
                  : (catalog?.modes.find(
                      (item) =>
                        item.domain_id === owner && item.task === "benchmark",
                    )?.task ?? "playground"),
            }),
      },
    }));
    await queryClient.invalidateQueries({
      queryKey: [
        resource.kind === "benchmark" ? "online-benchmarks" : "sample",
      ],
    });
  };
  const selectResource = async (kind: ResourceKind, id: string) => {
    if (!domain || !scoped) return;
    const entries =
      kind === "model"
        ? scoped.models
        : kind === "dataset"
          ? scoped.datasets
          : scoped.benchmarks;
    const entry = entries.find((value) => value.id === id);
    const verified =
      entry?.domain_id === domain.id &&
      (kind === "benchmark" ||
        (kind === "model"
          ? "method" in entry && entry.method
          : "profile" in entry && entry.profile));
    if (verified) {
      updateSelection(domain.id, { [kind]: id });
      return;
    }
    setConnecting(true);
    setResourceError("");
    try {
      const { data, error } = await api.POST(
        "/api/v1/resources/{kind}/{resource_id}/activate",
        {
          params: { path: { kind, resource_id: id } },
          body: { domain_id: domain.id },
        },
      );
      if (error || !data) throw apiError(error);
      await resourceLoaded(data);
    } catch (error) {
      setResourceError((error as Error).message);
    } finally {
      setConnecting(false);
    }
  };

  return (
    <div className="studio">
      <nav className="rail" aria-label="Navigation workspaces">
        <button
          className="brand brand-button"
          onClick={() => setDomainId(null)}
          aria-label="Choose a perspective"
        >
          <div className="brand-symbol">
            <Orbit size={26} />
          </div>
          <span>
            CoFL<small>STUDIO</small>
          </span>
        </button>
        <div className="rail-caption">PERSPECTIVE</div>
        {catalog?.domains.map((entry) => (
          <div
            className={`domain-group ${domainId === entry.id ? "active" : ""}`}
            key={entry.id}
          >
            <button
              className={`domain-link ${domainId === entry.id ? "active" : ""}`}
              onClick={() => enterDomain(entry.id)}
              aria-label={`Open ${entry.title} workspace`}
              aria-expanded={domainId === entry.id}
            >
              <PerspectiveIcon id={entry.id} size={18} />
              <span>
                {entry.title}
                <small>{entry.methods.join(" / ").toUpperCase()}</small>
              </span>
            </button>
            {domainId === entry.id && (
              <div className="domain-activities">
                {modes.map((item) => (
                  <button
                    key={item.id}
                    className={`mode-link ${mode?.id === item.id ? "active" : ""}`}
                    onClick={() =>
                      updateSelection(entry.id, { task: item.task })
                    }
                    aria-current={mode?.id === item.id ? "page" : undefined}
                  >
                    {item.task === "validation" ? (
                      <Database size={16} />
                    ) : item.task === "benchmark" ? (
                      <FlaskConical size={16} />
                    ) : item.task === "interactive" ? (
                      <MessagesSquare size={16} />
                    ) : (
                      <Box size={16} />
                    )}
                    <span>{item.title}</span>
                    {mode?.id === item.id && <ArrowUpRight size={13} />}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
        <div className="rail-footer">
          <div className="rail-footer-copy">
            <div className="rail-rule" />
            <Text size="1">
              Continuous flow fields.
              <br />A space to explore.
            </Text>
            <Badge variant="outline" color="gray" mt="3">
              LOCAL WORKSPACE
            </Badge>
          </div>
          <ShutdownControl onShutdown={onShutdown} />
        </div>
      </nav>
      <div className="studio-body">
        <header className="app-header">
          <div>
            <div className="eyebrow">
              {domain
                ? `${domain.title.toUpperCase()} / ${domain.methods.join(" / ").toUpperCase()}`
                : "COFL / INTERACTIVE NAVIGATION"}
            </div>
            <h1>{domain ? mode?.title : "Choose your perspective"}</h1>
            <p>
              {domain
                ? mode?.description
                : "Explore the world from above or from the robot's point of view."}
            </p>
          </div>
          {domain && (
            <Flex gap="3" align="center" wrap="wrap">
              <div className="model-select">
                <Text size="1" color="gray">
                  POLICY · {domain.methods.join(" / ").toUpperCase()}
                </Text>
                <Select.Root
                  value={modelId}
                  onValueChange={(id) => void selectResource("model", id)}
                  disabled={connecting || !scoped?.models.length}
                >
                  <Select.Trigger
                    aria-label="Policy checkpoint"
                    placeholder={
                      connecting ? "Loading policy…" : "Choose a checkpoint"
                    }
                  />
                  <Select.Content>
                    {scoped?.models.map((entry) => (
                      <Select.Item key={entry.id} value={entry.id}>
                        {entry.label}
                        {entry.method ? "" : " · verify"}
                      </Select.Item>
                    ))}
                  </Select.Content>
                </Select.Root>
              </div>
              <Button
                variant="soft"
                disabled={connecting}
                onClick={() => setResourceKind("model")}
              >
                {connecting ? (
                  <LoaderCircle className="spin" size={15} />
                ) : (
                  <FolderOpen size={15} />
                )}
                Load policy
              </Button>
              <Button
                variant="outline"
                color="gray"
                onClick={() =>
                  setResourceKind(
                    mode?.task === "validation"
                      ? "dataset"
                      : mode?.execution === "session"
                        ? "benchmark"
                        : "model",
                  )
                }
              >
                <Settings2 size={15} />
                Resources
              </Button>
              <DeviceControl fallbackDevice={catalog?.device ?? "cuda:0"} />
            </Flex>
          )}
        </header>
        {error ? (
          <Callout.Root className="connection-error" color="red">
            <Callout.Text>
              Studio API is unavailable. {error.message}
            </Callout.Text>
          </Callout.Root>
        ) : isPending ? (
          <div className="empty-workspace">Opening your workspace…</div>
        ) : !domain ? (
          <main className="perspective-home">
            <div className="perspective-intro">
              <span className="eyebrow">
                ONE STUDIO. DIFFERENT WAYS TO NAVIGATE.
              </span>
              <h2>Where do you want to begin?</h2>
              <p>
                Choose a perspective, then explore its models, scenes and tasks.
              </p>
            </div>
            <div className="perspective-cards">
              {catalog?.domains.map((entry) => (
                <button
                  className="perspective-card"
                  key={entry.id}
                  onClick={() => enterDomain(entry.id)}
                  aria-label={`Choose ${entry.title}`}
                >
                  <div className={`perspective-art ${entry.id}`}>
                    <PerspectiveIcon id={entry.id} size={72} />
                    <div className="perspective-orbit" />
                  </div>
                  <div className="perspective-card-copy">
                    <Badge color="teal">
                      {entry.methods.join(" / ").toUpperCase()}
                    </Badge>
                    <h2>
                      {entry.title}
                      <ArrowUpRight size={23} />
                    </h2>
                    <p>{entry.description}</p>
                    <div className="perspective-task-list">
                      {catalog.modes
                        .filter((item) => item.domain_id === entry.id)
                        .map((item) => (
                          <span key={item.id}>{item.title}</span>
                        ))}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </main>
        ) : (
          <>
            {resourceError && (
              <Callout.Root
                className="connection-error"
                color="red"
                role="alert"
              >
                <Callout.Text>{resourceError}</Callout.Text>
              </Callout.Root>
            )}
            {scoped && mode && View ? (
              <View
                key={
                  mode.task === "validation"
                    ? `${mode.id}:${datasetId}:${datasetSplit}`
                    : mode.id
                }
                catalog={scoped}
                domain={domain}
                mode={mode}
                modelId={modelId}
                modelRevision={selection.modelRevision ?? 0}
                datasetId={datasetId}
                benchmarkId={benchmarkId}
                onDatasetChange={(id) => void selectResource("dataset", id)}
                onBenchmarkChange={(id) => void selectResource("benchmark", id)}
                onOpenResource={setResourceKind}
                onNavigate={(task) => updateSelection(domain.id, { task })}
              />
            ) : (
              <div className="empty-workspace">
                <BookOpen />
                <h2>Renderer not installed</h2>
                <p>This workspace needs its frontend view to be registered.</p>
              </div>
            )}
          </>
        )}
      </div>
      {domain && (
        <ResourcePicker
          kind={resourceKind}
          domain={domain}
          device={catalog?.device ?? "cuda:0"}
          onKindChange={setResourceKind}
          onLoaded={resourceLoaded}
        />
      )}
    </div>
  );
}
