import { useCallback, useEffect, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  Badge,
  Button,
  Callout,
  Dialog,
  Flex,
  IconButton,
  ScrollArea,
  SegmentedControl,
  Text,
  TextField,
} from "@radix-ui/themes";
import {
  ArrowUp,
  Check,
  ChevronRight,
  FileBox,
  FileText,
  Folder,
  LoaderCircle,
  Search,
} from "lucide-react";
import { api, apiError } from "../api/client";
import type { Domain, ResourceKind, RegisteredResource } from "../api/client";

const resourceOptions = {
  model: {
    title: "Load a policy checkpoint",
    tabLabel: "Policy weights",
    pathPlaceholder: "Folder or checkpoint path",
    selectionPlaceholder: "Choose a checkpoint above.",
    selectionLabel: "Select checkpoint",
    help: "Select a policy checkpoint (.ckpt, .pt, .pth).",
    action: "Load policy",
    progress: (device: string) => `Loading policy on ${device}…`,
    icon: FileBox,
    suffixes: [".ckpt", ".pt", ".pth"],
    selectDirectory: false,
    requiresSplit: false,
  },
  dataset: {
    title: "Open a validation dataset",
    tabLabel: "Validation data",
    pathPlaceholder: "Dataset or collection folder",
    selectionPlaceholder: "Choose a dataset folder above.",
    selectionLabel: "Select dataset",
    help: "Open a native dataset or collection folder to select it, then choose its split.",
    action: "Open dataset",
    progress: () => "Checking dataset and reading its first sample…",
    icon: Folder,
    suffixes: [],
    selectDirectory: true,
    requiresSplit: true,
  },
  benchmark: {
    title: "Open an online benchmark",
    tabLabel: "Online benchmark",
    pathPlaceholder: "Folder or benchmark configuration path",
    selectionPlaceholder: "Choose a benchmark configuration above.",
    selectionLabel: "Select benchmark",
    help: "Select an online benchmark configuration (.yaml, .yml, .json).",
    action: "Open benchmark",
    progress: () => "Checking benchmark configuration…",
    icon: FileText,
    suffixes: [".yaml", ".yml", ".json"],
    selectDirectory: false,
    requiresSplit: false,
  },
} satisfies Record<
  ResourceKind,
  {
    title: string;
    tabLabel: string;
    pathPlaceholder: string;
    selectionPlaceholder: string;
    selectionLabel: string;
    help: string;
    action: string;
    progress: (device: string) => string;
    icon: typeof FileBox;
    suffixes: string[];
    selectDirectory: boolean;
    requiresSplit: boolean;
  }
>;

interface Props {
  domain: Domain;
  kind: ResourceKind | null;
  device: string;
  onKindChange(kind: ResourceKind | null): void;
  onLoaded(resource: RegisteredResource): Promise<void>;
}

/** Browse local resources within the selected policy domain. */
export function ResourcePicker({
  domain,
  kind,
  device,
  onKindChange,
  onLoaded,
}: Props) {
  const [recentDirectories, setRecentDirectories] = useState<
    Record<string, string>
  >({});
  const rememberDirectory = useCallback(
    (resourceKind: ResourceKind, path: string) => {
      const key = `${domain.id}:${resourceKind}`;
      setRecentDirectories((previous) =>
        previous[key] === path ? previous : { ...previous, [key]: path },
      );
    },
    [domain.id],
  );
  return (
    <Dialog.Root
      open={kind !== null}
      onOpenChange={(open) => {
        if (!open) onKindChange(null);
      }}
    >
      {kind && (
        <PickerContents
          key={`${domain.id}:${kind}`}
          domain={domain}
          kind={kind}
          device={device}
          initialDirectory={recentDirectories[`${domain.id}:${kind}`]}
          onDirectoryChange={rememberDirectory}
          onKindChange={onKindChange}
          onLoaded={onLoaded}
        />
      )}
    </Dialog.Root>
  );
}

function PickerContents({
  domain,
  kind,
  device,
  onKindChange,
  onLoaded,
  initialDirectory,
  onDirectoryChange,
}: Props & {
  kind: ResourceKind;
  initialDirectory?: string;
  onDirectoryChange(kind: ResourceKind, path: string): void;
}) {
  const options = resourceOptions[kind];
  const ActionIcon = options.icon;
  const availableKinds: ResourceKind[] =
    domain.resource_kinds ??
    (domain.id === "ego2d"
      ? ["model", "dataset", "benchmark"]
      : ["model", "dataset"]);
  const acceptsFile = (path: string) =>
    options.suffixes.some((suffix) => path.toLowerCase().endsWith(suffix));
  const [directory, setDirectory] = useState<string | undefined>(
    initialDirectory,
  );
  const [location, setLocation] = useState("");
  const [selectedPath, setSelectedPath] = useState("");
  const [filter, setFilter] = useState("");
  const [label, setLabel] = useState("");
  const [split, setSplit] = useState("val");
  const browser = useQuery({
    queryKey: ["resource-browser", domain.id, kind, directory],
    refetchOnWindowFocus: false,
    retry: false,
    queryFn: async ({ signal }) => {
      const { data, error } = await api.GET("/api/v1/resources/browse", {
        params: {
          query: {
            kind,
            domain_id: domain.id,
            ...(directory === undefined ? {} : { path: directory }),
          },
        },
        signal,
      });
      if (error || !data) throw apiError(error);
      return data;
    },
  });
  const registration = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/v1/resources", {
        body: {
          kind,
          domain_id: domain.id,
          path: selectedPath,
          label: label.trim() || undefined,
          ...(options.requiresSplit ? { split: split.trim() } : {}),
        },
      });
      if (error || !data) throw apiError(error);
      return data;
    },
    onSuccess: async (resource) => {
      await onLoaded(resource);
      onKindChange(null);
    },
  });
  const busy = registration.isPending;
  useEffect(() => {
    const data = browser.data;
    if (!data) return;
    const selectedFile = data.selected_path;
    setLocation(data.cwd);
    setSelectedPath(
      selectedFile &&
        options.suffixes.some((suffix) =>
          selectedFile.toLowerCase().endsWith(suffix),
        )
        ? selectedFile
        : options.selectDirectory && data.selectable
          ? data.cwd
          : "",
    );
    setFilter("");
    onDirectoryChange(kind, data.cwd);
  }, [browser.data, browser.dataUpdatedAt, kind, onDirectoryChange, options]);
  const navigate = (path: string) => {
    setDirectory(path);
    setLocation(path);
    setSelectedPath("");
    registration.reset();
    // Reopening the current location also refreshes files created since the last visit.
    if (path === directory) void browser.refetch();
  };
  const entries =
    browser.data?.entries.filter(
      (entry) =>
        entry.name.toLowerCase().includes(filter.toLowerCase()) &&
        (entry.is_dir || acceptsFile(entry.path)),
    ) ?? [];
  const error = registration.error ?? browser.error;
  return (
    <Dialog.Content
      maxWidth="760px"
      className="resource-picker"
      onEscapeKeyDown={(event) => {
        if (busy) event.preventDefault();
      }}
      onPointerDownOutside={(event) => {
        if (busy) event.preventDefault();
      }}
    >
      <Dialog.Title>{options.title}</Dialog.Title>
      <Dialog.Description size="2">
        Browse the machine running Studio. Files are read in place.
      </Dialog.Description>
      <Flex justify="between" align="center" gap="3" mt="4" wrap="wrap">
        <SegmentedControl.Root
          value={kind}
          disabled={busy}
          aria-label="Resource type"
          onValueChange={(value) => onKindChange(value as ResourceKind)}
        >
          {availableKinds.map((resourceKind) => (
            <SegmentedControl.Item key={resourceKind} value={resourceKind}>
              {resourceOptions[resourceKind].tabLabel}
            </SegmentedControl.Item>
          ))}
        </SegmentedControl.Root>
        <Flex gap="2">
          <Badge color="gray">{domain.title}</Badge>
          <Badge color="gray">{device}</Badge>
        </Flex>
      </Flex>
      <Flex gap="2" mt="4" wrap="wrap" aria-label="Quick locations">
        {browser.data?.shortcuts.map((shortcut) => (
          <Button
            key={shortcut.id}
            variant="soft"
            color="gray"
            size="1"
            disabled={busy}
            onClick={() => navigate(shortcut.path)}
          >
            {shortcut.label}
          </Button>
        ))}
      </Flex>
      <form
        className="resource-location"
        onSubmit={(event) => {
          event.preventDefault();
          if (!busy && location.trim()) navigate(location.trim());
        }}
      >
        <IconButton
          type="button"
          variant="soft"
          color="gray"
          aria-label="Parent folder"
          disabled={busy || !browser.data?.parent}
          onClick={() => {
            if (browser.data?.parent) navigate(browser.data.parent);
          }}
        >
          <ArrowUp size={16} />
        </IconButton>
        <TextField.Root
          aria-label="Resource path"
          placeholder={options.pathPlaceholder}
          value={location}
          disabled={busy}
          onChange={(event) => setLocation(event.target.value)}
        />
        <Button
          type="submit"
          variant="soft"
          disabled={busy || !location.trim()}
        >
          Go
        </Button>
      </form>
      {error && (
        <Callout.Root color="red" role="alert" mt="3">
          <Callout.Text>{error.message}</Callout.Text>
        </Callout.Root>
      )}
      <div className="resource-files" aria-busy={browser.isFetching}>
        <TextField.Root
          aria-label="Filter files and folders"
          placeholder="Filter this folder…"
          variant="soft"
          value={filter}
          disabled={busy}
          onChange={(event) => setFilter(event.target.value)}
        >
          <TextField.Slot>
            <Search size={14} />
          </TextField.Slot>
        </TextField.Root>
        <ScrollArea
          type="auto"
          scrollbars="vertical"
          style={{ height: "260px" }}
        >
          {browser.isFetching ? (
            <div className="resource-files-message" role="status">
              <LoaderCircle className="spin" size={18} />
              Opening folder…
            </div>
          ) : entries.length ? (
            entries.map((entry) => (
              <button
                key={`${entry.name}:${entry.path}`}
                className={`resource-entry ${selectedPath === entry.path ? "selected" : ""}`}
                disabled={busy || (!entry.is_dir && !entry.selectable)}
                aria-label={`${entry.is_dir ? "Open folder" : options.selectionLabel} ${entry.name}`}
                aria-pressed={
                  entry.is_dir ? undefined : selectedPath === entry.path
                }
                onClick={() => {
                  if (entry.is_dir) navigate(entry.path);
                  else {
                    setSelectedPath(entry.path);
                    registration.reset();
                  }
                }}
              >
                {entry.is_dir ? <Folder size={17} /> : <ActionIcon size={17} />}
                <span title={entry.name}>{entry.name}</span>
                {options.selectDirectory &&
                  entry.is_dir &&
                  entry.selectable && <Badge size="1">Dataset</Badge>}
                {entry.is_dir ? (
                  <ChevronRight size={14} />
                ) : selectedPath === entry.path ? (
                  <Check size={16} />
                ) : null}
              </button>
            ))
          ) : (
            <div className="resource-files-message">
              {browser.error
                ? "Enter another path to continue."
                : "No matching files or folders."}
            </div>
          )}
        </ScrollArea>
      </div>
      <Text size="1" color="gray" as="p" mt="2">
        {options.help}
      </Text>
      <Flex gap="3" mt="4" className="resource-options">
        <label>
          <Text size="1" color="gray" as="div" mb="1">
            Display name · optional
          </Text>
          <TextField.Root
            aria-label="Resource display name"
            placeholder="Use the file or folder name"
            value={label}
            disabled={busy}
            onChange={(event) => setLabel(event.target.value)}
          />
        </label>
        {options.requiresSplit && (
          <label className="resource-split">
            <Text size="1" color="gray" as="div" mb="1">
              Split
            </Text>
            <TextField.Root
              aria-label="Dataset split"
              placeholder="val"
              value={split}
              disabled={busy}
              onChange={(event) => {
                setSplit(event.target.value);
                registration.reset();
              }}
            />
          </label>
        )}
      </Flex>
      <div className="resource-selection">
        <Text size="1" color="gray">
          {selectedPath ? "SELECTED" : "SELECTION"}
        </Text>
        <Text as="p" size="2">
          {selectedPath || options.selectionPlaceholder}
        </Text>
      </div>
      <Flex gap="3" justify="end" align="center" wrap="wrap">
        {busy && (
          <Text size="1" color="gray" role="status">
            {options.progress(device)}
          </Text>
        )}
        <Dialog.Close>
          <Button variant="soft" color="gray" disabled={busy}>
            Done
          </Button>
        </Dialog.Close>
        <Button
          disabled={
            busy ||
            browser.isFetching ||
            !selectedPath ||
            (options.requiresSplit && !split.trim())
          }
          onClick={() => registration.mutate()}
        >
          {busy ? (
            <LoaderCircle className="spin" size={16} />
          ) : (
            <ActionIcon size={16} />
          )}
          {options.action}
        </Button>
      </Flex>
    </Dialog.Content>
  );
}
