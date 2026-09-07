import type { Catalog, Domain, Mode, ResourceKind } from "../api/client";
export interface ModeProps {
  catalog: Catalog;
  domain: Domain;
  mode: Mode;
  modelId: string;
  modelRevision: number;
  datasetId: string;
  benchmarkId: string;
  onDatasetChange(id: string): void;
  onBenchmarkChange(id: string): void;
  onOpenResource(kind: ResourceKind): void;
  onNavigate(task: string): void;
}
