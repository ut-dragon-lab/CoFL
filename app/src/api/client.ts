import createClient from "openapi-fetch";
import type { components, paths } from "./schema";

export const api = createClient<paths>();
export type Catalog = components["schemas"]["Catalog"];
export type Prediction = components["schemas"]["PredictionResult"];
export type PredictionRequest = components["schemas"]["PredictionRequest"];
export type DatasetView = components["schemas"]["DatasetView"];
export type Floor = components["schemas"]["FloorSpec"];
export type Mode = components["schemas"]["ModeDescriptor"];
export type Domain = components["schemas"]["DomainDescriptor"];
export type OnlineSession = components["schemas"]["OnlineSession"];
export type OnlineStep = components["schemas"]["OnlineStep"];
export type Point2 = [number, number];
export type ResourceKind =
  paths["/api/v1/resources"]["post"]["requestBody"]["content"]["application/json"]["kind"];
export type RegisteredResource =
  paths["/api/v1/resources"]["post"]["responses"][200]["content"]["application/json"];

export function apiError(error: unknown): Error {
  const detail = (error as { detail?: unknown })?.detail;
  if (typeof detail === "string") return new Error(detail);
  if (Array.isArray(detail))
    return new Error(detail.map((e) => e.msg).join("; "));
  return new Error(
    "The request could not be completed. Check that Studio is running.",
  );
}

export async function predict(body: PredictionRequest, signal?: AbortSignal) {
  const { data, error } = await api.POST("/api/v1/predict", { body, signal });
  if (error || !data) throw apiError(error);
  // openapi-fetch's JSON transport widens fixed-length tuples into arrays.
  // The response is validated against this generated contract by FastAPI.
  return data as Prediction;
}
