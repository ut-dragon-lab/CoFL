import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, apiError } from "../../api/client";
import type { OnlineSession, OnlineStep } from "../../api/client";
import type { components } from "../../api/schema";

export type SessionRequest = components["schemas"]["OnlineSessionRequest"];
export type ResetRequest = components["schemas"]["OnlineReset"];
export const isActive = (session?: OnlineSession | null) =>
  !!session &&
  ["starting", "running", "waiting", "paused", "stopping"].includes(
    session.status,
  );

/** One backend session owns the simulator; every view connects to that session. */
export function useOnlineSession(domainId: string) {
  const client = useQueryClient();
  const idKey = ["online-session-id", domainId];
  const [sessionId, setSessionId] = useState<string | null>(
    () => client.getQueryData(idKey) ?? null,
  );
  const cursor = useRef(-1);
  const latest = useRef<OnlineStep | undefined>(undefined);
  const active = useQuery<OnlineSession | null>({
    queryKey: ["online-active-session", domainId],
    enabled: !sessionId,
    retry: false,
    staleTime: 0,
    queryFn: async ({ signal }) => {
      const { data, error } = await api.GET(
        "/api/v1/domains/ego2d/sessions/active",
        { signal },
      );
      if (error) throw apiError(error);
      return (data as OnlineSession | null) ?? null;
    },
  });
  const remember = (value: OnlineSession) => {
    cursor.current = -1;
    latest.current = undefined;
    client.setQueryData(["online-session", value.id], value);
    client.setQueryData(idKey, value.id);
    setSessionId(value.id);
  };
  useEffect(() => {
    if (!sessionId && active.data) {
      client.setQueryData(["online-session", active.data.id], active.data);
      client.setQueryData(["online-session-id", domainId], active.data.id);
      setSessionId(active.data.id);
    }
  }, [active.data, client, domainId, sessionId]);
  const session = useQuery<OnlineSession>({
    queryKey: ["online-session", sessionId],
    enabled: !!sessionId,
    retry: false,
    refetchInterval: (query) =>
      !query.state.error && (!query.state.data || isActive(query.state.data))
        ? 350
        : false,
    queryFn: async ({ signal }) => {
      const { data, error } = await api.GET(
        "/api/v1/domains/ego2d/sessions/{session_id}",
        {
          params: {
            path: { session_id: sessionId! },
            query: { after_step: cursor.current },
          },
          signal,
        },
      );
      if (error || !data) throw apiError(error);
      const value = data as OnlineSession;
      const frame = value.steps?.at(-1);
      if (frame) {
        latest.current = frame;
        cursor.current = frame.step_index;
      }
      // Keep one prediction in browser memory. Epochs isolate pose resets.
      const valid =
        latest.current && (latest.current.epoch ?? 0) === (value.epoch ?? 0);
      return { ...value, steps: valid ? [latest.current!] : [] };
    },
  });
  const refresh = () =>
    client.invalidateQueries({ queryKey: ["online-session", sessionId] });
  const start = useMutation({
    mutationFn: async (body: SessionRequest) => {
      const { data, error } = await api.POST("/api/v1/domains/ego2d/sessions", {
        body,
      });
      if (error || !data) throw apiError(error);
      return data as OnlineSession;
    },
    onSuccess: remember,
  });
  const action = useMutation({
    mutationFn: async (name: "stop" | "pause" | "resume") => {
      const paths = {
        stop: "/api/v1/domains/ego2d/sessions/{session_id}/stop",
        pause: "/api/v1/domains/ego2d/sessions/{session_id}/pause",
        resume: "/api/v1/domains/ego2d/sessions/{session_id}/resume",
      } as const;
      const { data, error } = await api.POST(paths[name], {
        params: { path: { session_id: sessionId! } },
      });
      if (error || !data) throw apiError(error);
      return data as OnlineSession;
    },
    onSuccess: refresh,
  });
  const command = useMutation({
    mutationFn: async (instruction: string) => {
      const { data, error } = await api.POST(
        "/api/v1/domains/ego2d/sessions/{session_id}/commands",
        {
          params: { path: { session_id: sessionId! } },
          body: { instruction },
        },
      );
      if (error || !data) throw apiError(error);
      return data as OnlineSession;
    },
    onSuccess: refresh,
  });
  const reset = useMutation({
    mutationFn: async (body: ResetRequest) => {
      const { data, error } = await api.POST(
        "/api/v1/domains/ego2d/sessions/{session_id}/reset",
        {
          params: { path: { session_id: sessionId! } },
          body,
        },
      );
      if (error || !data) throw apiError(error);
      return data as OnlineSession;
    },
    onSuccess: refresh,
  });
  return {
    current: session.data,
    recovering: !sessionId && active.isPending,
    start,
    action,
    command,
    reset,
    error:
      start.error ??
      action.error ??
      command.error ??
      reset.error ??
      session.error ??
      active.error,
    clearErrors: () => {
      start.reset();
      action.reset();
      command.reset();
      reset.reset();
    },
  };
}
