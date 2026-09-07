import { useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Button, Callout, Text } from "@radix-ui/themes";
import { Check, LoaderCircle, Power } from "lucide-react";
import { api, apiError } from "../api/client";

type ShutdownState =
  | { phase: "closing" }
  | { phase: "closed"; serverStopping: boolean }
  | { phase: "failed"; message: string };

/** Event-driven shutdown: no unload handler, mount effect or automatic retry. */
export function useStudioShutdown() {
  const client = useQueryClient();
  const [state, setState] = useState<ShutdownState | null>(null);
  const pending = useRef(false);
  const request = async () => {
    if (pending.current) return;
    pending.current = true;
    setState({ phase: "closing" });
    try {
      await client.cancelQueries();
      const { data, error } = await api.POST("/api/v1/runtime/shutdown", {
        keepalive: true,
      });
      if (error || !data || data.status !== "closed") throw apiError(error);
      client.clear();
      setState({ phase: "closed", serverStopping: data.server_stopping });
    } catch (error) {
      setState({
        phase: "failed",
        message: error instanceof Error ? error.message : String(error),
      });
    } finally {
      pending.current = false;
    }
  };
  return { state, request };
}

export function ShutdownControl({ onShutdown }: { onShutdown: () => void }) {
  return (
    <button
      className="shutdown-button"
      onClick={onShutdown}
      title="End sessions and unload models and datasets"
      aria-label="Shut down Studio"
    >
      <Power size={16} />
      <span>Shut down Studio</span>
    </button>
  );
}

export function ShutdownScreen({
  state,
  onRetry,
}: {
  state: ShutdownState;
  onRetry: () => void;
}) {
  return (
    <main className="shutdown-screen">
      <section className="shutdown-card" aria-live="polite">
        <div className="shutdown-symbol">
          {state.phase === "closing" ? (
            <LoaderCircle size={32} className="spin" />
          ) : state.phase === "closed" ? (
            <Check size={32} />
          ) : (
            <Power size={32} />
          )}
        </div>
        <span className="eyebrow">COFL STUDIO</span>
        <h1>
          {state.phase === "closing"
            ? "Closing Studio…"
            : state.phase === "closed"
              ? "Studio closed"
              : "Shutdown not confirmed"}
        </h1>
        {state.phase === "closing" ? (
          <Text as="p" size="2" color="gray">
            Ending sessions and unloading models and datasets. Please wait for
            the server to finish.
          </Text>
        ) : state.phase === "closed" ? (
          <>
            <Text as="p" size="2" color="gray">
              Sessions have ended. Models and datasets have been unloaded. You
              can close this tab.
            </Text>
            <Text as="p" size="1" color="gray">
              {state.serverStopping
                ? "The Studio server is stopping."
                : "The host server remains available; Studio is closed."}
            </Text>
          </>
        ) : (
          <>
            <Callout.Root color="red" role="alert" mt="4">
              <Callout.Text>{state.message}</Callout.Text>
            </Callout.Root>
            <Text as="p" size="2" color="gray">
              The server has not confirmed that its resources were released.
              Workspace requests remain stopped.
            </Text>
            <Button onClick={onRetry} mt="3">
              Retry shutdown
            </Button>
          </>
        )}
      </section>
    </main>
  );
}
