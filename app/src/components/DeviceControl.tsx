import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertDialog, Badge, Button, Flex, Text } from "@radix-ui/themes";
import { Cpu, LoaderCircle } from "lucide-react";
import { api, apiError } from "../api/client";
import type { components } from "../api/schema";

const DEVICE_KEY = ["runtime-device"];
const DISMISSED_KEY = ["runtime-device-dismissed-failures"];

/** A CPU fallback is a user choice. Changing devices never retries an operation. */
export function DeviceControl({ fallbackDevice }: { fallbackDevice: string }) {
  const client = useQueryClient();
  const [open, setOpen] = useState(false);
  const [notice, setNotice] = useState("");
  const [dismissed, setDismissed] = useState<string[]>(
    () => client.getQueryData<string[]>(DISMISSED_KEY) ?? [],
  );
  const runtime = useQuery<components["schemas"]["RuntimeDevice"]>({
    queryKey: DEVICE_KEY,
    retry: false,
    refetchInterval: (query) => (query.state.error ? false : 2000),
    queryFn: async ({ signal }) => {
      const { data, error } = await api.GET("/api/v1/runtime/device", {
        signal,
      });
      if (error || !data) throw apiError(error);
      return data;
    },
  });
  const current = runtime.data;
  const device = current?.device ?? fallbackDevice;
  const gpu = device.startsWith("cuda");
  const failure = current?.failure;
  const dismiss = () => {
    if (failure && !dismissed.includes(failure.id)) {
      const next = [...dismissed, failure.id].slice(-64);
      setDismissed(next);
      client.setQueryData(DISMISSED_KEY, next);
    }
    setOpen(false);
  };
  useEffect(() => {
    if (failure && !dismissed.includes(failure.id)) setOpen(true);
  }, [failure, dismissed]);

  const change = useMutation({
    mutationFn: async (nextDevice: string) => {
      const { data, error } = await api.POST("/api/v1/runtime/device", {
        body: { device: nextDevice },
      });
      if (error || !data) throw apiError(error);
      return data;
    },
    onSuccess: (value) => {
      dismiss();
      client.setQueryData(DEVICE_KEY, value);
      void client.invalidateQueries({ queryKey: ["catalog"] });
      setNotice(
        `${value.device === "cpu" ? "CPU" : "GPU"} selected. Run the operation again.`,
      );
    },
    onError: () => {
      // A failed GPU switch retains the current device and reports its target failure.
      void client.invalidateQueries({ queryKey: DEVICE_KEY });
    },
  });
  const canChange = !!current?.can_change && !change.isPending;
  const useCPU = () => {
    change.reset();
    if (device === "cpu") {
      dismiss();
      setNotice("CPU remains selected. Run the operation again.");
    } else {
      change.mutate("cpu");
    }
  };

  return (
    <div className="device-control" aria-label="Inference device">
      <Flex gap="2" align="center" wrap="wrap">
        <Badge
          color={gpu ? "teal" : "gray"}
          title={current?.gpu_name ?? undefined}
        >
          <Cpu size={12} />
          {gpu ? `GPU · ${device}` : device.toUpperCase()}
        </Badge>
        <Button
          size="1"
          variant="ghost"
          color="gray"
          disabled={!canChange}
          onClick={() => {
            change.reset();
            setNotice("");
            if (gpu) setOpen(true);
            else change.mutate("cuda:0");
          }}
        >
          {change.isPending && <LoaderCircle className="spin" size={12} />}
          {gpu ? "Review device" : "Use GPU"}
        </Button>
      </Flex>
      {current && !current.can_change ? (
        <Text as="div" size="1" color="gray">
          Finish the current operation or end the active session to change
          device.
        </Text>
      ) : runtime.isError ? (
        <Text as="div" size="1" color="gray">
          Device status is unavailable.
        </Text>
      ) : gpu && current && !current.gpu_available ? (
        <Text as="div" size="1" color="amber">
          GPU is unavailable.
        </Text>
      ) : null}
      {notice && (
        <Text as="div" size="1" color="gray" role="status">
          {notice}
        </Text>
      )}
      {change.error && !open && (
        <Text as="div" size="1" color="red" role="alert">
          {change.error.message}
        </Text>
      )}
      <AlertDialog.Root
        open={open}
        onOpenChange={(value) => {
          if (!value) dismiss();
        }}
      >
        <AlertDialog.Content maxWidth="480px">
          <AlertDialog.Title>
            {failure
              ? "GPU execution is unavailable"
              : "Use CPU for inference?"}
          </AlertDialog.Title>
          <AlertDialog.Description>
            {failure
              ? `${failure.device}: ${failure.message}`
              : `Studio is using ${device}. You can explicitly switch inference to CPU.`}
          </AlertDialog.Description>
          <Text as="p" size="2" mt="3">
            {gpu
              ? "GPU remains selected until you confirm. CPU inference may be slower."
              : "CPU is still selected; the GPU switch did not succeed."}{" "}
            After changing devices, run your operation again.
          </Text>
          {current && !current.can_change && (
            <Text as="p" size="2" color="amber">
              Finish the current operation or end the active session before
              changing device.
            </Text>
          )}
          {change.error && (
            <Text as="p" size="2" color="red" role="alert">
              {change.error.message}
            </Text>
          )}
          <Flex gap="3" justify="end" mt="4">
            <AlertDialog.Cancel>
              <Button variant="soft" color="gray" disabled={change.isPending}>
                {gpu ? "Keep GPU" : "Keep current device"}
              </Button>
            </AlertDialog.Cancel>
            <Button disabled={!canChange} onClick={useCPU}>
              {change.isPending && <LoaderCircle className="spin" size={14} />}
              Use CPU
            </Button>
          </Flex>
        </AlertDialog.Content>
      </AlertDialog.Root>
    </div>
  );
}
