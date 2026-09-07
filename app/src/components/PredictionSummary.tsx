import { Badge, Button, Flex, Text } from "@radix-ui/themes";
import { Download, Route } from "lucide-react";
import type { Prediction } from "../api/client";

export function PredictionSummary({ result }: { result: Prediction | null }) {
  if (!result)
    return (
      <div className="result-placeholder">
        <Route size={20} />
        <Text size="2" color="gray">
          Your predicted path will appear here.
        </Text>
      </div>
    );
  const metric = result.metrics as Record<string, unknown>;
  return (
    <div className="prediction-summary">
      <Flex justify="between" align="center">
        <Text size="2" weight="medium">
          Prediction
        </Text>
        <Badge color="teal">{result.method.toUpperCase()}</Badge>
      </Flex>
      <div className="metrics-grid">
        <div>
          <strong>
            {Math.round(result.elapsed_ms)}
            <small> ms</small>
          </strong>
          <span>Inference</span>
        </div>
        <div>
          <strong>{result.trajectory.length}</strong>
          <span>Path points</span>
        </div>
      </div>
      <Text as="p" size="1" color="gray">
        {result.stop_reason.replaceAll("_", " ")} ·{" "}
        {result.cache_hit ? "reused observation" : "new observation"}
      </Text>
      {typeof metric.angular_error_deg_mean === "number" && (
        <Text as="p" size="1">
          Mean angular error: {metric.angular_error_deg_mean.toFixed(1)}°
        </Text>
      )}
      {typeof metric.vector_l2_mean === "number" && (
        <Text as="p" size="1">
          Mean vector error: {metric.vector_l2_mean.toPrecision(3)}
        </Text>
      )}
      {typeof metric.scope === "string" && (
        <Text as="p" size="1" color="gray">
          Metrics on sampled supervised cells · {result.vector_unit}
        </Text>
      )}
      <Button
        size="1"
        variant="soft"
        color="gray"
        onClick={() => {
          const url = URL.createObjectURL(
            new Blob([JSON.stringify(result, null, 2)], {
              type: "application/json",
            }),
          );
          const a = document.createElement("a");
          a.href = url;
          a.download = `cofl-${result.observation_id}.json`;
          a.click();
          URL.revokeObjectURL(url);
        }}
      >
        <Download size={13} />
        Export prediction
      </Button>
    </div>
  );
}
