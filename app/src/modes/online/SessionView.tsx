import { Badge, Flex, Text } from "@radix-ui/themes";
import { Camera, MessageSquare } from "lucide-react";
import type { OnlineSession, OnlineStep } from "../../api/client";
import { FieldView } from "../../components/FieldView";

function WorldPath({
  session,
  frame,
}: {
  session: OnlineSession;
  frame?: OnlineStep;
}) {
  const goal = session.goal_position
    ? [session.goal_position[0], session.goal_position[2]]
    : null;
  const position = session.current_position ?? frame?.agent_position;
  const agentPoint = position ? [position[0], position[2]] : null;
  const paths = [
    session.reference_path ?? [],
    frame?.world_path ?? [],
    frame?.world_trajectory ?? [],
  ];
  const points = [
    ...paths.flat(),
    ...(goal ? [goal] : []),
    ...(agentPoint ? [agentPoint] : []),
  ];
  if (!points.length)
    return (
      <div className="online-placeholder">
        The robot's world path appears here.
      </div>
    );
  let minX = Infinity,
    minZ = Infinity,
    maxX = -Infinity,
    maxZ = -Infinity;
  for (const p of points) {
    minX = Math.min(minX, p[0] - 1);
    maxX = Math.max(maxX, p[0] + 1);
    minZ = Math.min(minZ, p[1] - 1);
    maxZ = Math.max(maxZ, p[1] + 1);
  }
  const scale = 420 / Math.max(maxX - minX, maxZ - minZ);
  const project = (p: number[]) => [
    250 + (p[0] - (minX + maxX) / 2) * scale,
    250 + (p[1] - (minZ + maxZ) / 2) * scale,
  ];
  const agent = agentPoint && project(agentPoint),
    target = goal && project(goal);
  return (
    <svg
      className="world-path"
      viewBox="0 0 500 500"
      role="img"
      aria-label="Executed and planned paths in world XZ metres"
    >
      <rect width="500" height="500" fill="#edf1eb" />
      {paths.map((path, i) => (
        <polyline
          key={i}
          points={path.map((p) => project(p).join(",")).join(" ")}
          fill="none"
          stroke={["#c58b52", "#0b8a7e", "#83b5c0"][i]}
          strokeWidth={i === 1 ? 4 : 2}
          strokeDasharray={i === 0 ? "8 6" : undefined}
          strokeLinejoin="round"
        />
      ))}
      {target && (
        <circle
          cx={target[0]}
          cy={target[1]}
          r="7"
          fill="#c58b52"
          stroke="white"
          strokeWidth="2"
        />
      )}
      {agent && (
        <g
          transform={`translate(${agent[0]},${agent[1]}) rotate(${-(session.current_yaw_deg ?? 0)})`}
        >
          <path
            d="M 0 -12 L 7 7 L 0 4 L -7 7 Z"
            fill="#0b8a7e"
            stroke="white"
            strokeWidth="2"
          />
        </g>
      )}
      <text x="20" y="480" fill="#6d8075" fontSize="12">
        World XZ · metres
      </text>
    </svg>
  );
}

function MetricValues({ metrics }: { metrics: Record<string, unknown> }) {
  const entries = Object.entries(metrics).flatMap(([key, value]) =>
    value && typeof value === "object" && !Array.isArray(value)
      ? Object.entries(value)
          .filter(([, v]) => typeof v === "number")
          .map(([k, v]) => [`${key} · ${k}`, v] as const)
      : [[key, value] as const],
  );
  return (
    <div className="online-metrics">
      {entries.map(([key, value]) => (
        <div key={key}>
          <Text size="1" color="gray">
            {key}
          </Text>
          <Text as="p" size="3" weight="medium">
            {typeof value === "number" ? value.toFixed(3) : String(value)}
          </Text>
        </div>
      ))}
    </div>
  );
}

export function SessionView({
  current,
  interactive,
}: {
  current?: OnlineSession;
  interactive: boolean;
}) {
  const frame = current?.steps?.at(-1);
  const observation = current?.observation;
  const image = observation?.image ?? frame?.image;
  const active = current?.active_instruction;
  const text =
    typeof active?.text === "string" ? active.text : frame?.instruction;
  const pending = current?.command_state?.text;
  const source =
    typeof active?.source === "string"
      ? active.source
      : interactive
        ? "user"
        : "oracle";
  const log = current?.command_log ?? [];
  const metadata = active?.metadata;
  const progress =
    metadata &&
    typeof metadata === "object" &&
    "index" in metadata &&
    "count" in metadata
      ? `Sub-instruction ${Number(metadata.index) + 1} / ${Number(metadata.count)}`
      : null;
  const completionReason = current?.command_state?.completion_reason;
  const completionLabels: Record<string, string> = {
    timeout: "Command time limit reached",
    max_steps: "Command step limit reached",
    action_head_stop: "The policy stopped",
  };
  return (
    <>
      <Flex align="center" gap="2" wrap="wrap">
        <Badge>2D egocentric</Badge>
        <Badge color="gray">
          {interactive ? "Live instructions" : "Closed-loop VLN"}
        </Badge>
        {frame && (
          <Text size="1" color="gray">
            {frame.inference_ms.toFixed(0)} ms inference ·{" "}
            {frame.linear_mps.toFixed(2)} m/s · {frame.angular_radps.toFixed(2)}{" "}
            rad/s
          </Text>
        )}
      </Flex>
      <section className="active-instruction" aria-label="Current instruction">
        <Flex align="center" gap="2">
          <MessageSquare size={16} />
          <Text size="1" weight="medium">
            CURRENT INSTRUCTION
          </Text>
          <Badge color="gray">{source}</Badge>
        </Flex>
        <Text as="p" size="3">
          {text ||
            (interactive
              ? "Open a scene and tell the robot what to do."
              : "The current oracle sub-instruction appears when the episode starts.")}
        </Text>
        {typeof pending === "string" && pending && pending !== text && (
          <Text as="p" size="1" color="gray">
            Queued: {pending}
          </Text>
        )}
        {current?.status === "waiting" && (
          <Badge color="amber">Waiting for your next command</Badge>
        )}
        {progress && (
          <Badge color="gray" mt="2">
            {progress}
          </Badge>
        )}
        {current?.status === "waiting" &&
          typeof completionReason === "string" &&
          completionLabels[completionReason] && (
            <Text as="p" size="1" color="gray">
              {completionLabels[completionReason]}
            </Text>
          )}
        {current?.status === "paused" && (
          <Badge color="amber">Motion paused</Badge>
        )}
      </section>
      <div className="online-camera comparison-panel">
        <div className="panel-heading">
          <Text size="2" weight="medium">
            Robot observation
          </Text>
          <Badge color="gray">RGB-D</Badge>
        </div>
        {image ? (
          <img src={image} alt="Current egocentric robot observation" />
        ) : (
          <div className="online-placeholder">
            <Camera size={38} />
            <h2>
              {interactive
                ? "Your words. The robot's next move."
                : "See the policy navigate"}
            </h2>
            <p>
              {interactive
                ? "Choose a scene and a CoFL-S checkpoint. Send an instruction whenever you want the robot to change course."
                : "Select a CoFL-S checkpoint and an episode to watch oracle-guided navigation."}
            </p>
          </div>
        )}
      </div>
      <div className="comparison-grid">
        <section className="comparison-panel">
          <div className="panel-heading">
            <Text size="2" weight="medium">
              Local plan
            </Text>
            <Badge color="gray">Forward / left · m</Badge>
          </div>
          {frame ? (
            <FieldView
              groundRadius={Number(current?.geometry?.r_max_m ?? 3)}
              trajectory={frame.sector_trajectory}
              start={[0, 0]}
              showField={false}
            />
          ) : (
            <div className="online-placeholder">
              The latest CoFL-S trajectory.
            </div>
          )}
        </section>
        <section className="comparison-panel">
          <div className="panel-heading">
            <Text size="2" weight="medium">
              World path
            </Text>
            <Badge color="gray">
              {interactive ? "Executed / planned" : "Executed / reference"}
            </Badge>
          </div>
          {current ? (
            <WorldPath session={current} frame={frame} />
          ) : (
            <div className="online-placeholder">
              Track the robot's movement through the scene.
            </div>
          )}
        </section>
      </div>
      {interactive && log.length > 0 && (
        <section className="command-history" aria-label="Command history">
          <Text size="2" weight="medium">
            Command history
          </Text>
          <ol>
            {log.slice(-20).map((event, index) => (
              <li key={index}>
                <Text size="1" color="gray">
                  {String(
                    event.action ?? event.event ?? event.status ?? "command",
                  )}
                </Text>
                <Text size="2">
                  {String(event.text ?? event.instruction ?? "")}
                </Text>
              </li>
            ))}
          </ol>
        </section>
      )}
      {current &&
        !interactive &&
        Object.keys(current.metrics ?? {}).length > 0 && (
          <MetricValues metrics={current.metrics ?? {}} />
        )}
    </>
  );
}
