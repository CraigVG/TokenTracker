"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");

const {
  parseMuseIncremental,
  resolveMuseHome,
  resolveMuseSessionFiles,
} = require("../src/lib/rollout");

// 2026-10-13T14:35:00Z in epoch microseconds, the unit Muse records use.
const T0_US = Date.parse("2026-10-13T14:35:00Z") * 1000;

function modelRecord(runId, modelId) {
  return {
    schema_version: 1,
    id: `model-${runId}`,
    stream: { kind: "session", id: "session-1" },
    sequence: 1,
    recorded_at: T0_US,
    record_type: "event",
    payload_type: "run.model.configured",
    payload: { kind: "run_model", record: { run_stream: { kind: "run", id: runId }, model_id: modelId } },
  };
}

function usageRecord(usageId, runId, quantity, { family = "provider", atUs = T0_US } = {}) {
  return {
    schema_version: 1,
    id: usageId,
    stream: { kind: "session", id: "session-1" },
    sequence: 2,
    recorded_at: atUs,
    record_type: "event",
    payload_type: "runtime.session",
    payload: {
      kind: "run",
      run_id: runId,
      event: {
        kind: "goal_usage_attribution",
        record: {
          usage_id: usageId,
          usage_family: family,
          quantity: { unit: "tokens", reported: true, main_llm_steps: 1, ...quantity },
        },
      },
    },
  };
}

function writeSession(home, relDir, records) {
  const dir = path.join(home, ".local", "share", "muse", "sessions", relDir);
  fs.mkdirSync(dir, { recursive: true });
  const file = path.join(dir, "session.jsonl");
  fs.writeFileSync(file, records.map((r) => JSON.stringify(r)).join("\n") + "\n");
  return file;
}

function readRows(queuePath) {
  if (!fs.existsSync(queuePath)) return [];
  return fs.readFileSync(queuePath, "utf8").trim().split("\n").filter(Boolean).map(JSON.parse);
}

function withHome(fn) {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), "tt-muse-"));
  const env = { HOME: home, XDG_CONFIG_HOME: path.join(home, ".config") };
  return Promise.resolve(fn(home, env)).finally(() => fs.rmSync(home, { recursive: true, force: true }));
}

test("resolveMuseSessionFiles finds main and subagent logs and skips derived views", () =>
  withHome((home, env) => {
    const main = writeSession(home, "2026/10/13/s1", [modelRecord("r1", "muse-wonder-full-1.0")]);
    const sub = writeSession(home, "2026/10/13/s1/subagent/a1", [modelRecord("r2", "muse-wonder-full-1.0")]);
    writeSession(home, ".msp-view-v1/s1", [modelRecord("r3", "ignored")]);
    assert.deepEqual(resolveMuseSessionFiles(env).sort(), [main, sub].sort());
  }));

test("resolveMuseHome honours TOKENTRACKER_MUSE_HOME", () => {
  assert.equal(resolveMuseHome({ TOKENTRACKER_MUSE_HOME: "/tmp/custom-muse" }), "/tmp/custom-muse");
});

test("parseMuseIncremental splits cached input out and keeps reasoning a subset of output", () =>
  withHome(async (home, env) => {
    writeSession(home, "2026/10/13/s1", [
      modelRecord("r1", "muse-wonder-full-1.0"),
      usageRecord("u1", "r1", { input_tokens: 1000, cached_tokens: 900, output_tokens: 50, reasoning_tokens: 20 }),
      // Zero-token tool and reminder rows are not model usage.
      usageRecord("u2", "r1", { input_tokens: 0, output_tokens: 0, cached_tokens: 0, reasoning_tokens: 0 }, { family: "tool" }),
      usageRecord("u3", "r1", { input_tokens: 5, output_tokens: 5, cached_tokens: 0, reasoning_tokens: 0 }, { family: "reminder" }),
    ]);
    const queuePath = path.join(home, "queue.jsonl");
    const result = await parseMuseIncremental({ cursors: {}, queuePath, env });
    assert.equal(result.eventsAggregated, 1);
    const [row] = readRows(queuePath);
    assert.equal(row.source, "muse");
    assert.equal(row.model, "muse-wonder-full-1.0");
    assert.equal(row.hour_start, "2026-10-13T14:30:00.000Z");
    assert.equal(row.input_tokens, 100);
    assert.equal(row.cached_input_tokens, 900);
    assert.equal(row.cache_creation_input_tokens, 0);
    assert.equal(row.output_tokens, 50);
    assert.equal(row.reasoning_output_tokens, 20);
    assert.equal(row.total_tokens, 1050);
  }));

test("parseMuseIncremental attributes each run to its own model", () =>
  withHome(async (home, env) => {
    writeSession(home, "2026/10/13/s1", [
      modelRecord("r1", "muse-wonder-full-1.0"),
      modelRecord("r2", "muse-spark-1.3-contributor"),
      usageRecord("u1", "r1", { input_tokens: 10, cached_tokens: 0, output_tokens: 1, reasoning_tokens: 0 }),
      usageRecord("u2", "r2", { input_tokens: 20, cached_tokens: 0, output_tokens: 2, reasoning_tokens: 0 }),
    ]);
    const queuePath = path.join(home, "queue.jsonl");
    await parseMuseIncremental({ cursors: {}, queuePath, env });
    const byModel = Object.fromEntries(readRows(queuePath).map((r) => [r.model, r.input_tokens]));
    assert.deepEqual(byModel, { "muse-wonder-full-1.0": 10, "muse-spark-1.3-contributor": 20 });
  }));

test("parseMuseIncremental reads appended records once and keeps the model across syncs", () =>
  withHome(async (home, env) => {
    const records = [
      modelRecord("r1", "muse-wonder-full-1.0"),
      usageRecord("u1", "r1", { input_tokens: 100, cached_tokens: 0, output_tokens: 10, reasoning_tokens: 0 }),
    ];
    const file = writeSession(home, "2026/10/13/s1", records);
    const queuePath = path.join(home, "queue.jsonl");
    const cursors = {};
    await parseMuseIncremental({ cursors, queuePath, env });

    // Same file, no growth: nothing is counted again.
    const idle = await parseMuseIncremental({ cursors, queuePath, env });
    assert.equal(idle.eventsAggregated, 0);

    // A later usage row for the same run carries no model record in the new bytes.
    fs.appendFileSync(file, JSON.stringify(
      usageRecord("u2", "r1", { input_tokens: 40, cached_tokens: 0, output_tokens: 4, reasoning_tokens: 0 }),
    ) + "\n");
    const grown = await parseMuseIncremental({ cursors, queuePath, env });
    assert.equal(grown.eventsAggregated, 1);
    const rows = readRows(queuePath);
    const last = rows[rows.length - 1];
    assert.equal(last.model, "muse-wonder-full-1.0");
    assert.equal(last.input_tokens, 140);
    assert.equal(last.output_tokens, 14);
    assert.equal(last.conversation_count, 1, "a session is one conversation however many steps or syncs it spans");
  }));

test("parseMuseIncremental counts subagent logs as part of their parent conversation", () =>
  withHome(async (home, env) => {
    const step = { input_tokens: 10, cached_tokens: 0, output_tokens: 1, reasoning_tokens: 0 };
    writeSession(home, "2026/10/13/s1", [modelRecord("r1", "muse-wonder-full-1.0"), usageRecord("u1", "r1", step)]);
    writeSession(home, "2026/10/13/s1/subagent/a1", [
      modelRecord("r2", "muse-wonder-full-1.0"),
      usageRecord("u2", "r2", step),
      usageRecord("u3", "r2", step),
    ]);
    const queuePath = path.join(home, "queue.jsonl");
    await parseMuseIncremental({ cursors: {}, queuePath, env });
    const [row] = readRows(queuePath);
    assert.equal(row.input_tokens, 30);
    assert.equal(row.conversation_count, 1);
  }));

test("parseMuseIncremental leaves a half-written last line for the next sync", () =>
  withHome(async (home, env) => {
    const first = JSON.stringify(
      usageRecord("u1", "r1", { input_tokens: 10, cached_tokens: 0, output_tokens: 1, reasoning_tokens: 0 }),
    );
    const second = JSON.stringify(
      usageRecord("u2", "r1", { input_tokens: 30, cached_tokens: 0, output_tokens: 3, reasoning_tokens: 0 }),
    );
    const dir = path.join(home, ".local", "share", "muse", "sessions", "2026", "10", "13", "s1");
    fs.mkdirSync(dir, { recursive: true });
    const file = path.join(dir, "session.jsonl");
    fs.writeFileSync(file, `${first}\n${second.slice(0, 40)}`);
    const queuePath = path.join(home, "queue.jsonl");
    const cursors = {};
    const partial = await parseMuseIncremental({ cursors, queuePath, env });
    assert.equal(partial.eventsAggregated, 1);

    fs.writeFileSync(file, `${first}\n${second}\n`);
    const done = await parseMuseIncremental({ cursors, queuePath, env });
    assert.equal(done.eventsAggregated, 1);
    const rows = readRows(queuePath);
    assert.equal(rows[rows.length - 1].input_tokens, 40);
  }));

test("parseMuseIncremental does not recount a rewritten log", () =>
  withHome(async (home, env) => {
    const records = [
      modelRecord("r1", "muse-wonder-full-1.0"),
      usageRecord("u1", "r1", { input_tokens: 100, cached_tokens: 0, output_tokens: 10, reasoning_tokens: 0 }),
    ];
    const file = writeSession(home, "2026/10/13/s1", records);
    const queuePath = path.join(home, "queue.jsonl");
    const cursors = {};
    await parseMuseIncremental({ cursors, queuePath, env });
    // Rewrite smaller than before (compaction): the usage id was already counted.
    fs.writeFileSync(file, JSON.stringify(records[1]) + "\n");
    const again = await parseMuseIncremental({ cursors, queuePath, env });
    assert.equal(again.eventsAggregated, 0);
  }));

test("parseMuseIncremental does not recount a rewritten log whose ids fell out of the seen set", () =>
  withHome(async (home, env) => {
    const step = { input_tokens: 100, cached_tokens: 0, output_tokens: 10, reasoning_tokens: 0 };
    const model = modelRecord("r1", "muse-wonder-full-1.0");
    const first = usageRecord("u1", "r1", step, { atUs: T0_US });
    const file = writeSession(home, "2026/10/13/s1", [model, first]);
    const queuePath = path.join(home, "queue.jsonl");
    const cursors = {};
    await parseMuseIncremental({ cursors, queuePath, env });

    // The global id set is capped; simulate the id having been evicted.
    cursors.muse.seenIds = [];
    // Replace the log (compaction) and add one genuinely new row.
    const added = usageRecord("u2", "r1", step, { atUs: T0_US + 60_000_000 });
    const replacement = `${file}.tmp`;
    fs.writeFileSync(replacement, [model, first, added].map((r) => JSON.stringify(r)).join("\n") + "\n");
    fs.renameSync(replacement, file); // new inode: the log is re-read from the start
    const again = await parseMuseIncremental({ cursors, queuePath, env });
    assert.equal(again.eventsAggregated, 1, "only the new row is counted");
    const rows = readRows(queuePath);
    assert.equal(rows[rows.length - 1].input_tokens, 200);
  }));

test("parseMuseIncremental reads records inside a retained frame", () =>
  withHome(async (home, env) => {
    const frame = {
      retained_frame: "session_permission_transaction",
      frame_schema_version: 1,
      children: [
        { child_index: 0, record_json: JSON.stringify(modelRecord("r1", "muse-wonder-1.0")) },
        {
          child_index: 1,
          record_json: JSON.stringify(
            usageRecord("u1", "r1", { input_tokens: 7, cached_tokens: 2, output_tokens: 1, reasoning_tokens: 0 }),
          ),
        },
      ],
    };
    writeSession(home, "2026/10/13/s1", [frame]);
    const queuePath = path.join(home, "queue.jsonl");
    await parseMuseIncremental({ cursors: {}, queuePath, env });
    const [row] = readRows(queuePath);
    assert.equal(row.model, "muse-wonder-1.0");
    assert.equal(row.input_tokens, 5);
    assert.equal(row.cached_input_tokens, 2);
  }));

test("parseMuseIncremental falls back to the configured model, never reading prompt text", () =>
  withHome(async (home, env) => {
    const configDir = path.join(home, ".config", "muse");
    fs.mkdirSync(configDir, { recursive: true });
    fs.writeFileSync(path.join(configDir, "settings.json"), JSON.stringify({ model: "muse-wonder-full-1.0" }));
    writeSession(home, "2026/10/13/s1", [
      usageRecord("u1", "unknown-run", { input_tokens: 9, cached_tokens: 0, output_tokens: 1, reasoning_tokens: 0 }),
    ]);
    const queuePath = path.join(home, "queue.jsonl");
    await parseMuseIncremental({ cursors: {}, queuePath, env });
    const [row] = readRows(queuePath);
    assert.equal(row.model, "muse-wonder-full-1.0");
    assert.deepEqual(
      Object.keys(row).filter((k) => /prompt|text|content|path/i.test(k)),
      [],
    );
  }));
