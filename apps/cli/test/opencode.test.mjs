import { test } from "node:test";
import assert from "node:assert/strict";

import {
  buildTasks,
  cachesOnItsOwn,
  coldResumes,
  conversations,
  costOf,
  mapOpencodeRows,
  toolSignatureOf,
  toolTargetOf,
} from "@optimaizr/core";

const T0 = Date.parse("2026-10-10T12:00:00.000Z");

/** Rows as opencode stores them: JSON in `data`, times in ms. */
function rows({ sessions = [], messages = [], parts = [] }) {
  return { sessions, messages, parts };
}
const session = (over = {}) => ({
  id: "ses_a",
  parent_id: null,
  directory: "/home/someone/project",
  title: "Fix the parser",
  model: JSON.stringify({ id: "claude-sonnet-5", providerID: "anthropic", variant: "high" }),
  ...over,
});
const message = (id, data, at = T0, sid = "ses_a") => ({
  id,
  session_id: sid,
  time_created: at,
  data: JSON.stringify({ role: "assistant", parentID: "msg_user1", ...data }),
});
const step = (id, mid, tokens, at = T0 + 1000, reason = "stop") => ({
  id,
  message_id: mid,
  session_id: "ses_a",
  time_created: at,
  data: JSON.stringify({ type: "step-finish", reason, tokens, cost: 999 }),
});
const tool = (id, mid, name, input, status = "completed", outputChars = 120, at = T0 + 500) => ({
  id,
  message_id: mid,
  session_id: "ses_a",
  time_created: at,
  data: JSON.stringify({
    type: "tool",
    tool: name,
    callID: `call_${id}`,
    state: { status, input },
  }),
  output_chars: outputChars,
});
const TOKENS = { input: 1000, output: 200, reasoning: 50, cache: { read: 40_000, write: 3000 } };

test("a step-finish becomes one call with opencode's counts", () => {
  const out = mapOpencodeRows(
    rows({
      sessions: [session()],
      messages: [message("msg_1", { modelID: "claude-sonnet-5", variant: "max" })],
      parts: [step("prt_1", "msg_1", TOKENS)],
    }),
  );
  assert.equal(out.events.length, 1);
  const e = out.events[0];
  assert.equal(e.id, "ses_a#prt_1");
  assert.equal(e.source, "opencode");
  assert.equal(e.model, "claude-sonnet-5");
  assert.equal(e.provider, "anthropic");
  assert.equal(e.effort, "max");
  assert.equal(e.project, "/home/someone/project");
  assert.equal(e.turnId, "ses_a:msg_user1");
  assert.equal(e.ts, new Date(T0 + 1000).toISOString());
  assert.equal(e.stopReason, "end_turn");
  assert.deepEqual(
    [
      e.inputTokens,
      e.outputTokens,
      e.thinkingTokens,
      e.cacheReadTokens,
      e.cacheWrite5mTokens,
      e.cacheWrite1hTokens,
    ],
    [1000, 250, 50, 40_000, 3000, 0],
  );
  assert.equal(e.isSubagent, undefined);
  assert.deepEqual(out.titles, { ses_a: "Fix the parser" });
});

test("anthropic cost matches the same call priced directly", () => {
  const [e] = mapOpencodeRows(
    rows({
      sessions: [session()],
      messages: [message("msg_1", { modelID: "claude-sonnet-5" })],
      parts: [step("prt_1", "msg_1", TOKENS)],
    }),
  ).events;
  const direct = costOf(
    {
      input_tokens: 1000,
      output_tokens: 250,
      cache_read_input_tokens: 40_000,
      cache_creation_input_tokens: 3000,
    },
    "claude-sonnet-5",
    { at: e.ts },
  );
  assert.ok(e.cost.total > 0);
  assert.equal(e.cost.total, direct.total);
});

test("an OpenAI model's cache reads and writes are not subtracted twice", () => {
  const [e] = mapOpencodeRows(
    rows({
      sessions: [session({ model: null })],
      messages: [message("msg_1", { modelID: "gpt-5.4", providerID: "openai" })],
      parts: [step("prt_1", "msg_1", TOKENS)],
    }),
  ).events;
  const direct = costOf(
    {
      input_tokens: 44_000,
      input_tokens_details: { cached_tokens: 40_000, cache_write_tokens: 3000 },
      output_tokens: 250,
      output_tokens_details: { reasoning_tokens: 50 },
    },
    "gpt-5.4",
    { at: e.ts },
  );
  assert.equal(e.inputTokens, 1000);
  assert.ok(e.cost.total > 0);
  assert.equal(e.cost.total, direct.total);
});

test("model and effort fall back to the session's selection", () => {
  const out = mapOpencodeRows(
    rows({
      sessions: [session()],
      messages: [message("msg_1", {})],
      parts: [step("prt_1", "msg_1", TOKENS)],
    }),
  );
  assert.equal(out.events[0].model, "claude-sonnet-5");
  assert.equal(out.events[0].effort, "high");
  assert.deepEqual(out.warnings, []);
});

test("no model anywhere is priced as unknown, with a warning", () => {
  const out = mapOpencodeRows(
    rows({
      sessions: [session({ model: null })],
      messages: [message("msg_1", {})],
      parts: [step("prt_1", "msg_1", TOKENS)],
    }),
  );
  assert.equal(out.events[0].model, "unknown");
  assert.equal(out.events[0].effort, null);
  assert.equal(out.events[0].cost.unpriced, true);
  assert.ok(out.warnings.some((w) => w.includes("no model")));
});

test("tools attach to the step that called them", () => {
  const out = mapOpencodeRows(
    rows({
      sessions: [session()],
      messages: [message("msg_1", { modelID: "claude-sonnet-5" })],
      parts: [
        tool("prt_1", "msg_1", "read", { filePath: "/a.ts" }, "completed", 120, T0 + 100),
        tool("prt_2", "msg_1", "bash", { command: "false" }, "error", 9, T0 + 200),
        step("prt_3", "msg_1", TOKENS, T0 + 300, "tool-calls"),
        step("prt_4", "msg_1", TOKENS, T0 + 400),
      ],
    }),
  );
  const [first, second] = out.events;
  assert.deepEqual(
    first.tools.map((t) => [t.name, t.signature, t.resultChars, t.isError]),
    [
      ["Read", "Read:/a.ts", 120, undefined],
      ["Bash", "Bash:false", 9, true],
    ],
  );
  assert.equal(first.stopReason, "tool_use");
  assert.deepEqual(second.tools, []);
});

test("canonical tool names, read slices and edit targets use Claude's helpers", () => {
  const names = [
    "read",
    "write",
    "edit",
    "multiedit",
    "bash",
    "grep",
    "glob",
    "list",
    "todowrite",
    "todoread",
    "task",
    "webfetch",
    "websearch",
    "apply_patch",
    "patch",
    "custom",
  ];
  const canonical = [
    "Read",
    "Write",
    "Edit",
    "MultiEdit",
    "Bash",
    "Grep",
    "Glob",
    "LS",
    "TodoWrite",
    "TodoWrite",
    "Task",
    "WebFetch",
    "WebSearch",
    "apply_patch",
    "apply_patch",
    "custom",
  ];
  const input = {
    filePath: "/a.ts",
    offset: 2,
    limit: 10,
    command: "  git   status ",
    pattern: "foo",
    path: "/p",
    url: "https://example.com",
  };
  const translated = {
    offset: 2,
    limit: 10,
    command: "  git   status ",
    pattern: "foo",
    path: "/p",
    url: "https://example.com",
    file_path: "/a.ts",
  };
  const [e] = mapOpencodeRows(
    rows({
      sessions: [session()],
      messages: [message("msg_1", {})],
      parts: [
        ...names.map((name, i) =>
          tool(`prt_${i}`, "msg_1", name, input, "completed", 120, T0 + 500 + i),
        ),
        step("prt_end", "msg_1", TOKENS),
      ],
    }),
  ).events;
  assert.deepEqual(
    e.tools.map((t) => t.name),
    canonical,
  );
  for (const [i, t] of e.tools.entries()) {
    assert.equal(t.signature, toolSignatureOf(canonical[i], translated));
    assert.equal(t.target, toolTargetOf(translated));
  }
  assert.equal(e.tools[0].signature, "Read:/a.ts#2:10");
});

test("a completed read turn is a done lookup, not multi-step", () => {
  const out = mapOpencodeRows(
    rows({
      sessions: [session()],
      messages: [message("msg_1", {})],
      parts: [
        tool("prt_1", "msg_1", "read", { filePath: "/a.ts" }),
        step("prt_2", "msg_1", TOKENS),
      ],
    }),
  );
  const [task] = buildTasks(out.events);
  assert.equal(task.done, true);
  assert.equal(task.kind, "lookup");
});

test("stop reasons are canonical, with unknown reasons passed through", () => {
  const reasons = ["tool-calls", "stop", "length", "other"];
  const out = mapOpencodeRows(
    rows({
      sessions: [session()],
      messages: [message("msg_1", {})],
      parts: reasons.map((reason, i) => step(`prt_${i}`, "msg_1", TOKENS, T0 + i, reason)),
    }),
  );
  assert.deepEqual(
    out.events.map((e) => e.stopReason),
    ["tool_use", "end_turn", "max_tokens", "other"],
  );
});

test("a subagent session is flagged even when its parent row is absent", () => {
  const out = mapOpencodeRows(
    rows({
      sessions: [session({ parent_id: "ses_parent" })],
      messages: [message("msg_1", {})],
      parts: [step("prt_1", "msg_1", TOKENS)],
    }),
  );
  assert.equal(out.events[0].isSubagent, true);
});

test("subagents join the root conversation while retaining their own event and turn ids", () => {
  const out = mapOpencodeRows(
    rows({
      sessions: [
        session({ id: "ses_child", parent_id: "ses_middle", directory: null }),
        session({ id: "ses_middle", parent_id: "ses_a" }),
        session(),
      ],
      messages: [message("msg_1", {}), message("msg_child", {}, T0 + 2000, "ses_child")],
      parts: [
        step("prt_1", "msg_1", TOKENS),
        { ...step("prt_child", "msg_child", TOKENS, T0 + 3000), session_id: "ses_child" },
      ],
    }),
  );
  assert.deepEqual(
    out.events.map((e) => e.sessionId),
    ["ses_a", "ses_a"],
  );
  const child = out.events[1];
  assert.equal(child.isSubagent, true);
  assert.equal(child.id, "ses_child#prt_child");
  assert.equal(child.turnId, "ses_child:msg_user1");
  assert.equal(child.project, "/home/someone/project");
  // The opencode source filter is added in Task 3; exercise aggregation with a supported label.
  const convos = conversations({
    ...out,
    events: out.events.map((e) => ({ ...e, source: "claude-code" })),
  });
  assert.equal(convos.length, 1);
  assert.equal(convos[0].sessionId, "ses_a");
  assert.equal(convos[0].costUsd, out.events[0].cost.total + child.cost.total);
  assert.equal(convos[0].subagentUsd, child.cost.total);
});

test("a summary records full previous context and moves later calls to the next epoch", () => {
  const out = mapOpencodeRows(
    rows({
      sessions: [session()],
      messages: [
        message("msg_1", {}, T0),
        message("msg_2", { summary: true }, T0 + 10_000),
        message("msg_3", {}, T0 + 20_000),
      ],
      parts: [
        step("prt_1", "msg_1", TOKENS, T0 + 1000, "tool-calls"),
        step("prt_2", "msg_2", TOKENS, T0 + 11_000),
        step("prt_3", "msg_3", TOKENS, T0 + 21_000),
      ],
    }),
  );
  assert.deepEqual(
    out.events.map((e) => e.contextEpoch),
    [undefined, undefined, 1],
  );
  assert.equal(out.compactions.length, 1);
  assert.deepEqual(out.compactions[0], {
    source: "opencode",
    ts: new Date(T0 + 10_000).toISOString(),
    sessionId: "ses_a",
    project: "/home/someone/project",
    trigger: "unknown",
    preTokens: 44_000,
    midTask: true,
  });
});

test("days and project filter calls", () => {
  const now = Date.now();
  const out = mapOpencodeRows(
    rows({
      sessions: [session(), session({ id: "ses_b", directory: "/elsewhere" })],
      messages: [
        message("msg_old", {}, now - 10 * 86_400_000),
        message("msg_new", {}, now - 1000),
        message("msg_b", {}, now - 1000, "ses_b"),
      ],
      parts: [
        step("prt_old", "msg_old", TOKENS, now - 10 * 86_400_000),
        step("prt_new", "msg_new", TOKENS, now - 500),
        { ...step("prt_b", "msg_b", TOKENS, now - 500), session_id: "ses_b" },
      ],
    }),
    { days: 3, project: "PROJECT" },
  );
  assert.deepEqual(
    out.events.map((e) => e.id),
    ["ses_a#prt_new"],
  );
});

test("unparseable messages and parts are skipped with one warning", () => {
  const out = mapOpencodeRows(
    rows({
      sessions: [session()],
      messages: [{ ...message("msg_bad", {}), data: "{oops" }, message("msg_1", {})],
      parts: [
        { ...step("prt_bad", "msg_1", TOKENS), data: "{oops" },
        step("prt_1", "msg_1", TOKENS),
      ],
    }),
  );
  assert.equal(out.events.length, 1);
  assert.deepEqual(out.warnings, ["opencode: skipped 2 unparseable row(s)"]);
});

test("invalid message and part timestamps are skipped without losing valid calls", () => {
  const out = mapOpencodeRows(
    rows({
      sessions: [session()],
      messages: [
        message("msg_nan", {}, NaN),
        message("msg_huge", { summary: true }, 1e30),
        message("msg_1", {}),
      ],
      parts: [
        step("prt_nan_message", "msg_nan", TOKENS),
        step("prt_huge_message", "msg_huge", TOKENS),
        step("prt_nan", "msg_1", TOKENS, NaN),
        step("prt_huge", "msg_1", TOKENS, 1e30),
        step("prt_1", "msg_1", TOKENS),
      ],
    }),
  );
  assert.deepEqual(
    out.events.map((e) => e.id),
    ["ses_a#prt_1"],
  );
  assert.deepEqual(out.warnings, ["opencode: skipped 4 unparseable row(s)"]);
});

test("provider falls back for an uncatalogued model without inventing a price", () => {
  const out = mapOpencodeRows(
    rows({
      sessions: [session({ model: null })],
      messages: [message("msg_1", { modelID: "gpt-6.1-sol", providerID: "openai" })],
      parts: [step("prt_1", "msg_1", TOKENS)],
    }),
  );
  assert.equal(out.events[0].provider, "openai");
  assert.equal(out.events[0].cost.unpriced, true);
  assert.equal(out.events[0].cost.total, 0);
  assert.deepEqual(out.warnings, []);
});

test("the catalogue provider wins and unrecognised provider ids stay unknown", () => {
  const out = mapOpencodeRows(
    rows({
      sessions: [session()],
      messages: [
        message("msg_1", { modelID: "claude-sonnet-5", providerID: "openai" }),
        message("msg_2", { modelID: "custom-model", providerID: "not-a-provider" }, T0 + 2000),
      ],
      parts: [step("prt_1", "msg_1", TOKENS), step("prt_2", "msg_2", TOKENS, T0 + 3000)],
    }),
  );
  assert.deepEqual(
    out.events.map((e) => e.provider),
    ["anthropic", "unknown"],
  );
});

test("OpenAI opencode cold returns match identical Codex calls", () => {
  const out = mapOpencodeRows(
    rows({
      sessions: [session()],
      messages: [
        message("msg_1", { modelID: "gpt-5.4" }),
        message("msg_2", { modelID: "gpt-5.4" }, T0 + 600_000),
      ],
      parts: [
        step("prt_1", "msg_1", { ...TOKENS, cache: { read: 60_000, write: 0 } }),
        step(
          "prt_2",
          "msg_2",
          { ...TOKENS, input: 70_000, cache: { read: 0, write: 0 } },
          T0 + 601_000,
        ),
      ],
    }),
  );
  const cold = coldResumes(out.events);
  const codex = coldResumes(out.events.map((e) => ({ ...e, source: "codex" })));
  const metrics = (list) =>
    list.map(({ gapMinutes, ttlMinutes, rewriteTokens, rewriteUsd, warmUsd }) => ({
      gapMinutes,
      ttlMinutes,
      rewriteTokens,
      rewriteUsd,
      warmUsd,
    }));
  assert.equal(cold.length, 1);
  assert.deepEqual(metrics(cold), metrics(codex));
  assert.equal(cachesOnItsOwn({ ...out.events[0], provider: "anthropic" }), false);
  assert.equal(cachesOnItsOwn({ ...out.events[0], provider: "google" }), true);
  assert.equal(cachesOnItsOwn({ ...out.events[0], source: "sdk" }), false);
});

test("a parent_id cycle neither hangs nor drops calls", () => {
  const out = mapOpencodeRows(
    rows({
      sessions: [session({ parent_id: "ses_b" }), session({ id: "ses_b", parent_id: "ses_a" })],
      messages: [message("msg_1", {})],
      parts: [step("prt_1", "msg_1", TOKENS)],
    }),
  );
  assert.equal(out.events.length, 1);
});

test("a compaction after a finished reply is not mid-task", () => {
  const out = mapOpencodeRows(
    rows({
      sessions: [session()],
      messages: [message("msg_1", {}, T0), message("msg_2", { summary: true }, T0 + 10_000)],
      parts: [
        step("prt_1", "msg_1", TOKENS, T0 + 1000, "stop"),
        step("prt_2", "msg_2", TOKENS, T0 + 11_000),
      ],
    }),
  );
  assert.equal(out.compactions.length, 1);
  assert.equal(out.compactions[0].midTask, false);
});
