import { costOf, PROVIDERS, providerOf } from "../pricing.js";
import { toolSignatureOf, toolTargetOf } from "./claudecode.js";
import type { CompactionRecord, ToolCall, UsageEvent } from "../domain/types.js";

/**
 * opencode history, the third agent beside Claude Code and Codex. opencode
 * keeps everything in one SQLite file; the tables that matter:
 * - `session`: directory, title, `parent_id` for subagents, and `model`, the
 *   model and effort the session is set to now.
 * - `message`: assistant messages name the model that answered (`modelID`,
 *   `variant`) and the user message (`parentID`); a `summary` is a compaction.
 * - `part`: each `step-finish` is one billed call with its tokens; `tool`
 *   parts are the calls it made.
 *
 * opencode normalises counts across vendors: input excludes cache traffic,
 * and reasoning sits beside output. This mapping is pure, tested without SQLite.
 */

export interface OpencodeSessionRow {
  id: string;
  parent_id?: string | null;
  directory?: string | null;
  title?: string | null;
  /** JSON `{ id, providerID, variant }`. */
  model?: string | null;
}

export interface OpencodeMessageRow {
  id: string;
  session_id: string;
  time_created: number;
  data: string;
}

export interface OpencodePartRow {
  id: string;
  message_id: string;
  session_id: string;
  time_created: number;
  data: string;
  /** Length measured where the output is stored, so the text isn't copied. */
  output_chars?: number | null;
}

export interface OpencodeRows {
  sessions: OpencodeSessionRow[];
  messages: OpencodeMessageRow[];
  parts: OpencodePartRow[];
}

export interface OpencodeMapOptions {
  days?: number;
  project?: string;
}

export interface OpencodeMapped {
  events: UsageEvent[];
  compactions: CompactionRecord[];
  titles: Record<string, string>;
  warnings: string[];
}

/** opencode's per-step token counts. */
interface Tokens {
  input?: number;
  output?: number;
  reasoning?: number;
  cache?: { read?: number; write?: number };
}

function num(v: unknown): number {
  return typeof v === "number" && Number.isFinite(v) ? v : 0;
}

function str(v: unknown): string {
  return typeof v === "string" ? v : "";
}

function record(v: unknown): Record<string, unknown> {
  return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {};
}

function parse(text: string | null | undefined): Record<string, unknown> | null {
  if (!text) return null;
  try {
    const v: unknown = JSON.parse(text);
    return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

function iso(ms: unknown): string | null {
  if (typeof ms !== "number" || !Number.isFinite(ms) || Math.abs(ms) > 8.64e15) return null;
  const date = new Date(ms);
  return Number.isFinite(date.getTime()) ? date.toISOString() : null;
}

/**
 * opencode's counts in `normalizeUsage`'s vocabulary. Add cache traffic back
 * for providers whose own counts include it, so their policy subtracts it once.
 */
function usageShape(t: Tokens, provider: string): Record<string, unknown> {
  const reasoning = num(t.reasoning);
  const read = num(t.cache?.read);
  const write = num(t.cache?.write);
  const output = {
    output_tokens: num(t.output) + reasoning,
    output_tokens_details: { reasoning_tokens: reasoning },
  };
  if (PROVIDERS[provider]?.inputIncludesCacheReads) {
    return {
      ...output,
      input_tokens: num(t.input) + read + write,
      input_tokens_details: { cached_tokens: read, cache_write_tokens: write },
    };
  }
  return {
    ...output,
    input_tokens: num(t.input),
    cache_read_input_tokens: read,
    cache_creation_input_tokens: write,
  };
}

const TOOL_NAMES = new Map([
  ["read", "Read"],
  ["write", "Write"],
  ["edit", "Edit"],
  ["multiedit", "MultiEdit"],
  ["bash", "Bash"],
  ["grep", "Grep"],
  ["glob", "Glob"],
  ["list", "LS"],
  ["todowrite", "TodoWrite"],
  ["todoread", "TodoWrite"],
  ["task", "Task"],
  ["webfetch", "WebFetch"],
  ["websearch", "WebSearch"],
  ["patch", "apply_patch"],
]);

function toolInput(raw: unknown): Record<string, unknown> {
  const input = { ...record(raw) };
  if ("filePath" in input) {
    input.file_path = input.filePath;
    delete input.filePath;
  }
  return input;
}

function stopReason(raw: unknown): string | null {
  const reason = str(raw);
  switch (reason) {
    case "tool-calls":
      return "tool_use";
    case "stop":
      return "end_turn";
    case "length":
      return "max_tokens";
    default:
      return reason || null;
  }
}

function groupBy<T>(rows: T[], key: (row: T) => string): Map<string, T[]> {
  const out = new Map<string, T[]>();
  for (const row of rows) {
    const k = key(row);
    const list = out.get(k);
    if (list) list.push(row);
    else out.set(k, [row]);
  }
  return out;
}

/** Oldest first; opencode's ids sort by time, so they break ties. */
function byTime(a: { time_created: number; id: string }, b: { time_created: number; id: string }) {
  return a.time_created - b.time_created || a.id.localeCompare(b.id);
}

/** opencode's rows as calls. Bad JSON and timestamps become one warning. */
export function mapOpencodeRows(rows: OpencodeRows, opts: OpencodeMapOptions = {}): OpencodeMapped {
  const events: UsageEvent[] = [];
  const compactions: CompactionRecord[] = [];
  const titles: Record<string, string> = {};
  let bad = 0;
  let unnamed = 0;

  const sessions = new Map(rows.sessions.map((s) => [s.id, s]));
  const messagesOf = groupBy(rows.messages, (m) => m.session_id);
  const partsOf = groupBy(rows.parts, (p) => p.message_id);

  for (const s of rows.sessions) {
    if (s.title) titles[s.id] = s.title;
    // Sidechains share the root's conversation, but keep their own turns and epochs.
    let root = s;
    const ancestors = new Set([s.id]);
    while (root.parent_id) {
      const parent = sessions.get(root.parent_id);
      if (!parent || ancestors.has(parent.id)) break;
      ancestors.add(parent.id);
      root = parent;
    }
    const project = s.directory || root.directory || "unknown";
    const chosen = record(parse(s.model));
    let epoch = 0;
    let lastContext = 0;
    let toolLast = false;

    for (const m of (messagesOf.get(s.id) ?? []).sort(byTime)) {
      const data = parse(m.data);
      const messageTs = iso(m.time_created);
      if (!data || !messageTs) {
        bad++;
        continue;
      }
      if (data.role !== "assistant") continue;

      const model = str(data.modelID) || str(chosen.id) || "unknown";
      let provider = providerOf(model);
      const reportedProvider = str(data.providerID);
      if (provider === "unknown" && Object.hasOwn(PROVIDERS, reportedProvider))
        provider = reportedProvider;
      const effort = str(data.variant) || str(chosen.variant) || null;
      const parentID = str(data.parentID);
      const summary = data.summary === true;
      if (summary && lastContext > 0) {
        compactions.push({
          source: "opencode",
          ts: messageTs,
          sessionId: s.id,
          project,
          // opencode doesn't record what triggered it or how long it took.
          trigger: "unknown",
          preTokens: lastContext,
          midTask: toolLast,
        });
      }

      let tools: ToolCall[] = [];
      for (const p of (partsOf.get(m.id) ?? []).sort(byTime)) {
        const part = parse(p.data);
        const ts = iso(p.time_created);
        if (!part || !ts) {
          bad++;
          continue;
        }
        if (part.type === "tool") {
          const state = record(part.state);
          const rawName = str(part.tool) || "unknown";
          const name = TOOL_NAMES.get(rawName) ?? rawName;
          const input = toolInput(state.input);
          const target = toolTargetOf(input);
          tools.push({
            id: str(part.callID) || p.id,
            name,
            signature: toolSignatureOf(name, input),
            ...(target ? { target } : {}),
            ...(typeof p.output_chars === "number" ? { resultChars: p.output_chars } : {}),
            ...(state.status === "error" ? { isError: true } : {}),
          });
          continue;
        }
        // The billable unit.
        if (part.type !== "step-finish") continue;

        const t = record(part.tokens) as Tokens;
        const reason = stopReason(part.reason);
        if (model === "unknown") unnamed++;
        events.push({
          // Stable across re-reads, so decisions stay attached to their calls.
          id: `${s.id}#${p.id}`,
          source: "opencode",
          ts,
          model,
          provider,
          sessionId: root.id,
          project,
          inputTokens: num(t.input),
          outputTokens: num(t.output) + num(t.reasoning),
          thinkingTokens: num(t.reasoning),
          cacheReadTokens: num(t.cache?.read),
          // opencode reports one write figure; bill it at the short-TTL rate.
          cacheWrite5mTokens: num(t.cache?.write),
          cacheWrite1hTokens: 0,
          effort,
          stopReason: reason,
          ...(s.parent_id ? { isSubagent: true } : {}),
          ...(parentID ? { turnId: `${s.id}:${parentID}` } : {}),
          ...(epoch ? { contextEpoch: epoch } : {}),
          tools,
          // optimAIzr's own prices: opencode's `cost` is 0 on subscriptions.
          cost: costOf(usageShape(t, provider), model, { at: ts }),
        });
        tools = [];
        lastContext = num(t.input) + num(t.cache?.read) + num(t.cache?.write);
        toolLast = reason === "tool_use";
      }
      if (summary) epoch++;
    }
  }

  const cutoff = opts.days ? Date.now() - opts.days * 86_400_000 : null;
  const kept = (e: { ts: string; project: string }) =>
    !(cutoff && new Date(e.ts).getTime() < cutoff) &&
    !(opts.project && !e.project.toLowerCase().includes(opts.project.toLowerCase()));

  const warnings: string[] = [];
  if (bad) warnings.push(`opencode: skipped ${bad} unparseable row(s)`);
  if (unnamed) warnings.push(`opencode: ${unnamed} call(s) with no model, priced as unknown`);

  return {
    events: events.filter(kept).sort((a, b) => a.ts.localeCompare(b.ts)),
    compactions: compactions.filter(kept).sort((a, b) => a.ts.localeCompare(b.ts)),
    titles,
    warnings,
  };
}
