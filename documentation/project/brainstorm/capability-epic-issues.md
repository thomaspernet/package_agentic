# Capability Pattern - Epic & Child Issues

> Ready-to-post GitHub issues for `thomaspernet/sinan-agentic`.
> Source briefing: `capability-epic-briefing.md`.

**Suggested labels for all issues:** `capability-pattern`
**Per-issue labels:** noted inline.

---

## EPIC

**Title:** `[EPIC] Introduce Capability abstraction for pluggable agent behaviors`

**Labels:** `epic`, `capability-pattern`, `enhancement`

**Body:**

### Vision

Today, cross-cutting agent behaviors (turn budgets, tool-error recovery) are hardcoded inside `_CompositeHooks` in `sinan_agentic_core/core/base_runner.py:1037-1073`. Each new behavior - skills, compaction, memory, audit logging - requires editing `base_runner.py` again.

OpenAI's Agents SDK 0.14 (released 2026-04-15) introduced `agents.sandbox.Capability` as a clean primitive for exactly this kind of pluggable behavior. We've already built this pattern by hand twice (`TurnBudget`, `ToolErrorRecovery`); it's time to extract it.

### Goals

- Define a `Capability` protocol whose shape mirrors OpenAI's so future interop is cheap.
- Refactor `TurnBudget` and `ToolErrorRecovery` to implement it - **zero behavior change** for current users.
- Make capabilities pluggable via `AgentDefinition.capabilities` and the YAML catalog.
- Establish the extension point so adding a new behavior is "write a `Capability` subclass" instead of "edit `base_runner.py`."

### Non-goals (explicitly deferred)

- No `Manifest`, `Workspace`, sandbox client, or filesystem-isolation concept.
- No dependency on OpenAI's `agents.sandbox.*` internals.
- No backward-compat shims for renamed/removed helpers - dead code gets deleted.
- Snapshot/rehydrate is a **stretch** issue (#7); fine to ship the epic without it.

### Child issues (in dependency order)

- [ ] #1 - Define `Capability` protocol and base class (S)
- [ ] #2 - Refactor `TurnBudget` to implement `Capability` (M)
- [ ] #3 - Refactor `ToolErrorRecovery` to implement `Capability` (M)
- [ ] #4 - Wire `capabilities` field into `AgentDefinition` and generalize `_CompositeHooks` (M)
- [ ] #5 - YAML registry support for capabilities (M)
- [ ] #6 - Migration notes, custom-capability example, README update (S)
- [ ] #7 - **(Stretch)** Capability snapshot/rehydrate via `SQLiteSessionStore` (L)

**Dependency graph:** `1 -> (2, 3 in parallel) -> 4 -> 5 -> 6`. `7` depends on `4`, otherwise independent.

### Success criteria

- `pytest` is green at every issue boundary; no test regressions in `tests/test_turn_budget.py` or `tests/test_tool_error_recovery.py`.
- `mypy --strict sinan_agentic_core` passes.
- Adding a new capability requires **no edits** to `base_runner.py` after #4 lands.

### References

- Briefing: `documentation/project/brainstorm/capability-epic-briefing.md`
- OpenAI blog: https://openai.com/index/the-next-evolution-of-the-agents-sdk/
- SDK reference: https://openai.github.io/openai-agents-python/ref/sandbox/

---

## ISSUE 1 - Define `Capability` protocol and base class

**Title:** `feat(capabilities): define Capability protocol and base class`

**Labels:** `capability-pattern`, `enhancement`

**Complexity:** S

**Depends on:** none

**Description:**

Pure addition. Create the `Capability` protocol that subsequent issues will implement. No refactor of existing code in this issue.

### Implementation

- New module: `sinan_agentic_core/core/capabilities/__init__.py`
- New module: `sinan_agentic_core/core/capabilities/base.py`
- Define `Capability` (ABC or `Protocol` - choose whichever type-checks more cleanly under `mypy --strict`) with:
  - `instructions(self, ctx) -> str | None` - contribute a fragment to the system prompt; return `None` to contribute nothing.
  - `on_tool_start(self, ctx, tool, args) -> None` - default no-op.
  - `on_tool_end(self, ctx, tool, result) -> None` - default no-op.
  - `reset(self) -> None` - called at the start of each `execute()` call; default no-op.
  - `clone(self) -> "Capability"` - per-run copy; capabilities are stateful and must not leak across runs.
  - Optional: `tools(self) -> list[Tool]` - default `[]`. Reserved for future use.

### Acceptance criteria

- [ ] `from sinan_agentic_core.core.capabilities import Capability` succeeds.
- [ ] A trivial `DummyCapability(Capability)` in tests type-checks under `mypy --strict`.
- [ ] Unit tests in `tests/core/capabilities/test_base.py` cover: default no-op behavior of every lifecycle method, `clone()` returns an independent instance, `reset()` callable.
- [ ] `pytest` and `mypy --strict sinan_agentic_core` both pass.

---

## ISSUE 2 - Refactor `TurnBudget` to implement `Capability`

**Title:** `refactor(capabilities): make TurnBudget a Capability`

**Labels:** `capability-pattern`, `refactor`

**Complexity:** M

**Depends on:** #1

**Description:**

Adopt the `Capability` protocol on the existing `TurnBudget` class in `sinan_agentic_core/core/turn_budget.py`. **No behavior change.** This is purely a protocol-adoption refactor.

### Implementation

- `class TurnBudget(Capability):` (or implements the Protocol).
- Map existing methods to capability lifecycle:
  - `instructions(ctx)` -> wraps the existing `build_instruction_section()`.
  - `reset()` -> wraps the existing `reset()`.
  - `clone()` -> returns a fresh `TurnBudget` with the same configuration but zeroed counters.
- Public constructor and existing public methods stay unchanged - users importing `TurnBudget` directly see no break.
- Delete any helper paths that become dead after the refactor (per project rule: no backward-compat shims).

### Acceptance criteria

- [ ] `TurnBudget` satisfies the `Capability` protocol (verified by a `isinstance` / `issubclass` test or `mypy` check).
- [ ] All existing tests in `tests/test_turn_budget.py` pass **unchanged**.
- [ ] New test confirms `clone()` produces an independent instance with the same config but reset state.
- [ ] No diff in `base_runner.py` yet - that's #4.
- [ ] `pytest` and `mypy --strict` pass.

---

## ISSUE 3 - Refactor `ToolErrorRecovery` to implement `Capability`

**Title:** `refactor(capabilities): make ToolErrorRecovery a Capability`

**Labels:** `capability-pattern`, `refactor`

**Complexity:** M

**Depends on:** #1

**Description:**

Same shape as #2 but for `ToolErrorRecovery` in `sinan_agentic_core/core/tool_error_recovery.py`. Can be done in parallel with #2.

### Implementation

- `class ToolErrorRecovery(Capability):` (or implements the Protocol).
- Map existing methods:
  - `instructions(ctx)` -> wraps existing instruction-building helper.
  - `on_tool_end(ctx, tool, result)` -> wraps existing `RunHooks.on_tool_end` logic that records errors.
  - `reset()` -> clears the per-run error tally.
  - `clone()` -> fresh instance with same config, zeroed error state.
- Public API preserved. Dead helpers deleted.

### Acceptance criteria

- [ ] `ToolErrorRecovery` satisfies the `Capability` protocol.
- [ ] All existing tests in `tests/test_tool_error_recovery.py` pass **unchanged**.
- [ ] New test confirms `clone()` independence.
- [ ] No diff in `base_runner.py` yet.
- [ ] `pytest` and `mypy --strict` pass.

---

## ISSUE 4 - Wire `capabilities` into `AgentDefinition` and generalize `_CompositeHooks`

**Title:** `feat(capabilities): pluggable capabilities on AgentDefinition`

**Labels:** `capability-pattern`, `enhancement`, `refactor`

**Complexity:** M

**Depends on:** #2, #3

**Description:**

This is the integration step. After this lands, adding a new behavior requires no edits to `base_runner.py`.

### Implementation

- `sinan_agentic_core/registry/agent_registry.py` - add to `AgentDefinition`:
  ```python
  capabilities: list[Capability] = field(default_factory=list)
  ```
- `sinan_agentic_core/core/base_runner.py:1037-1073` - `_CompositeHooks` no longer references `TurnBudget` or `ToolErrorRecovery` by name. It iterates `agent_def.capabilities` and dispatches `on_tool_start` / `on_tool_end` / etc. to each.
- `sinan_agentic_core/core/base_runner.py:860-926` - `_apply_dynamic_instructions()` merges all `capability.instructions(ctx)` fragments (in registration order; skip `None`s).
- `sinan_agentic_core/core/base_runner.py:91-151` - `create_agent()` calls `capability.clone()` once per run for each capability and uses the clones for that run's hooks (per-run isolation).
- Reset all per-run capability state at the top of `execute()` via `capability.reset()`.
- **Delete** the now-unreachable hardcoded `TurnBudget` / `ToolErrorRecovery` branches in `_CompositeHooks` and the dynamic-instructions builder.

### Acceptance criteria

- [ ] Existing integration tests pass with **no behavior change** when `TurnBudget(...)` and `ToolErrorRecovery(...)` are passed in `AgentDefinition.capabilities`.
- [ ] New test: a custom `LoggingCapability` (records every `on_tool_start` call) wired via `capabilities=[LoggingCapability()]` is invoked correctly without any `base_runner.py` edits.
- [ ] New test: capability state does not leak across two sequential `execute()` calls (verifies `clone()` + `reset()`).
- [ ] `_CompositeHooks` no longer mentions `TurnBudget` or `ToolErrorRecovery` by name.
- [ ] `pytest` (with coverage gate) and `mypy --strict` pass.

---

## ISSUE 5 - YAML registry support for capabilities

**Title:** `feat(capabilities): YAML catalog support and CapabilityRegistry`

**Labels:** `capability-pattern`, `enhancement`

**Complexity:** M

**Depends on:** #4

**Description:**

Make capabilities declarable from `agents.yaml` so users don't need Python wiring.

### Implementation

- New module: `sinan_agentic_core/registry/capability_registry.py` - `CapabilityRegistry` parallel to `ToolRegistry`. Decorator `@register_capability("name")` to register a `Capability` factory.
- `sinan_agentic_core/registry/agent_catalog.py:54-89` - parse a `capabilities:` block on each agent entry. Two supported forms:
  - Built-in shorthand keys (resolved into the corresponding factory automatically):
    ```yaml
    turn_budget:
      max_turns: 10
    error_recovery:
      max_consecutive_errors: 3
    ```
  - Explicit list referencing registered names:
    ```yaml
    capabilities:
      - name: turn_budget
        config: { max_turns: 10 }
      - name: my_custom_capability
    ```
- Built-in shorthands `turn_budget` and `error_recovery` register on import.
- Validate at load time: unknown capability name -> clear `CapabilityNotFoundError`.

### Acceptance criteria

- [ ] `tests/registry/test_capability_registry.py` covers: registration, lookup, unknown-name error, factory invocation with config.
- [ ] `tests/registry/test_agent_catalog.py` covers: both YAML forms, mixing shorthand and explicit list, validation errors.
- [ ] An example YAML in `examples/` exercises a capability declared from YAML.
- [ ] `pytest` and `mypy --strict` pass.

---

## ISSUE 6 - Migration notes, custom-capability example, README update

**Title:** `docs(capabilities): migration notes, custom example, README extension section`

**Labels:** `capability-pattern`, `documentation`

**Complexity:** S

**Depends on:** #5

**Description:**

Document the new extension point. The migration story should be near-zero-change for existing users.

### Implementation

- `documentation/project/capabilities.md` (new) - protocol overview, lifecycle diagram (text is fine), how to write a custom capability.
- Update `README.md` "Extending sinan-agentic" section with a short capability example (e.g., a `ToolCallLogger` capability that prints every tool call).
- `examples/custom_capability.py` - runnable end-to-end sample.
- Migration notes section: existing `TurnBudget` / `ToolErrorRecovery` users see no change; YAML users get the new `capabilities:` block.

### Acceptance criteria

- [ ] `examples/custom_capability.py` runs without error.
- [ ] README links to the new docs page.
- [ ] Migration notes explicitly state "no required changes" for existing direct-Python users.
- [ ] `pytest` (in case the example is exercised) passes.

---

## ISSUE 7 - (Stretch) Capability snapshot/rehydrate

**Title:** `feat(capabilities): durable snapshot/rehydrate via SQLiteSessionStore`

**Labels:** `capability-pattern`, `enhancement`, `stretch`

**Complexity:** L

**Depends on:** #4 (independent of #5, #6)

**Description:**

Optional follow-up: persist capability state across process restarts so a session can resume a long-running agent mid-budget, mid-recovery, etc.

### Implementation

- Extend the `Capability` protocol with **optional** methods:
  - `to_snapshot(self) -> dict | None` - default returns `None` (capability is stateless / not durable).
  - `from_snapshot(self, data: dict) -> None` - default no-op.
- `TurnBudget` and `ToolErrorRecovery` implement both; serialize their counters.
- `sinan_agentic_core/session/sqlite_store.py` - persist a JSON blob per capability per session row.
- `sinan_agentic_core/session/agent_session.py` - on session load, call `from_snapshot()` for each capability whose snapshot is present.

### Acceptance criteria

- [ ] Round-trip test: configure a `TurnBudget` to 10 turns, advance counter to 4, snapshot, rehydrate in a new session, confirm 6 turns remain.
- [ ] Same round-trip for `ToolErrorRecovery`.
- [ ] Capabilities without snapshot support continue to work (no crash, no persistence).
- [ ] Schema migration for `SQLiteSessionStore` is forward-only and documented.
- [ ] `pytest` and `mypy --strict` pass.

---

## Posting

If you want to post these via `gh`:

```bash
# From repo root, with gh authenticated
gh issue create --repo thomaspernet/sinan-agentic \
  --title "[EPIC] Introduce Capability abstraction for pluggable agent behaviors" \
  --label epic,capability-pattern,enhancement \
  --body-file <(sed -n '/^## EPIC/,/^---$/p' documentation/project/brainstorm/capability-epic-issues.md)
# ... repeat per child issue, then edit the epic body to substitute real issue numbers for the #1..#7 placeholders.
```
