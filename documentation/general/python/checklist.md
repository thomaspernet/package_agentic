# Python Completion Checklist

Run the [general checklist](../checklist) first, then this one.

---

## Typing

- [ ] All functions have full type annotations — parameters and return type. No untyped functions. → [Strict Typing](typing#strict-type-checking)
- [ ] Union syntax: `X | None`, not `Optional[X]`. Built-in generics: `list[T]`, not `List[T]`. → [Modern Type Syntax](typing#modern-type-syntax-python-310)
- [ ] All data crossing layer boundaries is a Pydantic model. No raw `dict`, `tuple`, or `Any` returns. → [Pydantic Models](typing#pydantic-models)
- [ ] No `Any` in return types. `Any` allowed only in `dict[str, Any]` for genuinely heterogeneous data. → [No Any Abuse](typing#no-any-abuse)

## Architecture

- [ ] Interfaces (ABCs) live next to the abstractions they define; concrete implementations depend on interfaces, not vice versa. → [Interface-First Design](architecture#interface-first-design)
- [ ] Dependencies injected via constructor. No global singletons reached through module-level state. → [Dependency Injection](architecture#dependency-injection)
- [ ] Module organisation prefers depth over flat dumps; every package has explicit `__all__` in `__init__.py`. → [Module Organization](architecture#module-organization)

## Patterns

- [ ] All I/O is async. No sync alternatives. Repository, service, and API methods are all `async`. → [Async-First](patterns#async-first)
- [ ] Error handling uses `ValueError` (client, 400) and `RuntimeError` (server, 500). No custom exception hierarchies unless genuinely needed. → [Error Handling](patterns#error-handling)
- [ ] Exception chaining: `raise NewError(...) from original`. Never re-raise without `from`. → [Error Handling](patterns#error-handling)
- [ ] Guard clauses at the top of functions. Fail fast, early returns, no deep nesting. → [Guard Clauses](patterns#guard-clauses--early-returns)
- [ ] Entity lookups use `require_entity()` or equivalent. No silent None returns for required data. → [Guard Clauses](patterns#guard-clauses--early-returns)
- [ ] No hardcoded values. Thresholds, timeouts, limits come from config. → [No Hardcoded Values](patterns#no-hardcoded-values)
- [ ] Checked for existing helpers before writing new ones. Helpers placed in the right scope. → [Before Writing a New Function](patterns#before-writing-a-new-function-check-what-exists)

## Conventions

- [ ] Naming: PascalCase classes, snake_case functions, UPPER_SNAKE_CASE constants, I-prefix interfaces. → [Naming](conventions#naming)
- [ ] Absolute imports only. No relative imports. → [Import Organization](conventions#import-organization)
- [ ] Logging uses `%s` parameterized format, not f-strings. → [Logging](conventions#logging)
- [ ] `__init__.py` has explicit `__all__`. No star imports. → [\_\_init\_\_.py Pattern](conventions#__init__py-pattern)

## Secrets

- [ ] No API keys, tokens, or credentials in code or YAML. Always loaded from environment (`OPENAI_API_KEY`, etc.).

## Observability

- [ ] New behavior has appropriate logging. INFO for successful operations, WARNING for recoverable issues, ERROR for failures. → [Logging](conventions#logging)
- [ ] No silent exception swallowing. Every `except` block either logs, re-raises, or has a comment explaining why it is safe to ignore. → [Error Handling](patterns#error-handling)
- [ ] Log messages include enough context to debug without reproducing (entity UUID, operation name, relevant parameters). → [Logging](conventions#logging)

## Tests

- [ ] Async tests use `AsyncMock` and `@pytest.mark.asyncio`. → [Async Test Patterns](testing#async-test-patterns)
- [ ] Mocks patch at the import site, not the definition site. → [Mock Strategy](testing#mock-strategy)
- [ ] Test data uses factory functions with `**overrides`. → [Test Helpers](testing#test-helpers)
