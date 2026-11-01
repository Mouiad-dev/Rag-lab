# Scope, secrets and untrusted execution

- Obtain tenant/project scope from authenticated context. Recheck on retrieval, resume, artifact download and external action execution.
- Keep secrets out of source, prompts, traces, examples and browser bundles. Use placeholder-only `.env.example`; the owner provisions real environment values.
- Never print environment variables or credential files to debug a connection. Report variable presence or redacted errors instead.
- Do not run model-generated code in the API process or unrestricted host shell. Application code execution uses the designed sandbox port.
- SQL tools use a separate constrained readonly datasource, allowed objects/functions, timeout and result limits. SELECT/LIMIT alone are insufficient.
- Bind approvals to canonical action arguments, actor, scope and expiry; changed arguments need a new decision.
- Development hooks are partial guardrails, not an isolation or authorization boundary. Do not claim they cover shell/interpreter/MCP access.
- Use restrictive defaults and tests for sandbox resources, network, path traversal, symlinks and cleanup. No agent Docker socket or broad host mounts.
- Do not send messages, publish/deploy or perform destructive external actions without applicable user authorization.
- Do not copy third-party or proprietary code without a compatible license; record provenance of any reused code.
- Provider API keys come only from environment/secret stores; never write them to configs, logs, traces, fixtures or the UI bundle. Local providers must work with no keys.
