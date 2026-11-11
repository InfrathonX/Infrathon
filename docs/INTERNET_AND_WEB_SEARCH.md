# Internet access and web search

**Every task's Work container (the agent's working environment) has internet
access, and web search is enabled for both Codex and Claude Code. Prompts and
task documents neither forbid nor advertise this.**

The only switch is the task JSON's `internet` field. Every task under
`tasks/latest/*/tasks/*.json` sets `"internet": true`, and a new task must do
the same (an absent field also defaults to `true`; see
`sforge/harness/task_spec.py`). With `internet: true` the runtime prompt
contains no paragraph about the network or search.

## Rules for task documents and prompts

- Do not write "offline", "no internet", "network access is unavailable" or
  "do not access the network".
- Do not advertise it either ("internet access and web search are
  available"). The agent can find out for itself.
- Restrictions on **what is submitted** are stated as usual, for example "do
  not copy external implementations, published weights or reference outputs
  into the submission".
- Restrictions on **candidate code run during evaluation** are stated as usual,
  for example "candidate code must not access the network, files or
  subprocesses". These constrain the submitted code, not the agent's working
  environment.

What internet access does not change:

- **The Judge has no network.** A submission may depend only on what the task
  description says the Judge environment already provides.
- **The LLM integrity reviewer does not search.** The reviewer agent runs with
  `web_search = "disabled"` (`sforge/harness/llm_judge/agent_engine.py`) and
  reviews the submission against the task's `llm_judge/POLICY.md`; see
  [LLM_JUDGE.md](LLM_JUDGE.md).

## How a Work container reaches the internet

The local launchers place Work on a per-run `--internal` Docker network. Work
cannot reach the outside directly; its only route out is a host-controlled
**restricted CONNECT sidecar** on that network. The sidecar also carries a
fixed forwarder to the run's Judge server and, optionally, a fixed raw TCP
relay to one model-API endpoint the operator hosts (`--model-forward
LISTEN_PORT:HOST:PORT`). GPU leases and CPU pinning are unaffected by the
network mode.

The sidecar's CONNECT has two modes:

| Mode | What is tunnelled |
| --- | --- |
| allowlist (default) | TLS tunnels on port 443 only to the model-authentication hosts (`api.openai.com`, `auth.openai.com`, `chatgpt.com`); literal IP targets are refused |
| `--allow-public-egress` | in addition, a CONNECT to any hostname or literal IP on port **80 or 443**, provided the proxy's own resolution yields only globally routable unicast addresses (`ipaddress.is_global`; multicast, reserved and IPv4-embedding IPv6 forms are refused). The proxy then dials one of those vetted addresses itself instead of re-resolving the name, so a second DNS answer cannot rebind the tunnel onto an internal address. The host, the Docker bridges, other runs' containers, link-local and metadata addresses are refused |

Every policy refusal is answered with 403.

The task JSON's `internet` flag drives the rest:

| Component | Behaviour |
| --- | --- |
| Launcher | reads `internet` from the task JSON with `jq` (it must be a boolean). When `true`, it starts the sidecar with `--allow-public-egress`, exports `MLSYSFORGE_WORK_INTERNET=true`, and records `codex_web_search: "live"` in the run record |
| Network preflight | the internal network still cannot reach the outside directly. In public-egress mode the preflight also requires `CONNECT github.com:443` to return 200 and the Docker gateway (and other non-public addresses) to return 403, and prints `Public-egress network: PASS`. The dual-GPU launchers (flux, sdv2, orpo, gkd) have no network preflight |
| Task entry | `MLSYSFORGE_WORK_INTERNET` set to `true` or unset passes internet through (Codex gets `web_search="live"`, the prompt has no network paragraph); `false` gives the offline behaviour; any other value is an error |
| Work shell | uses the sidecar through `SFORGE_HTTP(S)_PROXY` → `HTTP(S)_PROXY`, so `curl`, `pip`, `git` and `huggingface_hub` work. Work does not set `PIP_NO_INDEX`. The Miles tasks mount `runtime/public-egress.profile.sh`, which keeps the proxy settings |
| Codex (`sforge/harness/agent/codex.py`) | passed `-c web_search="live"` explicitly (`"disabled"` when `internet: false`), so an image's `codex-config.toml` cannot override it |
| Claude Code (`sforge/harness/agent/claude_code.py`) | `WebSearch` / `WebFetch` stay enabled when `internet: true`; with `internet: false` they are passed to `--disallowedTools` |

### Limitations

- The sidecar implements only `CONNECT`, so a plain `http://` request through
  `HTTP_PROXY` (an absolute-URL GET) is refused. HTTPS (pip, git, curl https)
  is unaffected; apt against an http mirror does not work (the agent has no
  root anyway).
- The host's own public IP is still reachable on ports 80 and 443.
- The smoke scripts of flash-kmeans and eagle3 start the sidecar in allowlist
  mode.
- In internet mode concurrent-serving relies on its host proxy accepting
  CONNECT requests that target an IP address.
- `HF_*_OFFLINE=1` in Dockerfiles only makes the Hugging Face libraries prefer
  the local cache; the agent can unset it.
- The LLM integrity review sees the submission and the starting files, not the
  agent's command log, so it cannot tell whether an agent downloaded a
  published answer. Tasks whose upstream weights or outputs are public rely on
  their policy rules for this.

## Web search by model route

| Route | Search tool | Status |
| --- | --- | --- |
| Codex with a ChatGPT login | `web_search = "live"` (OpenAI-hosted search) | available |
| Claude Code with a Claude subscription | `WebSearch,WebFetch` | available |
| Codex through the model relay to a provider's Responses API | Codex sends `{"type":"web_search"}` and the relay forwards it unchanged | available when the upstream provider executes hosted search (it may be billed per search) |
| Codex through the model relay to an OpenAI-compatible Chat Completions provider | the chat bridge drops hosted search tools | **no hosted search**; the agent can still use the network from its shell |

The chat bridge (`sforge/harness/relay/chat_responses_bridge.py`) skips the hosted
search tools `web_search` and `web_search_preview` and converts every other
tool as usual.

## Model relays for providers without the Responses API

Codex speaks the OpenAI Responses API. A provider that does not offer it, or
whose credentials should not enter the Work container, is reached through the
optional host model relay:

```bash
python -m sforge.harness.relay.model_proxy --config <private>/relay.json
```

The config is a private JSON file holding credential file paths only:

| Key | Meaning |
| --- | --- |
| `key_file` | the upstream provider key (a bare token or one recognized dotenv field), mode 0600 |
| `relay_token_file` | the relay's own bearer token, created if absent; this is the only credential Work receives |
| `model`, `effort` | the model and reasoning effort pinned on every request |
| `upstream_base` | the provider's OpenAI-compatible base URL |
| `vendor` | the provider kind; `openrouter` enables `provider` pinning |
| `provider` | OpenRouter only: one upstream provider, no fallback |
| `wire` | `responses` (forward native Responses calls) or `chat` (convert to Chat Completions through the chat bridge) |
| `host`, `port` | the listen address; it must be reachable from the Work container (not loopback) |
| `log_file` | usage receipts (mode 0600; request bodies and upstream credentials are not logged) |

The relay pins the model and effort, drops `previous_response_id` so no
upstream conversation state carries between runs, and answers `GET /health`.
Point a run at it with `MODEL_API_BASE_URL=http://<relay-host>:<port>/v1` and
`MODEL_API_KEY_FILE=<relay token file>` in the launch config; see
[launch/README.md](../launch/README.md).
