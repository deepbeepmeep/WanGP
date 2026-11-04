# Remote LLMs

WanGP can use Codex, Claude Code, or OpenCode as the shared LLM engine for Deepy and Prompt Enhancer. Remote engines run outside WanGP's local model runtime, so they do not occupy the VRAM reserved for media generation and do not require WanGP to unload a diffusion model before an LLM turn.

Remote engines require **Deepy Prime** because Prime exposes WanGP's MCP tools. The selected engine is also used for on-demand prompt enhancement and visual inspection. There are no separate engine selectors for those roles.

## Privacy and credentials

Remote engines can send prompts, attached images, sampled video frames, and relevant tool results over the internet. Provider privacy, retention, and processing policies apply.

WanGP does not request or store provider passwords, API keys, access tokens, or refresh tokens. Authentication is owned by Codex, Claude Code, or OpenCode. The OpenCode custom-configuration field is saved verbatim, so do not put secrets in it; use OpenCode authentication or environment-variable references.

## Common WanGP setup

1. Open `Configuration` and select `Prompt Enhancer / Deepy`.
2. Set `Prompt Enhancer / Deepy LLM Engine` to Codex, Claude Code, or OpenCode.
3. Configure the selected engine as described below.
4. Select `Deepy Prime`.
5. Click `Save Settings`.
6. Return to `Media Generator`, open `Ask Deepy`, and send a short test request.

`Refresh` queries the external engine and caches its model catalog in `wgp_config.json`. Refreshing does not change the selected model or reasoning effort. `Automatic` leaves the corresponding choice to the external engine or provider.

The chat footer displays token consumption when the engine reports it. The context limit is taken from the selected engine's model metadata; it is not a WanGP context-size setting. Remote providers manage their own context compaction. WanGP displays a compaction notice when the external protocol reports one, but it cannot display a provider-internal summary that was not returned.

## Codex

### Prerequisites and authentication

WanGP can use a standalone Codex CLI, an npm installation, or the compatible Codex binary bundled with the Codex VS Code extension. Leave `Codex executable` as `codex` for automatic detection, or enter the absolute executable path.

If authentication is required, send a Deepy request. WanGP will show the secure Codex sign-in link in the chat. Complete the browser sign-in, return to WanGP, and retry the request. The credential remains in Codex.

See the [Codex CLI documentation](https://learn.chatgpt.com/docs/codex/cli).

### WanGP fields

- `Codex executable`: `codex` for automatic detection or an absolute path.
- `Codex model`: the model reported by Codex; `Automatic` uses the recommended account default.
- `Reasoning effort`: supported values depend on the selected model.
- `Refresh`: queries the signed-in Codex account and caches the current catalog.

## Claude Code

### Prerequisites and authentication

WanGP needs both Claude Code and the compatible Python bridge. Install the bridge in WanGP's Python environment:

```powershell
pip install --upgrade claude-agent-sdk==0.1.66
```

Use this exact version rather than an unpinned SDK upgrade. WanGP 0.1.66 support preserves its pinned MCP/Pydantic stack and enables summarized Claude thinking. WanGP detects Claude Code on `PATH` and compatible binaries bundled with VS Code, VS Code Insiders, Cursor, and Windsurf extensions.

Authenticate once with the configured executable:

```powershell
claude auth login --claudeai
```

See the [Claude Code authentication documentation](https://code.claude.com/docs/en/authentication).

### WanGP fields

- `Claude executable`: `claude` for automatic detection or an absolute path.
- `Claude model`: an account model or alias; `Automatic` uses the account default.
- `Reasoning effort`: sent only when explicitly selected and supported by the model.
- Summarized Claude thinking is displayed in Deepy when Claude provides it; hidden reasoning remains hidden.
- `Refresh`: queries Claude Code and caches its model catalog.

## OpenCode

OpenCode is the universal-provider option. WanGP talks to an OpenCode HTTP server and OpenCode talks to the selected provider. This makes providers such as OpenAI, DeepSeek, OpenRouter, local OpenAI-compatible servers, and others available without provider-specific WanGP backends.

### Installation and provider authentication

Install OpenCode outside WanGP and authenticate providers using OpenCode. In the OpenCode terminal UI, run `/connect`, choose a provider, and follow its authentication flow. You can also use the OpenCode authentication commands documented by the project.

Provider credentials remain in OpenCode. After adding or removing a provider, restart an already-running OpenCode server if necessary and click `Refresh` in WanGP.

See the [OpenCode provider documentation](https://opencode.ai/docs/providers/).

### WanGP fields

- `OpenCode executable`: `opencode` when it is on `PATH`, or an absolute executable path.
- `OpenCode server URL`: defaults to `http://127.0.0.1:4096`.
- `OpenCode provider`: providers currently configured and returned by OpenCode.
- `OpenCode model`: models returned for the selected provider.
- `Reasoning effort`: OpenCode model variant, when the model publishes variants.
- `OpenCode configuration`: optional JSON/JSONC passed through `OPENCODE_CONFIG_CONTENT` only when WanGP starts a local server.
- `Refresh`: queries `/config/providers` and caches the returned providers, models, context limits, defaults, and variants.

Refresh intentionally shows only providers currently available to OpenCode. It does not list every provider OpenCode supports. For example, to expose DeepSeek V4, connect the DeepSeek provider in OpenCode, supply a DeepSeek API key there, restart the server if needed, and refresh WanGP. DeepSeek V4 Pro and Flash then appear when reported by OpenCode's provider catalog.

### Server lifecycle

WanGP first checks the configured server URL. If no server is reachable and the URL is local, WanGP starts `opencode serve` automatically on the configured port. The server remains alive between Deepy turns to preserve the OpenCode session and context.

- Sending the first Deepy request starts a missing local OpenCode server.
- `Stop Deepy` interrupts the current turn but keeps the server and session.
- `Reset` clears the Deepy session and stops the OpenCode process only when that process was started by WanGP.
- A manually started or otherwise pre-existing OpenCode server is never terminated by WanGP.
- Before closing WanGP, use `Reset` if you want a WanGP-owned OpenCode process stopped cleanly.

To run the server manually:

```powershell
opencode serve --hostname 127.0.0.1 --port 4096
```

Stop a manually started server with `Ctrl+C` in the same terminal.

## Long generations

With a remote engine, Deepy can start generations in the background and keep working while they run, for example queueing several images and following them together, or preparing the next step while a video renders.

- Like any Deepy request, its generations run after the current task and ahead of the tasks you queued yourself. Several of them run in the order it requested them.
- A generation that takes more than ten minutes does not fail: Deepy gets control back while it keeps running, follows it to the end and does not submit it twice.
- `Stop Deepy` also cancels the generations it started in the background and that are still running or waiting, unless **Auto-abort** is off.

## Resuming saved sessions

A remote engine continues a saved session with what was said and done in it: when you resume the session, even after restarting WanGP, it receives the earlier messages and tool results. The same applies when you change engine during a session and later come back to the first one: it receives what it missed.

The words of the conversation and the most recent results are always sent whole. In a very long session, the oldest tool results are shortened, and the complete record is kept in the session's documents as `conversation_history.json`, where the engine can look up a detail when it needs one.

## Prompt Enhancer and visual inspection

Prompt Enhancer uses the same selected engine. Choose the normal on-demand prompt-enhancement action; no second remote-engine configuration is required.

When a remote engine is selected, Deepy visual inspection is routed to that remote engine. Images or sampled video frames may therefore leave the machine. The local inspection control is not exposed for this configuration.

## Troubleshooting

### Executable not found

An error mentioning `WinError 2` means the executable name was not found on `PATH`. Enter its absolute path in the engine's executable field, save settings, and retry.

### Refresh returns only a few OpenCode models

Run OpenCode's authentication listing command or `/connect` and confirm the desired provider is configured. OpenCode's catalog endpoint returns enabled providers, not its complete theoretical provider list.

### An OpenCode model selection is ignored

Select an explicit `OpenCode provider` before choosing an explicit model. With both fields on `Automatic`, OpenCode uses its provider and model defaults.

### Claude SDK dependency conflicts

Install exactly the pinned bridge version shown above in WanGP's environment. Do not add or upgrade WanGP's general requirements merely to obtain a newer Claude SDK.

### A turn ends with an OpenCode provider message

When a provider reports a limit that would keep OpenCode waiting for a long time, such as `Free usage exceeded`, Deepy ends the turn and shows that message instead of waiting. Shorter retries appear in the status line while OpenCode tries again. Choose another model or provider in `Configuration`, or send the request again once the limit has reset.

### Authentication or provider changes are not visible

Complete authentication in the external engine, restart an already-running external server when applicable, then click `Refresh`. Model catalogs are cached independently from the selected model.

---

> Applies to: Configuration, authentication and troubleshooting for WanGP's remote Codex, Claude Code and OpenCode engines. Provider setup and media-generation tool interfaces are documented separately.
