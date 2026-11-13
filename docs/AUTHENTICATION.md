# Authentication, HTTPS, and Reverse Proxies

This guide covers shared Gradio/Deepy web authentication, public proxy URLs, TLS certificates, and the separate MCP OAuth flow. Authentication is optional and disabled by default. The [CLI reference](CLI.md) lists the available flags.

## Web Authentication

Authentication is optional and off by default. `--auth` enables one password-only login for **all Gradio and Deepy web access**, including APIs, galleries, downloads, uploads and live connections. No username is needed.

```powershell
# Generate a new password and print it in the terminal
python wgp.py --listen --auth

# Choose a fixed passphrase
python wgp.py --listen --auth --auth-password "your long private passphrase"

# The same options work with the standalone Deepy Web app
python wgp.py --deepy-server --listen --auth
```

Open the usual Gradio or Deepy address and enter the password. A login covers both interfaces on the same hostname. Browser sessions expire after 24 hours; restarting WanGP invalidates every session. A generated password also changes at each launch. To avoid putting a fixed passphrase in command history, set `WANGP_AUTH_PASSWORD` in the launch environment and use `--auth`. An explicit `--auth-password` takes precedence. Passwords supplied by you are not printed by WanGP.

Login attempts are limited across all clients and web interfaces in this process. The first four failures have no delay. After failure 5, wait 30 seconds; each further failure adds 30 seconds, reaching 450 seconds after failure 19. From failure 20, only one attempt every ten minutes is allowed. Only one password check can run at a time. Requests during the waiting period do not extend it. A successful login resets the failure counter. Existing signed-in sessions keep working during a cooldown. Restarting WanGP resets the counter as well as all sessions.

Choose protection according to how the server is reached:

- **Only this PC:** the default localhost access usually needs no application password or certificate.
- **Trusted private LAN:** authentication is useful on shared networks. HTTPS protects the passphrase and generated media from network interception.
- **VPN-only access:** application authentication can be optional if firewall/VPN rules restrict access to trusted users and the entire connection is protected. Keep public port forwarding closed. A VPN ending at your router may leave the final LAN connection unencrypted.
- **Public access, including NAT port forwarding:** enable authentication and trusted HTTPS. NAT alone does not protect a forwarded port. Forward only the HTTPS port; never expose a password login over plain HTTP.

Network MCP has a **separate OAuth login**, enabled with `--mcp-auth`. The web password and browser cookie do not authorize MCP clients. See [MCP authentication](#mcp-authentication-and-https).

## HTTPS Certificates

The certificate options apply to Gradio, Deepy and network MCP. Obtain a certificate and private key for the exact hostname clients will use. Public access needs a certificate trusted by those clients, commonly issued for your domain by a public certificate authority or managed by an HTTPS reverse proxy. For a private LAN, [mkcert](https://github.com/FiloSottile/mkcert) can create a local certificate; each client device must trust that local certificate authority. Keep its CA private key and the server private key private.

Serve HTTPS directly on the main port:

```powershell
python wgp.py --listen --auth --server-port 7860 --ssl-certfile C:\certs\wangp.pem --ssl-keyfile C:\certs\wangp-key.pem
```

Open `https://<certificate-hostname>:7860/`, or `/deepy/` for the mobile Web app. Add `--deepy-server` for standalone Deepy at `/`.

To redirect HTTP on the main port to a separate HTTPS port:

```powershell
python wgp.py --listen --auth --server-port 7860 --https-port 7861 --ssl-certfile C:\certs\wangp.pem --ssl-keyfile C:\certs\wangp-key.pem
```

Use `https://<certificate-hostname>:7861/`. The HTTP port redirects; it does not serve a second unencrypted application. Alternatively, set `WANGP_SSL_CERT` and `WANGP_SSL_KEY` in the launch environment. Command-line certificate paths take precedence. Missing, mismatched or unreadable certificate/key files stop startup.

An HTTPS reverse proxy can manage certificates instead; see [Hosting Behind a Reverse Proxy](#hosting-behind-a-reverse-proxy).

Browser microphone recording can require trusted HTTPS even over a VPN. Native phone keyboard dictation does not use Deepy's microphone access.

## Hosting Behind a Reverse Proxy

By default, WanGP accepts browser origins using **HTTP or HTTPS** when the hostname and any explicit port match the request's Host header. This lets an HTTPS proxy forward HTTP to WanGP without a scheme mismatch. Different hostnames and explicit ports remain rejected. The rule applies to HTTP requests and shared WebSocket connections, with or without password authentication.

If Nginx, Traefik, Cloudflare, RunPod, or another proxy preserves the public Host header, no public URL option is needed. WanGP accepts either browser scheme for that host, and the Gradio page corrects its API root to the browser's scheme.

On HTTPS pages, the Gradio frontend also adds a `Content-Security-Policy` meta policy with `upgrade-insecure-requests`. This makes the browser upgrade HTTP subresource URLs, including absolute media URLs returned by later Gradio events, before requesting them. The policy applies to all insecure subresources, including third-party URLs; their HTTPS endpoints must be available because there is no HTTP fallback. Direct LAN access over HTTP does not receive this policy. This does not enable TLS on the backend, change authentication or origin checks, or replace HTTPS termination at the proxy. Test both audio/video playback and API requests after configuring a proxy.

To restrict browser access to one exact origin, or handle a proxy that rewrites the Host header, set `--public-url`:

```bash
python wgp.py --listen --public-url https://wangp.example.com

# Optional exact-origin restriction for RunPod.
python wgp.py --listen --public-url https://POD-ID-7860.proxy.runpod.net
```

Use only the scheme, hostname and optional public port (for example `https://wangp.example.com:8443`), without `/deepy/`, another path, a query or credentials. A trailing `/` is accepted. Add `--auth` to require a password; `--public-url` does not enable authentication itself.

With this option, only the configured browser origin is accepted: `--public-url https://wangp.example.com` rejects `http://wangp.example.com` as well as other hosts or ports. It also handles proxies that rewrite the backend Host header, where the default same-address rule cannot match. Forwarded headers do not override this restriction. Login cookies use the configured scheme, or the accepted browser scheme when the option is omitted. There is no provider-specific detection.

The default deliberately trusts the HTTP and HTTPS variants of the same address. Use `--public-url` if those variants serve different applications or you want to pin a single origin. Origin checks are browser protections, not authentication for arbitrary network clients; `--auth` remains independent.

The option also gives Gradio an explicit browser-facing origin when the proxy rewrites the Host header. It does not configure the proxy, change the listening port, mount the application under a URL prefix, or enable TLS inside WanGP. Keep browser HTTPS enabled and serve WanGP at the public origin's root. Deepy uses WebSocket when available and falls back to bounded HTTP polling if a proxy rejects the upgrade. Gradio uses HTTP streaming for its queue, so the proxy must pass streaming responses. No `FORWARDED_ALLOW_IPS='*'` override is needed for these origin checks. Use the same public origin for browser access to Gradio and its `/deepy/` app. For proxy-managed HTTPS, omit WanGP's `--https-port` redirect option; use certificate options only if the proxy also connects to WanGP over HTTPS. MCP OAuth has its separate `--mcp-auth-url` option.

### RunPod Setup

For a pod with WanGP already installed:

1. Update WanGP and use the pod's usual WanGP Python environment.
2. Expose WanGP's internal port, for example **7860**, in the pod/template's **Expose HTTP Ports** field. If it is already exposed, leave it there. Keep other ports the template needs. See [RunPod's port setup](https://docs.runpod.io/pods/configuration/expose-ports).
3. Copy the HTTPS address for that service from the pod's **Connect** panel. For example, `https://abc123xyz-7860.proxy.runpod.net`. Use your actual address, not this example.
4. In the WanGP directory, start WanGP on the exposed internal port:

   ```bash
   python wgp.py --listen --server-port 7860 --auth
   ```

   Keep your other usual WanGP arguments. If the template already starts WanGP, update its startup command and restart that instance instead of starting a second server on the same port. `--auth` prints a generated password in the terminal; use the password options above if you need a fixed passphrase.
5. Open the copied HTTPS address and sign in. Gradio is at `/`; the synchronized Deepy Web app is at `/deepy/`. Open a second tab at the same origin to use shared galleries/settings.

RunPod supplies browser HTTPS and forwards requests to WanGP's HTTP listener. When the public Host header is preserved, WanGP accepts the scheme difference and the Gradio page uses the browser's HTTPS origin even if the proxy omits `X-Forwarded-Proto`. No `--public-url` is needed for that case. Keep the browser on HTTPS; the HTTP port field describes the service inside the pod.

Optionally, add `--public-url https://abc123xyz-7860.proxy.runpod.net` to enforce that exact HTTPS origin, or if your template's additional proxy rewrites the hostname or port. Use your actual address, without `/deepy/`. For a reusable **shell startup script** with this stricter restriction, the URL can be built from the [pod ID RunPod supplies](https://docs.runpod.io/pods/templates/environment-variables):

```bash
python wgp.py --listen --server-port 7860 --public-url "https://${RUNPOD_POD_ID}-7860.proxy.runpod.net" --auth
```

Use the same internal port in the exposed HTTP ports, `--server-port`, and URL. If you replace the pod or change its exposed port, update a literal URL accordingly; the shell example uses the current pod ID automatically. This is script configuration, not provider-specific behavior inside WanGP.

If reconnection continues, check that the Gradio page's `window.gradio_config.root` is the public HTTPS origin, and inspect `/gradio_api/queue/data` for an active HTTP event stream. Check `/deepy/deepy_api/events` for a WebSocket **101** response; Deepy falls back to `/deepy/deepy_api/events/poll` when the WebSocket fails. A **403** points to origin validation or login; a failed queue stream or polling request needs its HTTP status and server logs checked.

## MCP Authentication and HTTPS

Network MCP supports optional OAuth 2.1 authorization, separate from the Gradio/Deepy browser login. Use it when external MCP clients can reach WanGP over an untrusted network. Local `stdio` connections do not use OAuth. Keep an unauthenticated network MCP server restricted to localhost or a trusted VPN.

Update the project dependencies first with `python -m pip install -r requirements.txt`. The network authentication uses the maintained MCP 1.x SDK; the MCP protocol version is independent of WanGP's API v1/v2 tool selection.

For direct HTTPS with a generated MCP approval passphrase:

```powershell
python wgp.py --mcp --mcp-transport streamable-http --mcp-host 0.0.0.0 --mcp-port 7866 --mcp-auth --mcp-auth-url https://wangp.example.com:7866 --ssl-certfile C:\certs\wangp.pem --ssl-keyfile C:\certs\wangp-key.pem
```

Connect your OAuth-capable MCP client to `https://wangp.example.com:7866/mcp`. `--mcp-auth-url` is the externally reachable **origin**, including a non-default port, without `/mcp`, other paths or query parameters. The certificate must cover that hostname and be trusted by the client. HTTPS is required for non-loopback OAuth origins.

The client discovers authorization settings, registers, then opens the **Authorize MCP Access** page. Check the client name and callback address, and enter the separate MCP passphrase printed at startup only if you initiated the connection. Approval gives that client access to the WanGP tools and media permitted by the server's configuration. The shared `wangp` scope is full server access, not a read-only or per-tool permission. OAuth does not expand filesystem permissions: the existing `--mcp-allow-read-file-system` option remains separate.

Use `--mcp-auth-password "your separate long passphrase"` with `--mcp-auth` for a fixed passphrase, or set `WANGP_MCP_AUTH_PASSWORD` in the launch environment. An explicit CLI passphrase takes precedence. Generated passphrases change at restart; supplied passphrases are not printed. Web `--auth` passwords and browser sessions do not authorize MCP requests.

The same options protect the MCP address when enabled with the web UI, for example `python wgp.py --listen --auth --mcp-webui --mcp-host 0.0.0.0 --mcp-port 7866 --mcp-auth --mcp-auth-url https://wangp.example.com:7866 --ssl-certfile C:\certs\wangp.pem --ssl-keyfile C:\certs\wangp-key.pem`. Web `--auth` then protects the browser pages and MCP OAuth the agents. The certificate options also serve the MCP port over HTTPS; `--https-port` redirects only the browser port.

For an HTTPS reverse proxy on the same PC:

```powershell
python wgp.py --mcp --mcp-transport streamable-http --mcp-host 127.0.0.1 --mcp-port 7866 --mcp-auth --mcp-auth-url https://wangp.example.com
```

The proxy handles the certificate. Forward the whole origin, including authorization and discovery endpoints, preserve the original Host header, and forward the correct scheme. Keep the backend private. Hosting under a URL subpath is not supported. For local development only, an origin such as `http://127.0.0.1:7866` is accepted without a certificate.

The same certificate flags work with `python -m shared.mcp`; use that entry point's `--transport`, `--host` and `--port` names. `--https-port` optionally adds HTTPS while redirecting the main HTTP port. Legacy SSE transport uses `/sse` and the same OAuth protection; Streamable HTTP is recommended for new client connections.

Client requirements and session behavior:

- Authorization-code flow with S256 PKCE, authorization-server and protected-resource discovery, and dynamic client registration are supported. Registration accepts public clients and clients using `client_secret_post` or `client_secret_basic`. Redirects must use HTTPS or loopback HTTP. URL-based client metadata documents and private-key client authentication are not supported.
- Clients send the issued access token in `Authorization: Bearer ...` on **every** MCP request and direct media upload/download. Never put a token or passphrase in a URL. A browser login cookie or the passphrase itself is not a bearer token.
- Authorization approvals expire after ten minutes and issued codes after two minutes. Access tokens expire after one hour. Refresh tokens rotate and expire at most seven days after the original approval. Reusing an old refresh token revokes that approval; reconnect the client. Clients can also revoke tokens through the advertised revocation endpoint.
- Server restart invalidates registered clients, approvals and tokens, even with a fixed passphrase. Reconnect or remove and re-add the server in clients that retain stale registration details. Unapproved registrations expire after ten minutes; approved registrations expire after 30 days.
- MCP password checks use the same [progressive delay rules](#web-authentication) as the web login, with a separate global counter. Existing authorized clients continue working during a login cooldown. Only one MCP password check runs at a time.

For NAT port forwarding, expose only trusted HTTPS with authentication enabled. See [HTTPS setup](#https-certificates) for certificate and VPN guidance.

---

> Applies to: WanGP and Deepy web authentication, HTTPS and reverse-proxy setup, and MCP OAuth configuration.

