<p align="center">
  <img src="frontend/public/smolagents.webp" alt="ML Intern" width="120" />
  <h1 align="center">ML Intern Local</h1>
  <p align="center">
    Autonomous ML engineering assistant with Telegram gateway, local job management,<br/>
    approval cockpit, and persistent scheduling — all running on your machine.
  </p>
</p>

---

## Features

- **Autonomous ML agent** — researches papers, writes training code, runs experiments
- **Web UI** — React + MUI dashboard with real-time streaming chat
- **Telegram gateway** — full bot with inline menus, model switching, approval flow
- **Local execution** — runs training directly on your GPU(s), no sandbox needed
- **HF sandbox mode** — optional HuggingFace Jobs for cloud compute
- **Persistent cron** — schedule recurring prompts that survive restarts
- **Job manager** — start/stop/kill/monitor local training jobs with per-job logs
- **Approval cockpit** — approve or reject sensitive operations from Telegram or Web UI
- **RBAC identity** — multi-user roles (owner/admin/user/viewer) for Telegram access
- **Event store** — append-only audit log for all gateway operations

---

## Quick Start

### Prerequisites

- **Python 3.11+**
- **Node.js 18+** (for frontend)
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager
- A **provider API key** (at least one): Anthropic, OpenAI, MiniMax, or Z.ai

### 1. Clone & Install

```bash
git clone https://github.com/jomangbp/ml-intern-local.git
cd ml-intern-local

# Install Python backend + CLI
uv sync
uv tool install -e .

# Install frontend (optional, for Web UI)
cd frontend
npm install
cd ..
```

### 2. Configure API Keys

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env` with your provider keys:

```ini
# HuggingFace (required for dataset/hub access)
HF_TOKEN=hf_...

# Anthropic (for Claude models)
ANTHROPIC_API_KEY=sk-ant-...

# MiniMax Token Plan — https://platform.minimax.io
MINIMAX_API_KEY=...

# Z.ai dev platform — https://docs.z.ai
ZAI_API_KEY=...
```

### OpenAI via Codex Subscription (free setup, no API key)

GPT-5.x models are routed through your **ChatGPT Pro/Plus Codex subscription** —
no paid API key required:

```bash
# 1. Install the Codex CLI
npm install -g @openai/codex

# 2. Authenticate (opens browser)
codex login --device-auth

# 3. Done — ml-intern auto-detects the token
```

The token is stored at `~/.config/codex/auth.json` and picked up automatically.
You can also set `CODEX_AUTH_TOKEN` in `.env` if you prefer.

### 3. Run

**CLI (interactive chat):**

```bash
ml-intern
```

**CLI (single prompt, auto-run):**

```bash
ml-intern "fine-tune llama on my dataset"
```

**Web UI + Backend:**

```bash
# Terminal 1: backend (API on :7860)
cd backend
uvicorn main:app --host 0.0.0.0 --port 7860

# Terminal 2: frontend (dev server on :5173)
cd frontend
npm run dev
```

Open `http://localhost:5173` for the dashboard.

---

## Telegram Bot Setup

### 1. Create a Bot

1. Open Telegram, search for **@BotFather**
2. Send `/newbot` and follow the prompts
3. Copy the bot token (format: `123456:ABC-DEF...`)

### 2. Get Your Chat ID

Send any message to your bot, then visit:

```
https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
```

Look for `"chat":{"id": 123456789}` — that number is your chat ID.

### 3. Configure

Add to your `.env`:

```ini
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_ALLOWED_CHAT_IDS=123456789
TELEGRAM_EXECUTION_MODE=local
```

Or configure from the **Web UI** → Settings → Telegram Bot.

### 4. Start the Gateway

The Telegram bot starts automatically when the backend launches if `TELEGRAM_BOT_TOKEN` is set.

### Bot Commands

| Command | Description |
|---|---|
| `/start` | Show interactive menu |
| `/new` | Create fresh session |
| `/models` | Choose model (inline keyboard) |
| `/status` | Session status |
| `/jobs` | List running jobs |
| `/logs <id>` | View job logs |
| `/kill <id>` | Kill a running job |
| `/cron <min> <prompt>` | Schedule recurring prompt |
| `/approvals` | List pending approvals |
| `/interrupt` | Stop current agent turn |

---

## Supported Models

| Model | Provider | Key / Auth |
|---|---|---|
| Claude Opus 4.6 | Anthropic | `ANTHROPIC_API_KEY` |
| GPT-5.3 / 5.4 / 5.5 | OpenAI | **Codex subscription** (no API key) |
| GPT-5.5 Codex | OpenAI | **Codex subscription** (no API key) |
| MiniMax M2.7 | MiniMax | `MINIMAX_API_KEY` |
| GLM 5.1 | Z.ai | `ZAI_API_KEY` |
| Kimi K2.6 | HuggingFace | `HF_TOKEN` |

> **OpenAI note:** GPT models use your ChatGPT Pro/Plus subscription via the
> Codex CLI. Run `codex login --device-auth` once — no `OPENAI_API_KEY` needed.

Switch models on-the-fly via Telegram (`/models`) or the Web UI model selector.

---

## Architecture

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   CLI User   │  │  Web UI      │  │  Telegram    │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                  │
       └────────┬────────┘                  │
                │                           │
        ┌───────▼───────┐          ┌────────▼────────┐
        │  FastAPI       │          │  Telegram Bot   │
        │  Backend       │◄────────►│  (subprocess    │
        │  (:7860)       │  events  │   polling)      │
        └───────┬───────┘          └─────────────────┘
                │
        ┌───────▼───────┐
        │  Agent Loop   │
        │  (max 300     │
        │   iterations) │
        └───────┬───────┘
                │
    ┌───────────┼───────────┐
    │           │           │
┌───▼──┐  ┌────▼────┐  ┌───▼──────┐
│ HF   │  │ Local   │  │ Research │
│ Tools│  │ Tools   │  │ Tools    │
└──────┘  └─────────┘  └──────────┘
```

### Key Components

| Component | Path | Description |
|---|---|---|
| Agent loop | `agent/core/agent_loop.py` | Iterative LLM tool-calling loop |
| Session manager | `backend/session_manager.py` | Creates/restores agent sessions |
| Telegram bot | `backend/telegram_bot.py` | Subprocess-based polling, Hermes-style display |
| Identity/RBAC | `backend/gateway/identity.py` | Roles, permissions, command auth |
| Approval store | `backend/approvals/approval_store.py` | Persistent approvals with auto-expiry |
| Job manager | `backend/jobs/local_job_manager.py` | Start/stop/kill/monitor local jobs |
| Event store | `backend/events/event_store.py` | Append-only JSONL audit trail |
| Prompt cron | `backend/prompt_cron.py` | Persistent recurring prompt scheduler |
| Model catalog | `backend/model_catalog.py` | Shared model list for UI + Telegram |
| REST API | `backend/routes/` | Agent, auth, gateway endpoints |

---

## API Reference

### Core Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/session` | Create new session |
| `POST` | `/api/session/restore-summary` | Restore session |
| `GET` | `/api/events/{session_id}` | SSE event stream |
| `POST` | `/api/session/{id}/message` | Send message |
| `POST` | `/api/session/{id}/interrupt` | Interrupt agent |
| `DELETE` | `/api/session/{id}` | Delete session |

### Gateway Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/gateway/health` | System health check |
| `GET/POST` | `/api/jobs` | List / start jobs |
| `POST` | `/api/jobs/{id}/stop` | Stop a job |
| `POST` | `/api/jobs/{id}/kill` | Force-kill a job |
| `GET` | `/api/jobs/{id}/logs` | Tail job logs |
| `GET` | `/api/approvals` | List approvals |
| `POST` | `/api/approvals/{id}/approve` | Approve request |
| `POST` | `/api/approvals/{id}/reject` | Reject request |
| `GET` | `/api/gateway/events` | Query event store |
| `GET/POST` | `/api/crons` | List / create crons |
| `DELETE` | `/api/crons/{id}` | Cancel a cron |

### Telegram Config

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/telegram/config` | Get bot configuration |
| `POST` | `/api/telegram/config` | Update bot config |
| `POST` | `/api/telegram/start` | Start the bot |
| `POST` | `/api/telegram/stop` | Stop the bot |

---

## Local Mode vs Sandbox Mode

| Feature | Local Mode | Sandbox Mode |
|---|---|---|
| Execution | Your machine | HF cloud sandbox |
| Training | Direct GPU access | HF Jobs |
| File system | Full access | Isolated sandbox |
| Tools | bash, read, write, edit, scheduler | sandbox_create, hf_jobs |
| Setup | No extra config | Requires `HF_TOKEN` |
| Default for Telegram | ✅ Yes | Optional |

Set execution mode via:

```ini
# .env
TELEGRAM_EXECUTION_MODE=local    # or sandbox
ML_INTERN_LOCAL_MODE=1           # for Web UI
```

---

## RBAC Roles

| Permission | Owner | Admin | User | Viewer |
|---|:---:|:---:|:---:|:---:|
| Run prompts | ✅ | ✅ | ✅ | ❌ |
| View status/logs | ✅ | ✅ | ✅ | ✅ |
| Select model | ✅ | ✅ | ✅ | ❌ |
| Create cron | ✅ | ✅ | ✅ | ❌ |
| Approve/reject | ✅ | ✅ | ❌ | ❌ |
| Kill jobs | ✅ | ✅ | ❌ | ❌ |
| Run bash/training | ✅ | ✅ | ❌ | ❌ |
| Gateway admin | ✅ | ❌ | ❌ | ❌ |

The first Telegram user is auto-assigned **owner** role. Subsequent users get **user** role.

Identity store: `~/.cache/ml-intern/identities.json`

---

## Persistent State

All runtime state lives under `~/.cache/ml-intern/`:

| Path | Description |
|---|---|
| `events/events.jsonl` | Append-only audit trail |
| `approvals/` | Pending/resolved approval records |
| `jobs/` | Job records and per-job logs |
| `crons/` | Persisted cron task definitions |
| `identities.json` | User identity store |
| `telegram_bot.json` | Telegram bot configuration |
| `gateway.pid` | Gateway process PID |

---

## CLI Reference

```bash
# Interactive chat
ml-intern

# Single prompt (non-interactive)
ml-intern "your prompt here"

# With options
ml-intern --model MiniMaxAI/MiniMax-M2.7 "prompt"
ml-intern --max-iterations 100 "prompt"
ml-intern --no-stream "prompt"
```

---

## Development

### Project Structure

```
ml-intern-local/
├── agent/                  # Core agent (tools, session, loop)
│   ├── core/               # Agent loop, session, tools
│   ├── prompts/            # System prompts (YAML)
│   └── tools/              # Built-in tools
├── backend/                # FastAPI backend
│   ├── approvals/          # Approval store
│   ├── events/             # Event store
│   ├── gateway/            # Identity, routing, health
│   ├── jobs/               # Local job manager
│   ├── routes/             # API endpoints
│   └── main.py             # App entrypoint
├── configs/                # Agent configs
├── frontend/               # React + MUI + Vite + TypeScript
│   └── src/
│       ├── components/     # UI components
│       └── hooks/          # React hooks
├── tests/                  # Test suite
├── .env.example            # Template for environment variables
└── pyproject.toml          # Python package config
```

### Running Tests

```bash
uv run pytest tests/ -v
```

### Frontend Development

```bash
cd frontend
npm run dev     # dev server with hot reload
npm run build   # production build → static/
```

### Adding a New Model

Edit `backend/model_catalog.py`:

```python
AVAILABLE_MODELS.append({
    "id": "provider/model-name",
    "label": "Display Name",
    "provider": "provider_key",
})
```

### Adding a New Tool

Edit `agent/core/tools.py` — add a `ToolSpec` with name, description, JSON schema, and async handler.

Model defaults live in:
- `configs/cli_agent_config.json` (CLI)
- `configs/frontend_agent_config.json` (web sessions)

---

## Troubleshooting

| Issue | Fix |
|---|---|
| Bot not receiving messages | Check `TELEGRAM_BOT_TOKEN` and `TELEGRAM_ALLOWED_CHAT_IDS` in `.env` |
| "Not authorized" on Telegram | Check `~/.cache/ml-intern/identities.json` — your user needs a valid role |
| Model not found | Ensure the provider API key is set in `.env` |
| Port 7860 in use | Kill stale process: `kill $(lsof -ti:7860)` |
| Frontend won't connect | Ensure backend is running on `:7860` and frontend proxies to it |

---

## License

This project is provided as-is. See individual dependency licenses for third-party terms.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## Security

See [SECURITY.md](SECURITY.md) for vulnerability reporting and credential handling.
