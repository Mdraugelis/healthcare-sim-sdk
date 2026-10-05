# Simulation Workbench

A local application around the Healthcare Intervention Simulation SDK. Its agent panel runs **the actual interactive Codex or Claude Code CLI in the project folder** through a PTY. The frontier model keeps its native reasoning, filesystem access, terminal tools, model selection, skills, approvals, and conversation controls. The app adds study context and evidence inspection; it does not replace the agent loop with a prompt chain or restrict the agent to a scenario catalog.

## Run

Requires Node.js 20+ (tested on macOS with Node 22), and the Python SDK environment described in the root README.

```sh
cd workbench
npm install
npm start
```

Open the complete URL printed by the server. It contains a one-time server-session access token; the browser exchanges it for an HttpOnly cookie and removes it from the address bar. The app listens only on `127.0.0.1:4317`.

Codex is installed locally with the app; it does not replace a globally installed CLI. Before first use, authenticate it:

```sh
./node_modules/.bin/codex login
```

For Claude Code, install and authenticate your own `claude` CLI. Select the provider and start a session. Native `/model` and other commands remain available inside the terminal. The app does not override the provider's model, reasoning effort, permissions, or authentication configuration. Choose your preferred frontier model through the CLI's native controls.

Optional environment variables:

- `SIM_PROJECT`: absolute SDK project folder (default: this repository).
- `PORT`: local port (default: 4317).
- `CODEX_BIN` / `CLAUDE_BIN`: alternate executable path. Arguments are passed directly, without a shell.
- `SHELL`: executable used for the project shell option.

The project's `.venv/bin` is prepended to the agent's PATH. The agent otherwise inherits the server's environment and runs as your user, like a CLI opened in this folder. Its native permission prompts remain interactive. Run this as a trusted local developer tool; it is not a multiuser service or isolated compute environment.

## Work on a study

1. Create a study and write the decision question. Unknown inputs can remain blank.
2. Save the brief and start an agent. Initial context points to the brief and the SDK's `CLAUDE.md` and simulation protocol. The agent can inspect the entire project, reason analytically, build custom scenarios, and run experiments.
3. Continue directly in the agent terminal. This is the full native CLI, including its approval and model controls.
4. Edit the brief as your understanding evolves. Save, then use **Copy brief update for agent** and paste it into the terminal. The app does not inject changes into an active reasoning turn.
5. Inspect experiments, compare raw metrics, and read configuration, validation scope, and artifacts. Missing checks remain incomplete; recorded passes apply only to their declared scope.

Study files are saved under `local/workbench/studies/<id>/`. `study.json` is the app-editable context; `brief.md` is generated from it for the agent. Ask the agent to save experiments under that study's `outputs/` directory with the SDK's `finalize_experiment()` lifecycle. All these paths inherit the repository's private-data gitignore rules.

The experiment browser discovers folders containing `metrics.json` under `outputs/` and `local/`; it also reads `config.json` and `validation.json`. Select **Entire project** to inspect existing work. Text artifacts are shown as plain text; HTML reports open in the isolated Reports preview. This initial version compares recorded metrics and exposes reports; it does not infer uncertainty intervals or automatically certify scientific validity.

## Sessions and limits

Browser refreshes and study switching detach the display while the server keeps the agent process alive. Starting the same study/provider reattaches and replays up to 1 MB of terminal output. The terminal retains its native interaction model; arbitrary command output is not parsed into a substitute chat transcript.

**End session** terminates the process. Server shutdown terminates all agent processes. After a server restart, use the CLI's native resume command or resume picker to recover its stored conversation. App study files persist independently. Detached processes retain the normal CLI's access to the project. Avoid simultaneously editing the same project through several agent sessions.

The app does not train predictive models or turn simulation output into clinical evidence. Study assumptions, mechanism verification, statistical uncertainty, and human review remain part of the SDK workflow.

## Verification

```sh
npm test
```

Tests cover persistent briefs and stale edit conflicts, artifact traversal/symlink boundaries, missing/failed/corrupt validation evidence, HTTP and WebSocket authentication, real PTY command execution in the project, and a browser flow for study editing, terminal use, run comparison, and safe artifact display. Browser tests use installed Google Chrome; set `PLAYWRIGHT_CHANNEL` for another installed Playwright browser channel.

The live agent integration uses native CLIs, not mocked model responses. Tests execute a project shell and check CLI availability; they do not make billable inference calls or establish the scientific validity of a generated simulation.

## HTML report side panel

Open **Reports** in the right-hand panel to preview agent-generated HTML while the agent terminal stays visible. The panel discovers `.html` and `.htm` files independently of experiment metrics, lets you choose the current study or entire project, and checks for new or updated reports every three seconds while visible. HTML artifact buttons in Experiments also open this preview.

Ask the agent to generate a **self-contained HTML report** in the study's outputs folder: embed styles, JavaScript, and images in the document. Existing reports under `local/`, `outputs/`, and `reports/` can be selected with Entire project. Inline JavaScript supports interactive charts and controls; external libraries, external assets, network requests, and workbench access are blocked in the isolated preview. Bundle libraries inline and embed image data for offline reports.
