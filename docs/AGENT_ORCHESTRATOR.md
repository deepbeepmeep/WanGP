# AI Agent Orchestrator

The AI Agent Orchestrator centralizes task intake, task classification, decomposition, route selection, and status tracking for WanGP.

## Where it lives

- Core orchestration logic: `shared/agent_orchestrator.py`
- App integration: `plugins/agent_orchestrator/plugin.py`
- Plugin metadata: `plugins/agent_orchestrator/plugin_info.json`

## Features

- Accepts a free-form task string
- Classifies the task into a standard category such as media generation, research, filesystem work, or workflow planning
- Decomposes tasks into ordered steps
- Routes each step to the best specialized agent
- Tracks status, results, and errors per task
- Exposes explicit tool metadata through the Deepy Zero plugin integration

## Example

```python
from shared.agent_orchestrator import default_ai_agent_orchestrator

orchestrator = default_ai_agent_orchestrator()
result = orchestrator.run_task("Generate a short teaser video and save a summary in the project folder.")
print(result["status"])
print(result["agent"])
```

The orchestrator is also available in the app through the `AI Agent Orchestrator` tab, where a task can be submitted directly from the existing WanGP UI.
