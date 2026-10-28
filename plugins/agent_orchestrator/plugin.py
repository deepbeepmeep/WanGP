import json

import gradio as gr

from shared.agent_orchestrator import default_ai_agent_orchestrator
from shared.utils.plugins import WAN2GPPlugin

PLUGIN_ID = "agent_orchestrator"
PLUGIN_NAME = "AI Agent Orchestrator"


class AgentOrchestratorPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = PLUGIN_NAME
        self.version = "0.1.0"
        self.description = "Central orchestration for task classification, decomposition, routing, and status tracking."
        self.type = ["app"]
        self._orchestrator = default_ai_agent_orchestrator()

    def setup_ui(self):
        self.add_tab(tab_id=PLUGIN_ID, label=PLUGIN_NAME, component_constructor=self.create_orchestrator_ui)
        self.register_deepy_zero_tool(
            self._orchestrator.run_task,
            name="orchestrate_task",
            display_name="Orchestrate Task",
            description="Accept a task, classify it, decompose it, route work to the best agent, and track the result.",
            parameters={
                "task": {"type": "string", "description": "The task description to route."},
                "agent_name": {"type": "string", "description": "Optional preferred agent override.", "required": False},
                "context": {"type": "object", "description": "Optional routing context and metadata.", "required": False},
            },
            pause_runtime=False,
        )
        self.register_deepy_zero_tool(
            self._orchestrator.list_tasks,
            name="list_agent_tasks",
            display_name="List Agent Tasks",
            description="List recent orchestrated tasks and their states.",
            pause_runtime=False,
        )
        self.register_deepy_zero_tool(
            self._orchestrator.get_task,
            name="get_agent_task",
            display_name="Get Agent Task",
            description="Fetch a single task by ID and include its status, steps, and result payload.",
            parameters={"task_id": {"type": "string", "description": "Task ID to look up."}},
            pause_runtime=False,
        )

    def create_orchestrator_ui(self, _api_session):
        def execute_task(task_text: str, preferred_agent: str = "") -> dict:
            if not task_text or not str(task_text).strip():
                return {"error": "Please enter a task before running the orchestrator."}
            context = {"preferred_agent": preferred_agent} if preferred_agent.strip() else {}
            return self._orchestrator.run_task(task_text, agent_name=preferred_agent or None, context=context)

        with gr.Column():
            gr.Markdown("### AI Agent Orchestrator\nRoutes a task to the best agent, decomposes it, and tracks progress and errors.")
            task_input = gr.Textbox(label="Task", lines=6, placeholder="Generate a short teaser video and export a summary file.")
            preferred_agent = gr.Textbox(label="Preferred agent (optional)", placeholder="media, research, filesystem, planner")
            run_btn = gr.Button("Run orchestrator")
            task_output = gr.JSON(label="Task result")
            recent_tasks = gr.JSON(label="Recent tasks")
            run_btn.click(fn=execute_task, inputs=[task_input, preferred_agent], outputs=[task_output])
            run_btn.click(fn=lambda: self._orchestrator.list_tasks(), outputs=[recent_tasks])
        return task_output


plugin = AgentOrchestratorPlugin
