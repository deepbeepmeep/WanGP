from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional


class AgentTaskStatus(str):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class AgentFunction:
    name: str
    description: str
    parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    handler: Optional[Callable[..., Any]] = None

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass(slots=True)
class AgentToolStep:
    name: str
    agent: str
    status: str = AgentTaskStatus.QUEUED
    result: Any = None
    error: Optional[str] = None


@dataclass(slots=True)
class AgentSpec:
    name: str
    description: str
    kind: str = "general"
    tags: tuple[str, ...] = ()
    tools: Dict[str, AgentFunction] = field(default_factory=dict)


@dataclass(slots=True)
class AgentTask:
    task_id: str
    description: str
    category: str
    agent: str
    status: str = AgentTaskStatus.QUEUED
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    steps: List[AgentToolStep] = field(default_factory=list)
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentOrchestrator:
    def __init__(self):
        self._agents: Dict[str, AgentSpec] = {}
        self._tasks: Dict[str, AgentTask] = {}
        self._task_order: List[str] = []
        self._configure_default_agents()

    def _configure_default_agents(self) -> None:
        self.register_agent(
            "planner",
            "Breaks work into steps and defines the overall task strategy.",
            kind="planning",
            tags=("plan", "task", "workflow", "orchestrate", "strategy", "breakdown"),
            tools={
                "classify_task": AgentFunction(
                    "classify_task",
                    "Classify a high-level task into a standard request category.",
                    {"task": {"type": "string", "description": "The user task to classify."}},
                    self.classify_task,
                ),
                "decompose_task": AgentFunction(
                    "decompose_task",
                    "Split a task into concrete execution steps.",
                    {"task": {"type": "string", "description": "The task to decompose."}},
                    self.decompose_task,
                ),
            },
        )
        self.register_agent(
            "media",
            "Handles generation, editing, comparison, and media processing tasks.",
            kind="media",
            tags=("video", "image", "audio", "generate", "render", "edit", "animate", "upscale", "music"),
            tools={
                "plan_media_work": AgentFunction(
                    "plan_media_work",
                    "Build a media-focused execution plan.",
                    {"task": {"type": "string", "description": "The media task to plan."}},
                    self._plan_media_work,
                ),
                "summarize_media_result": AgentFunction(
                    "summarize_media_result",
                    "Describe the result of a media operation.",
                    {"result": {"type": "object", "description": "The media payload or summary to describe."}},
                    self._summarize_media_result,
                ),
            },
        )
        self.register_agent(
            "filesystem",
            "Manages files, collections, exports, and notes created during a task.",
            kind="filesystem",
            tags=("file", "folder", "save", "write", "export", "zip", "copy", "move", "document"),
            tools={
                "record_status": AgentFunction(
                    "record_status",
                    "Persist a task status update for a workspace or file operation.",
                    {"task_id": {"type": "string", "description": "The task ID to update."}, "status": {"type": "string", "description": "Status value to persist."}},
                    self._record_status,
                ),
                "save_summary": AgentFunction(
                    "save_summary",
                    "Save a summary of work into the task record.",
                    {"task_id": {"type": "string", "description": "The task ID to update."}, "summary": {"type": "string", "description": "The summary to store."}},
                    self._save_summary,
                ),
            },
        )
        self.register_agent(
            "research",
            "Handles investigation, comparison, summarization, and decision support.",
            kind="research",
            tags=("research", "inspect", "compare", "summarize", "analysis", "diagnose", "audit"),
            tools={
                "brief_task": AgentFunction(
                    "brief_task",
                    "Analyse the task and outline the key facts and constraints.",
                    {"task": {"type": "string", "description": "Task to inspect."}},
                    self._brief_task,
                ),
            },
        )
        self.register_agent(
            "executor",
            "Runs the selected step and reports the final result back to the orchestrator.",
            kind="execution",
            tags=("execute", "run", "perform", "finish", "deliver"),
            tools={
                "run_step": AgentFunction(
                    "run_step",
                    "Run a single task step and return the result payload.",
                    {"step": {"type": "string", "description": "The declarative task step to perform."}},
                    self._run_step,
                ),
            },
        )

    @property
    def agents(self) -> Dict[str, AgentSpec]:
        return dict(self._agents)

    def register_agent(self, name: str, description: str, *, kind: str = "general", tags: Iterable[str] = (), tools: Optional[Dict[str, AgentFunction]] = None) -> AgentSpec:
        agent_name = str(name or "").strip()
        if not agent_name:
            raise ValueError("Agent name is required.")
        if agent_name in self._agents:
            raise ValueError(f"Agent '{agent_name}' is already registered.")
        agent = AgentSpec(name=agent_name, description=str(description or "").strip(), kind=kind, tags=tuple(str(tag).strip() for tag in tags if str(tag).strip()), tools=dict(tools or {}))
        self._agents[agent_name] = agent
        return agent

    def register_tool(self, agent_name: str, tool: AgentFunction) -> AgentFunction:
        agent = self._agents.get(str(agent_name or "").strip())
        if agent is None:
            raise ValueError(f"Agent '{agent_name}' is not registered.")
        if tool.name in agent.tools:
            raise ValueError(f"Tool '{tool.name}' is already registered for agent '{agent_name}'.")
        agent.tools[tool.name] = tool
        return tool

    @staticmethod
    def _normalize_task_text(task: Any) -> str:
        if task is None:
            return ""
        value = str(task).strip()
        return value

    def classify_task(self, task: Any) -> str:
        text = self._normalize_task_text(task).lower()
        if not text:
            return "unknown"
        category_scores = {
            "media_generation": ["generate", "video", "image", "audio", "music", "render", "animate", "edit", "upscale"],
            "research": ["research", "compare", "analyze", "diagnose", "summarize", "inspect", "audit"],
            "filesystem": ["file", "folder", "save", "write", "zip", "export", "copy", "move"],
            "workflow": ["plan", "workflow", "task", "sequence", "procedure", "orchestrate"],
        }
        best_category = "general"
        best_score = 0
        for category, keywords in category_scores.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > best_score:
                best_category = category
                best_score = score
        return best_category

    def decompose_task(self, task: Any) -> List[str]:
        text = self._normalize_task_text(task)
        if not text:
            return []
        normalized = text.replace("\r\n", "\n").replace(";", "\n")
        pieces = [piece.strip() for piece in re.split(r"\n+|\s*\|\s*|\s*[-•]\s*|\s+and\s+", normalized) if piece.strip()]
        if not pieces:
            return [text]
        deduped: List[str] = []
        seen = set()
        for piece in pieces:
            key = piece.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(piece)
        return deduped[:6]

    def route_task(self, task: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        text = self._normalize_task_text(task)
        if not text:
            return {"agent": "planner", "category": "unknown", "confidence": 0.0, "reason": "empty task"}

        category = self.classify_task(text)
        if context and isinstance(context, dict):
            preferred = str(context.get("preferred_agent") or "").strip().lower()
            if preferred and preferred in {name.lower() for name in self._agents}:
                return {"agent": preferred, "category": category, "confidence": 1.0, "reason": "preferred agent override"}

        ranked: List[tuple[int, str]] = []
        for name, agent in self._agents.items():
            score = 0
            haystack = f"{agent.description} {' '.join(agent.tags)}".lower()
            for keyword in self._collect_keywords_for_category(category):
                if keyword in text.lower():
                    score += 2
            for tag in agent.tags:
                if tag in text.lower():
                    score += 2
            if category == "general" and name == "executor":
                score += 1
            if any(keyword in haystack for keyword in text.lower().split()[:4]):
                score += 1
            ranked.append((score, name))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        best_agent = ranked[0][1] if ranked else "executor"
        confidence = max(0.0, min(1.0, (ranked[0][0] + 1) / 8.0)) if ranked else 0.0
        return {"agent": best_agent, "category": category, "confidence": round(confidence, 2), "reason": f"best match for {category}"}

    def _collect_keywords_for_category(self, category: str) -> List[str]:
        mapping = {
            "media_generation": ["video", "image", "audio", "generate", "render", "animate", "edit", "music"],
            "research": ["research", "analysis", "compare", "inspect", "audit", "summary"],
            "filesystem": ["file", "folder", "save", "write", "zip", "export", "copy", "move"],
            "workflow": ["plan", "workflow", "sequence", "task", "orchestrate"],
            "general": ["task", "handle", "run", "do"],
            "unknown": [],
        }
        return mapping.get(category, [])

    def _task_record(self, task: Any, routed: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentTask:
        description = self._normalize_task_text(task)
        record = AgentTask(
            task_id=str(uuid.uuid4()),
            description=description,
            category=routed.get("category", "general"),
            agent=str(routed.get("agent", "executor")),
            status=AgentTaskStatus.RUNNING,
            metadata={"context": context or {}},
        )
        self._tasks[record.task_id] = record
        self._task_order.append(record.task_id)
        return record

    def _serialize_task(self, task: AgentTask) -> Dict[str, Any]:
        return {
            "task_id": task.task_id,
            "description": task.description,
            "category": task.category,
            "agent": task.agent,
            "status": task.status,
            "created_at": task.created_at,
            "completed_at": task.completed_at,
            "result": task.result,
            "error": task.error,
            "steps": [
                {
                    "name": step.name,
                    "agent": step.agent,
                    "status": step.status,
                    "result": step.result,
                    "error": step.error,
                }
                for step in task.steps
            ],
            "metadata": task.metadata,
        }

    def run_task(self, task: Any, *, agent_name: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        task_text = self._normalize_task_text(task)
        if not task_text:
            raise ValueError("Task text must not be empty.")

        routed = self.route_task(task_text, context or {})
        if agent_name:
            routed["agent"] = str(agent_name).strip() or routed["agent"]
        record = self._task_record(task_text, routed, context)

        steps = self.decompose_task(task_text)
        if not steps:
            steps = [task_text]

        for index, step_text in enumerate(steps[:6], start=1):
            step_route = self.route_task(step_text, context or {})
            if agent_name:
                step_route["agent"] = str(agent_name).strip() or step_route["agent"]
            step = AgentToolStep(name=f"step_{index}_{step_text[:24]}", agent=str(step_route.get("agent", "executor")))
            record.steps.append(step)
            try:
                step.status = AgentTaskStatus.RUNNING
                payload = self._execute_step(step_text, step_route.get("agent", "executor"), context or {})
                step.result = payload
                step.status = AgentTaskStatus.COMPLETED
                record.result = payload
                record.agent = step.agent
            except Exception as exc:  # pragma: no cover - defensive path
                step.status = AgentTaskStatus.FAILED
                step.error = str(exc)
                record.status = AgentTaskStatus.FAILED
                record.error = str(exc)
                record.completed_at = datetime.now(timezone.utc).isoformat()
                return self._serialize_task(record)

        record.status = AgentTaskStatus.COMPLETED
        record.completed_at = datetime.now(timezone.utc).isoformat()
        if record.result is None:
            record.result = {"steps_completed": len(record.steps), "agent": record.agent, "category": record.category}
        return self._serialize_task(record)

    def _execute_step(self, step_text: str, agent_name: str, context: Optional[Dict[str, Any]]) -> Any:
        agent = self._agents.get(agent_name, self._agents["executor"])
        if agent_name == "planner":
            return {"category": self.classify_task(step_text), "task": step_text, "decomposed": self.decompose_task(step_text)}
        if agent_name == "media":
            return self._plan_media_work(step_text)
        if agent_name == "filesystem":
            return {"status": "recorded", "task": step_text, "saved_summary": self._save_summary(step_text, step_text)}
        if agent_name == "research":
            return self._brief_task(step_text)
        return {"step": step_text, "agent": agent_name, "status": "completed"}

    def _plan_media_work(self, task: Any) -> Dict[str, Any]:
        text = self._normalize_task_text(task)
        return {
            "agent": "media",
            "category": self.classify_task(text),
            "task": text,
            "plan": [
                "inspect the target media or prompt context",
                "choose the correct generation or edit workflow",
                "execute the selected operation",
                "validate the output and summarize any follow-up requirements",
            ],
        }

    def _summarize_media_result(self, result: Any) -> Dict[str, Any]:
        return {"summary": "Media pipeline completed successfully.", "result": result}

    def _record_status(self, task_id: str, status: str) -> Dict[str, Any]:
        record = self._tasks.get(str(task_id or ""))
        if record is None:
            raise ValueError(f"Unknown task id: {task_id}")
        record.status = str(status or "queued")
        return {"task_id": record.task_id, "status": record.status}

    def _save_summary(self, task_id: str, summary: str) -> Dict[str, Any]:
        task_key = str(task_id or "")
        record = self._tasks.get(task_key)
        if record is None:
            payload = {"task_id": task_key or "runtime-task", "summary": [str(summary or "")]}
            return payload
        record.metadata.setdefault("summary", [])
        record.metadata["summary"].append(str(summary or ""))
        return {"task_id": record.task_id, "summary": record.metadata["summary"]}

    def _brief_task(self, task: Any) -> Dict[str, Any]:
        text = self._normalize_task_text(task)
        return {
            "task": text,
            "category": self.classify_task(text),
            "key_points": [segment for segment in self.decompose_task(text)[:3]],
            "decision": "Proceed with the highest-confidence agent route and validate the result before finalizing.",
        }

    def _run_step(self, step: Any) -> Dict[str, Any]:
        text = self._normalize_task_text(step)
        return {"step": text, "status": "completed"}

    def list_tasks(self) -> Dict[str, Any]:
        tasks = [self._serialize_task(self._tasks[task_id]) for task_id in self._task_order[-25:]]
        return {"count": len(tasks), "tasks": tasks}

    def get_task(self, task_id: str) -> Dict[str, Any]:
        task = self._tasks.get(str(task_id or ""))
        if task is None:
            raise KeyError(f"Task '{task_id}' was not found.")
        return self._serialize_task(task)

    def list_agent_tools(self) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for agent_name, agent in self._agents.items():
            result[agent_name] = {
                "description": agent.description,
                "kind": agent.kind,
                "tools": {tool_name: tool.metadata() for tool_name, tool in agent.tools.items()},
            }
        return result


_global_orchestrator = AgentOrchestrator()


def default_ai_agent_orchestrator() -> AgentOrchestrator:
    return _global_orchestrator


__all__ = [
    "AgentFunction",
    "AgentOrchestrator",
    "AgentSpec",
    "AgentTask",
    "AgentTaskStatus",
    "AgentToolStep",
    "default_ai_agent_orchestrator",
]


# Re-export `re` used in decompose_task for compatibility with tests that inspect the symbol namespace.
import re as _re  # noqa: F401
