from shared.agent_orchestrator import default_ai_agent_orchestrator


def test_orchestrator_classifies_media_tasks():
    orchestrator = default_ai_agent_orchestrator()
    category = orchestrator.classify_task("generate a dramatic sunset video and render a still frame")
    assert category == "media_generation"


def test_orchestrator_routes_and_tracks_work():
    orchestrator = default_ai_agent_orchestrator()
    result = orchestrator.run_task("Generate a short teaser video and save the summary file.")
    assert result["status"] == "completed"
    assert result["task_id"]
    assert len(result["steps"]) >= 2
    assert result["agent"] in {"media", "filesystem", "executor", "research", "planner"}


def test_orchestrator_lists_recent_tasks():
    orchestrator = default_ai_agent_orchestrator()
    orchestrator.run_task("Write a project summary and compare the options.")
    tasks = orchestrator.list_tasks()
    assert tasks["count"] >= 1
    assert tasks["tasks"]
