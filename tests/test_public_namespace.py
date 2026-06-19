from threadswarm import ThreadSwarmConfig
from threadswarm.cli import main
from threadswarm.compiler import SubTask, TaskDAG
from threadswarm.engine import DAGOrchestrator, LocalToolRegistry
from threadswarm.models import OpenAICompatibleWorker
from threadswarm.tools import build_text_tool_registry


def test_public_namespace_exports_core_runtime():
    dag = TaskDAG(
        tasks=[
            SubTask(
                id="task_1",
                description="Normalize",
                instruction="Normalize input",
                dependencies=[],
                tool_name="normalize-text",
            )
        ]
    )
    registry = build_text_tool_registry()
    report = DAGOrchestrator(registry.create_hypervisor()).run(dag, context=" ciao ")

    assert ThreadSwarmConfig().llm_model
    assert LocalToolRegistry is not None
    assert OpenAICompatibleWorker().model
    assert main is not None
    assert report.final_result == {"text": "CIAO"}
