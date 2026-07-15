import json
from pathlib import Path

import src.cli as cli
from src.engine import DAGExecutionError


def test_default_golden_eval_is_independent_of_current_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main(["eval-golden", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["passed"] is True
    assert payload["total_cases"] == 1
    assert payload["cases"][0]["name"] == "text_pipeline"
    assert payload["cases"][0]["path"].endswith("text_pipeline.json")


def test_packaged_golden_fixture_matches_repository_fixture():
    packaged_fixture = cli.default_golden_path().joinpath("text_pipeline.json")
    repository_fixture = Path(__file__).parents[1] / "evals" / "golden" / "text_pipeline.json"

    assert json.loads(packaged_fixture.read_text(encoding="utf-8")) == json.loads(
        repository_fixture.read_text(encoding="utf-8")
    )


def test_cli_reports_dag_execution_error_without_traceback(tmp_path, monkeypatch, capsys):
    dag_file = tmp_path / "dag.json"
    dag_file.write_text(
        json.dumps(
            [
                {
                    "id": "task_1",
                    "description": "Normalize",
                    "instruction": "Normalize input",
                    "dependencies": [],
                    "tool_name": "normalize-text",
                }
            ]
        ),
        encoding="utf-8",
    )

    class FailingOrchestrator:
        def __init__(self, _hypervisor):
            pass

        def run(self, *_args, **_kwargs):
            raise DAGExecutionError("simulated DAG failure")

    monkeypatch.setattr(cli, "DAGOrchestrator", FailingOrchestrator)

    exit_code = cli.main(["run-dag", str(dag_file), "--payload", "hello"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "error: simulated DAG failure\n"
