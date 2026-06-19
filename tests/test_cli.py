import json

from threadswarm.cli import main


def test_validate_dag_cli_success(tmp_path, capsys):
    dag_file = tmp_path / "dag.json"
    dag_file.write_text(
        json.dumps(
            [
                {
                    "id": "task_1",
                    "description": "Prepare",
                    "instruction": "Prepare input",
                    "dependencies": [],
                },
                {
                    "id": "task_2",
                    "description": "Run",
                    "instruction": "Run task",
                    "dependencies": ["task_1"],
                },
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["validate-dag", str(dag_file)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert '"valid": true' in captured.out
    assert '"task_count": 2' in captured.out


def test_validate_dag_cli_reports_invalid_dependencies(tmp_path, capsys):
    dag_file = tmp_path / "dag.json"
    dag_file.write_text(
        json.dumps(
            [
                {
                    "id": "task_1",
                    "description": "Run",
                    "instruction": "Run task",
                    "dependencies": ["missing"],
                }
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["validate-dag", str(dag_file)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "references missing dependency missing" in captured.err


def test_demo_cli_writes_execution_report(tmp_path, capsys):
    report_file = tmp_path / "reports" / "incident.json"

    exit_code = main(["demo", "incident-triage", "--json", "--report-file", str(report_file)])

    captured = capsys.readouterr()
    report_payload = json.loads(report_file.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert "Incident Triage Report" in captured.out
    assert report_payload["summary"]["succeeded"] is True
    assert report_payload["summary"]["total_tasks"] == 5
    assert report_payload["task_results"]["task_5"]["tool_name"] == "build-incident-report"
    assert "task_4" in report_payload["task_results"]["task_5"]["dependency_results"]


def test_run_dag_cli_executes_json_dag_with_text_toolkit(tmp_path, capsys):
    dag_file = tmp_path / "dag.json"
    report_file = tmp_path / "report.json"
    dag_file.write_text(
        json.dumps(
            [
                {
                    "id": "task_1",
                    "description": "Normalize",
                    "instruction": "Normalize input",
                    "dependencies": [],
                    "tool_name": "normalize-text",
                },
                {
                    "id": "task_2",
                    "description": "Count",
                    "instruction": "Count words",
                    "dependencies": ["task_1"],
                    "tool_name": "word-count",
                },
                {
                    "id": "task_3",
                    "description": "Collect",
                    "instruction": "Collect outputs",
                    "dependencies": ["task_1", "task_2"],
                    "tool_name": "collect-json",
                },
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "run-dag",
            str(dag_file),
            "--payload",
            "hello local dag",
            "--json",
            "--report-file",
            str(report_file),
        ]
    )

    captured = capsys.readouterr()
    final_result = json.loads(captured.out)
    report_payload = json.loads(report_file.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert final_result["dependency_results"]["task_1"]["text"] == "HELLO LOCAL DAG"
    assert final_result["dependency_results"]["task_2"]["word_count"] == 3
    assert report_payload["summary"]["succeeded"] is True


def test_eval_golden_cli_runs_packaged_fixture(capsys):
    exit_code = main(["eval-golden", "evals/golden", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["passed"] is True
    assert payload["total_cases"] == 1
    assert payload["cases"][0]["name"] == "text_pipeline"


def test_eval_golden_cli_reports_failed_expectation(tmp_path, capsys):
    case_file = tmp_path / "bad_case.json"
    case_file.write_text(
        json.dumps(
            {
                "name": "bad_text_pipeline",
                "toolkit": "text",
                "payload": "hello local dag",
                "dag": [
                    {
                        "id": "task_1",
                        "description": "Count",
                        "instruction": "Count words",
                        "dependencies": [],
                        "tool_name": "word-count",
                    }
                ],
                "expect": {
                    "final_result": {
                        "word_count": 99,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["eval-golden", str(case_file), "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert payload["passed"] is False
    assert payload["failed_cases"] == 1
    assert "expected 99, got 3" in payload["cases"][0]["errors"][0]
