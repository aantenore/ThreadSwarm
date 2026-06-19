import json

from src.cli import main


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
