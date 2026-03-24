from examples.incident_triage import load_bundle_text, run_demo


def test_incident_triage_demo_runs_end_to_end():
    report = run_demo(load_bundle_text())
    positions = {task_id: index for index, task_id in enumerate(report.execution_order)}

    assert report.succeeded is True
    assert set(report.execution_order[:3]) == {"task_1", "task_2", "task_3"}
    assert positions["task_4"] > positions["task_1"]
    assert positions["task_4"] > positions["task_2"]
    assert positions["task_4"] > positions["task_3"]
    assert positions["task_5"] > positions["task_4"]
    assert report.final_result["report_markdown"].startswith("Incident Triage Report")
    assert "Primary service: payments" in report.final_result["report_markdown"]
    assert "Probable cause: payments database saturation caused connection pool exhaustion" in report.final_result["report_markdown"]
    assert "ERROR payments database connection pool exhausted" in report.final_result["report_markdown"]
