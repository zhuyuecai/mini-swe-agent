import json

from typer.testing import CliRunner

from minisweagent.run.utilities.get_tokens import app, total_tokens_in_experiment


def test_total_tokens_in_experiment_sums_child_trajectories(tmp_path):
    first = tmp_path / "instance-a"
    second = tmp_path / "instance-b"
    first.mkdir()
    second.mkdir()
    (first / "instance-a.traj.json").write_text(
        json.dumps({"messages": [{"extra": {"response": {"usage": {"total_tokens": 7}}}}]})
    )
    (second / "instance-b.traj.json").write_text(
        json.dumps(
            {
                "messages": [
                    {"extra": {"response": {"usage": {"total_tokens": 11}}}},
                    {"extra": {"response": {"usage": {"total_tokens": 13}}}},
                ]
            }
        )
    )
    (tmp_path / "top-level.traj.json").write_text(json.dumps({"usage": {"total_tokens": 1000}}))

    assert total_tokens_in_experiment(tmp_path) == 31


def test_get_tokens_cli_outputs_sum(tmp_path):
    child = tmp_path / "instance"
    child.mkdir()
    (child / "instance.traj.json").write_text(json.dumps({"usage": {"total_tokens": 42}}))

    result = CliRunner().invoke(app, [str(tmp_path)])

    assert result.exit_code == 0
    assert result.stdout == "42\n"
