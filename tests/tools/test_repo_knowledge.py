import os
import subprocess
from pathlib import Path

from minisweagent.tools.repo_knowledge import get_repo_knowledge


def _git(root: Path, *args: str, env: dict[str, str] | None = None) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, env=os.environ | (env or {}))


def _commit(root: Path, message: str, author_name: str, author_email: str) -> None:
    env = {
        "GIT_AUTHOR_NAME": author_name,
        "GIT_AUTHOR_EMAIL": author_email,
        "GIT_COMMITTER_NAME": author_name,
        "GIT_COMMITTER_EMAIL": author_email,
    }
    _git(root, "add", ".", env=env)
    _git(root, "commit", "-m", message, env=env)


def test_get_repo_knowledge_includes_author_context(tmp_path):
    _git(tmp_path, "init")
    Path(tmp_path, "owned.py").write_text("def owned_parser(value):\n    return value.strip()\n")
    _commit(tmp_path, "add owned parser", "Alice Example", "alice@example.com")
    Path(tmp_path, "other.py").write_text("def unrelated(value):\n    return value\n")
    _commit(tmp_path, "add other file", "Bob Example", "bob@example.com")

    result = get_repo_knowledge(
        tmp_path,
        query="parser",
        author="alice@example.com",
        recent_contributions=5,
    )

    assert result["author"] == "alice@example.com"
    assert result["author_context"]["recent_contributions"] == 1
    assert result["author_context"]["recent_commits"][0]["subject"] == "add owned parser"
    assert result["author_context"]["touched_files"] == [
        {"file": "owned.py", "commits": [result["author_context"]["recent_commits"][0]["short_commit"]], "exists": True}
    ]
    assert result["author_files"] == result["author_context"]["touched_files"]
    assert result["author_context"]["content"][0]["file"] == "owned.py"
    assert result["results"][0]["file"] == "owned.py"


def test_get_repo_knowledge_without_author_keeps_original_shape(tmp_path):
    Path(tmp_path, "sample.py").write_text("def parse_config(value):\n    return value\n")

    result = get_repo_knowledge(tmp_path, query="parse config")

    assert "author" not in result
    assert "author_files" not in result
    assert "author_context" not in result
    assert result["results"][0]["name"] == "parse_config"


def test_get_repo_knowledge_author_without_matches_is_empty(tmp_path):
    _git(tmp_path, "init")
    Path(tmp_path, "sample.py").write_text("def parse_config(value):\n    return value\n")
    _commit(tmp_path, "add sample", "Alice Example", "alice@example.com")

    result = get_repo_knowledge(tmp_path, query="parse config", author="nobody@example.com")

    assert result["results"][0]["name"] == "parse_config"
    assert result["author_files"] == []
    assert result["author_context"] == {"recent_contributions": 0, "recent_commits": [], "touched_files": [], "content": []}


def test_get_repo_knowledge_author_context_can_omit_content(tmp_path):
    _git(tmp_path, "init")
    Path(tmp_path, "sample.py").write_text("def parse_config(value):\n    return value\n")
    _commit(tmp_path, "add sample", "Alice Example", "alice@example.com")

    result = get_repo_knowledge(tmp_path, query="parse config", author="alice", include_author_content=False)

    assert result["author_context"]["touched_files"][0]["file"] == "sample.py"
    assert result["author_context"]["content"] == []
