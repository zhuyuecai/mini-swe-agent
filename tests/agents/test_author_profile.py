import pytest

from minisweagent.agents.author_profile import AuthorProfileAgent
from minisweagent.exceptions import FormatError
from minisweagent.models.test_models import DeterministicModel, make_output


class RecordingEnvironment:
    def __init__(self):
        self.actions = []

    def execute(self, action: dict) -> dict:
        self.actions.append(action)
        if action.get("tool") == "get_repo_knowledge":
            return {
                "output": '{"author_context": {"recent_contributions": 1}}',
                "returncode": 0,
                "exception_info": "",
            }
        if action.get("tool") == "record_developer_skill_profile":
            return {"output": '{"recorded": true}', "returncode": 0, "exception_info": ""}
        return {"output": "ok", "returncode": 0, "exception_info": ""}

    def get_template_vars(self, **kwargs):
        return {}

    def serialize(self):
        return {"info": {"config": {"environment": {}, "environment_type": "RecordingEnvironment"}}}


def _agent(outputs):
    return AuthorProfileAgent(
        DeterministicModel(outputs=outputs),
        RecordingEnvironment(),
        system_template="system",
        instance_template="task {{pr_author}}",
    )


def _author_context_action():
    return {
        "tool": "get_repo_knowledge",
        "query": "developer skill profile",
        "author": "alice",
        "recent_contributions": 20,
        "include_author_content": True,
    }


def _profile_action():
    return {
        "tool": "record_developer_skill_profile",
        "profile": {
            "identity": {"query": "alice"},
            "ownership": [],
            "technical_strengths": [],
            "style": {},
            "testing": {},
            "file_familiarity": {},
            "change_style": {},
            "mimicry_guidance": {},
            "confidence": {},
        },
    }


def test_requires_author_context_before_other_actions():
    agent = _agent([make_output("skip", [{"command": "echo hi"}])])
    agent.extra_template_vars["pr_author"] = "alice"
    with pytest.raises(FormatError) as exc_info:
        agent.step()
    assert "Before any other action" in exc_info.value.messages[0]["content"]


def test_requires_profile_after_author_context_before_bash():
    agent = _agent(
        [
            make_output("context", [_author_context_action()]),
            make_output("solve without profile", [{"command": "echo hi"}]),
        ]
    )
    agent.extra_template_vars["pr_author"] = "alice"
    agent.step()
    with pytest.raises(FormatError) as exc_info:
        agent.step()
    assert "record_developer_skill_profile" in exc_info.value.messages[0]["content"]


def test_allows_bash_after_recorded_profile():
    env = RecordingEnvironment()
    agent = AuthorProfileAgent(
        DeterministicModel(
            outputs=[
                make_output("context", [_author_context_action()]),
                make_output("profile", [_profile_action()]),
                make_output("solve", [{"command": "echo hi"}]),
            ]
        ),
        env,
        system_template="system",
        instance_template="task {{pr_author}}",
    )
    agent.extra_template_vars["pr_author"] = "alice"

    agent.step()
    agent.step()
    agent.step()

    assert env.actions[-1] == {"command": "echo hi"}
    assert agent.developer_skill_profile["identity"]["query"] == "alice"


def test_no_enforcement_without_pr_author():
    env = RecordingEnvironment()
    agent = AuthorProfileAgent(
        DeterministicModel(outputs=[make_output("normal", [{"command": "echo hi"}])]),
        env,
        system_template="system",
        instance_template="task",
    )

    agent.step()

    assert env.actions == [{"command": "echo hi"}]
