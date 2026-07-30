from typing import Any

from pydantic import Field

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import FormatError


class AuthorProfileAgentConfig(AgentConfig):
    require_author_profile: bool = True
    author_profile_marker: str = "DEVELOPER_SKILL_PROFILE"
    author_profile_tool: str = "get_repo_knowledge"
    author_profile_query: str = "developer skill profile"
    record_profile_tool: str = "record_developer_skill_profile"
    profile_required_tool_args: dict[str, Any] = Field(
        default_factory=lambda: {"recent_contributions": 20, "include_author_content": True}
    )


class AuthorProfileAgent(DefaultAgent):
    """Enforce author-context collection and profile summarization before solving."""

    def __init__(self, *args, config_class: type = AuthorProfileAgentConfig, **kwargs):
        super().__init__(*args, config_class=config_class, **kwargs)
        self._author_context_seen = False
        self._author_profile_seen = False
        self.developer_skill_profile: dict[str, Any] = {}

    @property
    def _pr_author(self) -> str:
        return str(self.extra_template_vars.get("pr_author") or "")

    def execute_actions(self, message: dict) -> list[dict]:
        actions = [self._apply_default_tool_args(action) for action in message.get("extra", {}).get("actions", [])]
        message.setdefault("extra", {})["actions"] = actions
        if self._must_enforce_author_profile():
            self._validate_author_profile_flow(message, actions)
        outputs = [self.env.execute(action) for action in actions]
        if self._must_enforce_author_profile():
            self._update_author_context_seen(outputs)
            self._update_author_profile_seen(actions)
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))

    def _must_enforce_author_profile(self) -> bool:
        return bool(self.config.require_author_profile and self._pr_author)

    def _validate_author_profile_flow(self, message: dict, actions: list[dict]) -> None:
        if not self._author_context_seen:
            if len(actions) != 1 or not self._is_author_context_action(actions[0]):
                raise self._format_error(
                    "Before any other action, call get_repo_knowledge exactly once with "
                    f'author="{self._pr_author}" to collect recent commits and touched files.'
                )
            return
        if not self._author_profile_seen:
            if len(actions) != 1 or not self._is_record_profile_action(actions[0]):
                raise self._format_error(
                    f"You received author_context for {self._pr_author}. Your next and only tool call must be "
                    "record_developer_skill_profile with a JSON-shaped profile inferred from author_context. "
                    "Do not inspect files, run bash, or solve the task before recording the profile."
                )

    def _is_author_context_action(self, action: dict) -> bool:
        return (
            action.get("tool") == self.config.author_profile_tool
            and action.get("author") == self._pr_author
            and self.config.author_profile_query in action.get("query", "").lower()
            and all(action.get(key) == value for key, value in self.config.profile_required_tool_args.items())
        )

    def _is_record_profile_action(self, action: dict) -> bool:
        return action.get("tool") == self.config.record_profile_tool and isinstance(action.get("profile"), dict)

    def _update_author_context_seen(self, outputs: list[dict]) -> None:
        if any('"author_context"' in output.get("output", "") for output in outputs):
            self._author_context_seen = True

    def _update_author_profile_seen(self, actions: list[dict]) -> None:
        for action in actions:
            if self._is_record_profile_action(action):
                self.developer_skill_profile = action["profile"]
                self._author_profile_seen = True

    def _format_error(self, error: str) -> FormatError:
        return FormatError(
            {
                "role": "user",
                "content": error,
                "extra": {"interrupt_type": "FormatError"},
            }
        )
