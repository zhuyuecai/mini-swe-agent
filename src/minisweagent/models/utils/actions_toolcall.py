"""Parse actions & format observations with toolcalls"""

import json
import time

from jinja2 import StrictUndefined, Template

from minisweagent.exceptions import FormatError
from minisweagent.models.utils.openai_multimodal import expand_multimodal_content

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}


GET_REPO_KNOWLEDGE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_repo_knowledge",
        "description": "Search the current repository for relevant Python files, symbols, and code blocks.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language or code search query.",
                },
                "path": {
                    "type": "string",
                    "description": "Optional repository-relative file or directory to search.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of matching code items to return.",
                },
                "include_code": {
                    "type": "boolean",
                    "description": "Whether to include source code snippets in the results.",
                },
                "author": {
                    "type": "string",
                    "description": "Optional git author name, email, or GitHub login to include recent author context.",
                },
                "recent_contributions": {
                    "type": "integer",
                    "description": "Number of recent commits by the author to inspect.",
                },
                "include_author_content": {
                    "type": "boolean",
                    "description": "Whether to include snippets from files touched by the author's recent commits.",
                },
            },
            "required": ["query"],
        },
    },
}

RECORD_DEVELOPER_SKILL_PROFILE_TOOL = {
    "type": "function",
    "function": {
        "name": "record_developer_skill_profile",
        "description": "Record the developer skill profile inferred from author_context before solving.",
        "parameters": {
            "type": "object",
            "properties": {
                "profile": {
                    "type": "object",
                    "description": "JSON-shaped developer skill profile inferred from repository author context.",
                }
            },
            "required": ["profile"],
        },
    },
}

TOOLS = [BASH_TOOL, GET_REPO_KNOWLEDGE_TOOL, RECORD_DEVELOPER_SKILL_PROFILE_TOOL]


def parse_toolcall_actions(
    tool_calls: list, *, format_error_template: str, template_kwargs: dict | None = None
) -> list[dict]:
    """Parse tool calls from the response. Raises FormatError if unknown tool or invalid args.

    ``template_kwargs`` are extra variables exposed to ``format_error_template`` (e.g.
    ``{"finish_reason": ...}`` so a template can distinguish a real format mistake from a
    ``max_tokens`` truncation).
    """
    template_kwargs = template_kwargs or {}
    if not tool_calls:
        raise FormatError(
            {
                "role": "user",
                "content": Template(format_error_template, undefined=StrictUndefined).render(
                    error="No tool calls found in the response. Every response MUST include at least one tool call.",
                    actions=[],
                    has_tool_calls=False,
                    **template_kwargs,
                ),
                "extra": {"interrupt_type": "FormatError"},
            }
        )
    actions = []
    for tool_call in tool_calls:
        error_msg = ""
        args = {}
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except Exception as e:
            error_msg = f"Error parsing tool call arguments: {e}."
        if name == "bash" and (not isinstance(args, dict) or "command" not in args):
            error_msg += "Missing 'command' argument in bash tool call."
        elif name == "get_repo_knowledge" and (not isinstance(args, dict) or "query" not in args):
            error_msg += "Missing 'query' argument in get_repo_knowledge tool call."
        elif (
            name == "record_developer_skill_profile"
            and (not isinstance(args, dict) or not isinstance(args.get("profile"), dict))
        ):
            error_msg += "Missing 'profile' argument in record_developer_skill_profile tool call."
        elif name not in ["bash", "get_repo_knowledge", "record_developer_skill_profile"]:
            error_msg += f"Unknown tool '{name}'."
        if error_msg:
            raise FormatError(
                {
                    "role": "user",
                    "content": Template(format_error_template, undefined=StrictUndefined).render(
                        actions=[], error=error_msg.strip(), has_tool_calls=True, **template_kwargs
                    ),
                    "extra": {"interrupt_type": "FormatError"},
                }
            )
        if name == "bash":
            actions.append({"command": args["command"], "tool_call_id": tool_call.id})
        elif name == "get_repo_knowledge":
            actions.append(
                {
                    "tool": "get_repo_knowledge",
                    "query": args["query"],
                    "path": args.get("path", ""),
                    "max_results": args.get("max_results", 8),
                    "include_code": args.get("include_code", True),
                    "author": args.get("author", ""),
                    "recent_contributions": args.get("recent_contributions", 5),
                    "include_author_content": args.get("include_author_content", True),
                    "tool_call_id": tool_call.id,
                }
            )
        else:
            actions.append(
                {
                    "tool": "record_developer_skill_profile",
                    "profile": args["profile"],
                    "tool_call_id": tool_call.id,
                }
            )
    return actions


def format_toolcall_observation_messages(
    *,
    actions: list[dict],
    outputs: list[dict],
    observation_template: str,
    template_vars: dict | None = None,
    multimodal_regex: str = "",
) -> list[dict]:
    """Format execution outputs into tool result messages."""
    not_executed = {"output": "", "returncode": -1, "exception_info": "action was not executed"}
    padded_outputs = outputs + [not_executed] * (len(actions) - len(outputs))
    results = []
    for action, output in zip(actions, padded_outputs):
        content = Template(observation_template, undefined=StrictUndefined).render(
            output=output, **(template_vars or {})
        )
        msg = {
            "content": content,
            "extra": {
                "raw_output": output.get("output", ""),
                "returncode": output.get("returncode"),
                "timestamp": time.time(),
                "exception_info": output.get("exception_info"),
                **output.get("extra", {}),
            },
        }
        if "tool_call_id" in action:
            msg["tool_call_id"] = action["tool_call_id"]
            msg["role"] = "tool"
        else:
            msg["role"] = "user"  # human issued commands
        if multimodal_regex:
            msg = expand_multimodal_content(msg, pattern=multimodal_regex)
        results.append(msg)
    return results
