import pytest

from minisweagent.exceptions import FormatError
from minisweagent.models.utils.actions_toolcall_response import (
    TOOLS_RESPONSE_API,
    parse_toolcall_actions_response,
)


def test_response_api_valid_get_repo_knowledge_tool_call():
    assert parse_toolcall_actions_response(
        [
            {
                "type": "function_call",
                "call_id": "call_repo",
                "name": "get_repo_knowledge",
                "arguments": '{"query": "config loading", "path": "src", "include_code": false}',
            }
        ],
        format_error_template="{{ error }}",
    ) == [
        {
            "tool": "get_repo_knowledge",
            "query": "config loading",
            "path": "src",
            "max_results": 8,
            "include_code": False,
            "tool_call_id": "call_repo",
        }
    ]


def test_response_api_missing_query_raises_format_error():
    with pytest.raises(FormatError) as exc_info:
        parse_toolcall_actions_response(
            [
                {
                    "type": "function_call",
                    "call_id": "call_repo",
                    "name": "get_repo_knowledge",
                    "arguments": '{"path": "src"}',
                }
            ],
            format_error_template="{{ error }}",
        )
    assert "Missing 'query' argument" in exc_info.value.messages[0]["content"][0]["text"]


def test_response_api_tools_include_repo_knowledge():
    assert [tool["name"] for tool in TOOLS_RESPONSE_API] == ["bash", "get_repo_knowledge"]
