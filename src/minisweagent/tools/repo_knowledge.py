import ast
import json
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IGNORED_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "site-packages",
    "venv",
}


@dataclass
class RepoKnowledgeNode:
    id: str
    kind: str
    name: str
    file: str
    start_line: int
    end_line: int
    content: str

    def to_result(self, *, score: float, include_code: bool) -> dict[str, Any]:
        result = {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "file": self.file,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "score": round(score, 3),
            "summary": _summarize(self.content),
        }
        if include_code:
            result["code"] = self.content
        return result


@dataclass
class RepoKnowledgeRelation:
    source: str
    target: str
    relation_type: str

    def to_dict(self) -> dict[str, str]:
        return {"from": self.source, "to": self.target, "type": self.relation_type}


def get_repo_knowledge(
    root: str | Path,
    *,
    query: str,
    path: str = "",
    max_results: int = 8,
    include_code: bool = True,
    author: str = "",
    recent_contributions: int = 5,
    include_author_content: bool = True,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    search_root = _resolve_search_root(root_path, path)
    nodes, relations = _build_index(root_path, search_root)
    author_context = _author_context(
        root_path,
        search_root,
        query=query,
        author=author,
        recent_contributions=recent_contributions,
        include_author_content=include_author_content,
        include_code=include_code,
    )
    author_files = author_context.get("touched_files", [])
    author_file_names = {file["file"] for file in author_files}
    scored_nodes = sorted(
        ((node, _score(node, query) + (4 if node.file in author_file_names else 0)) for node in nodes),
        key=lambda item: (-item[1], item[0].file, item[0].start_line),
    )
    results = [node.to_result(score=score, include_code=include_code) for node, score in scored_nodes[:max_results]]
    result_ids = {result["id"] for result in results}
    output = {
        "query": query,
        "root": str(root_path),
        "path": path,
        "result_count": len(results),
        "results": results,
        "relations": [
            relation.to_dict()
            for relation in relations
            if relation.source in result_ids or relation.target in result_ids
        ],
    }
    if author:
        output["author"] = author
        output["author_files"] = author_files
        output["author_context"] = author_context
    return output


def get_repo_knowledge_output(root: str | Path, action: dict) -> dict[str, Any]:
    try:
        return {
            "output": json.dumps(
                get_repo_knowledge(
                    root,
                    query=action["query"],
                    path=action.get("path", ""),
                    max_results=action.get("max_results", 8),
                    include_code=action.get("include_code", True),
                    author=action.get("author", ""),
                    recent_contributions=action.get("recent_contributions", 5),
                    include_author_content=action.get("include_author_content", True),
                ),
                indent=2,
            ),
            "returncode": 0,
            "exception_info": "",
            "extra": {"tool": "get_repo_knowledge"},
        }
    except Exception as e:
        return {
            "output": "",
            "returncode": -1,
            "exception_info": f"An error occurred while getting repository knowledge: {e}",
            "extra": {"tool": "get_repo_knowledge", "exception_type": type(e).__name__, "exception": str(e)},
        }


def get_repo_knowledge_command(action: dict) -> str:
    return "python3 -c " + shlex.quote(_DOCKER_SCRIPT) + " " + shlex.quote(json.dumps(action))


def _resolve_search_root(root: Path, path: str) -> Path:
    search_root = (root / path).resolve() if path else root
    search_root.relative_to(root)
    if not search_root.exists():
        raise FileNotFoundError(search_root)
    return search_root


def _build_index(root: Path, search_root: Path) -> tuple[list[RepoKnowledgeNode], list[RepoKnowledgeRelation]]:
    nodes = []
    relations = []
    for file in _iter_python_files(search_root):
        file_nodes = _nodes_from_file(root, file)
        nodes.extend(file_nodes)
        relations.extend(_relations_from_nodes(file_nodes))
    return nodes, relations


def _author_context(
    root: Path,
    search_root: Path,
    *,
    query: str,
    author: str,
    recent_contributions: int,
    include_author_content: bool,
    include_code: bool,
) -> dict[str, Any]:
    if not author:
        return {}
    commits = _recent_author_commits(root, author, recent_contributions)
    touched_files = _touched_files(root, search_root, commits)
    return {
        "recent_contributions": len(commits),
        "recent_commits": commits,
        "touched_files": touched_files,
        "content": _author_content(root, query, touched_files, include_code) if include_author_content else [],
    }


def _recent_author_commits(root: Path, author: str, limit: int) -> list[dict[str, Any]]:
    output = _git(root, "log", f"--author={author}", "-n", str(max(limit, 0)), "--date=iso-strict", "--pretty=%H%x00%an%x00%ae%x00%ad%x00%s")
    commits = []
    for line in output.splitlines():
        commit, name, email, date, subject = line.split("\0", 4)
        files = _git(root, "show", "--name-only", "--format=", commit).splitlines()
        commits.append(
            {
                "commit": commit,
                "short_commit": commit[:12],
                "author_name": name,
                "author_email": email,
                "date": date,
                "subject": subject,
                "files": [file for file in files if file],
            }
        )
    return commits


def _touched_files(root: Path, search_root: Path, commits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_file: dict[str, list[str]] = {}
    for commit in commits:
        for file in commit["files"]:
            file_path = (root / file).resolve()
            if _is_relative_to(file_path, search_root):
                by_file.setdefault(file, []).append(commit["short_commit"])
    return [
        {"file": file, "commits": commits, "exists": (root / file).exists()}
        for file, commits in sorted(by_file.items())
    ]


def _author_content(root: Path, query: str, touched_files: list[dict[str, Any]], include_code: bool) -> list[dict[str, Any]]:
    content = []
    for touched_file in touched_files:
        file = root / touched_file["file"]
        if not file.exists() or file.is_dir():
            continue
        if file.suffix == ".py":
            nodes = sorted(_nodes_from_file(root, file), key=lambda node: (-_score(node, query), node.start_line))
            if nodes:
                content.append(nodes[0].to_result(score=_score(nodes[0], query), include_code=include_code))
        elif _is_text_file(file):
            text = file.read_text(encoding="utf-8", errors="replace")
            item = {
                "file": touched_file["file"],
                "kind": "file",
                "name": touched_file["file"],
                "start_line": 1,
                "end_line": len(text.splitlines()),
                "summary": _summarize(text),
            }
            if include_code:
                item["code"] = "\n".join(text.splitlines()[:120])
            content.append(item)
    return content[:10]


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
    )
    return result.stdout if result.returncode == 0 else ""


def _is_text_file(file: Path) -> bool:
    return file.suffix.lower() in {".md", ".rst", ".txt", ".toml", ".yaml", ".yml", ".json"}


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _iter_python_files(root: Path):
    files = [root] if root.is_file() else root.rglob("*.py")
    for file in files:
        if file.suffix == ".py" and not any(part in IGNORED_DIRS for part in file.parts):
            yield file


def _nodes_from_file(root: Path, file: Path) -> list[RepoKnowledgeNode]:
    try:
        source = file.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return []
    lines = source.splitlines()
    rel_file = file.relative_to(root).as_posix()
    nodes = [
        RepoKnowledgeNode(
            id=f"{rel_file}:module",
            kind="module",
            name=rel_file,
            file=rel_file,
            start_line=1,
            end_line=len(lines),
            content=_module_summary(tree, rel_file),
        )
    ]
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            nodes.append(
                RepoKnowledgeNode(
                    id=f"{rel_file}:{node.lineno}:{node.name}",
                    kind=kind,
                    name=node.name,
                    file=rel_file,
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno),
                    content="\n".join(lines[node.lineno - 1 : getattr(node, "end_lineno", node.lineno)]),
                )
            )
    return nodes


def _relations_from_nodes(nodes: list[RepoKnowledgeNode]) -> list[RepoKnowledgeRelation]:
    relations = []
    modules = [node for node in nodes if node.kind == "module"]
    if not modules:
        return relations
    module = modules[0]
    for node in nodes:
        if node is not module:
            relations.append(RepoKnowledgeRelation(module.id, node.id, "contains"))
    for parent in nodes:
        for child in nodes:
            if parent is child or parent.kind == "module" or child.kind == "module":
                continue
            if parent.start_line < child.start_line and child.end_line <= parent.end_line:
                relations.append(RepoKnowledgeRelation(parent.id, child.id, "child"))
    return relations


def _module_summary(tree: ast.Module, rel_file: str) -> str:
    names = [node.name for node in tree.body if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)]
    return f"{rel_file}\nSymbols: {', '.join(names[:50])}"


def _score(node: RepoKnowledgeNode, query: str) -> float:
    query_terms = _tokens(query)
    if not query_terms:
        return 0.0
    name_terms = _tokens(node.name)
    path_terms = _tokens(node.file)
    content_terms = _tokens(node.content)
    score = 0.0
    score += 5 * len(query_terms & name_terms)
    score += 3 * len(query_terms & path_terms)
    score += len(query_terms & content_terms)
    if node.kind in ["class", "function"]:
        score += 0.5
    if query.lower() in node.content.lower() or query.lower() in node.file.lower():
        score += 10
    return score


def _tokens(text: str) -> set[str]:
    tokens = set()
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        tokens.add(token.lower())
        tokens.update(part.lower() for part in token.split("_") if part)
    return tokens


def _summarize(content: str) -> str:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    return "\n".join(lines[:6])


_DOCKER_SCRIPT = r'''
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

IGNORED_DIRS = {
    ".git", ".hg", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".venv",
    "__pycache__", "build", "dist", "node_modules", "site-packages", "venv",
}


def tokens(text):
    result = set()
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        result.add(token.lower())
        result.update(part.lower() for part in token.split("_") if part)
    return result


def summarize(content):
    return "\n".join([line.strip() for line in content.splitlines() if line.strip()][:6])


def score(node, query):
    query_terms = tokens(query)
    if not query_terms:
        return 0.0
    value = 5 * len(query_terms & tokens(node["name"]))
    value += 3 * len(query_terms & tokens(node["file"]))
    value += len(query_terms & tokens(node["content"]))
    if node["kind"] in ["class", "function"]:
        value += 0.5
    if query.lower() in node["content"].lower() or query.lower() in node["file"].lower():
        value += 10
    return value


def iter_python_files(root):
    files = [root] if root.is_file() else root.rglob("*.py")
    for file in files:
        if file.suffix == ".py" and not any(part in IGNORED_DIRS for part in file.parts):
            yield file


def nodes_from_file(root, file):
    try:
        source = file.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return []
    lines = source.splitlines()
    rel_file = file.relative_to(root).as_posix()
    names = [node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
    nodes = [{
        "id": f"{rel_file}:module",
        "kind": "module",
        "name": rel_file,
        "file": rel_file,
        "start_line": 1,
        "end_line": len(lines),
        "content": f"{rel_file}\nSymbols: {', '.join(names[:50])}",
    }]
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            nodes.append({
                "id": f"{rel_file}:{node.lineno}:{node.name}",
                "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                "name": node.name,
                "file": rel_file,
                "start_line": node.lineno,
                "end_line": getattr(node, "end_lineno", node.lineno),
                "content": "\n".join(lines[node.lineno - 1:getattr(node, "end_lineno", node.lineno)]),
            })
    return nodes


def relations_from_nodes(nodes):
    modules = [node for node in nodes if node["kind"] == "module"]
    if not modules:
        return []
    module = modules[0]
    relations = [{"from": module["id"], "to": node["id"], "type": "contains"} for node in nodes if node is not module]
    for parent in nodes:
        for child in nodes:
            if parent is child or parent["kind"] == "module" or child["kind"] == "module":
                continue
            if parent["start_line"] < child["start_line"] and child["end_line"] <= parent["end_line"]:
                relations.append({"from": parent["id"], "to": child["id"], "type": "child"})
    return relations


def run_git(root, *args):
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
    )
    return result.stdout if result.returncode == 0 else ""


def is_relative_to(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def is_text_file(file):
    return file.suffix.lower() in {".md", ".rst", ".txt", ".toml", ".yaml", ".yml", ".json"}


def recent_author_commits(root, author, limit):
    output = run_git(root, "log", f"--author={author}", "-n", str(max(limit, 0)), "--date=iso-strict", "--pretty=%H%x00%an%x00%ae%x00%ad%x00%s")
    commits = []
    for line in output.splitlines():
        commit, name, email, date, subject = line.split("\0", 4)
        files = run_git(root, "show", "--name-only", "--format=", commit).splitlines()
        commits.append({
            "commit": commit,
            "short_commit": commit[:12],
            "author_name": name,
            "author_email": email,
            "date": date,
            "subject": subject,
            "files": [file for file in files if file],
        })
    return commits


def touched_files(root, search_root, commits):
    by_file = {}
    for commit in commits:
        for file in commit["files"]:
            file_path = (root / file).resolve()
            if is_relative_to(file_path, search_root):
                by_file.setdefault(file, []).append(commit["short_commit"])
    return [
        {"file": file, "commits": commits, "exists": (root / file).exists()}
        for file, commits in sorted(by_file.items())
    ]


def author_content(root, query, touched, include_code):
    content = []
    for item in touched:
        file = root / item["file"]
        if not file.exists() or file.is_dir():
            continue
        if file.suffix == ".py":
            file_nodes = sorted(nodes_from_file(root, file), key=lambda node: (-score(node, query), node["start_line"]))
            if file_nodes:
                node = file_nodes[0]
                result = {key: node[key] for key in ["id", "kind", "name", "file", "start_line", "end_line"]}
                result["score"] = round(score(node, query), 3)
                result["summary"] = summarize(node["content"])
                if include_code:
                    result["code"] = node["content"]
                content.append(result)
        elif is_text_file(file):
            text = file.read_text(encoding="utf-8", errors="replace")
            result = {
                "file": item["file"],
                "kind": "file",
                "name": item["file"],
                "start_line": 1,
                "end_line": len(text.splitlines()),
                "summary": summarize(text),
            }
            if include_code:
                result["code"] = "\n".join(text.splitlines()[:120])
            content.append(result)
    return content[:10]


def author_context(root, search_root, action):
    if not action.get("author"):
        return {}
    commits = recent_author_commits(root, action["author"], action.get("recent_contributions", 5))
    touched = touched_files(root, search_root, commits)
    return {
        "recent_contributions": len(commits),
        "recent_commits": commits,
        "touched_files": touched,
        "content": author_content(root, action["query"], touched, action.get("include_code", True))
        if action.get("include_author_content", True) else [],
    }


action = json.loads(sys.argv[1])
root = Path.cwd().resolve()
search_root = (root / action.get("path", "")).resolve() if action.get("path") else root
search_root.relative_to(root)
nodes = []
relations = []
for file in iter_python_files(search_root):
    file_nodes = nodes_from_file(root, file)
    nodes.extend(file_nodes)
    relations.extend(relations_from_nodes(file_nodes))
context = author_context(root, search_root, action)
author_files = context.get("touched_files", [])
author_file_names = {file["file"] for file in author_files}
scored = sorted(
    ((node, score(node, action["query"]) + (4 if node["file"] in author_file_names else 0)) for node in nodes),
    key=lambda item: (-item[1], item[0]["file"], item[0]["start_line"]),
)
results = []
for node, node_score in scored[:action.get("max_results", 8)]:
    result = {key: node[key] for key in ["id", "kind", "name", "file", "start_line", "end_line"]}
    result["score"] = round(node_score, 3)
    result["summary"] = summarize(node["content"])
    if action.get("include_code", True):
        result["code"] = node["content"]
    results.append(result)
result_ids = {result["id"] for result in results}
payload = {
    "query": action["query"],
    "root": str(root),
    "path": action.get("path", ""),
    "result_count": len(results),
    "results": results,
    "relations": [relation for relation in relations if relation["from"] in result_ids or relation["to"] in result_ids],
}
if action.get("author"):
    payload["author"] = action["author"]
    payload["author_files"] = author_files
    payload["author_context"] = context
print(json.dumps(payload, indent=2))
'''
