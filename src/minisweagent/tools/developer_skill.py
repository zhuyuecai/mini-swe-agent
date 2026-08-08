import json
import re
import shlex
import subprocess
from collections import Counter
from pathlib import Path


class CommitChange:
    def __init__(self, hash_, author_name, author_email, date, subject, files, insertions, deletions):
        self.hash = hash_
        self.author_name = author_name
        self.author_email = author_email
        self.date = date
        self.subject = subject
        self.files = files
        self.insertions = insertions
        self.deletions = deletions


def get_developer_skill(root, *, developer):
    root_path = Path(root).resolve()
    commits = [commit for commit in _git_log(root_path) if _matches_developer(developer, commit)]
    file_counts = Counter(file for commit in commits for file in commit.files)
    extensions = Counter(Path(file).suffix or "<none>" for file in file_counts)
    directories = Counter(_ownership_path(file) for file in file_counts)
    style = _style_from_files(root_path, [file for file, _ in file_counts.most_common(25)])
    testing = _testing_style(root_path, file_counts)
    return {
        "identity": {
            "query": developer,
            "github": _github_name(developer, commits),
            "emails": sorted({commit.author_email for commit in commits}),
            "names": sorted({commit.author_name for commit in commits}),
            "commit_count": len(commits),
            "date_range": [commits[-1].date, commits[0].date] if commits else [],
        },
        "ownership": [
            {
                "path": path,
                "commit_count": count,
                "confidence": _confidence(count, len(commits)),
                "evidence": f"Changed in {count} of {len(commits)} matched commits",
            }
            for path, count in directories.most_common(8)
        ],
        "technical_strengths": _technical_strengths(extensions, directories, testing),
        "style": style,
        "testing": testing,
        "file_familiarity": {
            "frequently_changed_files": [
                {"file": file, "commit_count": count} for file, count in file_counts.most_common(15)
            ],
            "frequently_touched_symbols": _symbols_from_files(root_path, [file for file, _ in file_counts.most_common(10)]),
        },
        "change_style": {
            "typical_patch_size": _patch_size(commits),
            "average_files_per_commit": round(sum(len(commit.files) for commit in commits) / len(commits), 2)
            if commits
            else 0,
            "average_insertions_per_commit": round(sum(commit.insertions for commit in commits) / len(commits), 2)
            if commits
            else 0,
            "average_deletions_per_commit": round(sum(commit.deletions for commit in commits) / len(commits), 2)
            if commits
            else 0,
            "common_commit_words": [word for word, _ in _commit_words(commits).most_common(12)],
        },
        "mimicry_guidance": _mimicry_guidance(style, testing, directories, extensions),
        "confidence": {
            "overall": _overall_confidence(len(commits)),
            "strong_evidence": _strong_evidence(len(commits), directories, file_counts),
            "caveats": _caveats(developer, commits),
        },
    }


def get_developer_skill_output(root, action):
    try:
        return {
            "output": json.dumps(get_developer_skill(root, developer=action["developer"]), indent=2),
            "returncode": 0,
            "exception_info": "",
            "extra": {"tool": "get_developer_skill"},
        }
    except Exception as e:
        return {
            "output": "",
            "returncode": -1,
            "exception_info": f"An error occurred while getting developer skill: {e}",
            "extra": {"tool": "get_developer_skill", "exception_type": type(e).__name__, "exception": str(e)},
        }


def get_developer_skill_command(action):
    script = Path(__file__).read_text() + (
        "\nimport sys\n"
        "action = json.loads(sys.argv[1])\n"
        "print(json.dumps(get_developer_skill(Path.cwd(), developer=action['developer']), indent=2))\n"
    )
    return "python3 -c " + shlex.quote(script) + " " + shlex.quote(json.dumps(action))


def _git_log(root):
    output = subprocess.run(
        [
            "git",
            "log",
            "--all",
            "--date=short",
            "--format=%x1e%H%x1f%an%x1f%ae%x1f%ad%x1f%s",
            "--numstat",
        ],
        cwd=root,
        check=True,
        encoding="utf-8",
        errors="replace",
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    return [_parse_commit(record) for record in output.split("\x1e") if record.strip()]


def _parse_commit(record):
    lines = record.strip().splitlines()
    hash_, author_name, author_email, date, subject = lines[0].split("\x1f", maxsplit=4)
    files = []
    insertions = 0
    deletions = 0
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        files.append(parts[2])
        if parts[0].isdigit():
            insertions += int(parts[0])
        if parts[1].isdigit():
            deletions += int(parts[1])
    return CommitChange(hash_, author_name, author_email, date, subject, files, insertions, deletions)


def _matches_developer(developer, commit):
    query = developer.lower()
    email = commit.author_email.lower()
    candidates = {
        commit.author_name.lower(),
        email,
        email.split("@", maxsplit=1)[0],
        email.split("@", maxsplit=1)[0].split("+")[-1],
    }
    return query in candidates or query in commit.author_name.lower() or query in email


def _github_name(developer, commits):
    for commit in commits:
        local, _, domain = commit.author_email.partition("@")
        if domain == "users.noreply.github.com":
            return local.split("+")[-1]
    return developer if "@" not in developer else ""


def _ownership_path(file):
    parts = Path(file).parts
    return "/".join(parts[:2]) if len(parts) > 1 else file


def _style_from_files(root, files):
    text = "\n".join(_read_existing(root / file) for file in files if Path(file).suffix == ".py")
    if not text:
        return {
            "naming": "Insufficient Python file evidence.",
            "structure": "Insufficient Python file evidence.",
            "typing": "Insufficient Python file evidence.",
            "error_handling": "Insufficient Python file evidence.",
            "comments": "Insufficient Python file evidence.",
            "dependencies": "Insufficient Python file evidence.",
        }
    return {
        "naming": "Uses snake_case Python symbols." if re.search(r"def [a-z_][a-z0-9_]*", text) else "Mixed evidence.",
        "structure": "Keeps logic in functions/classes within existing modules.",
        "typing": "Uses return type annotations." if re.search(r"def .+ -> ", text) else "Limited return type annotations observed.",
        "error_handling": "Uses explicit exception handling." if "except " in text else "Little explicit exception handling observed.",
        "comments": "Uses comments sparingly." if text.count("#") < max(3, len(text.splitlines()) // 20) else "Uses inline comments regularly.",
        "dependencies": _dependency_style(text),
    }


def _testing_style(root, file_counts):
    test_files = [file for file in file_counts if _is_test_file(file)]
    text = "\n".join(_read_existing(root / file) for file in test_files[:20])
    return {
        "test_files_changed": len(test_files),
        "frameworks": ["pytest"] if "pytest" in text or any("test_" in Path(file).name for file in test_files) else [],
        "patterns": [
            pattern
            for pattern, present in {
                "parametrize": "pytest.mark.parametrize" in text,
                "fixtures": "@pytest.fixture" in text,
                "direct assertions": "assert " in text,
            }.items()
            if present
        ],
        "placement": sorted({_ownership_path(file) for file in test_files})[:8],
    }


def _technical_strengths(extensions, directories, testing):
    strengths = [
        {"area": extension, "evidence": f"Changed {count} files with this extension"}
        for extension, count in extensions.most_common(5)
    ]
    strengths.extend(
        {"area": path, "evidence": f"Frequent changes in {path}"} for path, _ in directories.most_common(3)
    )
    if testing["test_files_changed"]:
        strengths.append({"area": "testing", "evidence": f"Changed {testing['test_files_changed']} test files"})
    return strengths


def _symbols_from_files(root, files):
    symbols = []
    for file in files:
        source = _read_existing(root / file)
        for match in re.finditer(r"^(class|def|async def) ([A-Za-z_][A-Za-z0-9_]*)", source, re.MULTILINE):
            symbols.append({"symbol": match.group(2), "file": file, "kind": "class" if match.group(1) == "class" else "function"})
    return symbols[:20]


def _patch_size(commits):
    if not commits:
        return "unknown"
    average = sum(commit.insertions + commit.deletions for commit in commits) / len(commits)
    if average < 50:
        return "small"
    if average < 300:
        return "medium"
    return "large"


def _commit_words(commits):
    stop = {"a", "an", "and", "for", "in", "of", "the", "to", "with"}
    return Counter(
        word
        for commit in commits
        for word in re.findall(r"[a-z][a-z0-9_-]+", commit.subject.lower())
        if word not in stop
    )


def _mimicry_guidance(style, testing, directories, extensions):
    prefer = [f"Start in familiar areas: {', '.join(path for path, _ in directories.most_common(3))}"] if directories else []
    if extensions:
        prefer.append(f"Match common file types: {', '.join(extension for extension, _ in extensions.most_common(3))}")
    if testing["patterns"]:
        prefer.append(f"Follow observed test patterns: {', '.join(testing['patterns'])}")
    return {
        "prefer": prefer + [style["typing"], style["error_handling"]],
        "avoid": ["Adding broad abstractions without matching evidence in changed files."],
        "match": [style["naming"], style["comments"], style["dependencies"]],
    }


def _confidence(count, total):
    if total and count / total >= 0.4:
        return "high"
    if count >= 3:
        return "medium"
    return "low"


def _overall_confidence(commit_count):
    if commit_count >= 20:
        return "high"
    if commit_count >= 5:
        return "medium"
    return "low"


def _strong_evidence(commit_count, directories, file_counts):
    evidence = [f"{commit_count} matched commits"] if commit_count else []
    if directories:
        evidence.append(f"Top ownership path: {directories.most_common(1)[0][0]}")
    if file_counts:
        evidence.append(f"Top changed file: {file_counts.most_common(1)[0][0]}")
    return evidence


def _caveats(developer, commits):
    caveats = []
    if not commits:
        caveats.append(f"No commits matched {developer!r}.")
    elif len(commits) < 5:
        caveats.append("Profile is based on fewer than 5 commits.")
    caveats.append("Style is inferred from git history and current file contents, not from private review context.")
    return caveats


def _dependency_style(text):
    if "from pathlib import Path" in text:
        return "Uses pathlib where relevant."
    if "import os" in text:
        return "Uses standard-library OS helpers."
    return "No strong dependency preference observed."


def _is_test_file(file):
    path = Path(file)
    return "test" in path.parts or path.name.startswith("test_") or path.name.endswith("_test.py")


def _read_existing(path):
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
