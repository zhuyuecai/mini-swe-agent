#!/usr/bin/env python3

"""Add GitHub PR author metadata to SWE-bench datasets."""

import json
import os
import shutil
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import typer
from datasets import Dataset, load_dataset, load_from_disk
from rich.console import Console

from minisweagent import global_config_dir
from minisweagent.run.benchmarks.swebench import DATASET_MAPPING

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
console = Console()

DEFAULT_OUTPUT_ROOT = global_config_dir / "author_enrich"


def extract_pr_number(instance_id: str) -> str:
    pr_number = instance_id.rsplit("-", 1)[-1]
    if not pr_number.isdigit():
        msg = f"Could not extract PR number from instance_id={instance_id!r}"
        raise ValueError(msg)
    return pr_number


def dataset_path_from_subset(subset: str) -> str:
    return DATASET_MAPPING.get(subset, subset)


def dataset_path_slug(subset: str) -> str:
    return dataset_path_from_subset(subset).replace("/", "__")


def author_cache_path(subset: str, split: str, output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    return output_root / dataset_path_slug(subset) / split / "pr_author_cache.json"


def enriched_dataset_path(subset: str, split: str, output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    return output_root / dataset_path_slug(subset) / split / "dataset"


def load_author_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_author_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")


def github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "mini-swe-agent-pr-author-enricher",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token := os.getenv("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_pr_author(repo: str, pr_number: str, cache: dict[str, str], cache_path: Path) -> str:
    cache_key = f"{repo}#{pr_number}"
    if cache_key in cache:
        return cache[cache_key]

    try:
        with urlopen(
            Request(f"https://api.github.com/repos/{repo}/pulls/{pr_number}", headers=github_headers()), timeout=30
        ) as response:
            data = json.load(response)
    except HTTPError as error:
        if error.code == 403:
            msg = "GitHub returned 403. Set GITHUB_TOKEN to avoid the low unauthenticated rate limit, then rerun."
            raise RuntimeError(msg) from error
        msg = f"GitHub request failed for {cache_key}: HTTP {error.code}"
        raise RuntimeError(msg) from error
    except URLError as error:
        msg = f"GitHub request failed for {cache_key}: {error.reason}"
        raise RuntimeError(msg) from error

    author = data.get("user", {}).get("login")
    if not author:
        msg = f"GitHub response did not include an author for {cache_key}"
        raise RuntimeError(msg)

    cache[cache_key] = author
    save_author_cache(cache_path, cache)
    time.sleep(0.2)
    return author


def add_pr_author(record: dict, cache: dict[str, str], cache_path: Path, fetch_author=fetch_pr_author) -> dict:
    record["pr_author"] = fetch_author(record["repo"], extract_pr_number(record["instance_id"]), cache, cache_path)
    return record


def enrich_dataset(dataset: Dataset, cache: dict[str, str], cache_path: Path, fetch_author=fetch_pr_author) -> Dataset:
    return Dataset.from_list([add_pr_author(dict(record), cache, cache_path, fetch_author) for record in dataset])


# fmt: off
@app.command()
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    output_root: Path = typer.Option(DEFAULT_OUTPUT_ROOT, "--output-root", help="Root directory for the derived cache and output paths", rich_help_panel="Output"),
    force: bool = typer.Option(False, "--force", help="Regenerate the enriched dataset if it already exists", rich_help_panel="Output"),
) -> None:
    # fmt: on
    cache_path = author_cache_path(subset, split, output_root)
    dataset_output_path = enriched_dataset_path(subset, split, output_root)
    if dataset_output_path.exists() and not force:
        console.print(f"Loading existing enriched dataset from [bold green]{dataset_output_path}[/bold green]")
        console.print(load_from_disk(dataset_output_path))
        return

    dataset_path = dataset_path_from_subset(subset)
    console.print(f"Loading dataset [bold green]{dataset_path}[/bold green], split [bold green]{split}[/bold green]")
    enriched = enrich_dataset(load_dataset(dataset_path, split=split), load_author_cache(cache_path), cache_path)

    if dataset_output_path.exists():
        shutil.rmtree(dataset_output_path)
    dataset_output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.save_to_disk(dataset_output_path)
    console.print(f"Saved enriched dataset to [bold green]{dataset_output_path}[/bold green]")
    console.print(f"Saved author cache to [bold green]{cache_path}[/bold green]")


if __name__ == "__main__":
    app()
