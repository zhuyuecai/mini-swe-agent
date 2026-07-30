import json

import pytest
from datasets import Dataset

from minisweagent.run.utilities.author_enrich import (
    author_cache_path,
    dataset_path_from_subset,
    dataset_path_slug,
    enrich_dataset,
    enriched_dataset_path,
    extract_pr_number,
    load_author_cache,
    save_author_cache,
)


def test_dataset_paths_are_derived_from_mapping(tmp_path):
    assert dataset_path_from_subset("lite") == "princeton-nlp/SWE-Bench_Lite"
    assert dataset_path_slug("lite") == "princeton-nlp__SWE-Bench_Lite"
    assert author_cache_path("lite", "dev", tmp_path) == (
        tmp_path / "princeton-nlp__SWE-Bench_Lite" / "dev" / "pr_author_cache.json"
    )
    assert enriched_dataset_path("lite", "dev", tmp_path) == (
        tmp_path / "princeton-nlp__SWE-Bench_Lite" / "dev" / "dataset"
    )


def test_extract_pr_number():
    assert extract_pr_number("django__django-12345") == "12345"
    with pytest.raises(ValueError, match="Could not extract PR number"):
        extract_pr_number("django__django-not-a-pr")


def test_author_cache_roundtrip(tmp_path):
    save_author_cache(tmp_path / "cache" / "authors.json", {"django/django#1": "alice"})
    assert json.loads((tmp_path / "cache" / "authors.json").read_text()) == {"django/django#1": "alice"}
    assert load_author_cache(tmp_path / "cache" / "authors.json") == {"django/django#1": "alice"}
    assert load_author_cache(tmp_path / "missing.json") == {}


def test_enrich_dataset_adds_pr_author(tmp_path):
    calls = []

    def fetch_author(repo, pr_number, cache, cache_path):
        calls.append((repo, pr_number, cache_path))
        cache[f"{repo}#{pr_number}"] = f"{repo}:{pr_number}"
        return cache[f"{repo}#{pr_number}"]

    enriched = enrich_dataset(
        Dataset.from_list(
            [
                {"instance_id": "django__django-1", "repo": "django/django"},
                {"instance_id": "psf__requests-2", "repo": "psf/requests"},
            ]
        ),
        {},
        tmp_path / "authors.json",
        fetch_author,
    )

    assert list(enriched["pr_author"]) == ["django/django:1", "psf/requests:2"]
    assert calls == [
        ("django/django", "1", tmp_path / "authors.json"),
        ("psf/requests", "2", tmp_path / "authors.json"),
    ]
