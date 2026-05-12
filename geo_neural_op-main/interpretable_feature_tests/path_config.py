from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "gnp").is_dir() and (candidate / "interpretable_feature_tests").exists():
            return candidate
        if (candidate / "gnp").is_dir() and candidate.name == "geo_neural_op-main":
            return candidate
    raise RuntimeError(f"Could not find geo_neural_op-main from {start}")


def find_test_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if candidate.name == "interpretable_feature_tests" and (candidate / "data").is_dir():
            return candidate
    raise RuntimeError(f"Could not find interpretable_feature_tests from {start}")


def roots_for(script_file: str | Path) -> dict[str, Path]:
    script_path = Path(script_file).resolve()
    script_dir = script_path.parent
    repo_root = find_repo_root(script_dir)
    test_root = find_test_root(script_dir)
    workspace_root = repo_root.parent
    return {
        "SCRIPT_DIR": script_dir,
        "REPO_ROOT": repo_root,
        "TEST_ROOT": test_root,
        "WORKSPACE_ROOT": workspace_root,
        "MODELNET_ROOT": workspace_root / "ModelNet10",
        "DATA_ROOT": test_root / "data",
        "OUTPUT_ROOT": test_root / "output",
        "FEATURE_SCRIPTS_ROOT": test_root / "feature_scripts",
        "GAUSSIAN_SCRIPTS_ROOT": test_root / "gaussian_dataset_creation",
    }
