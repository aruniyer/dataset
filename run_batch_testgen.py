#!/usr/bin/env python3
"""Batch test generation across all SWE-CARE instances.

Processes every instance in the dataset with optional parallel workers.
Each worker gets its own isolated repo working directory (via git clone --shared)
and virtual environment to avoid data races. Source repos are cached and reused.

Usage:
  python run_batch_testgen.py --split dev test             # Both splits
  python run_batch_testgen.py --split dev --keep-repos     # Keep clones
  python run_batch_testgen.py --no-resume                  # Re-process everything
  python run_batch_testgen.py --split dev --workers 4      # Parallel processing
  python run_batch_testgen.py --repo tobymao/sqlglot       # Single repo
  python run_batch_testgen.py --split dev --limit 10       # First 10 instances
"""

import argparse
import json
import logging
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from execution.container_runtime import docker_image_exists, get_docker_image_name
from pipeline import dataset_utils, repo_manager, test_generator
from pipeline.llm_client import LLMUsage
from run_testgen import process_instance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_existing_result(output_dir: Path, instance_id: str) -> dict | None:
    """Load an existing result.json for an instance, or None if not found."""
    instance_dir = output_dir / instance_id.replace("/", "__")
    result_file = instance_dir / "result.json"
    if result_file.exists():
        try:
            return json.loads(result_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not load existing result for %s: %s", instance_id, e)
    return None


def cleanup_repo(repo: str, repos_dir: Path) -> None:
    """Remove a cloned repo directory."""
    repo_dir = repos_dir / repo.replace("/", "__")
    if repo_dir.exists():
        logger.info("Cleaning up repo: %s", repo_dir)
        shutil.rmtree(repo_dir)


def write_summary(output_dir: Path, all_results: list[dict], elapsed: float) -> dict:
    """Write summary.json with per-repo aggregation and overall stats."""
    # Per-repo aggregation
    repo_data: dict[str, dict] = {}
    for r in all_results:
        repo = r["repo"]
        if repo not in repo_data:
            repo_data[repo] = {
                "repo": repo,
                "instances": 0,
                "comments": 0,
                "tests_generated": 0,
                "successes": 0,
            }
        rd = repo_data[repo]
        rd["instances"] += 1
        rd["comments"] += r.get("num_comments", 0)
        rd["tests_generated"] += sum(
            1 for c in r.get("results", []) if c.get("test_code")
        )
        rd["successes"] += sum(
            1 for c in r.get("results", []) if c.get("success")
        )

    for rd in repo_data.values():
        rd["success_rate"] = (
            rd["successes"] / rd["tests_generated"] if rd["tests_generated"] > 0 else 0.0
        )

    total_comments = sum(r.get("num_comments", 0) for r in all_results)
    total_tests = sum(
        sum(1 for c in r.get("results", []) if c.get("test_code"))
        for r in all_results
    )
    total_successes = sum(
        sum(1 for c in r.get("results", []) if c.get("success"))
        for r in all_results
    )
    total_attempts = sum(
        sum(c.get("attempts_used", 0) for c in r.get("results", []))
        for r in all_results
    )
    total_errors = sum(1 for r in all_results if r.get("error"))

    # Aggregate usage across all instances
    total_usage = LLMUsage()
    for r in all_results:
        if "usage" in r:
            u = r["usage"]
            total_usage = total_usage + LLMUsage(
                prompt_tokens=u.get("prompt_tokens", 0),
                completion_tokens=u.get("completion_tokens", 0),
                total_tokens=u.get("total_tokens", 0),
                cost_usd=u.get("cost_usd", 0.0),
            )

    # Use model from the first result that has one
    model = ""
    for r in all_results:
        if r.get("model"):
            model = r["model"]
            break

    summary: dict = {
        "total_instances": len(all_results),
        "total_comments": total_comments,
        "total_tests_generated": total_tests,
        "total_successes": total_successes,
        "total_attempts": total_attempts,
        "total_errors": total_errors,
        "overall_success_rate": total_successes / total_tests if total_tests > 0 else 0.0,
        "elapsed_seconds": elapsed,
        "repo_summary": list(repo_data.values()),
        "instance_results": [
            {
                "instance_id": r["instance_id"],
                "repo": r["repo"],
                "num_comments": r.get("num_comments", 0),
                "success_rate": r.get("overall_success_rate", 0.0),
                "error": r.get("error"),
            }
            for r in all_results
        ],
    }
    if model:
        summary["model"] = model
    summary["usage"] = total_usage.to_dict()

    summary_file = output_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Batch test generation across all SWE-CARE instances"
    )

    parser.add_argument(
        "--split", nargs="+", default=["dev"],
        choices=["dev", "test"],
        help="Dataset split(s) to process (default: dev)",
    )
    parser.add_argument(
        "--repo", type=str, default=None,
        help="Only process instances for this repo (e.g. 'tobymao/sqlglot')",
    )
    parser.add_argument(
        "--instances-file", type=str, default=None,
        help="File with instance IDs to process (one per line, # comments ignored)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of instances to process",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results_testgen",
        help="Where to save results (default: results_testgen/)",
    )
    parser.add_argument(
        "--repos-dir", type=str, default="repos",
        help="Where to cache repo clones (default: repos/)",
    )
    parser.add_argument(
        "--model", type=str, default=test_generator.DEFAULT_MODEL,
        help=f"LLM model to use (default: {test_generator.DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-attempts", type=int, default=3,
        help="Max generate/execute/feedback attempts per comment (default: 3)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Re-process all instances even if results already exist",
    )
    parser.add_argument(
        "--keep-repos", action="store_true",
        help="Keep cloned repos instead of cleaning up after processing",
    )
    parser.add_argument(
        "--skip-execution", action="store_true",
        help="Generate tests only, don't run them",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--use-docker", action=argparse.BooleanOptionalAction, default=True,
        help="Run tests inside Docker containers (requires pre-built images). "
             "Enabled by default; use --no-use-docker to disable.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    repos_dir = Path(args.repos_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume = not args.no_resume
    workers = args.workers

    # Load instances from all requested splits
    all_instances = []
    for split in args.split:
        instances = dataset_utils.load_instances(split=split, repo=args.repo)
        logger.info("Loaded %d instances from split '%s'", len(instances), split)
        all_instances.extend(instances)

    # Filter to specific instance IDs if a file is provided
    if args.instances_file:
        id_file = Path(args.instances_file)
        if not id_file.exists():
            logger.error("Instances file not found: %s", id_file)
            sys.exit(1)
        wanted_ids = set()
        for line in id_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                wanted_ids.add(line)
        before = len(all_instances)
        all_instances = [i for i in all_instances if i["instance_id"] in wanted_ids]
        logger.info(
            "Filtered to %d instances from %s (%d IDs, %d matched)",
            len(all_instances), id_file, len(wanted_ids), len(all_instances),
        )
        missing = wanted_ids - {i["instance_id"] for i in all_instances}
        if missing:
            logger.warning("Instance IDs not found in dataset: %s", missing)

    if args.limit and len(all_instances) > args.limit:
        all_instances = all_instances[: args.limit]
        logger.info("Limited to %d instances", len(all_instances))

    if not all_instances:
        logger.error("No instances found.")
        sys.exit(1)

    # Group by repo
    by_repo: dict[str, list[dict]] = {}
    for inst in all_instances:
        by_repo.setdefault(inst["repo"], []).append(inst)

    logger.info(
        "Total: %d instances across %d repos (splits: %s)",
        len(all_instances), len(by_repo), ", ".join(args.split),
    )

    # --- Phase 1: Sequential setup (clone source repos, fetch PR commits) ---
    source_repos: dict[str, Path] = {}  # repo name -> source repo path
    all_results: list[dict] = []
    to_process: list[dict] = []

    for repo, repo_instances in by_repo.items():
        logger.info("=== Setup: %s (%d instances) ===", repo, len(repo_instances))

        # Check resumability — load existing results, determine which need processing
        repo_to_process = []
        for inst in repo_instances:
            if resume:
                existing = load_existing_result(output_dir, inst["instance_id"])
                if existing is not None:
                    logger.info("Skipping (already done): %s", inst["instance_id"])
                    all_results.append(existing)
                    continue
            repo_to_process.append(inst)

        if not repo_to_process:
            logger.info("All instances for %s already processed, skipping repo.", repo)
            continue

        logger.info("%d instance(s) to process for %s", len(repo_to_process), repo)

        # In Docker mode, partition instances by image availability.
        # Instances with images don't need a local clone; instances without
        # are skipped (no fallback to local execution).
        if args.use_docker:
            have_image = []
            no_image = []
            for inst in repo_to_process:
                if docker_image_exists(get_docker_image_name(inst["instance_id"])):
                    have_image.append(inst)
                else:
                    no_image.append(inst)

            if no_image:
                for inst in no_image:
                    logger.warning(
                        "[%s] No Docker image, skipping", inst["instance_id"],
                    )
                # These will be handled in _process_one (returns error result)
                to_process.extend(no_image)

            if have_image:
                logger.info(
                    "%d/%d instances for %s have Docker images, skipping repo clone",
                    len(have_image), len(repo_to_process), repo,
                )
                source_repos.setdefault(repo, None)  # no local repo needed
                to_process.extend(have_image)

            if not have_image:
                # No instances with images — nothing needs a clone
                source_repos.setdefault(repo, None)
            continue

        # Clone source repo + fetch PR commits (needed for local execution)
        repo_path = repo_manager.clone_repo(repo, cache_dir=repos_dir)

        # Collect unique PR numbers to avoid redundant fetches
        fetched_prs: set[int] = set()
        for inst in repo_to_process:
            pn = inst.get("pull_number")
            if pn and pn not in fetched_prs:
                repo_manager.fetch_pr_commits(repo_path, pn)
                fetched_prs.add(pn)

        source_repos[repo] = repo_path
        to_process.extend(repo_to_process)

    if not to_process:
        logger.info("All instances already processed.")
        total_elapsed = 0.0
        summary = write_summary(output_dir, all_results, total_elapsed)
        logger.info(
            "Summary: %d instances, %d tests, %d successes (%.1f%%), cost $%.4f",
            summary["total_instances"], summary["total_tests_generated"],
            summary["total_successes"], summary["overall_success_rate"] * 100,
            summary.get("usage", {}).get("cost_usd", 0.0),
        )
        return

    # --- Phase 2: Parallel processing ---
    workdir_root = repos_dir / "workdirs"
    start_time = time.time()
    processed = 0
    lock = threading.Lock()

    logger.info(
        "Processing %d instance(s) with %d worker(s)",
        len(to_process), workers,
    )

    def _process_one(inst: dict) -> dict:
        """Process a single instance in an isolated workdir or Docker container."""
        instance_id = inst["instance_id"]
        repo = inst["repo"]
        source_path = source_repos[repo]
        workdir = None

        # Check if we should use Docker for this instance
        docker_image = None
        if args.use_docker:
            image = get_docker_image_name(instance_id)
            if docker_image_exists(image):
                docker_image = image
                logger.info("[%s] Using Docker image: %s", instance_id, image)
            else:
                logger.warning(
                    "[%s] Docker image not found (%s), skipping instance",
                    instance_id, image,
                )
                return {
                    "instance_id": instance_id,
                    "repo": repo,
                    "model": args.model,
                    "usage": LLMUsage().to_dict(),
                    "num_comments": len(inst.get("reference_review_comments", [])),
                    "error": f"Docker image not found: {image}",
                    "results": [],
                    "overall_success_rate": 0.0,
                }

        try:
            if docker_image:
                # Docker mode: no workdir or venv needed — container has everything
                # including git history for context extraction.
                result = process_instance(
                    instance=inst,
                    repo_path=source_path,  # may be None; unused in Docker mode
                    venv_path=None,
                    output_dir=output_dir,
                    model=args.model,
                    skip_execution=args.skip_execution,
                    max_attempts=args.max_attempts,
                    docker_image=docker_image,
                )
                return result

            # Local mode: create isolated workdir
            workdir = repo_manager.create_instance_workdir(
                source_path, instance_id, workdir_root
            )

            # Detect languages used by this instance's comments
            comments = inst.get("reference_review_comments", [])
            languages = {
                test_generator.detect_language(c["path"])
                or test_generator.DEFAULT_LANGUAGE
                for c in comments
            }

            # Setup execution environments if needed
            venv_path = None
            if not args.skip_execution:
                head = inst["commit_to_review"]["head_commit"]
                repo_manager.checkout_commit(workdir, head)

                if "python" in languages:
                    try:
                        venv_path = repo_manager.setup_venv(workdir)
                    except Exception as e:
                        logger.error(
                            "[%s] Failed to setup Python venv: %s",
                            instance_id, e,
                        )

                if languages & {"javascript", "typescript"}:
                    try:
                        repo_manager.setup_node_env(workdir)
                    except Exception as e:
                        logger.error(
                            "[%s] Failed to setup Node env: %s",
                            instance_id, e,
                        )

            result = process_instance(
                instance=inst,
                repo_path=workdir,
                venv_path=venv_path,
                output_dir=output_dir,
                model=args.model,
                skip_execution=args.skip_execution,
                max_attempts=args.max_attempts,
            )
            return result
        except Exception:
            logger.exception("[%s] Error processing instance", instance_id)
            return {
                "instance_id": instance_id,
                "repo": repo,
                "model": args.model,
                "usage": LLMUsage().to_dict(),
                "num_comments": len(inst.get("reference_review_comments", [])),
                "error": "Processing failed",
                "results": [],
                "overall_success_rate": 0.0,
            }
        finally:
            # Always clean up the per-instance workdir (only exists in local mode)
            if workdir and workdir.exists():
                try:
                    shutil.rmtree(workdir)
                except OSError as e:
                    logger.warning("[%s] Failed to clean up workdir: %s", instance_id, e)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_inst = {
            executor.submit(_process_one, inst): inst for inst in to_process
        }

        for future in as_completed(future_to_inst):
            inst = future_to_inst[future]
            t0 = time.time()
            result = future.result()

            with lock:
                all_results.append(result)
                processed += 1
                count = processed

            elapsed = time.time() - t0
            logger.info(
                "[%d/%d] Completed %s in %.1fs",
                len(all_results), len(all_instances),
                inst["instance_id"], elapsed,
            )

            # Write progressive summary every 10 completions
            if count % 10 == 0:
                with lock:
                    elapsed_so_far = time.time() - start_time
                    summary = write_summary(output_dir, list(all_results), elapsed_so_far)
                logger.info(
                    "Progress: %d/%d instances, %d tests generated, %d successes (%.1f%%), cost $%.4f",
                    summary["total_instances"], len(all_instances),
                    summary["total_tests_generated"], summary["total_successes"],
                    summary["overall_success_rate"] * 100,
                    summary.get("usage", {}).get("cost_usd", 0.0),
                )

    # Cleanup source repos
    if not args.keep_repos:
        for repo in source_repos:
            cleanup_repo(repo, repos_dir)
        # Remove workdirs root if empty
        if workdir_root.exists():
            try:
                workdir_root.rmdir()
            except OSError:
                pass

    # Final summary
    total_elapsed = time.time() - start_time
    summary = write_summary(output_dir, all_results, total_elapsed)
    logger.info(
        "=== DONE === %d instances, %d tests generated, %d successes (%.1f%%), "
        "%d errors, %.0fs elapsed, cost $%.4f",
        summary["total_instances"],
        summary["total_tests_generated"],
        summary["total_successes"],
        summary["overall_success_rate"] * 100,
        summary["total_errors"],
        total_elapsed,
        summary.get("usage", {}).get("cost_usd", 0.0),
    )
    logger.info("Summary: %s", output_dir / "summary.json")


if __name__ == "__main__":
    main()
