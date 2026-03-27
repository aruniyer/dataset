#!/usr/bin/env python3
"""Main CLI entry point for the test generation pipeline.

Generates executable tests from code review comments and validates them
against before/after code versions. Supports multiple languages — the test
language is auto-detected from the reviewed file's extension.

Usage:
  python run_testgen.py --instance-id <id>          # Single instance
  python run_testgen.py --repo tobymao/sqlglot       # All instances for a repo
  python run_testgen.py --split dev --limit 10       # Batch mode

Options:
  --output-dir results/         Where to save results
  --repos-dir repos/            Where to cache repo clones
  --skip-execution              Generate tests only, don't run them
  --model gpt-5.2              LLM model to use
  --max-attempts 3              Max generate→execute→feedback loops per comment
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from execution.container_runtime import DockerContainerSession
from pipeline import dataset_utils, diff_analyzer, repo_manager, test_generator, test_runner
from pipeline.llm_client import LLMUsage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def process_instance(
    instance: dict,
    repo_path: Path,
    venv_path: Path | None,
    output_dir: Path,
    model: str,
    skip_execution: bool,
    max_attempts: int = 3,
    docker_image: str | None = None,
) -> dict:
    """Process a single instance: generate tests and optionally run them.

    For each comment, runs a generate → execute → feedback loop up to
    max_attempts times until the test achieves the expected fail-then-pass
    pattern, or we exhaust attempts.

    Args:
        docker_image: If provided, run tests inside a Docker container using
            this image instead of local execution. The container has all deps
            pre-installed and source code at /workspace.

    Returns:
        Result dict with test generation and execution results.
    """
    instance_id = instance["instance_id"]
    repo = instance["repo"]
    comments = instance["reference_review_comments"]
    head_commit = instance["commit_to_review"]["head_commit"]
    merged_commit = instance["merged_commit"]
    patch_to_review = instance["commit_to_review"]["patch_to_review"]
    merged_patch = instance["merged_patch"]

    logger.info("Processing instance: %s (%d comments)", instance_id, len(comments))

    instance_dir = output_dir / instance_id.replace("/", "__")
    instance_dir.mkdir(parents=True, exist_ok=True)

    comment_results = []
    instance_usage = LLMUsage()

    use_docker = docker_image is not None and not skip_execution
    session = None

    try:
        if use_docker:
            safe_name = instance_id.replace("/", "--").replace("@", "-")
            container_name = f"rb-{safe_name}"
            # Remove any stale container with the same name from a previous run
            rm_result = subprocess.run(["docker", "rm", "-f", container_name],
                                       capture_output=True, text=True)
            if rm_result.returncode == 0 and rm_result.stdout.strip():
                # A container was actually removed — give the daemon time to
                # clean up its filesystem layers to avoid "RWLayer unexpectedly nil"
                time.sleep(2)
            session = DockerContainerSession(docker_image, name=container_name)
            session.start()
            logger.info("Started Docker container for %s (image: %s)", instance_id, docker_image)

        # Create a get_file function — uses container in Docker mode, local repo otherwise
        if use_docker:
            def get_file_fn(commit, filepath):
                result = session.run_command(
                    ["git", "show", f"{commit}:{filepath}"],
                    timeout=30,
                )
                if result.returncode != 0:
                    return ""
                return result.stdout
        else:
            def get_file_fn(commit, filepath):
                return repo_manager.get_file_at_commit(repo_path, commit, filepath)

        for i, comment in enumerate(comments):
            logger.info("--- Comment %d/%d ---", i + 1, len(comments))
            logger.info("  File: %s", comment["path"])
            logger.info("  Text: %s", comment["text"][:100])

            # 1. Detect language from the commented file
            language = (
                test_generator.detect_language(comment["path"])
                or test_generator.DEFAULT_LANGUAGE
            )
            logger.info("  Language: %s", language)

            # 1b. Skip languages we can't execute
            _EXECUTABLE_LANGUAGES = {"python", "javascript", "typescript", "go"}
            if language not in _EXECUTABLE_LANGUAGES:
                logger.info("  Skipping: language '%s' is not executable by test runner", language)
                comment_results.append({
                    "comment_index": i,
                    "comment_text": comment["text"],
                    "comment_type": "unknown",
                    "language": language,
                    "test_file": None,
                    "test_code": None,
                    "error": f"Language '{language}' not supported for test execution",
                    "before_passed": None,
                    "after_passed": None,
                    "success": False,
                    "before_output": "",
                    "after_output": "",
                    "attempts_used": 0,
                    "usage": LLMUsage().to_dict(),
                })
                continue

            # 2. Extract context
            context = diff_analyzer.extract_comment_context(
                comment=comment,
                patch_to_review=patch_to_review,
                merged_patch=merged_patch,
                get_file_fn=get_file_fn,
                head_commit=head_commit,
                merged_commit=merged_commit,
            )

            # 3. Classify comment
            comment_type = test_generator.classify_comment(
                comment["text"], comment.get("diff_hunk", "")
            )
            logger.info("  Type: %s", comment_type)

            # 4. Pre-flight: probe execution environment
            environment_notes = ""
            if use_docker and language == "python":
                module_path = comment["path"].replace("/", ".").replace(".py", "")
                # Check import on the "after" (merged) commit — the correct version
                probe = session.run_command(
                    f"git checkout --force {merged_commit} --quiet 2>/dev/null; "
                    f"python -c \"import {module_path}\" 2>&1",
                    timeout=30,
                )
                if probe.returncode != 0:
                    error_snippet = (probe.stdout + probe.stderr).strip()[-300:]
                    environment_notes = (
                        f"WARNING: `import {module_path}` FAILS in this environment.\n"
                        f"Error: {error_snippet}\n\n"
                        f"You MUST use source file inspection instead of importing the module. "
                        f"Read the file with `open('/workspace/{comment['path']}').read()` "
                        f"and check for the specific code pattern that changed in the diff. "
                        f"Do NOT attempt to import {module_path} or any of its parent packages."
                    )
                    logger.info("  Pre-flight: import %s FAILED — will use source inspection", module_path)
                else:
                    environment_notes = (
                        f"`import {module_path}` works. Prefer functional tests that "
                        f"import and exercise the actual code."
                    )
                    logger.info("  Pre-flight: import %s OK", module_path)

            # 5. Generate initial test
            test_code, gen_usage = test_generator.generate_test(
                context=context,
                repo=repo,
                comment_type=comment_type,
                model=model,
                language=language,
                environment_notes=environment_notes,
            )
            comment_usage = gen_usage

            if test_code is None:
                logger.warning("  Failed to generate test for comment %d", i)
                instance_usage = instance_usage + comment_usage
                comment_results.append({
                    "comment_index": i,
                    "comment_text": comment["text"],
                    "comment_type": comment_type,
                    "language": language,
                    "test_file": None,
                    "test_code": None,
                    "error": "Test generation failed",
                    "before_passed": None,
                    "after_passed": None,
                    "success": False,
                    "before_output": "",
                    "after_output": "",
                    "attempts_used": 0,
                    "usage": comment_usage.to_dict(),
                })
                continue

            # 5. Generate → Execute → Feedback loop
            test_ext = test_generator.get_test_file_ext(language)
            test_filename = f"test_review_comment_{i}{test_ext}"
            test_file = (instance_dir / test_filename).resolve()
            test_file_rel = str(Path(instance_dir.name) / test_filename)
            attempts_used = 0
            result_entry = None

            # Docker can execute all languages; local needs venv for Python
            if use_docker:
                can_execute = True
            else:
                can_execute = not skip_execution and (
                    (language == "python" and venv_path is not None)
                    or language in ("javascript", "typescript", "go")
                )
            effective_attempts = max_attempts if can_execute else 1

            for attempt in range(effective_attempts):
                attempts_used = attempt + 1

                # Write test file
                test_file.write_text(test_code)
                logger.info(
                    "  Attempt %d/%d: %s (%d lines)",
                    attempts_used, effective_attempts,
                    test_file.name, test_code.count("\n") + 1,
                )

                if not can_execute:
                    # No execution — just record the generated test
                    result_entry = {
                        "comment_index": i,
                        "comment_text": comment["text"],
                        "comment_type": comment_type,
                        "language": language,
                        "test_file": test_file_rel,
                        "test_code": test_code,
                        "before_passed": None,
                        "after_passed": None,
                        "success": None,
                        "before_output": "",
                        "after_output": "",
                        "attempts_used": attempts_used,
                        "usage": comment_usage.to_dict(),
                    }
                    break

                # Execute on both versions
                if use_docker:
                    tr = test_runner.run_test_pair_docker(
                        session=session,
                        test_file=test_file,
                        head_commit=head_commit,
                        merged_commit=merged_commit,
                        comment_index=i,
                        language=language,
                    )
                else:
                    tr = test_runner.run_test_pair(
                        repo_path=repo_path,
                        venv_path=venv_path,
                        test_file=test_file,
                        head_commit=head_commit,
                        merged_commit=merged_commit,
                        comment_index=i,
                        language=language,
                    )

                result_entry = {
                    "comment_index": i,
                    "comment_text": comment["text"],
                    "comment_type": comment_type,
                    "language": language,
                    "test_file": test_file_rel,
                    "test_code": test_code,
                    "before_passed": tr.before_passed,
                    "after_passed": tr.after_passed,
                    "success": tr.success,
                    "before_output": tr.before_output,
                    "after_output": tr.after_output,
                    "attempts_used": attempts_used,
                    "usage": comment_usage.to_dict(),
                }

                if tr.success:
                    logger.info("  Test succeeded (before=FAIL, after=PASS) on attempt %d", attempts_used)
                    break

                # Not successful — try to regenerate if we have attempts left
                if attempt < effective_attempts - 1:
                    logger.info(
                        "  Test did not achieve fail-then-pass (before=%s, after=%s), regenerating...",
                        "PASS" if tr.before_passed else "FAIL",
                        "PASS" if tr.after_passed else "FAIL",
                    )
                    test_code, regen_usage = test_generator.regenerate_test(
                        context=context,
                        repo=repo,
                        comment_type=comment_type,
                        previous_test=test_code,
                        before_passed=tr.before_passed,
                        before_output=tr.before_output,
                        after_passed=tr.after_passed,
                        after_output=tr.after_output,
                        model=model,
                        language=language,
                        environment_notes=environment_notes,
                    )
                    comment_usage = comment_usage + regen_usage
                    if test_code is None:
                        logger.warning("  Regeneration failed, stopping retry loop")
                        break
                else:
                    logger.info(
                        "  Exhausted %d attempts (before=%s, after=%s)",
                        effective_attempts,
                        "PASS" if tr.before_passed else "FAIL",
                        "PASS" if tr.after_passed else "FAIL",
                    )

            # Update usage in the final result_entry (may have accumulated regen usage)
            if result_entry is not None:
                result_entry["usage"] = comment_usage.to_dict()
            instance_usage = instance_usage + comment_usage
            comment_results.append(result_entry)

    finally:
        if session is not None:
            session.remove(force=True)

    # Compute overall success rate
    tested = [r for r in comment_results if r["success"] is not None]
    successes = sum(1 for r in tested if r["success"])
    success_rate = successes / len(tested) if tested else 0.0

    result = {
        "instance_id": instance_id,
        "repo": repo,
        "model": model,
        "usage": instance_usage.to_dict(),
        "num_comments": len(comments),
        "results": comment_results,
        "overall_success_rate": success_rate,
    }

    # Save result JSON
    result_file = instance_dir / "result.json"
    result_file.write_text(json.dumps(result, indent=2, default=str))
    logger.info("Result saved to %s (success rate: %.1f%%)", result_file, success_rate * 100)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate and run tests from code review comments"
    )

    # Instance selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--instance-id", type=str, help="Single instance ID to process")
    group.add_argument("--repo", type=str, help="Process all instances for a repo")

    # Filtering
    parser.add_argument("--split", default="dev", choices=["dev", "test"],
                        help="Dataset split (default: dev)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of instances to process")
    parser.add_argument("--difficulty", type=str, default=None,
                        help="Filter by difficulty level")
    parser.add_argument("--max-comments", type=int, default=None,
                        help="Only process instances with at most N comments")

    # Paths
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Where to save results (default: results/)")
    parser.add_argument("--repos-dir", type=str, default="repos",
                        help="Where to cache repo clones (default: repos/)")

    # Execution options
    parser.add_argument("--skip-execution", action="store_true",
                        help="Generate tests only, don't run them")
    parser.add_argument("--model", type=str, default=test_generator.DEFAULT_MODEL,
                        help=f"LLM model to use (default: {test_generator.DEFAULT_MODEL})")
    parser.add_argument("--max-attempts", type=int, default=3,
                        help="Max generate→execute→feedback attempts per comment (default: 3)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    repos_dir = Path(args.repos_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load instances
    if args.instance_id:
        instance = dataset_utils.load_instance(args.instance_id, split=args.split)
        if instance is None:
            logger.error("Instance not found: %s", args.instance_id)
            sys.exit(1)
        instances = [instance]
    else:
        instances = dataset_utils.load_instances(
            split=args.split,
            repo=args.repo,
            difficulty=args.difficulty,
            max_comments=args.max_comments,
            limit=args.limit,
        )

    if not instances:
        logger.error("No instances matched the given filters.")
        sys.exit(1)

    logger.info("Processing %d instance(s)", len(instances))

    # Group instances by repo for efficient cloning
    by_repo: dict[str, list[dict]] = {}
    for inst in instances:
        by_repo.setdefault(inst["repo"], []).append(inst)

    all_results = []

    for repo, repo_instances in by_repo.items():
        logger.info("=== Repo: %s (%d instances) ===", repo, len(repo_instances))

        # Clone repo
        repo_path = repo_manager.clone_repo(repo, cache_dir=repos_dir)

        # Fetch PR commits for all instances (they may not be on any branch)
        for inst in repo_instances:
            pn = inst.get("pull_number")
            if pn:
                repo_manager.fetch_pr_commits(repo_path, pn)

        # Detect languages across all instances for this repo
        all_languages: set[str] = set()
        for inst in repo_instances:
            for c in inst.get("reference_review_comments", []):
                lang = test_generator.detect_language(c["path"])
                all_languages.add(lang or test_generator.DEFAULT_LANGUAGE)

        # Setup execution environments (only needed if running tests)
        venv_path = None
        if not args.skip_execution:
            head = repo_instances[0]["commit_to_review"]["head_commit"]
            repo_manager.checkout_commit(repo_path, head)

            if "python" in all_languages:
                try:
                    venv_path = repo_manager.setup_venv(repo_path)
                except Exception as e:
                    logger.error("Failed to setup Python venv for %s: %s", repo, e)

            if all_languages & {"javascript", "typescript"}:
                try:
                    repo_manager.setup_node_env(repo_path)
                except Exception as e:
                    logger.error("Failed to setup Node env for %s: %s", repo, e)

        for inst in repo_instances:
            t0 = time.time()
            try:
                result = process_instance(
                    instance=inst,
                    repo_path=repo_path,
                    venv_path=venv_path,
                    output_dir=output_dir,
                    model=args.model,
                    skip_execution=args.skip_execution,
                    max_attempts=args.max_attempts,
                )
                all_results.append(result)
            except Exception:
                logger.exception("Error processing %s", inst["instance_id"])
                all_results.append({
                    "instance_id": inst["instance_id"],
                    "repo": repo,
                    "model": args.model,
                    "usage": LLMUsage().to_dict(),
                    "num_comments": len(inst.get("reference_review_comments", [])),
                    "error": "Processing failed",
                    "results": [],
                    "overall_success_rate": 0.0,
                })
            elapsed = time.time() - t0
            logger.info("Completed %s in %.1fs", inst["instance_id"], elapsed)

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

    # Write summary
    summary_file = output_dir / "summary.json"
    summary = {
        "total_instances": len(all_results),
        "total_comments": sum(r.get("num_comments", 0) for r in all_results),
        "total_tests_generated": sum(
            sum(1 for c in r.get("results", []) if c.get("test_code"))
            for r in all_results
        ),
        "total_successes": sum(
            sum(1 for c in r.get("results", []) if c.get("success"))
            for r in all_results
        ),
        "model": args.model,
        "usage": total_usage.to_dict(),
        "instance_results": [
            {
                "instance_id": r["instance_id"],
                "repo": r["repo"],
                "num_comments": r.get("num_comments", 0),
                "success_rate": r.get("overall_success_rate", 0.0),
            }
            for r in all_results
        ],
    }
    total_gen = summary["total_tests_generated"]
    overall_rate = (
        summary["total_successes"] / total_gen if total_gen > 0 else 0.0
    )
    summary["overall_success_rate"] = overall_rate

    summary_file.write_text(json.dumps(summary, indent=2))
    logger.info(
        "=== DONE === %d instances, %d tests generated, %d successes (%.1f%%), cost $%.4f",
        summary["total_instances"],
        total_gen,
        summary["total_successes"],
        overall_rate * 100,
        total_usage.cost_usd,
    )
    logger.info("Summary: %s", summary_file)


if __name__ == "__main__":
    main()
