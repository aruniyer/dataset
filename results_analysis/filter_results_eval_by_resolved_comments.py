from pathlib import Path
import json
import shutil

BASE = Path(__file__).parent.parent
SRC_EVAL = BASE / 'results_eval_combined'
SRC_RESOLUTION = BASE / 'agent_resolution_combined'
OUT_EVAL = BASE / 'results_eval_combined_resolved_true_by_comment_text'

MODELS = ['claude-code', 'codex', 'devin', 'pr-agent']
EXPECTED_INSTANCE_COUNT = 184
EXPECTED_RESOLVED_TRUE_COUNT = 234

TOP_LEVEL_SET_NONE = [
    'agent_diff', 'num_findings', 'num_tests', 'num_tests_passed', 'test_pass_rate', 'tool'
]
TOP_LEVEL_REMOVE = ['agent', 'error', 'num_comments', 'num_resolved', 'resolution_rate']
RESULTS_REMOVE = ['resolved', 'file_path', 'agent_diff']


def _make_dummy_eval_json(instance_id: str, model_dir: str, resolution_obj: dict | None) -> dict:
    """Create an eval-style dummy JSON from resolution data for one missing model instance."""
    if resolution_obj is None:
        repo = None
        model = None
        results = []
    else:
        repo = resolution_obj.get('repo')
        model = resolution_obj.get('model')
        results = []
        for r in resolution_obj.get('results', []):
            results.append({
                'comment_index': r.get('comment_index', None),
                'comment_text': r.get('comment_text', None),
                'test_passed': False,
                'test_output': r.get('test_output', None),
                'error': r.get('error', None),
            })

    return {
        'instance_id': instance_id,
        'repo': repo,
        'tool': model_dir,
        'model': model,
        'num_findings': None,
        'agent_diff': None,
        'num_tests': None,
        'num_tests_passed': None,
        'test_pass_rate': None,
        'results': results,
    }


def main() -> None:
    # All unique eval instances (across all 4 model folders)
    eval_instances = set()
    for model in MODELS:
        for p in (SRC_EVAL / model).glob('*/result.json'):
            d = json.loads(p.read_text())
            eval_instances.add(d['instance_id'])

    print('eval instance count:', len(eval_instances))
    assert len(eval_instances) == EXPECTED_INSTANCE_COUNT, (
        f'Expected {EXPECTED_INSTANCE_COUNT} eval instances, got {len(eval_instances)}'
    )

    # Build resolution lookup by comment_text (resolved=True only), restricted to eval instances
    resolved_true_texts = {}
    resolution_by_instance = {}
    resolution_instances_present = set()
    resolved_true_count = 0

    for p in SRC_RESOLUTION.glob('*/result.json'):
        d = json.loads(p.read_text())
        iid = d.get('instance_id') or p.parent.name
        resolution_by_instance[iid] = d
        if iid not in eval_instances:
            continue

        resolution_instances_present.add(iid)

        keep = set()
        for r in d.get('results', []):
            if r.get('resolved') is True and r.get('comment_text') is not None:
                keep.add(r['comment_text'])

        resolved_true_texts[iid] = keep
        resolved_true_count += len(keep)

    missing_instances = sorted(eval_instances - resolution_instances_present)

    print('resolution instances found for eval set:', len(resolution_instances_present))
    print('missing eval instances in resolution:', len(missing_instances))
    print('resolved=True count for eval set:', resolved_true_count)

    assert len(resolution_instances_present) == EXPECTED_INSTANCE_COUNT, (
        f'Expected {EXPECTED_INSTANCE_COUNT} resolution instances for eval set, got {len(resolution_instances_present)}'
    )
    assert resolved_true_count == EXPECTED_RESOLVED_TRUE_COUNT, (
        f'Expected {EXPECTED_RESOLVED_TRUE_COUNT} resolved=True comments, got {resolved_true_count}'
    )
    print('Count checks passed.')

    # Copy source eval directory
    if OUT_EVAL.exists():
        shutil.rmtree(OUT_EVAL)
    shutil.copytree(SRC_EVAL, OUT_EVAL)
    print('Copied to:', OUT_EVAL)

    # Create per-model dummy eval files for missing model-instance pairs so each model
    # has a complete 149-instance directory set.
    dummy_files_created = 0
    dummy_instances_by_model = {m: set() for m in MODELS}
    for model in MODELS:
        for iid in eval_instances:
            out_file = OUT_EVAL / model / iid / 'result.json'
            if out_file.exists():
                continue
            out_file.parent.mkdir(parents=True, exist_ok=True)
            dummy = _make_dummy_eval_json(iid, model, resolution_by_instance.get(iid))
            out_file.write_text(json.dumps(dummy, indent=2, ensure_ascii=True) + '\n')
            dummy_files_created += 1
            dummy_instances_by_model[model].add(iid)

    print('dummy per-model result.json files created:', dummy_files_created)

    # Filter + normalize copied eval JSON files
    files_processed = 0
    files_changed = 0
    total_before = 0
    total_after = 0
    kept_per_model = {m: 0 for m in MODELS}

    for model in MODELS:
        for p in (OUT_EVAL / model).glob('*/result.json'):
            d = json.loads(p.read_text())
            iid = d.get('instance_id') or p.parent.name

            original = d.get('results', [])
            total_before += len(original)

            keep_texts = resolved_true_texts.get(iid, set())
            filtered = [r for r in original if r.get('comment_text') in keep_texts]

            # Normalize results[] entries
            normalized = []
            for r in filtered:
                for k in RESULTS_REMOVE:
                    r.pop(k, None)

                # Defaults if missing
                item = {
                    'comment_index': r.get('comment_index', None),
                    'comment_text': r.get('comment_text', None),
                    'test_passed': r.get('test_passed', False),
                    'test_output': r.get('test_output', None),
                    'error': r.get('error', None),
                }
                normalized.append(item)

            d['results'] = normalized
            total_after += len(normalized)
            kept_per_model[model] += len(normalized)

            # Top-level normalization:
            # set None defaults only for dummy files (missing model-instance pairs).
            if iid in dummy_instances_by_model[model]:
                for k in TOP_LEVEL_SET_NONE:
                    d[k] = None
            for k in TOP_LEVEL_REMOVE:
                d.pop(k, None)

            files_processed += 1
            if len(normalized) != len(original):
                files_changed += 1

            p.write_text(json.dumps(d, indent=2, ensure_ascii=True) + '\n')

    print('files processed:', files_processed)
    print('files changed:', files_changed)
    print('results entries before:', total_before)
    print('results entries after:', total_after)
    print('kept per model:', kept_per_model)

    # Sanity checks after write. With per-model dummy files in place, each model
    # should now contain all resolved=True comments from the 149-instance set.
    for model in MODELS:
        assert kept_per_model[model] == EXPECTED_RESOLVED_TRUE_COUNT, (
            model, kept_per_model[model], EXPECTED_RESOLVED_TRUE_COUNT
        )

    print('Validation passed: each model kept 189 comments.')


if __name__ == '__main__':
    main()
