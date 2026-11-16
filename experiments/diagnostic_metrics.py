"""Objective diagnostic criteria, separate from the benchmark's canonical score."""
from collections import defaultdict
import json
from pathlib import Path


def repeated_span(tokens, width=30, occurrences=3):
    """Does any exact span appear at least `occurrences` times without overlap?"""
    seen = {}
    for i in range(max(0, len(tokens) - width + 1)):
        key = tuple(tokens[i:i + width])
        previous, count = seen.get(key, (-width, 0))
        if i >= previous + width:
            count += 1
            if count >= occurrences:
                return True
            seen[key] = i, count
    return False


def measure(folder):
    folder = Path(folder)
    raw = [json.loads(x) for x in (folder / 'artifacts/output/original.jsonl').read_text().split('\n') if x.strip()]
    graded = json.loads((folder / 'artifacts/output/original_per_problem.json').read_text())
    grades = {(r['id'], r['sample']): r for r in graded}
    rows = []
    for row in raw:
        grade = grades[row['id'], row['sample']]
        complete = row['finish_reason'] == 'stop' and grade['pred'] is not None
        tokens = row['token_ids']
        grams = [tuple(tokens[i:i+20]) for i in range(max(0, len(tokens)-19))]
        rows.append(dict(id=row['id'], source=row.get('source', 'unknown'), tokens=len(tokens),
            finish_reason=row['finish_reason'], canonical_correct=grade['correct'],
            diagnostic_complete=complete, diagnostic_correct=complete and grade['correct'],
            repeated_30token_span_three_nonoverlapping=repeated_span(tokens),
            duplicate_20gram_fraction=1-len(set(grams))/len(grams) if grams else 0))
    groups = defaultdict(list)
    for row in rows:
        groups[row['source']].append(row)
    fields = ['canonical_correct', 'diagnostic_complete', 'diagnostic_correct',
              'repeated_30token_span_three_nonoverlapping']
    summary = dict(problems=len(rows), **{key: sum(r[key] for r in rows) for key in fields})
    summary['by_source'] = {source: dict(problems=len(items), **{key: sum(r[key] for r in items) for key in fields})
                            for source, items in groups.items()}
    result = dict(criteria='Completion: stop with extracted boxed answer; repetition: an exact 30-token span at >=3 nonoverlapping positions',
                  summary=summary, per_problem=rows)
    (folder / 'prospective_metrics.json').write_text(json.dumps(result, indent=2))
    return result
