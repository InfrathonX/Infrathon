"""Content uniqueness, including token-equivalent messages and mixed batches."""
import json
from pathlib import Path
import pytest
import prepare_prompts as prepare
from conftest import ROOT, private_data

class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(messages[0]['content'].strip().encode()) + [int(kwargs['enable_thinking'])]

def test_allocator_resolves_rendered_tokens():
    allocator = prepare.UniquePrompts(Tokenizer())
    rows = [{'prompt_id': str(i), 'messages': [{'content': s}]} for i, s in enumerate(['a', ' a ', 'b'])]
    assert [r['prompt_id'] for r in allocator.take(rows, 2, False)] == ['0', '2']
    with pytest.raises(ValueError, match='unique'):
        allocator.take(rows, 1, False)

def test_shipped_content_disjoint():
    # Exact message equality is a sufficient witness before token audit at generation.
    seen = set()
    public = ROOT / 'images/work/workdir/spec_lab/public_prompts'
    private = private_data()
    for split, directory in [('public', public), ('anchor', private/'anchor'), ('final', private/'final'), ('warmup', private/'warmup')]:
        for path in sorted(directory.glob('*.jsonl')):
            thinking = path.stem == 'think_bs1'
            for line in path.read_text().splitlines():
                row = json.loads(line)
                key = (json.dumps(row['messages'], sort_keys=True), thinking)
                assert key not in seen, (split, path.stem, row['prompt_id'])
                seen.add(key)
