import json
from pathlib import Path
p = Path(r"c:\work\Startups\ai-station\llm-data-forge\datasets_new\topics\ai\design_creativity\ai.design_creativity.reasoning.json")
if not p.exists():
    print('ERROR: file not found:', p)
    raise SystemExit(1)
with p.open(encoding='utf-8') as f:
    data = json.load(f)
ids = []

def collect(obj):
    if isinstance(obj, dict):
        if 'id' in obj:
            try:
                ids.append(int(obj['id']))
            except:
                pass
        for v in obj.values():
            collect(v)
    elif isinstance(obj, list):
        for v in obj:
            collect(v)

collect(data)
ids_set = sorted(set(ids))
print('Found unique IDs:', len(ids_set))
missing = [i for i in range(1,1001) if i not in ids_set]
print('Missing count:', len(missing))
if missing:
    print('Missing IDs:')
    print(','.join(map(str, missing)))
else:
    print('No missing IDs')
