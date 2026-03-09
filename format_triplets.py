import pandas as pd
import json

df = pd.read_parquet('data/triplets/triplets.parquet')

print("| Original Clean Record (Positive) | Messy Search Query (Anchor) | Corruption Applied |")
print("|----------------------------------|-----------------------------|--------------------|")

# Hand pick some diverse examples
samples = []
# 1. Abbreviation
samples.append(df[df['corruption_types'] == '["abbreviation"]'].iloc[0])
# 2. Typos
samples.append(df[df['corruption_types'] == '["levenshtein_1"]'].iloc[2])
# 3. Double Field Drop
samples.append(df[df['corruption_types'] == '["field_drop_double"]'].iloc[5])
# 4. Domain Swap
samples.append(df[df['corruption_types'] == '["domain_swap"]'].iloc[1])
# 5. Schema Confusion (swapped)
samples.append(df[df['corruption_types'] == '["swapped_attributes"]'].iloc[0] if len(df[df['corruption_types'] == '["swapped_attributes"]']) > 0 else df[df['corruption_types'] == '["case_mutation"]'].iloc[0])


for row in samples:
    pos = str(row['positive_text_pipe']).replace('|', '\\|')
    anch = str(row['anchor_text_pipe']).replace('|', '\\|')
    corr = ", ".join(json.loads(row['corruption_types'])).title().replace('_', ' ')
    print(f"| `{pos}` | `{anch}` | {corr} |")

