# Dataset Design Specification

**Project:** entity-resolution-poc
**Version:** v1
**Status:** Reference document — all scripts implement this spec exactly

---

## 1. Schema Definition

Every profile in the dataset adheres to the following schema. All fields are nullable except `profile_id` and `last_name`.

| Field | Type | Nullable | Constraints | Example |
|-------|------|----------|-------------|---------|
| profile_id | str (UUID4) | No | Globally unique, stable across corruption | `a3f7b2c1-...` |
| first_name | str | Yes | 1–50 chars; alpha + hyphen allowed; no numbers | `Jonathan` |
| last_name | str | No | 1–60 chars; alpha + hyphen + apostrophe; no numbers | `O'Brien` |
| company | str | Yes | 1–120 chars; any printable ASCII | `Acme Corp International` |
| email | str | Yes | Valid RFC-5321-ish format; max 254 chars | `jon.smith@acme.com` |
| country | str | Yes | ISO 3166-1 alpha-3 preferred; stored as display name | `USA` |

### Field Semantics

- **first_name:** Given name. May be a common nickname (Bill, Bob, Liz). May be hyphenated (Mary-Jane). May contain just an initial ("J.") in some real data.
- **last_name:** Family name. May be hyphenated (Smith-Jones). May have prefixes (van der Berg → stored as one string). Required because last_name is the highest-cardinality reliable identifier.
- **company:** Current employer or most recent. May include legal suffix (Inc., LLC) or not. May be abbreviated (IBM vs International Business Machines). Key ambiguity source.
- **email:** Work or personal. Work emails follow `{pattern}@{company_domain}` convention. Personal emails have arbitrary local parts. Email domain is a strong matching signal despite domain_swap corruption.
- **country:** Country of residence or employment. B2B context means this correlates with company headquarters location.

---

## 2. Country and Email Distribution Rationale

### Country Distribution

```
USA: 60%
UK:  10%
India: 10%
Canada: 5%
Germany: 5%
Australia: 3%
France: 3%
Singapore: 2%
Brazil: 2%
```

**Rationale:** This reflects a realistic B2B data platform's customer base. US-headquartered companies dominate B2B data (Salesforce, HubSpot, etc.). India has 10% share due to large tech workforce and outsourcing. UK is the primary non-US English-speaking market. Total = 100%.

**Implementation:** Faker locales are selected proportionally. `en_US` for USA (60% of records), `en_GB` for UK (10%), `en_IN` for India (10%), `en_CA` for Canada (5%), `de_DE` for Germany (5%), `en_AU` for Australia (3%), `fr_FR` for France (3%), `en_SG` for Singapore (2%), `pt_BR` for Brazil (2%).

**Why this matters for entity resolution:** US names are over-represented, so "John Smith" ambiguity is real — our hard negatives will include many "John Smith @ DifferentCompany" records. The model must learn to use company + email to disambiguate common names.

### Email Distribution

```
Work emails: 70%
Personal emails: 30%
```

**Work email patterns (sampled uniformly among 70%):**
- `{first}.{last}@{company_domain}` — most common (40%)
- `{first_initial}{last}@{company_domain}` — second most common (30%)
- `{first}_{last}@{company_domain}` — less common (20%)
- `{first}@{company_domain}` — first name only (10%)

**Personal email domains (sampled from):**
- gmail.com (50% of personal)
- yahoo.com (15%)
- hotmail.com (10%)
- outlook.com (10%)
- icloud.com (8%)
- protonmail.com (7%)

**Rationale:** 70% work email is realistic for B2B data. Personal emails are harder to match because they break the company-email correlation assumptions.

---

## 3. Corruption Types — Detailed Specification

### Corruption Implementation Rules

1. Corruptions are applied to a clean base profile to produce the anchor record.
2. The positive is always the original clean base profile.
3. Multiple corruptions can stack (applied sequentially, each with its independent probability).
4. Profile IDs are never corrupted.
5. Fields dropped by `field_drop_*` are set to `None` in the data struct; serialized as `[MISSING]`.

### Corruption Type 1: Abbreviation

**Probability:** 0.15
**Applies to:** first_name
**Effect:** Replace full first name with common short form using a lookup table.

**Lookup table sample (stored in `src/data/name_abbreviations.json`):**
```json
{
  "Jonathan": ["Jon", "Jonny", "Jono"],
  "Robert": ["Rob", "Bob", "Robby"],
  "William": ["Will", "Bill", "Willie"],
  "Elizabeth": ["Liz", "Beth", "Eliza", "Bette"],
  "Michael": ["Mike", "Mikey", "Mick"],
  "Katherine": ["Kate", "Kathy", "Kat"],
  "Jennifer": ["Jen", "Jenny"],
  "Christopher": ["Chris"],
  "Patricia": ["Pat", "Patty", "Trish"],
  "Anthony": ["Tony", "Ant"]
}
```

**Fallback:** If name not in lookup table, apply truncation to first 3 characters instead.

**Example:**
```
Before: first_name="Jonathan", last_name="Smith"
After:  first_name="Jon", last_name="Smith"
```

### Corruption Type 2: Truncation

**Probability:** 0.10
**Applies to:** last_name, company
**Effect:** Truncate to a random length between `min_chars` (3) and `len-1`.

**Algorithm:**
```python
def truncate(value: str, min_chars: int = 3) -> str:
    if len(value) <= min_chars:
        return value  # Can't truncate further
    keep = random.randint(min_chars, len(value) - 1)
    return value[:keep]
```

**Examples:**
```
"Schmidt" → "Schm" (keep=4)
"Google Inc" → "Goo" (keep=3)
"International" → "Intern" (keep=6)
```

### Corruption Type 3: Levenshtein-1 (Single Edit)

**Probability:** 0.20
**Applies to:** first_name, last_name
**Effect:** Apply one random edit operation: substitution, insertion, deletion, or transposition.

**Algorithm:**
```python
def levenshtein_1(value: str) -> str:
    ops = ['substitute', 'insert', 'delete', 'transpose']
    op = random.choice(ops)
    if op == 'substitute':
        pos = random.randint(0, len(value)-1)
        char = random.choice('abcdefghijklmnopqrstuvwxyz')
        return value[:pos] + char + value[pos+1:]
    elif op == 'insert':
        pos = random.randint(0, len(value))
        char = random.choice('abcdefghijklmnopqrstuvwxyz')
        return value[:pos] + char + value[pos:]
    elif op == 'delete':
        if len(value) <= 2: return value
        pos = random.randint(0, len(value)-1)
        return value[:pos] + value[pos+1:]
    elif op == 'transpose':
        if len(value) <= 1: return value
        pos = random.randint(0, len(value)-2)
        s = list(value)
        s[pos], s[pos+1] = s[pos+1], s[pos]
        return ''.join(s)
```

**Examples:**
```
"Smith" → "Smyth"  (substitute: i→y)
"Smith" → "Smiht"  (transpose: t↔h)
"Smith" → "Smit"   (delete: h)
"John"  → "Jonh"   (transpose: h↔n)
```

### Corruption Type 4: Levenshtein-2 (Double Edit)

**Probability:** 0.10
**Applies to:** first_name, last_name, company
**Effect:** Apply levenshtein_1 twice sequentially (not necessarily same operation).

**Examples:**
```
"Jonathan" → "Jonathon" → "Jonathen"  (two substitutions)
"Schneider" → "Shneider" → "Shneidur" (two substitutions)
```

### Corruption Type 5: Field Drop (Single)

**Probability:** 0.20
**Applies to:** any one field from [first_name, last_name, company, email, country]
**Effect:** Set one randomly chosen field to None. Serialized as `[MISSING]`.

**Weighting:**
- first_name: 0.25 (most commonly missing)
- email: 0.25 (second most commonly missing)
- company: 0.20
- country: 0.20
- last_name: 0.10 (least likely to be missing — required field in many forms)

**Example:**
```
Before: Jonathan | Smith | Google | jon.smith@google.com | USA
After:  Jonathan | Smith | [MISSING] | jon.smith@google.com | USA
# company field dropped
```

### Corruption Type 6: Field Drop (Double)

**Probability:** 0.10
**Applies to:** any two distinct fields
**Effect:** Set two randomly chosen fields to None.

**Field pair weights (from most to least likely to both be missing):**
- (email, company): 0.35
- (first_name, country): 0.20
- (first_name, email): 0.20
- (company, country): 0.15
- (email, country): 0.10

**Example:**
```
Before: Jonathan | Smith | Google | jon.smith@google.com | USA
After:  [MISSING] | Smith | [MISSING] | jon.smith@google.com | USA
# first_name and company dropped
```

### Corruption Type 7: Domain Swap

**Probability:** 0.10
**Applies to:** email (only profiles with valid email)
**Effect:** Replace email domain with a different domain. Three sub-types:

- **work-to-personal (60%):** `j.smith@google.com` → `j.smith@gmail.com`
- **work-to-work (20%):** `j.smith@google.com` → `j.smith@googlemail.com`
- **personal-to-different-personal (20%):** `j.smith@gmail.com` → `j.smith@yahoo.com`

**Example:**
```
Before: email="jon.smith@google.com"
After:  email="jon.smith@gmail.com"
# Domain swapped, local part unchanged — same person, different email domain
```

### Corruption Type 8: Company Abbreviation

**Probability:** 0.05
**Applies to:** company
**Effect:** Replace company name with an abbreviation or common short form.

**Abbreviation strategies:**
- Acronym from initials: "International Business Machines" → "IBM"
- Common short name: "General Electric" → "GE", "Goldman Sachs" → "GS"
- Truncation: "Google Incorporated" → "Goog"
- Strip suffix: "Google Inc." → "Google"

**Lookup table sample:**
```json
{
  "International Business Machines": "IBM",
  "General Electric": "GE",
  "Goldman Sachs": "GS",
  "JPMorgan Chase": "JPM",
  "Procter & Gamble": "P&G",
  "Johnson & Johnson": "J&J",
  "Hewlett-Packard": "HP",
  "Microsoft Corporation": "MSFT"
}
```

**Fallback:** Strip legal suffix (Inc., LLC, Ltd., Corp., Co., Limited, Corporation).

### Corruption Type 9: Case Mutation

**Probability:** 0.05
**Applies to:** first_name, last_name, company
**Effect:** Apply one of three case transformations.

- **lowercase (40%):** "Jonathan Smith" → "jonathan smith"
- **uppercase (40%):** "Jonathan Smith" → "JONATHAN SMITH"
- **mixed (20%):** "Jonathan Smith" → "jOnAtHaN sMiTh" (random capitalization per character)

**Only applied to the selected field,** not all fields simultaneously.

### Corruption Type 10: Nickname Substitution

**Probability:** 0.05
**Applies to:** first_name
**Effect:** Apply a common nickname abbreviation from the lookup table.

**This is the curated version of Type 1.** The difference: abbreviation applies common short forms (Jon, Rob, Bill), nickname applies informal nicknames that are less obvious (Bobby, Robby, Liz, Bette, Mickey).

**Nickname lookup table sample:**
```json
{
  "Robert":    ["Bobby", "Robby", "Rob", "Bob"],
  "Elizabeth": ["Liz", "Lizzie", "Beth", "Bette", "Betty"],
  "Michael":   ["Mickey", "Mikey", "Micky"],
  "William":   ["Billy", "Will", "Willy"],
  "Margaret":  ["Maggie", "Peggy", "Meg"],
  "Richard":   ["Rick", "Ricky", "Dick"],
  "James":     ["Jimmy", "Jim", "Jamie"]
}
```

---

## 4. Quality Pipeline — 7 Steps with Assertions

All generated records pass through this pipeline before being written to disk. Any record failing an assertion is discarded (logged, not raised).

### Step 1: Deduplication

**Goal:** Ensure no two profiles are the same person.

**Exact deduplication:**
```python
assert df['email'].notna() implies df['email'].is_unique()
```

**Near-deduplication (Levenshtein):**
- Group by `last_name[:3]` (blocking key for efficiency)
- Within each block, flag pairs where:
  - `levenshtein(first_name_A, first_name_B) < 2` AND
  - `levenshtein(last_name_A, last_name_B) < 2` AND
  - `company_A == company_B`
- Keep one from each near-duplicate pair (deterministic: keep lower profile_id).

### Step 2: Email Validation

**Assertions:**
```python
assert re.match(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$', email)
assert len(email) <= 254
assert '@' in email and email.count('@') == 1
assert not email.startswith('.') and not email.endswith('.')
assert '..' not in email
```

### Step 3: Name Validation

**Assertions:**
```python
# first_name (nullable):
if first_name is not None:
    assert len(first_name) >= 2
    assert len(first_name) <= 50
    assert re.match(r"^[A-Za-z\-' ]+$", first_name)
    assert not first_name.isdigit()

# last_name (required):
assert last_name is not None
assert len(last_name) >= 2
assert len(last_name) <= 60
assert re.match(r"^[A-Za-z\-' ]+$", last_name)
```

### Step 4: Company Normalization

**Processing:**
1. Strip leading/trailing whitespace.
2. Normalize multiple spaces to single space.
3. Create `company_canonical` by stripping legal suffixes for dedup purposes only (the original `company` field is preserved).

```python
LEGAL_SUFFIXES = ['Inc.', 'Inc', 'LLC', 'Ltd.', 'Ltd', 'Corp.', 'Corp',
                   'Co.', 'Co', 'Limited', 'Corporation', 'Group', 'Holdings']

def canonical_company(name: str) -> str:
    for suffix in LEGAL_SUFFIXES:
        name = re.sub(rf'\b{re.escape(suffix)}\s*$', '', name, flags=re.IGNORECASE)
    return name.strip()
```

**Assertion:**
```python
assert len(company) >= 1 if company is not None else True
assert len(company) <= 120 if company is not None else True
```

### Step 5: Country Canonicalization

**Processing:**
1. Map all country inputs to canonical display name from an alias dict.
2. Filter to only countries in `country_distribution` config.

**Alias dict sample:**
```python
COUNTRY_ALIASES = {
    "US": "USA", "United States": "USA", "United States of America": "USA",
    "GB": "UK", "Great Britain": "UK", "England": "UK",
    "IN": "India", "IND": "India",
    "CA": "Canada", "CAN": "Canada",
    "DE": "Germany", "DEU": "Germany", "Deutschland": "Germany",
}
```

**Assertion:**
```python
assert country in ALLOWED_COUNTRIES or country is None
```

### Step 6: Cross-Field Consistency Check

**Goal:** Ensure work emails are plausible for the assigned company.

**For work email profiles:**
```python
company_slug = canonical_company(company).lower().replace(' ', '').replace('-', '')[:8]
email_domain = email.split('@')[1].replace('.com', '').replace('.io', '')
overlap = levenshtein_distance(company_slug, email_domain)
assert overlap <= 3 or email_domain in KNOWN_FREEMAIL_DOMAINS
# Allow up to 3 edit distance between company slug and email domain
# (e.g., "google" vs "gmail" = distance 3 → allowed)
```

### Step 7: Triplet Validity Assertions

Applied after triplet construction, before writing triplets to disk.

```python
def assert_triplet_valid(triplet: Triplet, profiles: dict) -> None:
    anchor = profiles[triplet.anchor_id]
    positive = profiles[triplet.positive_id]
    negative = profiles[triplet.negative_id]

    # IDs must be distinct
    assert triplet.anchor_id != triplet.positive_id
    assert triplet.anchor_id != triplet.negative_id
    assert triplet.positive_id != triplet.negative_id

    # Positive IS the true match (same canonical identity)
    assert positive.profile_id == triplet.positive_id

    # Negative is NOT the true match
    assert negative.profile_id != triplet.positive_id

    # Corruptions were actually applied
    if 'field_drop_single' in triplet.corruption_types_applied:
        missing_fields = [f for f in FIELDS if getattr(anchor_record, f) is None]
        assert len(missing_fields) >= 1

    # Text strings are not identical (corruption must change something)
    assert triplet.anchor_text != triplet.positive_text

    # Anchor text is not empty
    assert len(triplet.anchor_text.strip()) > 0
```

---

## 5. Triplet Format

Each training sample is a triplet with the following schema:

```python
class Triplet(BaseModel):
    anchor_id: str           # UUID of anchor profile (base identity)
    positive_id: str         # UUID of positive (= anchor_id — same person, clean record)
    negative_id: str         # UUID of negative (different person)
    anchor_text: str         # Serialized corrupted record (model input)
    positive_text: str       # Serialized clean record (true match)
    negative_text: str       # Serialized negative record
    corruption_types_applied: list[str]  # e.g., ["levenshtein_1", "field_drop_single"]
    corrupted_fields: list[str]          # e.g., ["last_name", "email"]
    negative_source: str     # "same_company_prefix" | "bm25_mined" | "random"
    round: int               # 1 = easy negatives, 2 = hard negatives
    dataset_version: str     # "v1"
```

**Example triplet (JSON):**
```json
{
  "anchor_id": "a3f7b2c1-4a12-4b3c-8d2e-1234abcd5678",
  "positive_id": "a3f7b2c1-4a12-4b3c-8d2e-1234abcd5678",
  "negative_id": "b9c1d4e5-7f23-4a5b-9c6d-2345bcde6789",
  "anchor_text": "Jon Smyth | Googl | jsmith@gmail.com | USA",
  "positive_text": "Jonathan Smith | Google Inc | jonathan.smith@google.com | USA",
  "negative_text": "Jane Smith | Google LLC | jane.smith@google.com | USA",
  "corruption_types_applied": ["abbreviation", "levenshtein_1", "domain_swap"],
  "corrupted_fields": ["first_name", "last_name", "email"],
  "negative_source": "same_company_prefix",
  "round": 1,
  "dataset_version": "v1"
}
```

**Triplet sizes:**
- Round 1: ~200K anchors × 3 negatives = ~600K triplets
- Round 2: ~200K anchors × 5 hard negatives = ~1M triplets
- Total training data: ~1.6M triplets

---

## 6. Evaluation Set Design

### Design Principles

1. Each eval query has exactly ONE correct match in the 1M-record index.
2. The correct match is always the clean (uncorrupted) version of the profile.
3. Eval profiles are disjoint from both training triplet anchors and the index (where possible — for 1M index scale this isn't fully possible, so the true match IS in the index).
4. Each bucket has ~1667 queries for 10K total (last bucket gets 1665 to hit round number).

### Per-Bucket Query Construction

**Bucket 1: pristine (n=1667)**
- Query = clean serialized record (no corruption)
- True match = same record (by ID) in index
- Tests: baseline retrieval quality; minimum viable performance threshold
- A model that can't get high recall here is broken

**Bucket 2: missing_firstname (n=1667)**
- Query = `[MISSING] | Smith | Google | j.smith@google.com | USA`
- True match = `Jonathan Smith | Google | jonathan.smith@google.com | USA`
- Tests: Can the model find the match using only last_name + company + email + country?
- Failure mode: Model over-weights first_name; without it, rank drops significantly

**Bucket 3: missing_email_company (n=1667)**
- Query = `Jonathan | Smith | [MISSING] | [MISSING] | USA`
- True match = `Jonathan Smith | Google | jonathan.smith@google.com | USA`
- Tests: Hardest partial record — only name + country. Can model do name-only search?
- Failure mode: Too many "Jonathan Smith" in USA (common name ambiguity)
- BM25 expected Recall@1 ≈ 0.10-0.20 for this bucket

**Bucket 4: typo_name (n=1667)**
- Query has Levenshtein-1 or Levenshtein-2 applied to first_name or last_name
- Mix: 50% Lev-1, 50% Lev-2
- Tests: Typo robustness — the #1 real-world corruption in manual data entry
- Example: "Smyth | Google | j.smith@google.com | USA" → still finds Jonathan Smith

**Bucket 5: domain_mismatch (n=1667)**
- Query email domain is swapped (work→personal or differently-hosted)
- Example: `Jonathan Smith | Google | jonathan.smith@gmail.com | USA`
- Tests: Can model match when email domain is wrong but local part is right?
- BM25 failure mode: "jonathan.smith" token matches, but "gmail" vs "google" pulls toward wrong records

**Bucket 6: swapped_attributes (n=1665)**
- Two field values are swapped in the query
- Example: first_name and company swapped: `Google | Smith | Jonathan | j.smith@google.com | USA`
- Tests: Schema confusion robustness (rare but occurs in ETL bugs)
- This is deliberately the hardest bucket — even humans might fail

### Target Metrics Per Bucket

| Bucket | BM25 R@1 (est.) | Target R@1 (finetuned) |
|--------|----------------|------------------------|
| pristine | ~0.85 | ≥ 0.87 |
| missing_firstname | ~0.65 | ≥ 0.80 |
| missing_email_company | ~0.12 | ≥ 0.65 |
| typo_name | ~0.35 | ≥ 0.78 |
| domain_mismatch | ~0.55 | ≥ 0.82 |
| swapped_attributes | ~0.10 | ≥ 0.50 |

---

## 7. File Formats

### profiles.parquet
- **Path:** `data/raw/profiles.parquet`
- **Schema:** profile_id (str), first_name (str?), last_name (str), company (str?), email (str?), country (str?)
- **Rows:** 1,200,000
- **Encoding:** Snappy compression
- **Partitioning:** None (single file at this scale)

### index_profiles.parquet
- **Path:** `data/processed/index_profiles.parquet`
- **Schema:** profile_id + pipe_text + kv_text (pre-serialized strings)
- **Rows:** 1,000,000
- **Notes:** Pre-serialized for fast eval loading; both formats stored to avoid re-serializing per eval run

### triplets_round1.parquet
- **Path:** `data/triplets/triplets_round1.parquet`
- **Schema:** Full Triplet schema (see Section 5)
- **Rows:** ~600,000
- **Notes:** Easy negatives (same_company_prefix)

### triplets_round2.parquet
- **Path:** `data/triplets/triplets_round2.parquet`
- **Schema:** Full Triplet schema
- **Rows:** ~1,000,000
- **Notes:** Hard negatives (bm25_mined)

### eval_{bucket}.parquet
- **Path:** `data/eval/eval_{bucket}.parquet`
- **Schema:** query_id, query_text_pipe, query_text_kv, true_match_id, true_match_text_pipe, corruption_types_applied, bucket
- **Rows:** ~1,667 per bucket

### eval_{bucket}.json
- **Path:** `data/eval/eval_{bucket}.json`
- **Format:** JSON Lines (one query per line)
- **Contents:** Same as parquet but human-readable for debugging
- **Example line:**
```json
{"query_id": "q_001234", "bucket": "typo_name", "query_text_pipe": "Jon Smyth | Google | j.smith@google.com | USA", "true_match_id": "a3f7b2c1-...", "corruption_types_applied": ["levenshtein_1"]}
```
