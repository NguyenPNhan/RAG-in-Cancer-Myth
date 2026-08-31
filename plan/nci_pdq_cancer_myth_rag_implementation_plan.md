# NCI PDQ Cancer-Myth RAG: Detailed Implementation Plan

## 1. Goal

Build a local, evidence-grounded RAG system that answers questions from
`data/cancermyth_screening_dataset.json` using only the NCI PDQ documents in:

- `data/nci_pdq/cancers/`
- `data/nci_pdq/general_topics/`

The system must do more than retrieve a generally relevant cancer page. It must:

1. identify medically important claims or assumptions in the question;
2. retrieve passages that can verify those claims;
3. decide whether the evidence supports, contradicts, qualifies, or cannot resolve each claim;
4. correct a misleading premise without overcorrecting a reasonable premise;
5. answer the user's underlying question in plain language;
6. attach citations that point to the exact NCI page and section used; and
7. abstain or qualify the response when the local PDQ corpus is insufficient.

This is a research and information system, not a diagnostic or personalized treatment system.

## 2. What is in the repository now

The following inventory was measured from the repository on 2026-09-01.

### Benchmark dataset

`data/cancermyth_screening_dataset.json` contains:

- 735 rows;
- 585 rows with `split = cancer_myth` and `correct_answer = true`;
- 150 rows with `split = cancer_myth_nfp` and `correct_answer = false`;
- 126 distinct values in `cancer`;
- unique IDs from 1 through 735;
- no blank questions; and
- one duplicated question with conflicting labels: IDs 329 and 593.

The benchmark schema is:

```json
{
  "id": 3,
  "split": "cancer_myth",
  "cancer": "Bladder cancer",
  "question": "My father was recently diagnosed ...",
  "correct_answer": true
}
```

The file does not contain a reference answer, the false premise as a separate field, or gold evidence. Therefore it can directly score only a binary prediction once the meaning of `correct_answer` has been confirmed. It cannot by itself establish that a generated explanation is medically correct or properly cited.

### NCI PDQ source corpus

The two requested source folders currently contain:

- 224 JSON files: 166 cancer files and 58 general-topic files;
- 378 page records;
- 223 health-professional pages and 155 patient pages;
- 16,385 sections in total;
- 13,962 non-boilerplate, non-empty sections; and
- approximately 24.9 million characters of non-boilerplate text.

The page topics are:

| Topic | Pages |
|---|---:|
| Adult treatment | 134 |
| Pediatric treatment | 101 |
| Integrative therapies | 48 |
| Supportive care | 37 |
| Screening | 27 |
| Prevention | 24 |
| Genetics | 7 |

The section-size distribution shows why section-aware token splitting is required. The median non-boilerplate section is 837 characters, the 90th percentile is 4,051, and the largest is 130,619.

### Data issues to resolve before indexing

There are two known integrity problems.

1. `data/nci_pdq/all_pages.json` contains 381 pages, but the requested grouped folders contain 378. The three pages absent from the grouped folders are:

   - Cognitive Impairment in Adults With Cancer;
   - Metastatic Squamous Neck Cancer With Occult Primary Treatment; and
   - Myelodysplastic/Myeloproliferative Neoplasms Treatment.

2. `manifest.json` reports 168 cancer files and 59 general-topic files, while 166 and 58 are present. The three filenames missing on disk correspond to the pages above.

Do not silently index an incomplete corpus. Restore or regenerate those three grouped files, then require the manifest, grouped folders, and `all_pages.json` to agree. If the project intentionally excludes a page, record that exclusion and its reason in a machine-readable report.

All JSON must be read explicitly as UTF-8. The NCI titles contain characters such as `PDQ®` and an en dash; reading UTF-8 without the correct encoding can produce mojibake even when the source file itself is valid.

## 3. Non-negotiable data boundaries

### Use the benchmark only for evaluation

Never embed or index any of the following benchmark fields:

- `question`;
- `correct_answer`;
- `split`; or
- benchmark `id`.

The knowledge index must be generated exclusively from the PDQ source folders. Otherwise a retriever may memorize benchmark wording or labels rather than retrieve medical evidence.

### Validate the target label before model development

Do not assume that `correct_answer = true` necessarily means “a false premise is present” without locating the dataset specification or manually auditing the labels. Some `false` examples appear to contain claims that may still need correction, and IDs 329 and 593 contain the same question with opposing labels.

Create a label-audit table for at least 100 stratified rows, including:

- 50 `true` and 50 `false` rows;
- common and rare cancers;
- adult and pediatric cases;
- treatment, prognosis, screening, side-effect, support, and alternative-therapy claims; and
- the conflicting duplicate.

For each audited row, record:

```json
{
  "benchmark_id": 3,
  "label_interpretation": "false_premise_present",
  "candidate_claim": "Surgery is the only treatment for muscle-invasive bladder cancer.",
  "auditor_verdict": "contradicted",
  "evidence_chunk_ids": ["..."],
  "notes": "..."
}
```

Quarantine exact duplicates with conflicting labels from headline metrics. Report both raw-dataset results and cleaned-dataset results.

## 4. Target system architecture

```text
Offline indexing
PDQ JSON -> validation -> normalization -> section-aware chunks
         -> dense embeddings + BM25 index + metadata/alias catalog

Online answering
question -> structured question analysis -> query plan
         -> cancer-specific and general-topic retrieval
         -> reciprocal-rank fusion -> evidence-aware reranking
         -> per-claim evidence verdicts -> grounded response
         -> citations, confidence, safety flags, and trace log
```

Keep the major stages separate. A single prompt that performs extraction, retrieval planning, adjudication, and answer writing at once will be difficult to debug and evaluate.

## 5. Offline pipeline

### 5.1 Validate source records

Implement `scripts/validate_data.py` first. It should fail with a nonzero exit code when it finds:

- malformed JSON or non-UTF-8 input;
- a missing required page field;
- a page without a canonical URL, title, topic, audience, or content hash;
- duplicate canonical URLs with different content hashes;
- an empty `pages` list;
- an empty non-boilerplate page;
- manifest paths that do not exist;
- page URLs present in `all_pages.json` but absent from both grouped folders;
- benchmark IDs that are missing or repeated; or
- duplicate benchmark questions with conflicting labels.

Produce `artifacts/data_validation_report.json` with counts, warnings, exclusions, and corpus hashes. Warnings may be allowed; integrity errors must block indexing.

### 5.2 Parse both PDQ folder types into one page model

Use one Pydantic model for cancer and general-topic pages. Preserve the origin folder as metadata instead of building two incompatible loaders.

Required normalized page fields:

```python
class PdqPage(BaseModel):
    page_id: str
    source_group: Literal["cancer", "general_topic"]
    source_file: str
    source: str
    collection: str
    cancer_type: str | None
    topic: str
    audience: Literal["health_professional", "patient", "unspecified"]
    title: str
    summary_name: str
    url: str
    last_updated: str | None
    retrieved_at: str
    content_sha256: str
    sections: list[PdqSection]
```

For `general_topics`, `cancer_type` may be null. Do not manufacture a single cancer assignment for cross-cutting topics such as pain, fatigue, nutrition, genetics, or complementary therapies.

### 5.3 Normalize without destroying source truth

Store two forms for names:

- `display_name`: the exact NCI or benchmark text;
- `normalized_name`: a lowercase comparison key with punctuation and parenthetical expansions normalized.

Build `config/cancer_aliases.yaml` for explicit, reviewed mappings. Examples that need mapping include:

- `Acute Myeloid Leukemia (AML)` -> `acute_myeloid_leukemia`;
- `Gastric (Stomach) Cancer` -> `stomach_gastric_cancer`;
- `Pregnancy and Breast Cancer` -> `breast_cancer_treatment_during_pregnancy`;
- `Primary CNS Lymphoma (Lymphoma)` -> `primary_cns_lymphoma` and related PDQ pages; and
- `Vascular Tumors (Soft Tissue Sarcoma)` -> the relevant childhood vascular-tumor and soft-tissue-sarcoma pages.

An alias may map to multiple related source groups with weights. Never merge distinct diseases merely because they share tokens such as “lymphoma,” “leukemia,” or “brain tumor.”

The benchmark `cancer` field should not be required in production. The primary experiment extracts cancer entities from question text. A separate “label-assisted retrieval” experiment may use the benchmark cancer field as an oracle hint, but its results must be reported separately.

### 5.4 Remove retrieval noise

Exclude sections with `is_boilerplate = true` from the default indexes. Also detect and exclude or strongly downweight:

- “About PDQ” and generic disclaimer sections;
- permission and contact sections;
- citation-only reference lists;
- navigation remnants; and
- empty or near-duplicate sections.

Keep excluded text in the canonical corpus for auditability. Record `index_status` and `exclusion_reason` rather than deleting source content.

### 5.5 Create section-aware chunks

Use the section hierarchy as the first boundary and tokens as the second boundary.

Initial chunking configuration:

- target: 450-650 tokens;
- hard maximum: 800 tokens;
- minimum useful body: 60 tokens;
- overlap for split long sections: 60-90 tokens;
- no overlap across different pages or major heading branches; and
- merge short adjacent sibling sections only when their parent path is the same.

Do not split in the middle of a list item or table row. Very long sections should first be separated at paragraphs, lists, or table boundaries and only then by sentences.

Every embedding text should carry its context in a short header:

```text
Title: Bladder Cancer Treatment
Cancer/topic: Bladder Cancer
Audience: Health Professional
Section: General Information > Histopathology

[section passage]
```

Do not repeat the full heading path multiple times in the body.

### 5.6 Define stable chunk records

Write canonical chunks to `artifacts/chunks/nci_pdq_chunks.jsonl` before building either index. Both sparse and dense retrieval must use this exact chunk set.

```json
{
  "chunk_id": "nci_pdq:<page_id>:<section_hash>:000",
  "page_id": "nci_pdq:<url_hash>",
  "source_group": "cancer",
  "source_file": "cancers/bladder_cancer.json",
  "cancer_type": "Bladder Cancer",
  "topic": "adult_treatment",
  "audience": "health_professional",
  "title": "Bladder Cancer Treatment (PDQ®)–Health Professional Version",
  "section": "Histopathology",
  "section_path": ["General Information About Bladder Cancer", "Histopathology"],
  "url": "https://www.cancer.gov/...",
  "last_updated": "May 2, 2025",
  "retrieved_at": "2026-08-31T07:08:28Z",
  "content_sha256": "...",
  "chunk_index": 0,
  "text": "...",
  "embedding_text": "..."
}
```

Generate stable IDs from canonical URL, normalized section path, passage text hash, and chunk ordinal. A rerun with unchanged inputs and configuration must produce identical IDs and byte-identical JSONL.

### 5.7 Build two retrieval indexes

Use the libraries already declared in `pyproject.toml`:

- `sentence-transformers` for dense embeddings;
- Qdrant in local persistent mode for vectors and metadata payloads; and
- `rank-bm25` for sparse lexical retrieval.

Pin the exact embedding model name, revision, dimensionality, normalization setting, and query/document prefixes in the experiment configuration. Select the final model through retrieval evaluation rather than assuming one model is best.

Persist:

```text
artifacts/indexes/<index_version>/
  manifest.json
  qdrant/
  bm25.pkl
  chunk_lookup.jsonl
```

The index manifest must include source corpus hash, chunking-config hash, embedding model revision, number of chunks, creation time, and code commit when available.

### 5.8 Guard against leakage

During index construction, normalize and hash every benchmark question and every chunk. Fail if an exact benchmark question appears in the chunk corpus. Also run a high-similarity lexical check to detect accidental benchmark ingestion.

## 6. Online question-answering pipeline

### 6.1 Structured question analysis

The analyzer must be neutral: ask it to identify important assumptions, allowing zero assumptions, rather than asking it to find “the false assumption.”

Target schema:

```json
{
  "cancer_mentions": ["muscle-invasive bladder cancer"],
  "mapped_cancer_keys": ["bladder_cancer"],
  "patient_context": {
    "age_group": "older_adult",
    "stage": null,
    "subtype": "muscle-invasive",
    "treatment_status": "newly_diagnosed"
  },
  "intent": ["treatment_options", "support_resources"],
  "candidate_claims": [
    {
      "claim_id": "c1",
      "text": "Surgery is the only treatment for muscle-invasive bladder cancer.",
      "importance": "high"
    }
  ],
  "urgency_flags": []
}
```

Candidate claims should be declarative, specific, and independently verifiable. Preserve qualifiers such as stage, subtype, age group, pregnancy, prior treatment, and words like “only,” “always,” “never,” or “inevitable.”

### 6.2 Query planning

Generate separate query groups:

1. `claim_queries`: seek evidence that can resolve each candidate claim;
2. `answer_queries`: seek evidence for the user's direct concern; and
3. `general_topic_queries`: seek supportive-care, screening, prevention, genetics, or integrative-therapy evidence when relevant.

For the bladder example:

```json
{
  "claim_queries": [
    "muscle-invasive bladder cancer treatment options",
    "muscle-invasive bladder cancer alternatives to cystectomy",
    "bladder preservation radiation chemotherapy"
  ],
  "answer_queries": [
    "bladder cancer surgery preparation support"
  ],
  "general_topic_queries": [
    "communication and emotional support during cancer treatment"
  ]
}
```

Cap query expansion to avoid flooding the retriever. Start with no more than three queries per important claim and two for the direct answer.

### 6.3 Route to both source groups

Retrieve from the cancer collection and general-topic collection in parallel conceptually, then fuse the results.

Use metadata as a boost, not an absolute filter:

- strong boost for exact mapped cancer and matching stage/subtype terms;
- moderate boost for related cancer aliases;
- boost for matching topic (`screening`, `prevention`, `supportive_care`, and so on);
- retain global general-topic results; and
- allow unfiltered fallback when mapped-cancer retrieval is weak.

A hard cancer filter can hide the correct passage for benchmark labels that differ from NCI terminology or for cross-cutting questions about pain, nutrition, anxiety, genetics, and complementary therapies.

### 6.4 Retrieve and fuse candidates

Recommended starting values for each query:

- dense top 20;
- BM25 top 20;
- deduplicate by `chunk_id`;
- fuse rankings using reciprocal-rank fusion, initially `k = 60`; and
- retain the best 30 candidates for reranking.

Do not combine raw cosine similarity with raw BM25 score through an uncalibrated weighted sum. Their score scales are unrelated. Reciprocal-rank fusion gives a reliable first implementation.

### 6.5 Rerank for evidentiary value

Reranking should answer more than “is this passage about the same cancer?” For every claim-candidate pair, estimate:

- cancer/stage/subtype match;
- whether the passage directly addresses the claim;
- whether it contains support, contradiction, or a material qualifier;
- source audience;
- section specificity; and
- whether it is boilerplate or reference-heavy.

Use a cross-encoder reranker if available, but retain a deterministic metadata-aware fallback. Keep the top 6-10 diverse chunks, normally no more than two nearly identical chunks from the same page section.

Health-professional passages should usually lead factual adjudication. Patient passages may be preferred for accessible explanation. Audience is a role signal, not a claim that one page is always more correct.

### 6.6 Build an evidence matrix before answering

For each candidate claim, create a structured verdict using only retrieved text:

```json
{
  "claim_id": "c1",
  "claim": "Surgery is the only treatment for muscle-invasive bladder cancer.",
  "verdict": "contradicted",
  "confidence": 0.91,
  "supporting_chunk_ids": [],
  "contradicting_chunk_ids": ["nci_pdq:..."],
  "qualifying_chunk_ids": ["nci_pdq:..."],
  "reason": "The passage describes both removal of the bladder and bladder-preserving radiation plus chemotherapy.",
  "evidence_sufficient": true
}
```

Allowed verdicts:

- `supported`;
- `contradicted`;
- `qualified`;
- `insufficient_evidence`; and
- `not_medical_or_not_verifiable`.

Use `insufficient_evidence` rather than letting the language model fill gaps from memory. A contradiction should require a directly relevant passage, not merely the absence of supporting text.

### 6.7 Generate the patient-facing answer

The response generator receives:

- the original question;
- structured question analysis;
- evidence matrix;
- selected chunk text and metadata; and
- the response policy.

Recommended response order:

1. acknowledge the concern briefly;
2. correct or qualify an important misleading assumption if evidence warrants it;
3. answer the direct question;
4. state important uncertainty or missing individual context;
5. suggest discussion with the treating oncology team where personalized decisions are involved; and
6. list compact NCI citations.

Do not expose internal labels such as `contradicted` to the patient unless a structured API consumer requests them.

Every medical sentence should be traceable to at least one supplied chunk. Citation output should contain title, section path, canonical URL, NCI update date when available, and retrieval date.

### 6.8 Safety and scope behavior

The system should:

- avoid diagnosis and individualized treatment selection;
- avoid exact survival predictions from incomplete patient context;
- not present a static corpus as current clinical-trial availability;
- clearly say when the local PDQ corpus does not answer a location-specific or current-services question;
- distinguish evidence about complementary symptom management from claims that an alternative therapy cures cancer;
- preserve pregnancy, pediatric, stage, and subtype qualifiers; and
- surface urgent-care guidance only when the question indicates a possible emergency, without using retrieval failure as reassurance.

## 7. API and output contract

Return both a natural-language answer and an auditable structured record:

```json
{
  "request_id": "...",
  "answer": "...",
  "premise_status": "contradicted",
  "claims": [],
  "citations": [
    {
      "citation_id": 1,
      "chunk_id": "...",
      "title": "...",
      "section_path": ["..."],
      "url": "https://www.cancer.gov/...",
      "last_updated": "..."
    }
  ],
  "confidence": "medium",
  "insufficient_evidence": false,
  "safety_flags": [],
  "retrieval_trace_id": "...",
  "index_version": "..."
}
```

Do not derive user-visible confidence from the language model's self-reported probability alone. Base it on retrieval coverage, agreement between evidence passages, reranker scores, and whether each important claim has direct evidence.

## 8. Evaluation plan

### 8.1 Create development and test partitions correctly

The file has a `split` field, but it denotes dataset category rather than a train/dev/test partition. Create a reproducible evaluation partition that:

- groups exact and near-duplicate questions together;
- stratifies by the verified binary label;
- balances common and rare cancers as far as possible;
- keeps related paraphrases in the same partition; and
- reserves the final test set until prompts and thresholds are frozen.

The RAG corpus remains the same for all partitions. Only benchmark questions and any manually created annotations are partitioned.

### 8.2 Binary premise-detection metrics

After confirming the label semantics, report:

- precision, recall, and F1 for the positive class;
- specificity and negative predictive value;
- balanced accuracy;
- Matthews correlation coefficient;
- confusion matrix; and
- per-cancer and per-intent slices where sample size is sufficient.

Because the classes are 585 versus 150, raw accuracy is not an adequate headline metric.

Tune a decision threshold only on the development set. Freeze it before running the final test.

### 8.3 Retrieval evaluation requires new gold annotations

The current benchmark contains no evidence IDs. Build a small gold retrieval set, initially 100-150 questions, with:

- normalized cancer mapping;
- explicit claim(s);
- relevant PDQ page(s);
- decisive section(s) or chunk IDs;
- verdict for each claim; and
- whether the local corpus has enough evidence.

Measure:

- page Recall@5 and Recall@10;
- decisive-chunk Recall@5 and Recall@10;
- mean reciprocal rank;
- nDCG@10;
- evidence coverage per claim;
- general-topic routing recall; and
- rate of retrieving the right disease but the wrong stage, subtype, or population.

### 8.4 Generation evaluation

Use a balanced, stratified sample for clinician or expert review. Score each answer on:

- correct handling of the premise;
- factual consistency with cited chunks;
- directness and completeness;
- citation correctness;
- preservation of stage/subtype/population qualifiers;
- appropriate uncertainty;
- patient-friendly wording; and
- safety.

Also calculate citation entailment automatically: for each cited sentence, check whether the cited chunk actually supports it. Use automated judging as a screening tool, not the sole medical correctness measure.

### 8.5 Baselines and ablations

Run at least these systems on the same frozen evaluation set:

1. language model without retrieval;
2. dense-only vanilla RAG using the raw question;
3. hybrid vanilla RAG using the raw question;
4. hybrid RAG with query decomposition;
5. full premise-aware hybrid RAG with evidence adjudication.

Useful ablations:

- cancer files only versus cancer plus general topics;
- BM25 only versus dense only versus hybrid;
- question-only versus claim-specific queries;
- no alias mapping versus reviewed aliases;
- no reranker versus relevance reranker versus evidence-aware reranker;
- professional pages only versus both audiences; and
- hard cancer filter versus metadata boost with fallback.

Save all prompts, configs, retrieved chunk IDs, model versions, and raw outputs for every run.

## 9. Testing strategy

### Unit tests

Cover:

- UTF-8 JSON loading;
- page-schema validation;
- cancer alias normalization;
- boilerplate filtering;
- stable chunk IDs;
- paragraph/list/table chunk boundaries;
- token-limit enforcement;
- BM25 and dense result shapes;
- reciprocal-rank fusion;
- citation rendering; and
- refusal/insufficient-evidence behavior.

### Integration tests

Use a small fixed corpus fixture and representative questions for:

- treatment “only option” myths;
- screening “only test” myths;
- prognosis absolutes;
- pediatric/adult confusion;
- side effects and supportive care;
- complementary-therapy cure claims;
- a reasonable assumption that should not be corrected; and
- a question the corpus cannot answer.

Assert that cited chunk IDs exist in the selected index and that every URL in the response matches chunk metadata.

### Regression tests

Maintain a small set of high-value examples. A change should fail CI when it:

- loses decisive evidence from top 10;
- changes a correct premise verdict to an unsupported contradiction;
- produces an uncited medical claim;
- leaks benchmark fields into the index; or
- changes chunk IDs without an intentional index-version change.

## 10. Recommended project structure

```text
rag-cancer-myth/
  config/
    cancer_aliases.yaml
    chunking.yaml
    retrieval.yaml
    prompts/
      analyze_question.txt
      adjudicate_claims.txt
      generate_answer.txt
  src/cancer_myth_rag/
    schemas.py
    data_validation.py
    pdq_loader.py
    normalization.py
    chunking.py
    dense_index.py
    sparse_index.py
    retrieval.py
    query_planner.py
    reranker.py
    adjudication.py
    generation.py
    citations.py
    safety.py
    pipeline.py
  scripts/
    validate_data.py
    prepare_chunks.py
    build_indexes.py
    answer_question.py
    run_benchmark.py
    evaluate_retrieval.py
    evaluate_answers.py
  tests/
    fixtures/
    unit/
    integration/
    regression/
  artifacts/
    data_validation_report.json
    chunks/
    indexes/
    runs/
    evaluations/
```

Generated artifacts should normally be ignored by Git unless the project explicitly chooses to version a small benchmark fixture or summary report.

## 11. Initial configuration

Start with a simple, reproducible configuration and tune only after measuring it.

```yaml
chunking:
  target_tokens: 550
  max_tokens: 800
  min_tokens: 60
  overlap_tokens: 80
  exclude_boilerplate: true

retrieval:
  dense_top_k_per_query: 20
  bm25_top_k_per_query: 20
  fusion: rrf
  rrf_k: 60
  rerank_candidates: 30
  final_context_chunks: 8
  max_chunks_per_page_section: 2
  allow_global_fallback: true

answering:
  max_claim_queries: 3
  max_answer_queries: 2
  require_direct_evidence_for_contradiction: true
  cite_each_medical_claim: true
  abstain_on_insufficient_evidence: true
```

Treat these as baseline values, not facts. Tune them on retrieval annotations and development data, never on the final test set.

## 12. Implementation phases and acceptance criteria

### Phase 0: Clarify labels and repair data

Work:

- confirm the source definition of `correct_answer` and `cancer_myth_nfp`;
- audit a stratified sample;
- quarantine conflicting duplicate IDs 329/593;
- restore the three missing grouped PDQ files; and
- add the validation command and report.

Exit criteria:

- benchmark target semantics are documented;
- all label conflicts have a declared handling rule;
- manifest, grouped files, and `all_pages.json` agree; and
- validation passes from a clean checkout.

### Phase 1: Canonical chunk corpus

Work:

- implement schemas, loaders, normalization, filtering, chunking, and stable IDs;
- build and review `cancer_aliases.yaml`; and
- write deterministic JSONL plus a chunking report.

Exit criteria:

- zero boilerplate chunks in the default index;
- all chunks fit the token limit;
- all chunks have valid provenance;
- repeated builds are byte-identical; and
- manual review of at least 50 chunks finds no serious boundary or context loss.

### Phase 2: Retrieval baseline

Work:

- build dense and BM25 indexes;
- implement metadata boosts, reciprocal-rank fusion, and trace logging; and
- create the first gold evidence set.

Exit criteria:

- retrieval is deterministic for fixed versions;
- every result resolves to a canonical chunk;
- decisive-chunk Recall@10 is measured; and
- errors are categorized before tuning.

### Phase 3: Premise-aware retrieval

Work:

- implement structured question analysis and query planning;
- retrieve separately for claims and direct-answer intents;
- add evidence-aware reranking and evidence matrices; and
- compare against raw-question hybrid retrieval.

Exit criteria:

- schema-valid outputs on at least 99% of benchmark questions;
- measurable improvement in decisive-evidence recall or premise F1 on development data; and
- no unacceptable increase in false corrections on audited negative cases.

### Phase 4: Grounded answer generation

Work:

- add patient-facing generation, citations, abstention, and safety rules;
- verify citation entailment; and
- add integration and regression tests.

Exit criteria:

- no fabricated URLs or chunk IDs in the test suite;
- no uncited substantive medical claims in the reviewed sample;
- insufficient-evidence cases are not answered as established fact; and
- expert review meets the agreed safety and faithfulness thresholds.

### Phase 5: Frozen benchmark evaluation

Work:

- freeze code, prompts, thresholds, corpus, and index versions;
- run baselines, full system, and planned ablations;
- report raw and cleaned benchmark metrics; and
- publish error analysis and reproducibility metadata.

Exit criteria:

- one command can reproduce each reported run;
- all results reference immutable run and index manifests; and
- limitations caused by label quality and missing gold answers are explicit.

## 13. Observability and reproducibility

For each request, log without storing unnecessary personal information:

- normalized question hash;
- extracted entities, intents, and claims;
- generated queries;
- dense and sparse rankings;
- fusion and reranker scores;
- selected context chunk IDs;
- evidence verdicts;
- final citations;
- model and prompt versions;
- index version; and
- latency for each stage.

Never log secrets. If real patient questions are introduced later, define retention, access control, and de-identification policies before storing raw text.

## 14. Important failure modes

| Failure | Likely cause | Mitigation |
|---|---|---|
| Relevant cancer, wrong stage | broad semantic match | preserve qualifiers; evidence-aware reranking |
| No correction of an “only” claim | retrieval follows downstream support intent | separate claim queries from answer queries |
| Invented myth in a normal question | analyzer is forced to find one | allow zero claims; require direct contradiction evidence |
| Supportive-care page never appears | hard cancer filtering | retrieve general topics separately and fuse |
| Correct page, useless reference chunk | references dominate lexical match | filter/downweight reference-only sections |
| Conflicting NCI patient/professional wording | different detail levels or update states | retain audience/date metadata and adjudicate qualifiers |
| Fabricated answer despite weak evidence | generator relies on model memory | evidence matrix plus explicit abstention |
| Good binary score, poor medical answer | benchmark has only a Boolean label | add gold evidence and expert generation evaluation |
| Inflated score from duplicates | paraphrase or exact duplicate split leakage | group duplicates before partitioning |
| Irreproducible results | mutable models or corpus | pin revisions and write run/index manifests |

## 15. Definition of done for version 1

Version 1 is complete when:

1. the benchmark label semantics and known conflicts are documented;
2. the requested PDQ folders form a complete, validated 381-page grouped corpus or have explicit exclusions;
3. a deterministic, provenance-preserving chunk corpus can be rebuilt with one command;
4. dense and BM25 indexes are built from the same chunks with no benchmark leakage;
5. the online pipeline extracts zero or more claims, retrieves claim-specific evidence, adjudicates it, and generates a cited answer;
6. every citation resolves to an exact local chunk and canonical NCI URL;
7. the system abstains when the corpus lacks decisive evidence;
8. retrieval, classification, citation, and answer-quality metrics are reported separately;
9. results include baseline and ablation comparisons; and
10. the complete experiment is reproducible from saved configuration and manifests.

The best first engineering milestone is not a chatbot UI. It is a command-line pipeline that can take one benchmark ID, print the extracted claims and retrieval trace, show the decisive NCI passages, produce a structured verdict, and then render the cited patient-facing answer. Once that trace is reliable, an API or interface can be added without hiding errors in the core evidence pipeline.
