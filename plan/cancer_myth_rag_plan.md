# Cancer Myth RAG — Implementation Plan

## 1. Project objective

Build a **premise-aware Retrieval-Augmented Generation (RAG) system** that can answer patient-facing cancer questions while detecting and correcting clinically important false assumptions embedded in those questions.

This is not a conventional cancer question-answering task. Many questions are phrased so that the user asks for advice *after assuming an incorrect medical fact*. A standard RAG system may retrieve relevant information for the surface question and accidentally reinforce the false assumption.

Example pattern:

> “My father has muscle-invasive bladder cancer and believes surgery is the only treatment. Are there support groups for preparing for surgery?”

The system should not immediately answer with surgery support resources. It should first verify the assumption that surgery is the only treatment, correct it if unsupported, and then address the user’s actual concern.

The core goal is therefore:

> **Retrieve authoritative evidence that allows the model to verify the assumptions in a cancer question, then generate an evidence-grounded answer that corrects misinformation without inventing a misconception when none exists.**

---

## 2. Dataset summary

Uploaded dataset:

`cancermyth_screening_dataset.json`

### Dataset size

- **735 total questions**
- **585 `cancer_myth` questions**
  - `correct_answer = true`
  - These contain a false or misleading presupposition that should be identified/corrected.
- **150 `cancer_myth_nfp` questions**
  - `correct_answer = false`
  - These are negative controls where the system should *not* invent a false presupposition.
- **126 unique cancer labels**

### Current schema

```json
{
  "id": 1,
  "split": "cancer_myth",
  "cancer": "Lymphoma",
  "question": "My 70-year-old mom was just diagnosed with lymphoma...",
  "correct_answer": true
}
```

### Important implication

The dataset gives the benchmark label, but it does **not** contain the external evidence that the RAG system should retrieve.

Therefore:

- Use this dataset for **evaluation**
- Do **not** index the benchmark questions or gold labels into the retrieval corpus
- Build the knowledge base independently from authoritative oncology sources

Otherwise, retrieval may simply recover benchmark-specific answers and create data leakage.

---

# 3. Main research question

Primary research question:

> **Does premise-aware RAG improve the correction of false cancer presuppositions compared with a general LLM and vanilla RAG, while maintaining a low false-positive rate on questions without false presuppositions?**

Recommended experimental systems:

| System | Retrieval | Explicit premise analysis |
|---|---:|---:|
| LLM baseline | No | No |
| Vanilla RAG | Yes | No |
| Premise-aware RAG | Yes | Yes |

This design lets us test whether retrieval alone is sufficient, or whether the model requires an explicit premise-verification stage.

---

# 4. Data sources for the RAG knowledge base

Use authoritative, evidence-based, preferably reusable sources.

## Tier 1 — Core corpus

### 4.1 National Cancer Institute (NCI) / Cancer.gov / PDQ

This should be the **main knowledge source**.

Collect:

- Cancer-specific treatment summaries
- Diagnosis and staging
- Screening
- Prevention
- Genetics
- Supportive care
- Palliative care
- Treatment modalities
- Treatment-related adverse effects
- Pediatric cancer information

Prefer both:

- **PDQ Health Professional Version**
- **PDQ Patient Version**

Why both?

- Professional pages: more precise clinical information
- Patient pages: useful for patient-friendly wording

Recommended metadata:

```json
{
  "source": "NCI",
  "collection": "PDQ",
  "audience": "health_professional",
  "cancer_type": "Bladder Cancer",
  "topic": "Treatment",
  "section": "Treatment of Stage II and Stage III Bladder Cancer",
  "title": "Bladder Cancer Treatment (PDQ)",
  "url": "...",
  "last_updated": "...",
  "text": "..."
}
```

---

### 4.2 MedlinePlus

Use as a second high-quality source, especially for:

- Patient education
- Treatment effects
- Common symptoms
- General supportive care
- Patient-friendly explanations

Prefer downloading or consuming structured data rather than scraping arbitrary pages.

---

# 5. Secondary and dynamic sources

## 5.1 ClinicalTrials.gov

Use for questions asking about:

- Available trials
- Recruiting trials
- Experimental treatments
- Trial eligibility

Do **not** rely only on a static vector index for current trial availability.

Instead:

```text
Question
   ↓
Trial-intent classifier
   ↓
ClinicalTrials.gov API
   ↓
Current trial records
   ↓
LLM response
```

Clinical trial data changes over time, so dynamic retrieval is preferable.

---

## 5.2 PubMed

Use PubMed mainly when:

- NCI does not provide enough detail
- The cancer is rare
- The question involves emerging evidence
- A specific clinical controversy must be resolved

Start with abstracts rather than ingesting the entire PubMed database.

---

## 5.3 PubMed Central Open Access

Use PMC Open Access for full text when needed.

Possible document types:

- Systematic reviews
- Clinical reviews
- Guidelines
- Consensus statements
- High-quality original studies

Do not indiscriminately index millions of papers in the first version.

---

# 6. Sources to avoid as the primary corpus

Avoid making these the main evidence base:

- Random cancer blogs
- SEO medical websites
- Forums
- Reddit
- Social media posts
- Unverified hospital marketing pages
- General web search snippets
- Automatically generated medical summaries

They may still be useful for nonclinical resources, but not as the primary evidence for medical claims.

---

# 7. Proposed system architecture

```text
                       ┌───────────────────────┐
                       │    Patient Question    │
                       └───────────┬───────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │ Question Understanding    │
                    │ - cancer entity           │
                    │ - stage/subtype           │
                    │ - intent                  │
                    │ - candidate assumptions   │
                    └───────────┬──────────────┘
                                │
                                ▼
                    ┌──────────────────────────┐
                    │ Premise Verification Plan │
                    │ Generate search queries   │
                    └───────────┬──────────────┘
                                │
                     ┌──────────┴──────────┐
                     ▼                     ▼
              ┌────────────┐        ┌─────────────┐
              │   BM25     │        │ Dense Search │
              └─────┬──────┘        └──────┬──────┘
                    │                      │
                    └──────────┬───────────┘
                               ▼
                     ┌────────────────────┐
                     │ Hybrid Candidate Set│
                     └──────────┬─────────┘
                                ▼
                     ┌────────────────────┐
                     │     Reranker       │
                     └──────────┬─────────┘
                                ▼
                     ┌────────────────────┐
                     │ Evidence Evaluation │
                     │ support / contradict│
                     │ qualify / insufficient
                     └──────────┬─────────┘
                                ▼
                     ┌────────────────────┐
                     │ Final LLM Answer    │
                     │ - correct premise   │
                     │ - answer concern    │
                     │ - cite evidence     │
                     └────────────────────┘
```

---

# 8. Step 1 — Ingest and clean the documents

Each source should be converted into a common document format.

Example internal representation:

```python
Document(
    text="...",
    metadata={
        "source": "NCI",
        "cancer": "bladder cancer",
        "topic": "treatment",
        "stage": "muscle-invasive",
        "section": "Treatment options",
        "url": "...",
        "updated": "..."
    }
)
```

Cleaning should remove:

- Navigation menus
- Repeated headers
- Cookie banners
- Footers
- Related-link widgets
- Duplicated text
- JavaScript
- Empty fragments

Preserve:

- Headings
- Cancer names
- Treatment names
- Stage information
- Tables converted to readable text
- Source URLs
- Update dates

---

# 9. Step 2 — Section-aware chunking

Do **not** split documents only by character count.

Bad:

```python
text[0:2000]
text[2000:4000]
```

Better:

```text
Bladder Cancer Treatment
└── Treatment of Muscle-Invasive Disease
    ├── Radical cystectomy
    ├── Bladder-preserving therapy
    └── Systemic therapy
```

Recommended first configuration:

- Chunk size: **300–700 tokens**
- Overlap: **50–100 tokens**
- Preserve heading hierarchy
- Include cancer name in every chunk
- Include source and section metadata

Example chunk:

```text
Cancer: Bladder Cancer
Section: Treatment of Muscle-Invasive Bladder Cancer

[chunk text...]
```

This helps the retriever distinguish clinically similar concepts.

---

# 10. Step 3 — Build two retrieval indexes

Use **hybrid retrieval**.

## Dense retrieval

Purpose:

- Semantic similarity
- Patient language ↔ medical terminology
- Paraphrased questions

Possible embedding models:

- `BAAI/bge-m3`
- another strong biomedical/general retrieval embedding model tested empirically

Vector stores:

- FAISS for a simple local baseline
- Qdrant for a more production-like system

### Recommended V1

Use **Qdrant** if metadata filtering and hybrid search are important.

Use **FAISS** if the goal is the fastest local research prototype.

---

## Sparse retrieval

Use BM25 for:

- Drug names
- Regimen names
- Cancer subtype names
- Acronyms
- Exact medical terminology

Examples:

```text
R-CHOP
BEACOPP
CLL
PCNSL
CAR-T
TURBT
HCC
BRAF V600E
```

Dense embeddings can occasionally blur these terms; BM25 helps preserve exact matching.

---

# 11. Step 4 — Premise extraction

Before retrieval, analyze the question.

Do **not** ask:

> “What is the false assumption?”

That wording forces the model to assume one exists.

Instead ask:

> “Identify clinically important assumptions in the question. There may be zero, one, or multiple assumptions.”

Expected structured output:

```json
{
  "cancer": "Bladder cancer",
  "subtype": "muscle-invasive bladder cancer",
  "user_intent": "find support resources",
  "candidate_assumptions": [
    "Surgery is the only treatment for muscle-invasive bladder cancer."
  ]
}
```

For a negative-control question:

```json
{
  "cancer": "Wilms tumor",
  "user_intent": "support for chemotherapy-related hair loss",
  "candidate_assumptions": []
}
```

This is critical for avoiding overcorrection.

---

# 12. Step 5 — Query decomposition and rewriting

Do not retrieve using only the raw patient question.

Example raw question:

```text
My father has muscle-invasive bladder cancer and believes surgery
is the only treatment. Are there support groups for preparing for surgery?
```

Generate several evidence-oriented queries:

```text
muscle-invasive bladder cancer treatment options
is radical cystectomy the only treatment for muscle-invasive bladder cancer
bladder-preserving trimodality therapy muscle-invasive bladder cancer
chemoradiation versus cystectomy muscle-invasive bladder cancer
```

Recommended output:

```json
{
  "verification_queries": [
    "muscle-invasive bladder cancer treatment options",
    "radical cystectomy alternatives",
    "bladder-preserving chemoradiation"
  ],
  "answer_queries": [
    "support resources bladder cancer surgery"
  ]
}
```

This separates:

1. **Premise verification**
2. **Surface-question answering**

---

# 13. Step 6 — Metadata-aware retrieval

Use cancer labels and extracted entities as a retrieval prior.

Example:

```python
filter = {
    "cancer": "bladder cancer"
}
```

Do not always make filters strict.

Better approach:

- Strongly boost same-cancer chunks
- Allow related/general oncology chunks
- Retrieve supportive-care information across diseases when appropriate

This prevents incorrect filtering in questions where the dataset cancer label is broad or imperfect.

---

# 14. Step 7 — Retrieve candidates

Example V1:

```text
Dense search: top 20
BM25 search: top 20
        ↓
Deduplicate / merge
        ↓
~30 candidate chunks
```

Then rerank.

---

# 15. Step 8 — Reranking

Ordinary reranking asks:

> Is this passage relevant to the question?

For this project, relevance is not enough.

The reranker should favor evidence that can **resolve an assumption**.

Desired criterion:

> Does this passage contain information capable of supporting, contradicting, or qualifying the candidate premise?

Example:

### Candidate A

> Support services are available after radical cystectomy.

### Candidate B

> Muscle-invasive bladder cancer may be treated with radical cystectomy or bladder-preserving trimodality therapy in selected patients.

For premise verification, Candidate B should rank substantially higher.

Possible rerankers:

- `bge-reranker-v2-m3`
- Cross-encoder reranker
- LLM-based reranking for experiments

Recommended top context after reranking:

- **5–8 chunks**

---

# 16. Step 9 — Evidence classification

Before producing the final answer, classify evidence for each candidate assumption.

Possible labels:

```text
SUPPORTED
CONTRADICTED
QUALIFIED
INSUFFICIENT_EVIDENCE
```

Example:

```json
{
  "assumption": "Surgery is the only treatment for muscle-invasive bladder cancer.",
  "verdict": "CONTRADICTED",
  "evidence_ids": ["nci_3821", "nci_3824"],
  "confidence": 0.94
}
```

This intermediate step improves interpretability and allows detailed error analysis.

---

# 17. Step 10 — Final answer generation

Recommended answer policy:

1. Detect whether a clinically important premise exists
2. Correct it if contradicted or materially incomplete
3. Do not manufacture a misconception
4. Answer the actual question
5. Use retrieved evidence only for factual medical claims
6. Cite sources
7. Communicate uncertainty
8. Avoid individualized treatment recommendations beyond available information

Suggested internal output:

```json
{
  "false_presupposition_present": true,
  "presupposition": "Surgery is the only treatment for muscle-invasive bladder cancer.",
  "correction": "Surgery is an important option, but it is not the only possible treatment...",
  "answer": "...",
  "citations": [
    {
      "source": "NCI",
      "title": "...",
      "url": "..."
    }
  ]
}
```

Then render this as natural patient-facing prose.

---

# 18. Prompt template

```text
You are an evidence-grounded cancer information assistant.

The patient's question may or may not contain medically incorrect,
unsupported, or overly absolute assumptions.

Using only the supplied evidence:

1. Identify clinically important assumptions in the patient's question.
2. Determine whether each assumption is supported, contradicted,
   qualified, or unresolved by the evidence.
3. If an important assumption is incorrect or misleading, correct it
   clearly before answering the user's downstream question.
4. If no incorrect assumption is present, do not invent one.
5. Answer the patient's underlying concern directly and compassionately.
6. Do not make unsupported prognostic claims.
7. Do not recommend a specific personalized treatment when the necessary
   clinical information is unavailable.
8. Cite the supporting evidence for medical claims.
9. If evidence is insufficient, say so explicitly.
```

---

# 19. Special routing

Some questions should use external live tools instead of only static RAG.

## Clinical trial question

```text
Question
   ↓
Clinical-trial intent = true
   ↓
Cancer + stage + location + intervention extraction
   ↓
ClinicalTrials.gov API
   ↓
Relevant active studies
```

---

## Location-dependent question

Examples:

- “support groups near me”
- “best cancer center in my area”
- “wig shops in my city”

The medical premise should still be checked using RAG.

After premise verification:

```text
medical evidence verification
        +
location/business search
```

Do not let local-resource retrieval replace medical premise verification.

---

# 20. Recommended technology stack

## Language

```text
Python 3.11+
```

## Parsing

```text
requests
beautifulsoup4
lxml
trafilatura
```

## Data

```text
pandas
pydantic
```

## Embeddings

```text
sentence-transformers
transformers
```

## Vector database

Option A:

```text
FAISS
```

Option B:

```text
Qdrant
```

Recommended for this project:

```text
Qdrant
```

because metadata and hybrid retrieval are useful.

## Sparse retrieval

```text
rank-bm25
```

or Qdrant sparse retrieval.

## Reranking

```text
BAAI/bge-reranker-v2-m3
```

## RAG framework

Keep V1 lightweight.

Options:

```text
LlamaIndex
Haystack
LangChain
custom Python
```

Recommended:

> Start with either **LlamaIndex** or a small custom pipeline.

Avoid an overly complicated agent framework until retrieval quality is understood.

---

# 21. Suggested repository structure

```text
cancer-myth-rag/
│
├── data/
│   ├── benchmark/
│   │   └── cancermyth_screening_dataset.json
│   │
│   ├── raw/
│   │   ├── nci/
│   │   ├── medlineplus/
│   │   └── pmc/
│   │
│   └── processed/
│       ├── documents.jsonl
│       └── chunks.jsonl
│
├── scripts/
│   ├── download_nci.py
│   ├── download_medlineplus.py
│   ├── parse_documents.py
│   ├── chunk_documents.py
│   └── build_index.py
│
├── src/
│   ├── ingestion/
│   │   ├── nci.py
│   │   ├── medlineplus.py
│   │   └── pubmed.py
│   │
│   ├── retrieval/
│   │   ├── dense.py
│   │   ├── bm25.py
│   │   ├── hybrid.py
│   │   └── reranker.py
│   │
│   ├── premise/
│   │   ├── extractor.py
│   │   ├── query_rewriter.py
│   │   └── verifier.py
│   │
│   ├── generation/
│   │   ├── prompts.py
│   │   └── answer.py
│   │
│   └── evaluation/
│       ├── retrieval.py
│       ├── myth_detection.py
│       └── answer_quality.py
│
├── experiments/
│   ├── llm_baseline.py
│   ├── vanilla_rag.py
│   └── premise_aware_rag.py
│
├── tests/
│
├── notebooks/
│   └── error_analysis.ipynb
│
├── requirements.txt
└── README.md
```

---

# 22. Evaluation design

Evaluation should be separated into several levels.

## 22.1 False-premise detection

For all 735 questions:

```text
Predicted premise present: yes/no
Gold: correct_answer
```

Metrics:

- Accuracy
- Sensitivity / Recall
- Specificity
- Precision
- F1
- Balanced accuracy

Because the data are imbalanced:

```text
585 positive
150 negative
```

balanced accuracy, sensitivity, and specificity are especially important.

---

## 22.2 Premise correction

For the 585 Cancer-Myth examples:

Assess:

- Was the problematic assumption identified?
- Was it corrected correctly?
- Was the correction clinically meaningful?

Recommended primary metric:

```text
Premise Correction Rate =
number of questions with correctly identified + correctly corrected premise
---------------------------------------------------------------------------
585
```

---

## 22.3 False-positive correction

For the 150 NFP questions:

```text
False Presupposition Rate =
NFP questions where the model invented a false premise
------------------------------------------------------
150
```

This is one of the most important safety metrics.

---

# 23. Retrieval evaluation

Answer-generation performance alone cannot tell us *why* RAG failed.

Manually create a smaller evidence-labeled retrieval set.

Suggested:

```text
100–200 benchmark questions
```

For each question, label whether each retrieved passage:

- directly verifies the premise
- partially helps
- is irrelevant
- may reinforce the false premise

Metrics:

- Recall@1
- Recall@3
- Recall@5
- Recall@10
- MRR
- nDCG
- premise-resolving evidence recall

A useful project-specific metric:

```text
Evidence Resolution Recall@K =
fraction of questions where at least one top-K chunk contains enough
information to verify or refute the target premise
```

---

# 24. Final-answer evaluation

Possible dimensions:

### Correctness

Does the answer reflect accepted medical knowledge?

### Premise handling

Does it identify and correct the misleading assumption?

### Completeness

Does it still answer the user's actual concern?

### Evidence faithfulness

Are claims supported by retrieved evidence?

### Citation correctness

Does each citation actually support the associated claim?

### Communication quality

Is the response understandable to a patient?

### Safety

Does it avoid:

- unsupported prognosis
- inappropriate end-of-life framing
- false reassurance
- harmful alternative-treatment recommendations
- unsupported personalized treatment recommendations

---

# 25. Experimental comparisons

Recommended experiments:

## Experiment A — LLM only

```text
Question → LLM → Answer
```

Purpose:

Baseline knowledge without retrieval.

---

## Experiment B — Vanilla RAG

```text
Question
   ↓
Top-K retrieval
   ↓
LLM
```

Purpose:

Determine whether conventional RAG solves the benchmark.

Expected failure:

It may retrieve relevant information to the downstream request without challenging the embedded premise.

---

## Experiment C — Premise-aware RAG

```text
Question
   ↓
Assumption extraction
   ↓
Verification-oriented query rewriting
   ↓
Hybrid retrieval
   ↓
Reranking
   ↓
Premise verification
   ↓
Answer
```

This should be the proposed main system.

---

# 26. Ablation experiments

After the full system works, test which components matter.

### A1

Dense retrieval only

vs.

Hybrid retrieval

### A2

Raw question

vs.

Query rewriting

### A3

No reranker

vs.

Reranker

### A4

No cancer metadata

vs.

Cancer metadata boosting

### A5

Vanilla RAG prompt

vs.

Premise-aware prompt

### A6

Single retrieval query

vs.

Multiple verification queries

### A7

NCI only

vs.

NCI + MedlinePlus

### A8

NCI + MedlinePlus

vs.

NCI + MedlinePlus + PubMed/PMC

This will determine whether adding a very large biomedical corpus actually improves the task.

---

# 27. Error taxonomy

Every failed example should be assigned a reason.

Recommended taxonomy:

```text
1. Premise extraction failure
2. Cancer/entity extraction failure
3. Query rewriting failure
4. Retrieval failure
5. Reranking failure
6. Correct evidence retrieved but ignored
7. Incorrect evidence interpretation
8. Hallucinated correction
9. False-positive myth detection
10. Correct correction but downstream question unanswered
11. Unsupported treatment recommendation
12. Citation mismatch
13. Knowledge-source coverage gap
14. Temporal/current-information failure
```

This analysis will likely be more scientifically valuable than reporting accuracy alone.

---

# 28. Data leakage prevention

Do not insert the benchmark directly into the vector database.

Avoid indexing:

```text
question → benchmark correction
```

Avoid training on the held-out test questions.

Recommended separation:

```text
External evidence:
NCI / MedlinePlus / PMC

Benchmark:
Cancer-Myth dataset

Purpose:
evaluation only
```

If prompt development or model tuning uses benchmark items, create explicit:

```text
development split
test split
```

and freeze the final test set.

---

# 29. Recommended development phases

## Phase 1 — Benchmark exploration

Tasks:

- Load the 735 questions
- Inspect cancer labels
- Inspect positive vs negative controls
- Create an error taxonomy
- Sample approximately 100 questions manually

Deliverable:

```text
dataset_analysis.ipynb
```

---

## Phase 2 — NCI corpus

Tasks:

- Download NCI cancer pages / PDQ
- Parse documents
- Normalize metadata
- Create chunks

Deliverable:

```text
nci_chunks.jsonl
```

---

## Phase 3 — Baseline dense RAG

Tasks:

- Build embeddings
- Create FAISS/Qdrant index
- Retrieve top-K
- Generate answers

Deliverable:

```text
vanilla_rag.py
```

---

## Phase 4 — Hybrid retrieval

Add:

- BM25
- dense search
- result merging
- reranker

Deliverable:

```text
hybrid_retriever.py
```

---

## Phase 5 — Premise-aware pipeline

Add:

- assumption extraction
- premise-oriented query generation
- evidence classification
- final correction logic

Deliverable:

```text
premise_aware_rag.py
```

---

## Phase 6 — MedlinePlus expansion

Add patient-facing medical content.

Compare:

```text
NCI
vs.
NCI + MedlinePlus
```

---

## Phase 7 — Rare-cancer fallback

Add PubMed/PMC retrieval when core sources are insufficient.

Do not automatically query PubMed for every question.

Possible router:

```python
if core_retrieval_confidence < threshold:
    retrieve_pubmed()
```

---

## Phase 8 — Dynamic clinical-trial retrieval

Add ClinicalTrials.gov API routing.

---

## Phase 9 — Full evaluation

Run:

```text
LLM baseline
Vanilla RAG
Premise-aware RAG
```

on the fixed benchmark.

---

# 30. Minimal viable implementation

Do not start with:

- Agents
- GraphRAG
- knowledge graphs
- fine-tuning
- multi-agent debate
- massive PubMed ingestion

Start with:

```text
NCI + MedlinePlus
        ↓
section-aware chunking
        ↓
BM25 + dense retrieval
        ↓
reranker
        ↓
premise extraction
        ↓
query rewriting
        ↓
evidence verification
        ↓
answer + citations
```

This is enough to test the central hypothesis.

---

# 31. Suggested V1 configuration

```yaml
corpus:
  sources:
    - NCI_PD​​Q
    - MedlinePlus

chunking:
  target_tokens: 500
  overlap_tokens: 75
  preserve_headings: true

dense_retrieval:
  model: BAAI/bge-m3
  top_k: 20

sparse_retrieval:
  method: BM25
  top_k: 20

fusion:
  method: reciprocal_rank_fusion

reranking:
  model: BAAI/bge-reranker-v2-m3
  candidates: 30
  final_k: 6

generation:
  structured_output: true
  citations_required: true

premise_verification:
  labels:
    - SUPPORTED
    - CONTRADICTED
    - QUALIFIED
    - INSUFFICIENT_EVIDENCE
```

Treat these as starting values, not fixed choices. Tune them on a development set.

---

# 32. Example end-to-end workflow

Input:

```text
My 70-year-old mom has advanced lymphoma.
We were told there is no treatment because it is advanced.
What should we expect?
```

## Step A — Analyze

```json
{
  "cancer": "lymphoma",
  "candidate_assumptions": [
    "Advanced lymphoma cannot be treated."
  ],
  "user_intent": "understand expected next steps"
}
```

## Step B — Generate verification queries

```text
advanced lymphoma treatment
stage III IV lymphoma treatment options
can advanced lymphoma be treated
treatment of aggressive advanced lymphoma
```

## Step C — Retrieve evidence

Retrieve NCI lymphoma treatment sections rather than primarily hospice pages.

## Step D — Verify

```json
{
  "assumption": "Advanced lymphoma cannot be treated.",
  "verdict": "CONTRADICTED"
}
```

## Step E — Generate

Desired answer pattern:

```text
Advanced lymphoma does not automatically mean that no treatment is available.
Treatment depends on the lymphoma subtype, stage, symptoms, overall health,
and prior therapy.

[Explain evidence-supported treatment context.]

The next useful step is to clarify the exact lymphoma subtype and what the
oncology team is recommending.

[Answer the user's concern.]
```

The system should not simply accept the question's original end-of-life framing.

---

# 33. Core principle

The key idea of this project is:

> **Retrieve against the assumptions, not only against the surface question.**

A standard QA retriever optimizes:

```text
"What information answers this question?"
```

The proposed retriever should additionally optimize:

```text
"What evidence determines whether the assumptions required by this
question are medically valid?"
```

That is the central difference between **vanilla cancer RAG** and **premise-aware cancer RAG**.

---

# 34. Immediate next steps

Recommended order:

1. Create the repository structure.
2. Write a script to analyze the 735-question benchmark.
3. Create a crawler/downloader for NCI PDQ.
4. Normalize NCI pages into JSONL.
5. Implement section-aware chunking.
6. Build a dense retrieval baseline.
7. Add BM25 + reciprocal rank fusion.
8. Add a reranker.
9. Implement premise extraction.
10. Implement verification-query rewriting.
11. Implement evidence classification.
12. Implement structured answer generation.
13. Run the three-system comparison.
14. Perform retrieval and error analysis.
15. Add MedlinePlus.
16. Add PubMed/PMC only where failure analysis shows a need.
17. Add live ClinicalTrials.gov retrieval as a specialized route.

---

# 35. Success criteria for the first research prototype

The first prototype is successful if it can demonstrate that:

1. The external corpus contains premise-resolving evidence for a large fraction of benchmark questions.
2. Premise-aware query rewriting improves retrieval of that evidence.
3. Premise-aware RAG improves correction of the 585 myth questions compared with vanilla RAG.
4. The improvement does not come from simply flagging every question as misleading.
5. False-positive corrections remain low on the 150 NFP questions.
6. Generated claims are traceable to retrieved evidence.
7. Failure cases can be assigned to interpretable pipeline components.

If these conditions are met, there is a strong foundation for later work involving fine-tuning, biomedical embedding comparisons, retrieval optimization, or more sophisticated agentic verification.
