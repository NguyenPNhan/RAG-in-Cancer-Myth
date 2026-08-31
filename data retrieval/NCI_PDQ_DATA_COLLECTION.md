# NCI / Cancer.gov / PDQ Data Collection for the Cancer-Myth RAG Corpus

**Project:** Cancer Myth RAG  
**Primary source:** National Cancer Institute (NCI), Cancer.gov  
**Collection:** PDQ® Cancer Information Summaries  
**Collection script:** `nci_pdq_crawler.py`  
**Source review date:** 2026-08-31

---

## 1. Why NCI PDQ is the primary knowledge source

The National Cancer Institute's PDQ cancer information summaries are a strong Tier-1 source for a cancer-myth RAG system because they are authoritative, evidence-based, structured, referenced, and maintained by NCI editorial boards and subject-matter experts.

NCI describes its health-professional PDQ summaries as comprehensive, evidence-based, up-to-date cancer content. The summaries are written and reviewed by subject-matter experts, report the strength of evidence for interventions, and contain references to the scientific literature.

NCI also explicitly states that the PDQ summaries are **not clinical practice guidelines or treatment recommendations**. This distinction should be preserved in any publication or system description.

Official PDQ overview:

- https://www.cancer.gov/publications/pdq/information-summaries

The corpus is useful for the Cancer Myth task because false-premise questions commonly involve:

- whether a cancer is treatable;
- whether surgery is mandatory;
- whether surgery causes cancer to spread;
- whether a biopsy spreads cancer;
- whether all patients need chemotherapy or radiation;
- whether a colostomy, ostomy, dialysis, transplant, or disability is inevitable;
- incorrect assumptions about prognosis or hospice;
- screening misconceptions;
- prevention and risk-factor misconceptions;
- hereditary/genetic misconceptions;
- treatment adverse effects;
- supportive and palliative care;
- complementary or alternative therapies;
- pediatric cancers.

PDQ covers most of these information needs directly.

---

## 2. PDQ collections used

The crawler discovers pages from NCI-maintained index pages rather than maintaining a hard-coded list of individual cancer URLs.

### 2.1 Adult treatment

Index:

https://www.cancer.gov/publications/pdq/information-summaries/adult-treatment

NCI states that the adult treatment collection covers more than 70 common and rare adult cancers and contains information on cancer subtypes, staging, treatment, and prognosis.

This is the most important collection for treatment-related cancer myths.

Examples include:

- bladder cancer;
- breast cancer;
- cervical cancer;
- colorectal cancer;
- leukemia;
- lymphoma;
- liver cancer;
- lung cancer;
- melanoma;
- ovarian cancer;
- pancreatic cancer;
- prostate cancer;
- thyroid cancer;
- many rare cancers.

### 2.2 Pediatric treatment

Index:

https://www.cancer.gov/publications/pdq/information-summaries/pediatric-treatment

The pediatric collection includes major and rare childhood cancers and provides information on subtypes, staging, treatment, and prognosis.

This collection is required because the benchmark contains questions about childhood leukemia, brain tumors, neuroblastoma, retinoblastoma, sarcomas, germ-cell tumors, and other pediatric malignancies.

### 2.3 Screening

Index:

https://www.cancer.gov/publications/pdq/information-summaries/screening

The screening collection contains cancer-site-specific summaries and evidence about screening interventions.

Examples include:

- breast cancer screening;
- cervical cancer screening;
- colorectal cancer screening;
- liver cancer screening;
- lung cancer screening;
- prostate cancer screening;
- skin cancer screening;
- stomach cancer screening;
- thyroid cancer screening.

This collection is particularly important for false assumptions such as:

> Colonoscopy is the only way to screen for colorectal cancer.

or

> A cancer cannot be present if the patient has no symptoms.

### 2.4 Prevention

Index:

https://www.cancer.gov/publications/pdq/information-summaries/prevention

The prevention summaries cover risk factors and prevention interventions for a number of common cancers and discuss the strength of evidence supporting those interventions.

This collection supports questions about:

- tobacco;
- HPV;
- inherited versus modifiable risks;
- diet and supplements;
- cancer prevention myths;
- screening-versus-prevention confusion.

### 2.5 Genetics

Index:

https://www.cancer.gov/publications/pdq/information-summaries/genetics

The genetics collection covers the genetic basis of selected cancers, family-history risk, cancer susceptibility genes and syndromes, high-risk interventions, genetic counseling, and ethical/legal/social issues in genetic testing.

Examples include summaries concerning:

- breast and gynecologic cancer genetics;
- colorectal cancer genetics;
- gastric cancer genetics;
- kidney cancer genetics;
- prostate cancer genetics;
- skin cancer genetics;
- endocrine and neuroendocrine neoplasia;
- hereditary hematologic malignancies.

Some genetics summaries cover several cancer types at once. The crawler therefore keeps them under `general_topics/` rather than assigning them incorrectly to one cancer file.

### 2.6 Supportive and palliative care

Index:

https://www.cancer.gov/publications/pdq/information-summaries/supportive-care

This collection covers physical and psychosocial complications of cancer and treatment. Examples include:

- cancer pain;
- anxiety and distress;
- cognitive impairment;
- communication;
- depression;
- fatigue;
- grief and bereavement;
- hospice;
- last days of life;
- lymphedema;
- nausea and vomiting;
- nutrition;
- oral complications;
- end-of-life care;
- sleep disorders.

These summaries are cross-cutting rather than cancer-site-specific, so the crawler saves them under `general_topics/`.

### 2.7 Integrative, alternative, and complementary therapies

Index:

https://www.cancer.gov/publications/pdq/information-summaries/cam

This collection contains evidence summaries concerning complementary and alternative approaches, including topics such as acupuncture, aromatherapy, dietary supplements, cannabis/cannabinoids, curcumin, intravenous vitamin C, and other therapies.

This content is highly relevant to cancer-myth questions that incorrectly assume that an alternative therapy can replace conventional cancer treatment.

---

## 3. Why both Patient and Health Professional versions are collected

Where NCI provides both versions, both should be retained.

### Health Professional version

Advantages:

- greater clinical precision;
- staging details;
- treatment options;
- evidence summaries;
- prognosis information;
- references to primary literature;
- more detailed disease classification.

Primary RAG role:

**clinical grounding and myth correction.**

### Patient version

Advantages:

- simpler terminology;
- clearer explanation of treatments and tests;
- patient-oriented descriptions;
- useful language for the final answer.

Primary RAG role:

**patient-friendly explanation.**

The two versions must not be concatenated into one undifferentiated document. The `audience` metadata field is retained so retrieval or reranking can prefer the professional version while still retrieving patient text when useful.

---

## 4. Why the crawler discovers URLs from index pages

Individual NCI URLs should not be maintained manually.

A manually curated URL list becomes stale when NCI:

- adds a new summary;
- removes a summary;
- changes a URL;
- merges or separates disease summaries;
- adds a patient version;
- changes a disease name.

The crawler therefore starts from the NCI PDQ collection indexes and discovers the linked PDQ pages at crawl time.

Pipeline:

```text
NCI PDQ index pages
        |
        v
Discover current PDQ URLs
        |
        v
Deduplicate canonical URLs
        |
        +-------------------+
        |                   |
        v                   v
Patient version      Health Professional version
        |                   |
        +---------+---------+
                  |
                  v
          Download HTML
                  |
                  v
       Extract page metadata
                  |
                  v
      Preserve heading hierarchy
                  |
                  v
       Group by cancer/topic
                  |
                  v
       Write JSON + manifest
```

---

## 5. Collection method

### 5.1 HTTP retrieval

The crawler uses `requests.Session` with:

- a descriptive User-Agent;
- retry handling;
- exponential backoff;
- support for `Retry-After` on rate limiting;
- a configurable delay between requests;
- a configurable timeout.

The default delay is deliberately conservative.

For a sustained research crawl, replace the default User-Agent with a project identifier that includes a contact address or project URL.

Example:

```bash
python nci_pdq_crawler.py \
  --output data/nci_pdq \
  --user-agent "CancerMythRAG/1.0 (+https://example.edu/project)"
```

### 5.2 robots.txt

The script reads Cancer.gov's `robots.txt` before crawling and checks discovered URLs before requesting them.

If `robots.txt` cannot be retrieved, the crawler exits rather than silently bypassing it.

An `--ignore-robots` option exists only for situations where permission has been independently verified. It should not be used casually.

### 5.3 Retry behavior

The crawler retries common temporary failures, including:

- HTTP 429;
- HTTP 500;
- HTTP 502;
- HTTP 503;
- HTTP 504.

Permanent failures are added to the output manifest instead of silently disappearing.

---

## 6. HTML parsing

The live Cancer.gov page is treated as the canonical source document.

The crawler extracts text from the main page content and removes obvious interface elements such as:

- navigation;
- forms;
- scripts;
- styles;
- footer/aside elements.

The parser retains meaningful document elements:

- H1-H6 headings;
- paragraphs;
- list items;
- blockquotes;
- tables.

Tables are converted to plain text rows separated by `|` so that the information is retained for later retrieval.

---

## 7. Section hierarchy

Arbitrary fixed-width chunking is not performed during collection.

Instead, the collector first preserves the semantic hierarchy of the source document.

Example:

```text
Bladder Cancer Treatment
    -> General Information About Bladder Cancer
        -> Histopathology
        -> Diagnostics
        -> Prognostic Factors
    -> Stage Information for Bladder Cancer
    -> Treatment Option Overview
```

A section is stored as:

```json
{
  "section": "Diagnostics",
  "section_path": [
    "Bladder Cancer Treatment (PDQ...)...",
    "General Information About Bladder Cancer",
    "Diagnostics"
  ],
  "is_boilerplate": false,
  "block_count": 5,
  "text": "..."
}
```

This hierarchy should be preserved during later RAG chunking because clinical meaning is frequently dependent on disease stage, subtype, treatment setting, or section context.

---

## 8. Metadata collected for every PDQ page

Each page record contains approximately the following fields:

```json
{
  "source": "National Cancer Institute (NCI)",
  "source_domain": "cancer.gov",
  "collection": "PDQ",
  "collections": ["adult_treatment"],
  "topic": "adult_treatment",
  "audience": "health_professional",
  "language": "en",
  "cancer_type": "Bladder Cancer",
  "title": "Bladder Cancer Treatment (PDQ...)...",
  "summary_name": "Bladder Cancer Treatment",
  "url": "https://www.cancer.gov/...",
  "discovered_from": ["https://www.cancer.gov/.../adult-treatment"],
  "last_updated": "...",
  "retrieved_at": "...",
  "content_sha256": "...",
  "section_count": 42,
  "sections": []
}
```

### Important fields

`source`
: Provenance of the information.

`collection`
: Identifies the source as PDQ.

`collections`
: The NCI index collection(s) through which the page was discovered.

`topic`
: Treatment, screening, prevention, genetics, supportive care, etc.

`audience`
: `patient`, `health_professional`, or `unspecified`.

`cancer_type`
: Grouping key used for per-cancer JSON output when a safe assignment is possible.

`url`
: Canonical live Cancer.gov URL.

`last_updated`
: NCI's displayed update/review date when the parser can identify it.

`retrieved_at`
: Timestamp of this local crawl.

`content_sha256`
: Hash of extracted textual content. This allows later update detection.

`section_path`
: Hierarchical context for retrieval.

---

## 9. Output structure

A complete crawl produces a directory like:

```text
data/nci_pdq/
|
|-- manifest.json
|-- discovered_pages.json
|-- all_pages.json
|
|-- cancers/
|   |-- bladder_cancer.json
|   |-- breast_cancer.json
|   |-- cervical_cancer.json
|   |-- colon_cancer.json
|   |-- prostate_cancer.json
|   |-- ...
|
|-- general_topics/
|   |-- cancer_pain.json
|   |-- fatigue.json
|   |-- genetics_of_colorectal_cancer.json
|   |-- acupuncture.json
|   |-- ...
|
`-- raw_html/
    |-- <url-hash>.html
    `-- ...
```

### `cancers/`

Contains cancer-specific treatment, screening, and prevention pages grouped by the derived cancer name.

For example, if the titles align, `breast_cancer.json` may contain:

- Breast Cancer Treatment - Patient;
- Breast Cancer Treatment - Health Professional;
- Breast Cancer Screening - Patient;
- Breast Cancer Screening - Health Professional;
- Breast Cancer Prevention - Patient;
- Breast Cancer Prevention - Health Professional.

### `general_topics/`

Contains cross-cutting or multi-cancer summaries that cannot safely be assigned to one cancer type.

Examples:

- hospice;
- cancer pain;
- nutrition;
- fatigue;
- genetics of breast and gynecologic cancers;
- acupuncture;
- cancer therapy interactions with foods and supplements.

These documents should still be indexed in the final RAG database.

### `raw_html/`

Contains the original downloaded HTML snapshot for reproducibility and parser development.

Do not repeatedly download the website when merely changing the parsing or chunking logic. Use the stored HTML with:

```bash
python nci_pdq_crawler.py --output data/nci_pdq --use-cache
```

---

## 10. Per-cancer JSON format

Example structure:

```json
{
  "schema_version": "1.0",
  "source": "National Cancer Institute (NCI)",
  "collection": "PDQ",
  "cancer_type": "Bladder Cancer",
  "generated_at": "2026-08-31T...Z",
  "page_count": 4,
  "pages": [
    {
      "topic": "adult_treatment",
      "audience": "health_professional",
      "title": "Bladder Cancer Treatment ...",
      "url": "...",
      "last_updated": "...",
      "sections": [
        {
          "section": "Treatment Option Overview",
          "section_path": ["..."],
          "text": "..."
        }
      ]
    }
  ]
}
```

The file remains a source-document representation. Embedding chunks should be generated as a later processing step rather than modifying the canonical downloaded corpus.

---

## 11. Cancer-name normalization

Cancer naming is not perfectly uniform across all NCI collections.

For example, NCI may use related labels such as:

- gastric cancer;
- stomach (gastric) cancer;
- renal cell cancer;
- kidney (renal cell) cancer;
- hepatocellular cancer;
- liver cancer.

The crawler performs only a **small number of high-confidence canonical aliases**.

This is intentional.

Aggressive automatic normalization can incorrectly merge distinct diseases, especially for:

- leukemia subtypes;
- lymphoma subtypes;
- brain and CNS tumors;
- germ-cell tumors;
- sarcomas;
- head and neck cancers;
- adult versus childhood diseases.

For research reproducibility, preserve NCI's original `summary_name` and `title` even when a `cancer_type` grouping key is normalized.

A later ontology-mapping step can map NCI labels to a controlled vocabulary such as NCI Thesaurus without altering the raw source records.

---

## 12. Raw corpus versus RAG chunks

Do not use the downloaded JSON directly as the final embedding unit.

Recommended processing pipeline:

```text
Raw NCI JSON
    |
    v
Remove/flag boilerplate
    |
    v
Section-aware chunking
    |
    v
Add inherited metadata
    |
    v
Embedding
    |
    +------> dense vector index
    |
    +------> BM25 / sparse index
    |
    v
Hybrid retrieval + reranking
```

Recommended initial chunking parameters:

- target: 600-900 tokens;
- maximum: approximately 1,200 tokens;
- overlap: 80-120 tokens;
- do not overlap across major section boundaries;
- always retain `section_path`;
- retain the original URL and update date on every chunk.

For very short sections, adjacent blocks under the same heading may be merged.

---

## 13. Recommended retrieval metadata

A downstream chunk should inherit at least:

```json
{
  "source": "NCI",
  "collection": "PDQ",
  "audience": "health_professional",
  "cancer_type": "Bladder Cancer",
  "topic": "adult_treatment",
  "title": "Bladder Cancer Treatment",
  "section": "...",
  "section_path": ["..."],
  "url": "...",
  "last_updated": "...",
  "retrieved_at": "...",
  "chunk_index": 3,
  "text": "..."
}
```

Do not discard provenance after embedding.

---

## 14. Retrieval strategy for the Cancer Myth task

The task is not simply question answering. Many queries contain a false premise.

For example:

> My father has muscle-invasive bladder cancer and believes surgery is the only treatment...

The model should not accept the assumption and answer only the secondary request. It should first determine whether the premise is medically supported.

Recommended retrieval sequence:

```text
User question
    |
    v
Cancer/type recognition
    |
    v
Retrieve cancer-specific PDQ chunks
    |
    +--> Health Professional evidence
    |
    +--> Patient explanation
    |
    +--> Cross-cutting supportive/genetics/CAM evidence when needed
    |
    v
Rerank evidence
    |
    v
Premise assessment
    |
    +--> supported
    +--> unsupported / myth
    +--> context-dependent
    |
    v
Generate corrected, patient-friendly answer
```

A useful prompting rule for the answer generator is:

> Before answering the user's explicit request, identify any medically important assumption in the question. If the retrieved evidence contradicts that assumption, correct it clearly before addressing the secondary request.

---

## 15. Health Professional versus Patient retrieval weighting

For factual adjudication, a reasonable first configuration is:

```text
health_professional: 1.00
patient:             0.80-0.90
```

This is not an evidence-quality score. It is a retrieval preference reflecting the greater clinical detail of the professional summaries.

The generator can still use the patient version to phrase the final explanation clearly.

---

## 16. Update strategy

Cancer information changes over time. A healthcare RAG corpus should not be treated as static.

Each record stores:

- `last_updated` from the NCI page when detectable;
- `retrieved_at` from the local crawl;
- `content_sha256` of extracted text.

Recommended refresh workflow:

```text
Re-crawl index
    |
    v
Compare discovered URLs with previous manifest
    |
    +--> new URL -> download
    |
    +--> removed URL -> mark inactive
    |
    v
Fetch current source
    |
    v
Compare content hash / update date
    |
    +--> unchanged -> keep existing embeddings
    |
    `--> changed -> re-parse and re-embed affected chunks
```

For a research benchmark, save the crawl date and corpus version used for every experiment.

---

## 17. Provenance and reproducibility

Every experimental result should be traceable to:

1. the original Cancer.gov URL;
2. the retrieval date;
3. the NCI update date when available;
4. the raw HTML snapshot;
5. the parser version;
6. the chunking version;
7. the embedding model/version;
8. the vector-index version.

A model output without source provenance is difficult to audit in a medical setting.

---

## 18. Copyright and reuse

NCI's current reuse policy states that, unless otherwise indicated, text in NCI products is free of copyright and may be reused without permission, with the National Cancer Institute credited as the source.

Official policy:

https://www.cancer.gov/policies/copyright-reuse

The same page notes that graphics may have separate copyright restrictions. Therefore this crawler is designed as a **text-first corpus** and does not download images as RAG content.

For digital reproduction, follow NCI's attribution and linking requirements.

The NCI logo and the PDQ trademark have additional restrictions and should not be reused as branding for the RAG system.

---

## 19. Why live Cancer.gov pages are used instead of PDQ XML

NCI describes an XML dissemination service for content partners. Its current syndication page states that the cancer information distributed through that mechanism is not currently being updated and that NCI is not accepting new syndication partners.

Official page:

https://www.cancer.gov/syndication

For this project, the practical reproducible source is therefore the **live Cancer.gov PDQ webpages discovered through the official PDQ indexes**, rather than relying on access to the XML partner feed.

If NCI later provides a current public bulk API or downloadable structured dataset, that route should be reconsidered because structured first-party data is preferable to HTML parsing.

---

## 20. Installation and execution

Install dependencies:

```bash
pip install requests beautifulsoup4
```

Run all configured PDQ collections:

```bash
python nci_pdq_crawler.py --output data/nci_pdq
```

Run only adult and pediatric cancer treatment pages:

```bash
python nci_pdq_crawler.py \
  --collections adult_treatment pediatric_treatment \
  --output data/nci_pdq_treatment
```

Run treatment + screening + prevention only:

```bash
python nci_pdq_crawler.py \
  --collections adult_treatment pediatric_treatment screening prevention \
  --output data/nci_pdq_cancer_specific
```

Reuse previously downloaded HTML while modifying parsing code:

```bash
python nci_pdq_crawler.py \
  --output data/nci_pdq \
  --use-cache
```

Use a slower request rate:

```bash
python nci_pdq_crawler.py \
  --output data/nci_pdq \
  --delay 1.5
```

---

## 21. Validation after crawling

Do not assume a crawl is successful merely because the program exits.

Check `manifest.json` for:

- number of discovered URLs;
- successful pages;
- failed pages;
- number of cancer files;
- number of general-topic files;
- failure reasons.

Then manually inspect several diseases spanning different structures, for example:

- bladder cancer;
- breast cancer;
- colorectal cancer;
- acute lymphoblastic leukemia;
- lymphoma;
- lung cancer;
- melanoma;
- pancreatic cancer;
- a childhood brain tumor;
- a rare cancer.

For each sample, verify:

- title;
- audience;
- source URL;
- update date;
- heading hierarchy;
- list extraction;
- table extraction;
- absence of navigation/footer contamination;
- no major missing sections.

The parser should be regression-tested whenever Cancer.gov changes its HTML templates.

---

## 22. Known limitations

### HTML structure can change

The crawler does not use an official bulk PDQ API. If Cancer.gov changes its templates, parser behavior must be revalidated.

### Cancer names are not a perfect ontology

The filename grouping is designed for practical RAG ingestion, not as a formal cancer taxonomy.

### General summaries are not duplicated into every cancer file

Supportive-care, genetics, and complementary-therapy pages may apply to multiple cancers. Duplicating them into every cancer file would create unnecessary duplication and retrieval bias. They are stored once under `general_topics/` and should be included in the global index.

### An NCI summary is not an individualized treatment recommendation

PDQ is an evidence information resource. The final RAG system must not transform general evidence summaries into unsupported patient-specific recommendations.

### References can dominate retrieval

Health-professional pages may contain long reference sections. The collector retains them for provenance, but they are marked by section hierarchy and can be filtered or downweighted during embedding.

---

## 23. Recommended next preprocessing step

After this collection step, create a second script such as:

```text
prepare_nci_rag_chunks.py
```

Its responsibilities should be:

1. read every file under `cancers/` and `general_topics/`;
2. remove or downweight `is_boilerplate=true` sections;
3. split long sections into 600-900-token chunks;
4. inherit page and section metadata;
5. assign stable `document_id` and `chunk_id` values;
6. output one JSONL file for indexing;
7. optionally generate a BM25 corpus and embedding corpus from the same canonical chunks.

Recommended final chunk format:

```json
{
  "chunk_id": "nci_pdq_bladd..._0003",
  "source": "NCI",
  "collection": "PDQ",
  "audience": "health_professional",
  "cancer_type": "Bladder Cancer",
  "topic": "adult_treatment",
  "title": "Bladder Cancer Treatment",
  "section": "Treatment of Stage II and Stage III Bladder Cancer",
  "section_path": [
    "Bladder Cancer Treatment",
    "Treatment of Stage II and Stage III Bladder Cancer"
  ],
  "url": "https://www.cancer.gov/...",
  "last_updated": "...",
  "text": "..."
}
```

This chunk corpus, rather than the raw crawler output, should be sent to the embedding/indexing stage.

---

## 24. Official source URLs

- PDQ overview: https://www.cancer.gov/publications/pdq/information-summaries
- Adult treatment: https://www.cancer.gov/publications/pdq/information-summaries/adult-treatment
- Pediatric treatment: https://www.cancer.gov/publications/pdq/information-summaries/pediatric-treatment
- Screening: https://www.cancer.gov/publications/pdq/information-summaries/screening
- Prevention: https://www.cancer.gov/publications/pdq/information-summaries/prevention
- Genetics: https://www.cancer.gov/publications/pdq/information-summaries/genetics
- Supportive and palliative care: https://www.cancer.gov/publications/pdq/information-summaries/supportive-care
- Integrative/complementary therapies: https://www.cancer.gov/publications/pdq/information-summaries/cam
- NCI reuse policy: https://www.cancer.gov/policies/copyright-reuse
- NCI syndication information: https://www.cancer.gov/syndication
