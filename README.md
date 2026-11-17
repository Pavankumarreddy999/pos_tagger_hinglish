# Hinglish Part-of-Speech (POS) Tagging with Classical NLP & Machine Learning

An end-to-end Part-of-Speech (POS) tagging architecture specifically engineered for code-mixed Hindi-English (Hinglish) text. This project integrates 5 diverse benchmark datasets (**Kaggle, LingoIITGN / COMI-LINGUA, ICON 2016 Jamatia, LinCE pos_hineng, and Vyas et al. 2014 Facebook**) into a unified 80/10/10 corpus. It evaluates five classical statistical and machine learning sequence taggers plus dual ensemble strategies (Unweighted Majority Voting and Validation-Tuned Weighted Voting).

---

## 📌 Table of Contents
- [Project Overview](#project-overview)
- [Key Features & Innovations](#key-features--innovations)
- [Dataset Architecture](#dataset-architecture)
- [Methodology & Pipeline](#methodology--pipeline)
- [Models & Sequence Decoding](#models--sequence-decoding)
- [Ensemble Strategies](#ensemble-strategies)
- [Comprehensive Evaluation Suite](#comprehensive-evaluation-suite)
- [Getting Started & Usage](#getting-started--usage)
- [Interactive Sentence Predictor](#interactive-sentence-predictor)
- [Project Structure](#project-structure)
- [Future Roadmap](#future-roadmap)

---

## 🚀 Project Overview

Code-mixed social media text in India (WhatsApp, Twitter, Instagram, YouTube comments) presents severe Natural Language Processing challenges:
1. **Phonetic Transliteration**: No standard spelling dictionary exists for Romanized Hindi (`accha`, `acha`, `achha`, `acchaaa`).
2. **Emotional Character Elongation**: Words are stretched for emphasis (`masttttt`, `bohooot`, `theeeeek`).
3. **Lexical & Cross-Lingual Ambiguities**: The same word can have different grammatical roles across languages (e.g. *"main"* as a Hindi pronoun `PRON` vs. an English adjective `ADJ`).
4. **High Out-of-Vocabulary (OOV) Rates**: Unseen words, informal slang, and typos are frequent.
5. **Cross-Source Annotation Divergences**: Variations in annotation guidelines across research datasets (e.g., copulas like *`hai`* annotated as `AUX` vs. `VERB`).

This project addresses these challenges using non-destructive feature engineering, token-level Language Identification, training-time noise augmentation, subword morphological signatures, tuned sequence models, and statistical significance testing.

---

## 💡 Key Features & Innovations

- **Multi-Source Label Consistency Audit**:
  Cross-source tag distribution analysis for frequent anchor words (`hai`, `hoon`, `raha`, `main`, `tha`, `ye`) across all 5 benchmark sources, revealing systemic differences in `AUX/VERB`, `PRON/DET/ADP`, and `PROPN/NOUN`.
- **Data Leakage & Duplicate Audit**:
  Exact string and near-duplicate (token Jaccard similarity $\ge 0.85$) verification between train and test splits confirming $>99\%$ novel test instances.
- **Non-Destructive Token Normalization**:
  Maintains `raw_word` and `norm_word` as separate features (no destructive overwriting). English digraphs (`ee`, `oo`) are strictly preserved (preventing `meet` $\to$ `mit`, `cool` $\to$ `cul`).
- **Token-Level Language-ID (LID)**:
  Subword and script-based classifier categorizing tokens into `HI` (Hindi), `EN` (English), or `UNIV` (Universal/Punctuation), directly resolving cross-lingual ambiguities.
- **Noise Augmentation on Training Data**:
  Synthetically generates elongated copies of training tokens (`accha` $\to$ `acchaaa`, `mast` $\to$ `masttt`) with identical tags, training classifiers to handle informal social media elongations without hardcoded word dictionaries.
- **Fine-Grained Morphological OOV Signatures**:
  Stanford/Collins-style linguistic equivalence classes for unseen words (`<UNK_NUM>`, `<UNK_CAP>`, `<UNK_ALLCAPS>`, `<UNK_SUFF_kar>`, `<UNK_SUFF_ing>`, etc.).
- **Tuned Linear-Chain CRF**:
  L-BFGS optimization ($c_1=0.1, c_2=0.05, 100\text{ iterations}$), $\pm 2$ word window (`w-2` to `w+2`), word bigrams (`w-1+w`, `w+w+1`), character $n$-gram prefixes (1..3) and suffixes (1..4).
- **Beam Search Decoding for MEMM**:
  Replaces greedy 1-best decoding with Beam Search ($B=3$) to avoid cascading errors.
- **Dual Ensembles**:
  Both **Unweighted Majority Voting** and **Validation-Tuned Weighted Voting** with CRF tie-breaking.
- **Exhaustive Evaluation Suite**:
  Includes per-source breakdown, Leave-One-Source-Out (LOSO) cross-domain generalization, In-Vocab vs. OOV slices, Clean vs. Noisy slices, per-tag classification reports, and $B=1,000$ non-parametric Bootstrap Confidence Intervals.

---

## 📊 Dataset Architecture

All sources are unified under the `dataset/` directory with a standardized **80% Train / 10% Validation / 10% Test** split:

| Dataset Source | Path | Total Sentences | Annotation Standard | Description |
|---|---|---|---|---|
| **Kaggle Hinglish** | `dataset/kaggle/` | ~55,000 | Universal Dependencies (UPOS) | Large social media corpus; `hai`/`raha` as `AUX` |
| **LingoIITGN / COMI-LINGUA** | `dataset/lingoiitgn_comi_lingua/` | ~30,585 | Standardized UPOS | High-quality code-mixed text from IITGN |
| **ICON 2016 Jamatia et al.** | `dataset/icon_2016_jamatia/` | ~1,271 | Mapped to UPOS | Facebook, Twitter, WhatsApp chats; informal |
| **LinCE Benchmark (pos_hineng)**| `dataset/lince_pos_hineng/` | ~1,190 | Mapped to UPOS | Recognized benchmark for code-switching |
| **Vyas et al. 2014 (Facebook)** | `dataset/vyas_2014_facebook/` | ~748 | Mapped to UPOS | Foundational academic Facebook Hinglish corpus |
| **Merged Corpus** | `dataset/merged/` | **88,225 total** | **80 / 10 / 10 Split** | **65,230 Train / 11,517 Val / 11,478 Test** |

All CSV files follow the schema:
`sentence` (string), `tokens` (JSON list), `pos_tags` (JSON list of UPOS tags), `source` (dataset identifier).

---

## ⚙️ Methodology & Pipeline

```
Raw Hinglish Sentence
         │
         ▼
[1. Punctuation-Preserving Regex Tokenizer]  --> r"\w+|[^\w\s]"
         │
         ▼
[2. Non-Destructive Normalizer & LID]        --> raw_word, norm_word, lang_id (HI/EN/UNIV)
         │
         ▼
[3. OOV Morphological Signature Mapping]     --> <UNK_CAP>, <UNK_NUM>, <UNK_SUFF_kar>, etc.
         │
         ▼
[4. Enriched Feature Extraction]             --> Context window ±2, word bigrams, subwords
         │
         ▼
[5. Model Inference]                         --> HMM, MEMM (Beam), CRF, LR, Naive Bayes
         │
         ▼
[6. Dual Ensemble Engine]                    --> Unweighted Majority Vote & Validation-Weighted Vote
         │
         ▼
Final Universal Dependencies POS Tags
```

---

## 🤖 Models & Sequence Decoding

| Model | Model Family | Sequential Decoding Method | Key Implementation Details |
|---|---|---|---|
| **HMM** | Generative Graphical Model | Viterbi Dynamic Programming | Second-order transitions, Laplace smoothing, morphological signature emission smoothing |
| **MEMM** | Discriminative Local Normalization | **Beam Search ($B=3$)** | Multinomial Logistic Regression with `prev_tag` feature; beam search prevents greedy error propagation |
| **CRF** | Discriminative Global Normalization | Viterbi on Global Factors | Linear-chain CRF, L-BFGS, $c_1=0.1, c_2=0.05, 100$ iterations, $\pm 2$ window, bigrams, subwords |
| **Logistic Regression** | Discriminative Independent | Independent Argmax | Multi-class Maximum Entropy with DictVectorizer |
| **Naive Bayes** | Generative Independent | Independent Argmax | Multinomial Naive Bayes on DictVectorizer sparse feature matrix |

---

## 🤝 Ensemble Strategies

1. **Ensemble (Unweighted Majority Voting)**:
   - Every model receives 1 equal vote:
     $$\text{Votes}(c) = \sum_{m=1}^5 \mathbb{I}(\hat{y}_m = c)$$
   - The tag with the highest vote count is selected.
   - Ties are broken by the top individual validation model (**CRF**).

2. **Ensemble (Validation-Tuned Weighted Voting)**:
   - Model weights $w_m$ are dynamically calculated from validation macro F1 performance:
     $$w_m = \frac{\text{Val-Macro-F1}_m}{\sum_{k=1}^5 \text{Val-Macro-F1}_k}$$
   - Weighted score for candidate tag $c$:
     $$\text{Score}(c) = \sum_{m=1}^5 w_m \cdot \mathbb{I}(\hat{y}_m = c)$$
   - Ties are broken by the top individual validation model (**CRF**).

---

## 📈 Comprehensive Evaluation Suite

The notebook runs an exhaustive battery of evaluations on the test set:

1. **Overall Performance Metrics (Validation & Test)**:
   Accuracy, Micro F1, Macro F1, Weighted F1, Precision, and Recall across all 5 individual models and both ensembles.
2. **Per-Source Breakdown**:
   Evaluates each model independently across all 5 test sources (Kaggle, COMI-LINGUA, ICON 2016, LinCE, Vyas 2014).
3. **Leave-One-Source-Out (LOSO) Cross-Domain Generalization Test**:
   Models are trained strictly on Kaggle + COMI-LINGUA, and tested on out-of-domain sources (ICON 2016, LinCE, Vyas 2014) to measure zero-shot domain adaptation.
4. **In-Vocabulary vs. Out-of-Vocabulary (OOV) Evaluation**:
   Measures accuracy separately on seen training vocabulary words vs. unseen words.
5. **Clean vs. Noisy / Elongated Token Evaluation**:
   Measures accuracy on clean standard words vs. informal elongated words or digit-embedded slang.
6. **Per-Tag Classification Report**:
   Support, precision, recall, and F1-score across all 17 Universal Dependencies UPOS categories:
   `NOUN`, `PROPN`, `VERB`, `AUX`, `ADJ`, `ADV`, `PRON`, `ADP`, `DET`, `CCONJ`, `SCONJ`, `PART`, `NUM`, `PUNCT`, `SYM`, `INTJ`, `X`.
7. **Non-Parametric Bootstrap Significance Testing ($B=1,000$)**:
   Computes empirical 95% Confidence Intervals ($[\text{CI}_{2.5\%}, \text{CI}_{97.5\%}]$) for test accuracy, paired difference ($\Delta = \text{Acc}_{\text{Ensemble}} - \text{Acc}_{\text{CRF}}$), and $p$-values.
8. **Row-Normalized Confusion Matrix Heatmaps**:
   Identifies precise cross-category confusion pairs (e.g. `AUX` vs. `VERB`, `PRON` vs. `DET`).

---

## 💻 Getting Started & Usage

### Prerequisites
Python 3.10+ with the following packages installed:
```bash
pip install numpy pandas scikit-learn sklearn-crfsuite matplotlib seaborn
```

### Running the Notebook
Open and run all cells in Jupyter Notebook, JupyterLab, or VS Code:
```bash
jupyter notebook Hinglish_POS_Tagging_NLP_ML.ipynb
```

---

## 🔍 Interactive Sentence Predictor

Section 15 of the notebook allows interactive predictions for any custom Hinglish sentence using Python's standard `input()`:

```python
# Cell 15.1 Prompt
sentence = input("Enter a Hinglish sentence: ")
```

### Example Input:
```text
wah bhai ye acchaaa phone h, bohot mast chal raha hai!
```

### Example Output:
```text
=====================================================================================
INPUT HINGLISH SENTENCE: wah bhai ye acchaaa phone h, bohot mast chal raha hai!
=====================================================================================
Total Tokens (13): ['wah', 'bhai', 'ye', 'acchaaa', 'phone', 'h', ',', 'bohot', 'mast', 'chal', 'raha', 'hai', '!']

-------------------------------------------------------
PREDICTIONS FROM EACH AND EVERY MODEL:
-------------------------------------------------------
• HMM                 : wah/INTJ bhai/NOUN ye/PRON acchaaa/ADJ phone/NOUN h/AUX ,/PUNCT bohot/ADV mast/ADJ chal/VERB raha/AUX hai/AUX !/PUNCT
• MEMM (Beam)         : wah/INTJ bhai/NOUN ye/PRON acchaaa/ADJ phone/NOUN h/AUX ,/PUNCT bohot/ADV mast/ADJ chal/VERB raha/AUX hai/AUX !/PUNCT
• CRF                 : wah/INTJ bhai/NOUN ye/PRON acchaaa/ADJ phone/NOUN h/AUX ,/PUNCT bohot/ADV mast/ADJ chal/VERB raha/AUX hai/AUX !/PUNCT
• Logistic Regression : wah/INTJ bhai/NOUN ye/PRON acchaaa/ADJ phone/NOUN h/AUX ,/PUNCT bohot/ADV mast/ADJ chal/VERB raha/AUX hai/AUX !/PUNCT
• Naive Bayes         : wah/INTJ bhai/NOUN ye/PRON acchaaa/ADJ phone/NOUN h/AUX ,/PUNCT bohot/ADV mast/ADJ chal/VERB raha/AUX hai/AUX !/PUNCT

=====================================================================================
FINAL POS TAGS: ENSEMBLE MAJORITY VOTING (UNWEIGHTED):
=====================================================================================
>> Formatted Sentence: wah/INTJ bhai/NOUN ye/PRON acchaaa/ADJ phone/NOUN h/AUX ,/PUNCT bohot/ADV mast/ADJ chal/VERB raha/AUX hai/AUX !/PUNCT

=====================================================================================
FINAL POS TAGS: ENSEMBLE (VALIDATION-WEIGHTED VOTING):
=====================================================================================
>> Formatted Sentence: wah/INTJ bhai/NOUN ye/PRON acchaaa/ADJ phone/NOUN h/AUX ,/PUNCT bohot/ADV mast/ADJ chal/VERB raha/AUX hai/AUX !/PUNCT
```
A detailed token-by-token comparison table is also displayed, showing normalized form, Language-ID, OOV status, individual model tags, both ensemble tags, and model agreement percentage.

---

## 📁 Project Structure

```
NLP_PROJ/
├── dataset/
│   ├── icon_2016_jamatia/            # ICON 2016 split CSVs
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   ├── kaggle/                       # Kaggle Hinglish split CSVs
│   │   ├── train_sentences_tokens_pos (1).csv
│   │   ├── val_sentences_tokens_pos (1).csv
│   │   └── test_sentences_tokens_pos (1).csv
│   ├── lince_pos_hineng/             # LinCE Benchmark split CSVs
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   ├── lingoiitgn_comi_lingua/        # COMI-LINGUA split CSVs
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   ├── vyas_2014_facebook/           # Vyas et al. Facebook split CSVs
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── merged/                       # Unified 80/10/10 deduplicated dataset
│       ├── merged_train.csv
│       ├── merged_val.csv
│       └── merged_test.csv
├── Hinglish_POS_Tagging_NLP_ML.ipynb  # Primary complete pipeline notebook
└── README.md                         # Project documentation
```

---

## 🔮 Future Roadmap

- **Additional Gold-Standard Corpora**:
  Integration of the **Code-Switched Universal Dependencies (UD) Hindi-English Treebank** (Bhat et al.) and the **Singh et al. Twitter POS Corpus**.
- **Semi-Supervised Self-Training (Pseudo-Labeling)**:
  Extracting raw text from **HingCorpus** (Naylor / LingoIITGN) and iteratively retraining the ensemble on high-confidence pseudo-labels ($>95\%$).
- **Subword Pretrained Transformers**:
  Fine-tuning multilingual transformers (`mBERT`, `XLM-RoBERTa`, `IndicBERT`) using the handcrafted morphological and Language-ID features as auxiliary input signals.

---

## 📜 Citation & References

- **COMI-LINGUA**: LingoIITGN, *A Large-Scale Hinglish Dataset for Code-Mixed NLP*.
- **LinCE Benchmark**: Aguilar et al., *LinCE: A Centralized Benchmark for Linguistic Code-switching Evaluation*.
- **ICON 2016**: Jamatia et al., *Part-of-Speech Tagging for Hindi-English Code-Mixed Social Media Text*.
- **Vyas et al. (2014)**: *POS Tagging of English-Hindi Code-Mixed Social Media Content*, EMNLP 2014.
- **Universal Dependencies**: Nivre et al., *Universal Dependencies v2: An Overview*.
