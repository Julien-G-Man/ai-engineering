# Hugging Face Examples — Learning Guide

This folder is a compact, concept-first playground for Hugging Face. It was built around the DataCamp "Associate AI Engineer for Developers" track (chapter 4), and is meant to teach the ideas you need to understand models, pipelines, and document workflows — not just copy-paste snippets. The goal: anyone should be able to open this repo years from now and learn the concepts and trade-offs.

---

## Learning goals
- Understand what a `pipeline` does and when to use it.
- Know the difference between `Auto*` classes and explicit model classes.
- See the two flavors of summarization and when to choose each.
- Learn why document QA is different from text QA and how retrieval fits in.

---

## Core concepts 

- **Pipelines**: high-level wrappers for common tasks (generation, classification, summarization). They wire model, tokenizer, and pre/post-processing so you can iterate quickly.
- **Auto classes**: `AutoModel`, `AutoTokenizer`, and friends let you load a model from an ID without hardcoding the class. Use them for portability and to avoid mismatches between model IDs and class names. See [01.1-huggingface/pipeline_with_autoclasses.py](01.1-huggingface/pipeline_with_autoclasses.py).
- **Model vs tokenization vs config**: Models hold weights and layer definitions; tokenizers translate text ↔ token ids; configs describe hyperparameters and architecture choices. All three can be loaded independently with `from_pretrained`.
- **Document QA vs Text QA**: Document QA often needs layout and image inputs (scanned pages) and uses layout-aware models (LayoutLM family). Text QA is span-based over plain text. If your pipeline complains about `image` or model compatibility, you're mixing these modes.
- **Summarization types**: Extractive summarizers pick sentences from the input; abstractive summarizers rewrite content in new words. Use extractive for fidelity and abstractive for readability or compression. See [01.1-huggingface/text_summarization.py](01.1-huggingface/text_summarization.py).
- **Retrieval + generation**: For long documents, retrieve relevant chunks (via embeddings or simple heuristics) and then prompt a generator. This is more reliable and cheaper than sending entire documents to a model.

---

## What’s in this folder

- [01.1-huggingface/pipeline_intro.py](01.1-huggingface/pipeline_intro.py) — quick text-generation demo using `pipeline`.
- [01.1-huggingface/pipeline_with_autoclasses.py](01.1-huggingface/pipeline_with_autoclasses.py) — `AutoModel` + `AutoTokenizer` example for sentiment.
- [01.1-huggingface/auto_models_and_tokenizers.py](01.1-huggingface/auto_models_and_tokenizers.py) — note and tiny snippet showing `AutoModelForSequenceClassification`.
- [01.1-huggingface/text_classification.py](01.1-huggingface/text_classification.py) — multiple classification recipes (sentiment, grammar, zero-shot).
- [01.1-huggingface/text_summarization.py](01.1-huggingface/text_summarization.py) — extractive vs abstractive examples and notes.
- [01.1-huggingface/document_qna.py](01.1-huggingface/document_qna.py) — PDF-to-text + a pragmatic text-only QA flow (retrieval is recommended; see notes).
- [01.1-huggingface/inference_providers.py](01.1-huggingface/inference_providers.py) and [01.1-huggingface/deepseek_v4_pro.py](01.1-huggingface/deepseek_v4_pro.py) — examples using `huggingface_hub.InferenceClient` and provider-specific chat calls.
- [01.1-huggingface/datasets_manipulation.py](01.1-huggingface/datasets_manipulation.py) — starting point for loading/slicing datasets.

---

## How to run examples (practical)

1. Activate the virtualenv (Windows PowerShell):

```powershell
& venv\Scripts\Activate.ps1
```

2. Install dependencies from the repo root:

```powershell
pip install -r requirements.txt
```

3. Optional: create a `.env` with `HF_TOKEN=your_token` to avoid rate limits and speed up downloads.

4. Run a file, e.g.:

```powershell
python 01.1-huggingface\pipeline_intro.py
python 01.1-huggingface\text_summarization.py
python 01.1-huggingface\document_qna.py
```

Notes: large models download on first run and can take minutes. Use smaller demo models when experimenting.
