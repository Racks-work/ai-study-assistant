[README.md](https://github.com/user-attachments/files/33441895/README.md)
# 📚 StudyLens — AI Study Assistant

A **document-grounded study application** built with Python and Streamlit. Upload PDF, TXT, or Markdown notes; search them using TF-IDF retrieval; optionally generate AI answers with source citations; and practice with automatically generated fill-in-the-blank questions.

## Features

- **Upload and parse** text-based PDF, TXT, and Markdown documents (12 MB per file).
- **Ask questions** with ranked excerpts and file/page references in **free offline mode**.
- **Optional OpenAI-powered answers** grounded in retrieved passages, with numbered citations.
- **Practice quiz** generated locally from sentences in the uploaded notes, with revealable answers.
- **Browse source passages** and see which document/page each excerpt came from.
- **Responsive Streamlit UI**, dark theme, session-based uploads, and basic automated tests.

> **Accuracy note:** AI answers may still contain mistakes. Verify important claims against the cited source excerpts. The offline mode is an extractive search tool, not a generative AI model.

## Quick start

Requires **Python 3.10+**.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Streamlit displays a local address, usually `http://localhost:8501`.

**Try it:** Upload `samples/intro_to_ml.txt`, ask “What is overfitting?”, and open the **Practice quiz** tab.

## Optional AI answer mode

1. Select **AI answer (OpenAI API)** in the sidebar.
2. Enter your own API key (or set the `OPENAI_API_KEY` environment variable).
3. Use a model your account can access; the default is `gpt-4o-mini`.

**API usage may incur charges.** An API key is **not needed** for offline document search or quizzes. Never commit API keys or `.env` files. Relevant document excerpts and your question are transmitted to OpenAI when you use AI answer mode. Do not upload private or sensitive documents without permission.

## Architecture

```text
Upload PDF/TXT/MD
       ↓
Extract selectable text + preserve PDF page references
       ↓
Overlapping text chunks
       ↓
TF-IDF vector search (scikit-learn)
       ↓
Top matching excerpts
       ├── Offline: show literal excerpts and source labels
       └── AI: send excerpts + question to OpenAI, request grounded answer with citations
```

Quiz generation is a separate local pipeline that converts source sentences into cloze questions.

## Project structure

```text
ai-study-assistant/
├── app.py
├── study_assistant/
│   ├── __init__.py
│   ├── core.py
│   └── llm.py
├── tests/test_core.py
├── samples/intro_to_ml.txt
├── .streamlit/config.toml
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Tests

```bash
python -m unittest discover -s tests -v
python -m compileall -q app.py study_assistant
```

## Known limitations / next steps

- Scanned PDFs without embedded text need OCR (not implemented).
- TF-IDF is lexical retrieval, so paraphrased questions may miss relevant passages; future work: embeddings and vector search.
- Citations refer to retrieved excerpts; the app does not independently fact-check AI output.
- The quiz is deterministic cloze practice, not adaptive assessment.
- Uploads are held in the active Streamlit session, not stored in a user account or database. A hosted Streamlit server processes uploads in its environment.
- There is no login, persistent history, or multi-user data isolation beyond Streamlit session state; do not use this as a sensitive-data service.

## GitHub publishing

Create a repository named `ai-study-assistant` and upload **the contents of this folder** (not the ZIP itself). Keep the directory structure. Add the repository link to your portfolio and include screenshots after you run the app locally.

## Resume / portfolio description

**StudyLens — AI Study Assistant** | Python, Streamlit, scikit-learn, OpenAI API

Built a document-grounded study application supporting PDF/text uploads, TF-IDF retrieval, optional LLM-generated answers with source citations, and automated cloze quizzes. Added modular Python code, privacy-conscious API configuration, and unit tests.

## License

MIT — see `LICENSE`.
