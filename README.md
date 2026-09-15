# Custom Chatbot — RAG over 2023 Fashion Trends

A question-answering chatbot that grounds `gpt-3.5-turbo-instruct` in a dataset the
model has never seen, using embedding-based retrieval to inject relevant context into
the prompt.

## Why this dataset

`data/2023_fashion_trends.csv` holds 82 rows of editorial fashion commentary from
Refinery29, Who What Wear and similar outlets, each with its source URL.

It is a good fit precisely because the model *cannot* know it. `gpt-3.5-turbo-instruct`
has a 2021 training cutoff, so every question about 2023 runway trends falls outside
its parametric knowledge — the base model has no option but to guess. That makes the
contribution of retrieval unambiguous.

## Results

Each question is asked twice — once as a bare completion, once through the
retrieval-augmented prompt.

| Question | Without context | With context |
|---|---|---|
| "I want a bold color trend from 2023 runways — what color should I pick?" | "…tangerine orange" (invented) | **Cobalt blue** |
| "What do you know about 2023 Fashion Trend: Shine For The Daytime?" | "…'daytime sparkle' or 'casual shimmer'" (invented) | **2023 Fashion Trend Shine For The Daytime** |
| "Which 2023 trend mentions tailored cargo pants made from silk or organza?" | "the 'Luxury Cargo' trend" (invented) | **Cargo Pants** |

In all three cases the ungrounded model produces a confident, fluent, wrong answer.
The retrieval-augmented version answers from the source text.

## How it works

1. **Wrangle** — load the CSV, strip non-standard characters and boilerplate, expose a
   single `text` column.
2. **Embed** — batch all 82 rows through `text-embedding-ada-002` (batch size 100).
3. **Retrieve** — embed the question, rank rows by cosine distance to it.
4. **Pack** — fill the prompt with the closest rows up to a 1000-token budget, measured
   with `tiktoken` (`cl100k_base`), leaving 150 tokens for the answer.
5. **Complete** — call `gpt-3.5-turbo-instruct` with instructions to answer from the
   context, or say "I don't know" if the context is insufficient.

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env     # then add your OpenAI key
jupyter notebook project.ipynb
```

The notebook reads `OPENAI_API_KEY` from the environment via `python-dotenv` — no key
is ever written into the notebook. Set `OPENAI_API_BASE` if you are routing through a
proxy or an alternative OpenAI-compatible endpoint.

Embedding all 82 rows costs a fraction of a cent.

## Repository layout

```
project.ipynb                        The full pipeline, with executed outputs
data/2023_fashion_trends.csv         Dataset used (82 rows)
data/character_descriptions.csv      Alternative datasets offered by the course,
data/nyc_food_scrap_drop_off_sites.csv   included unused for reference
.env.example                         Template for your API key
```

## Attribution

Built as a project for the Udacity Generative AI Nanodegree. The notebook scaffold and
the datasets under `data/` are Udacity course material, provided under Udacity's
educational-content license. One volunteer email address in the NYC open-data CSV has
been redacted.
