# datomatisation

Create a .streamlit folder where the config.toml and secrets.toml should be
Content of  .streamlit/secrets.toml:

[settings]
USE_GEMINI = true

[services.gemini]
GEMINI_API_KEY = "your key here"
GEMINI_CHAT_MODEL = "gemini-1.5-flash"
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"

[services.gpt]
GPT_BASE= "your gpt_base link here"
GPT_VERSION="2024-08-01-preview"
GPT_KEY= "your key here"
GPT_ENGINE= "gpt-4o-mini"

Requires Python 3.10 or higher

Run using:
```bash
streamlit run src/app.py
```
