# Datomatisation

This document outlines the necessary configuration and setup for the
**Datomatisation** application.

------------------------------------------------------------------------

## Directory Structure

Create a `.streamlit` folder at the root of your project directory.\
This folder will contain your application's configuration and secrets
files.

    your_project/
    ├── .streamlit/
    │   ├── config.toml
    │   └── secrets.toml
    └── src/
        └── app.py

------------------------------------------------------------------------

## Configuration

### `.streamlit/secrets.toml`

Create a file named `secrets.toml` inside the `.streamlit` folder and
add the following content.\
Remember to replace the placeholder values with your actual API keys.

``` toml
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
```

> **Note:** The `config.toml` file, which is also placed in the
> `.streamlit` folder, is typically used for general application
> settings such as theme and layout options.

------------------------------------------------------------------------

## Requirements

-   Python **3.10** or higher

------------------------------------------------------------------------

## How to Run

Execute the following command from the root of your project directory to
run the Streamlit application:

``` bash
streamlit run src/app.py
```
