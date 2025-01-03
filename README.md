# llm-web-search

[![PyPI](https://img.shields.io/pypi/v/llm-web-search.svg)](https://pypi.org/project/llm-web-search/)
[![Changelog](https://img.shields.io/github/v/release/simonw/llm-web-search?include_prereleases&amp;label=changelog)](https://github.com/simonw/llm-web-search/releases)
[![Tests](https://github.com/simonw/llm-web-search/actions/workflows/test.yml/badge.svg)](https://github.com/simonw/llm-web-search/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/llm-web-search/blob/main/LICENSE)

A plugin for LLM that enables web searches using Bing or Google, with configurable LLM models for processing results and fallback options.

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).

```bash
llm install llm-web-search
```

## Configuration

### API Keys

You will need API keys for either Bing or Google, depending on which search engine you want to use.

#### Bing

1. Obtain a Bing Search API key from the Azure portal.
2. Set the key using `llm keys set bing`:

    ```bash
    llm keys set bing
    ```
    ```
    paste bing api key here
    ```
    You can also set the key using the environment variable `BING_API_KEY`.

#### Google
1. Set up a [Google Custom Search Engine](https://programmablesearchengine.google.com/) and obtain an API key.
2. Set the key using `llm keys set google`

    ```bash
    llm keys set google
    ```
    ```
    paste google api key here
    ```
3. Set your custom search engine ID using `llm keys set google_search_id`

    ```bash
    llm keys set google_search_id
    ```
    ```
    paste google search id here
    ```

    You can also use the environment variables `GOOGLE_API_KEY` and `GOOGLE_SEARCH_ID`.

## Usage

### Basic Search
To perform a web search, use the `llm web-search search` command followed by your search query:

```bash
llm web-search search "your search query"
```

This will use the google search engine by default, processing each page using `gemini-1.5-pro-latest`, and the summary will be generated using `gemini-1.5-pro-latest` as well.

### Specifying a different search engine

Use the `-e` or `--engine` flag to specify bing or google

```bash
llm web-search search -e google "your search query"
```
```bash
llm web-search search --engine bing "your search query"
```

### Model Selection
You can change the models using the following options:
*   `--model`:  Model used to process the content of each page
*   `--fallback-model`:  Fallback model to use if the content of a page is too large for the default model
*   `--summary-model`:  Model used to create a summary of all of the results
*   `--summary-fallback-model`: Fallback model to use if the combined results are too large for the `summary-model`

```bash
llm web-search search -e google --model gpt-4 --fallback-model gpt-4-32k --summary-model gpt-3.5-turbo --summary-fallback-model gpt-3.5-turbo-16k "your search query"
```
### Result Limits

    Use the `-n` or `--num-results` option to set the number of results to fetch from the search API. Note that each search API has it's own limits, and the plugin will use the smallest of the limits between the requested number of results, and the API limits.

```bash
llm web-search search -n 20 "your search query"
```
### API options

Use the `-o` or `--option` flag to access any parameter that is available for either search engine's API. For example, to use the bing `freshness` parameter:

```bash
llm web-search search -e bing "your search query" -o freshness "day"
```
Or to use the google `cr` parameter for a specific country:

```bash
    llm web-search search -e google "your search query" -o cr "countryUS"
```
Multiple options can be used at once:
```bash
    llm web-search search -e bing "your search query" -o freshness "day" -o textFormat "raw"
```
    Please see the Bing and Google documentation for all available parameters

* Bing Search API:  https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/reference/query-parameters
* Google Custom Search API: https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list

### Default Models

The plugin defines default and fallback models for each search engine. You can also use the aliases:

    ```bash
    llm web-search search "query" -m google_search_default -f google_search_fallback -m google_summary_default -f google_summary_fallback
    llm web-search search "query" -e bing -m bing_search_default -f bing_search_fallback -m bing_summary_default -f bing_summary_fallback
    ```

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:

```bash
cd llm-web-search
python3 -m venv venv
source venv/bin/activate
```

Now install the dependencies and test dependencies:

```bash
llm install -e '.[test]'
```
To run the tests:
```bash
pytest
```