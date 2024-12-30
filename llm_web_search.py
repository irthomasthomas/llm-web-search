import click
import asyncio
import json
import os
import pathlib
import sqlite_utils
import re
import hashlib
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import sys
import httpx
from urllib.parse import quote
import aiofiles
import llm
import concurrent.futures

def user_dir():
    """Get or create user directory for storing application data."""
    llm_user_path = os.environ.get("LLM_USER_PATH")
    if llm_user_path:
        path = pathlib.Path(llm_user_path)
    else:
        path = pathlib.Path(click.get_app_dir("io.datasette.llm"))
    path.mkdir(exist_ok=True, parents=True)
    return path

def logs_db_path():
    """Get path to logs database."""
    return user_dir() / "logs.db"

def setup_logging():
    """Configure logging to write to both file and console."""
    log_path = user_dir() / "llm_search.log"

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler(str(log_path))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)
logger.debug("llm_search module is being imported")

class DatabaseConnection:
    _instance: Optional['DatabaseConnection'] = None

    def __init__(self):
        self.db = sqlite_utils.Database(logs_db_path())

    @classmethod
    def get_connection(cls) -> sqlite_utils.Database:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance.db

def log_response(response, model):
    """Log model response to database and log file."""
    try:
        db = DatabaseConnection.get_connection()
        response.log_to_db(db)
        logger.debug(f"Response from {model} logged to database")
    except Exception as e:
        logger.error(f"Error logging to database: {e}")

def estimate_token_count(text: str) -> int:
    """Estimates token count for a given text."""
    word_count = len(text.split())
    return int(word_count * 4 / 3)

async def fetch_url(url: str, cache_dir: pathlib.Path) -> Optional[str]:
        """Fetches a URL and returns its text content using httpx or from cache."""
        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_file = cache_dir / cache_key

        try:
            # Check if cached result exists and is less than 1 hour old
            if cache_file.exists() and (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).total_seconds() < 3600:
                async with aiofiles.open(str(cache_file), mode='r') as f:
                    logger.debug(f"Using cached result for {url}")
                    return await f.read()
            
            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(url, timeout=10)
                response.raise_for_status()
                content = response.text
                # Save to cache
                cache_dir.mkdir(parents=True, exist_ok=True)
                async with aiofiles.open(str(cache_file), mode='w') as f:
                    await f.write(content)
                logger.debug(f"Retrieved and cached content from {url}")
                return content

        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch content from {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error processing {url}: {e}")
            return None

async def convert_html_to_text(html_content: str, url: str, use_shot_scraper: bool=True) -> Optional[str]:
    """Converts HTML content to text using html2text and shot-scraper when needed."""
    try:
        import html2text
        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        h.body_width = 0  
        text_content = h.handle(html_content)
        if text_content and not "Enable JavaScript and cookies to continue" in text_content:
                logger.debug(f"Successfully converted HTML to text using html2text")
                return text_content
        
        if not use_shot_scraper:
            logger.warning(f"html2text failed, and shot-scraper disabled for {url}, returning None")
            return None
        
        import subprocess
        process = await asyncio.create_subprocess_exec(
            "shot-scraper", "html", url,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            logger.error(f"shot-scraper failed to process {url}: {stderr.decode()}")
            return None
        
        html_content = stdout.decode()
        text_content = h.handle(html_content)
        if text_content:
            logger.debug(f"Successfully converted HTML to text using shot-scraper")
            return text_content
        
        logger.error(f"Failed to convert HTML to text for {url} with html2text and shot-scraper")
        return None
        
    except ImportError:
        logger.error(f"html2text or shot-scraper not installed for {url}")
        return None

async def convert_pdf_to_text(pdf_content: bytes) -> Optional[str]:
    """Converts PDF content to text."""
    try:
        import subprocess
        process = await asyncio.create_subprocess_exec(
            "pdftotext", "-",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = await process.communicate(input=pdf_content)
        if process.returncode != 0:
            logger.error(f"pdftotext failed: {stderr.decode()}")
            return None
        
        text_content = stdout.decode()
        logger.debug(f"Successfully converted PDF to text")
        return text_content
    
    except FileNotFoundError:
        logger.error("pdftotext is not installed")
        return None

async def process_url_content(
    url: str,
    search_query: str,
    cache_dir: pathlib.Path,
    default_llm_model: str,
    fallback_llm_model: str,
    use_shot_scraper:bool=True,
) -> Optional[str]:
        """Processes the content of a single URL, extracts relevant quotes and returns the output."""
        logger.debug(f"Processing URL: {url}")
        
        if url.endswith('.pdf'):
            try:
                content = await fetch_url(url, cache_dir)
                if content:
                    text_content = await convert_pdf_to_text(content.encode())
                else:
                    return None
                
            except Exception as e:
                logger.error(f"Error fetching or converting pdf {url}: {e}")
                return None
        else:
            html_content = await fetch_url(url, cache_dir)
            if not html_content:
                return None
            text_content = await convert_html_to_text(html_content, url, use_shot_scraper=use_shot_scraper)
        
        if not text_content:
            logger.warning(f"Could not get text content for: {url}")
            return None

        token_count = estimate_token_count(text_content)
        llm_model = default_llm_model if token_count < 8000 else fallback_llm_model

        try:
            llm_prompt = f"""Extract any relevant passages that may be in any way related to the search query "{search_query}" if they exist in the scraped page along with their URL. Provide a relevance score from 1-3 for each passage. If two or more passages score 2 or more for relevance, provide the whole page text minus any irrelevant content."""
            response = llm.get_model(llm_model).prompt(
                text_content, 
                system=llm_prompt
            )
            log_response(response, llm_model)

            extracted_text = response.text()
            output = f"""URL: {url}
Relevant Quotes:
{extracted_text}
--------------------------------------------------
"""
            logger.debug(f"Processed content from {url} with {llm_model}")
            return output
        except Exception as e:
            logger.error(f"Failed to process text content with llm {url}: {e}")
            return None
        
async def summarize_results(
    results: str,
    search_query: str,
    default_llm_model: str,
    fallback_llm_model: str,
) -> Optional[str]:
        """Summarizes the search results using an LLM."""
        token_count = estimate_token_count(results)
        llm_model = default_llm_model if token_count < 8000 else fallback_llm_model
        
        try:
            llm_prompt = f"Summarize the key points from the search results related to '{search_query}'. Focus on the most relevant and highest-scored quotes. Provide a concise overview of the main findings."
            response = llm.get_model(llm_model).prompt(
                results,
                system=llm_prompt
            )
            log_response(response, llm_model)
            summary = response.text()
            logger.debug(f"Successfully generated summary using {llm_model}")
            return summary
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return None

def get_default_llm_model() -> str:
    """Get the default model from llm config or return a fallback"""
    try:
        return llm.get_default_model()
    except:
            return "gpt-3.5-turbo"

async def search_bing(
    search_query: str,
    bing_api_key: str,
    count: int = 10,
    offset: int = 0,
    format: str = "json",
    config_id: Optional[str] = None,
    region: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
    """Performs a Bing search using the Bing Search API."""

    base_url = "https://api.bing.microsoft.com/v7.0/search"

    if not bing_api_key:
        logger.error("Bing API key is missing.")
        raise ValueError("Bing API key is required for Bing search.")

    headers = {"Ocp-Apim-Subscription-Key": bing_api_key}
    params = {
        "q": search_query,
        "count": count,
        "offset": offset,
        "responseFormat": format
    }
    if config_id:
        params["customConfig"] = config_id
    if region:
        params["mkt"] = region

    logger.debug(f"Bing API request URL: {base_url}")
    logger.debug(f"Bing API request headers: {headers}")
    logger.debug(f"Bing API request params: {params}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(base_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        if e.response.status_code == 401:
            logger.error(f"Bing search API authentication failed. Check your API key.")
            raise ValueError("Bing API authentication failed. Invalid or missing API key.")
        else:
            logger.error(f"Bing search API error: {e}")
            return None
    except Exception as e:
        logger.error(f"Unexpected error during Bing search: {e}")
        return None

@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def search():
        """Search the web using various search engines and LLMs."""
        pass

    @search.command()
    @click.argument("search_query")
    @click.option(
        "-e",
        "--engine",
        type=click.Choice(["bing", "local", "auto"]),
        default="auto",
        help="Search engine to use: 'local' (default) uses the original script, 'bing' uses Bing API directly, 'auto' tries local first, then Bing if needed.",
    )
    @click.option(
        "-n",
        "--num-results",
        type=int,
        default=10,
        help="Number of search results to fetch.",
    )
    @click.option(
        "-o",
        "--output-file",
        type=click.Path(writable=True),
        help="Output file to save the search results.",
    )
    @click.option(
        "--default-llm-model",
        type=str,
        default=get_default_llm_model(),
        help="Default LLM model to use for content processing and summarization.",
    )
    @click.option(
        "--fallback-llm-model",
        type=str,
        default="gemini-1.5-flash-8b-latest",
        help="Fallback LLM model to use when requests are too large.",
    )
    @click.option(
        "--cache-dir",
        type=click.Path(),
        default=str(user_dir() / "llm_search_cache"),
        help="Directory to store cached results.",
    )
    @click.option(
        "--bing-api-key",
        type=str,
        required=False,
        envvar="BING_API_KEY",
        help="Bing API key (required if using 'bing' engine or if 'auto' falls back to Bing).",
    )
    @click.option(
        "--no-shot-scraper",
        is_flag=True,
            help="Disable shot-scraper for rendering javascript heavy pages"
    )
    @click.option(
        "--bing-config-id",
            type=str,
        help="Bing Custom Config ID"
    )
    @click.option(
        "--bing-region",
        type=str,
        help="Bing Market region (e.g. en-US)"
    )    
    @click.option(
        "--bing-format",
        type=click.Choice(["json", "text"]),
        default="json",
        help="Bing API output format"
    )
    @click.option(
        "--local-script-path",
        type=str,
        default="/home/ShellLM/Projects/claude.sh/utils/search/bing-search.sh",
        help="Path to the local bing-search.sh script.",
    )
    def web_search(
        search_query: str,
        engine: str,
        num_results: int,
        output_file: Optional[str],
        default_llm_model: str,
        fallback_llm_model: str,
        cache_dir: str,
        bing_api_key: Optional[str],
        no_shot_scraper: bool,
        bing_config_id: Optional[str],
        bing_region: Optional[str],
        bing_format: str,
        local_script_path: str
    ):
        """Searches the web, extracts content, and provides a summary."""

        cache_path = pathlib.Path(cache_dir)

        async def _async_search():
            
            async def run_local_search():
                if not local_script_path or not os.path.exists(local_script_path):
                    raise ValueError("Local script path is invalid or not provided.")
                try:
                    import subprocess
                    process = subprocess.run([local_script_path, "-q", search_query, "-n", str(num_results), "-f", "json"], capture_output=True, text=True, check=True)

                    # Handle potential JSON decoding errors
                    try:
                        search_results = json.loads(process.stdout)
                    except json.JSONDecodeError:
                        logger.error(f"Error decoding JSON from local script: {process.stdout}")
                        return None

                    if not search_results or "webPages" not in search_results or not search_results["webPages"].get("value"):
                        logger.error(f"No valid results from local script for {search_query}")
                        return None
                    urls = [item["url"] for item in search_results["webPages"]["value"]]
                    return urls
                except subprocess.CalledProcessError as e:
                    logger.error(f"Local script execution failed: {e}")
                    return None
                except Exception as e:
                    logger.exception(f"Error running local script: {e}")
                    return None
            
            async def run_bing_api_search():
                if not bing_api_key:
                    raise click.ClickException(
                        "Bing API key is required when using the 'bing' engine or if 'auto' falls back to Bing. Please provide it using --bing-api-key or set the BING_API_KEY environment variable."
                    )
                search_results = await search_bing(
                    search_query=search_query,
                    bing_api_key=bing_api_key,
                    count=num_results,
                    config_id=bing_config_id,
                    region=bing_region,
                    format=bing_format,
                )
                if search_results is None:
                    raise ValueError("Bing search failed. Likely due to an invalid or missing API key.")
                if not search_results or "webPages" not in search_results or not search_results["webPages"].get("value"):
                    logger.error(f"No valid results from Bing search for {search_query}")
                    return None
                urls = [item["url"] for item in search_results["webPages"]["value"]]
                return urls

            try:
                if engine == "local":
                    urls = await run_local_search()
                elif engine == "bing":
                    urls = await run_bing_api_search()
                elif engine == "auto":
                    urls = await run_local_search()
                    if not urls:
                        logger.info("Local search did not return results, falling back to Bing API.")
                        urls = await run_bing_api_search()
                else:
                    raise ValueError(f"Invalid search engine specified: {engine}")
                
                if not urls:
                    logger.error(f"No URLs found for search query: {search_query}")
                    return None

                # Process URLs with timeout and error handling
                tasks = [
                    asyncio.create_task(
                        asyncio.wait_for(
                            process_url_content(
                                url,
                                search_query,
                                cache_path,
                                default_llm_model,
                                fallback_llm_model,
                                use_shot_scraper=not no_shot_scraper
                            ),
                            timeout=30
                        )
                    ) for url in urls[:num_results]
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                processed_results = [r for r in results if not isinstance(r, Exception) and r is not None]

                if not processed_results:
                    logger.warning("No content processed from search results")
                    return None

                combined_results = "\n".join(processed_results)
                summary = await summarize_results(
                    combined_results, search_query, default_llm_model, fallback_llm_model
                )

                return f"""{combined_results}
Summary:
{summary or 'No summary generated'}"""

            except asyncio.TimeoutError:
                logger.error("Search or processing timed out")
                return None
            except ValueError as e:
                logger.error(f"Search failed: {e}")
                raise click.ClickException(str(e))
            except Exception as e:
                logger.exception(f"Unexpected error in search: {e}")
                return None

        try:
            output = asyncio.run(_async_search())
            if output:
                if output_file:
                    with open(output_file, "w") as f:
                        f.write(output)
                    click.echo(f"Results saved to {output_file}")
                else:
                    click.echo(output)
            else:
                click.echo(f"No search results for {search_query}")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise click.ClickException(str(e))

logger.debug("llm_search module finished loading")