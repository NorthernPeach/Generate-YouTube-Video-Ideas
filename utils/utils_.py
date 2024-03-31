import concurrent.futures
import asyncio
from utils.summarizer import Summarizer
from langchain_openai import ChatOpenAI


def process_summary(url):
    url = url.replace(" ", "")   # remove any white space in the URL
    
    llm        = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
    summarizer = Summarizer(url, llm)
    summarizer.create_chunks()
    return asyncio.run(summarizer.summarize())


def pool_executor(urls):
    results = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit each URL to the executor
        futures = [executor.submit(process_summary, url) for url in urls]

        # Wait for all futures to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing URL: {e}")
    return results

def format_summaries(data):
    formatted_strings = []
    for i, obj in enumerate(data, start=1):
        summary = obj["summary"]
        views = obj["metadata"]["view_count"]
        title = obj["metadata"]["title"].split("|")[0].strip()

        formatted_string = (
            f"Video {i}\nTitle: {title}\nView Count: {views}\nSummary: {summary}\n"
        )
        formatted_strings.append(formatted_string)

    result = "\n".join(formatted_strings)
    return result