import requests
from bs4 import BeautifulSoup


def refactor_extracted_comments(comments: list) -> str:
    """
    Refactors and cleans the extracted comments

    :param comments: comments being refactored
    :return: refactored comments
    """
    comments = "".join(comments)
    lines = comments.split("\n")
    cleaned_lines = [line.strip() for line in lines]

    refactored_lines = [line for line in cleaned_lines if line]
    refactored_text = "\n".join(refactored_lines).replace("\n\n", "\n").replace("reply", "\nreply:")

    return refactored_text


def extract_page_content(url: str) -> dict:
    """
    Extracts the page content from the specified url

    :param url: url for which content is extracted
    :return: extracted page content
    """

    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for non-200 status codes

    soup        = BeautifulSoup(response.content, "lxml")
    title_text  = soup.select_one("title").text.strip() if soup.select_one("title") else None
    paragraphs  = soup.find_all('p')
    
    # Extract and merge text from each paragraph
    descript_text = ""
    for paragraph in paragraphs:
        descript_text += paragraph.get_text(strip=True) + "\n"  # Add newline between paragraphs

    comments = ""
    comm_divs = soup.find_all("div", class_="comment") 
    if comm_divs:
        for div in comm_divs:
            # Access the text content of the div
            comments += div.get_text(strip=True)  # Strips leading/trailing whitespace

    try:
        refactored_comments = refactor_extracted_comments(comments)
    except: 
        refactored_comments = ""
    return {
        "title": "".join(title_text),
        "description": "".join(descript_text),
        "comments": refactored_comments
    }