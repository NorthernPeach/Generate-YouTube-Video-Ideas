# from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# load_dotenv()

def generate_youtube_ideas_from_content(page_content: dict, openai_api_key: str) -> str:
    """
    Generates youtube ideas for the given content

    :param page_content: extracted title, description and comments/replies
    :return: generated ideas
    """
    human_message = """
    I've the following topic
    {title}
    with the description
    {description}
    And the following are the comments on that
    {comments}
    
    Generate 5 different viral YouTube content ideas related to this. For each idea, please provide
    title, description and youtube content script (with timestamps)
    
    Output the result in the following format using markdown:
    
    Idea 1:
    Title:
    Description:
    YouTube content script(with timestamps):
    0:00 some script content
    0:30 other script content
    
    Idea 2:
    Title:
    Description:
    YouTube content script(with timestamps):
    0:00 some script content
    0:30 other script content

    ...
    """

    system_message_prompt = SystemMessagePromptTemplate.from_template("You are an expert youtube content creator.")
    human_message_prompt  = HumanMessagePromptTemplate.from_template(human_message)

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            system_message_prompt,
            human_message_prompt
        ]
    )

    chat_model = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo", openai_api_key=openai_api_key)

    llm_chain = LLMChain(prompt=chat_prompt, llm=chat_model)
    response = llm_chain.invoke(
        {
            "title": page_content["title"],
            "description": page_content["description"],
            "comments": page_content["comments"]}
    )

    response = response['text'].replace("\n", "\n\n")

    return response