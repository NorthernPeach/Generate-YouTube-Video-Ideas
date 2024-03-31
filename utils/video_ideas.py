from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
import pandas as pd
from io import StringIO

from utils.utils_ import format_summaries, pool_executor

# from dotenv import load_dotenv
# load_dotenv()

def build_llm(high_engagement_vids_urls, low_engagement_vids_urls, openai_api_key):
    
    # max len for the videos ~1h 15min
    high_engagement_summaries = pool_executor(high_engagement_vids_urls)
    low_engagement_summaries  = pool_executor(low_engagement_vids_urls)

    high_eng_prompt = format_summaries(high_engagement_summaries)
    low_eng_prompt  = format_summaries(low_engagement_summaries)

    prompt_template = """ 
    You are helpful AI assistant that helps to increase the engagement of youtube videos by analyzing the scripts of old videos.\
    Looking at the given videos below in High_engagement_Videos and Low_engagement_videos sections,\
    come up with new ideas for next videos.\
        
    High_Engagement_Videos:
    {high_engagement_videos}
        
    Low_Engagement_Videos:
    {low_engagement_videos}

    Task elaboration:
    Given the above High_Engagement_Videos and Low_Engagement_Videos, generate new ideas and themes.
    New ideas should be related to the High_Engagement_Videos by keeping the titles and summaries of the episodes in context\
    and must be based on the common patterns between the High_Engagement_Videos and the guests in those episodes.\
    The new videos should not have any content from Low_Engagement_Videos.\
    Make sure to not include any speaker name in your suggested video topics or themes.

    Output format: 
    Make sure to return at least 10 new ideas. Your response must be a csv file which contains the following columns:\
    Topic, Theme, Summary. Summary should contain points to talk on the show and must be 100 words at max. Use | as a seperator\
    and do not append any extra line in your csv response. Each row must have proper data and columns in each row must be three.
    If you don't know the answer, just say "Hmm, I'm not sure."\
    Don't try to make up an answer. 
    """

    llm    = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, openai_api_key=openai_api_key)
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["high_engagement_videos", "low_engagement_videos"],
    )
    ideas_chain = LLMChain(llm=llm, prompt=PROMPT)
    
    response = ideas_chain.invoke(
        {
            "high_engagement_videos": high_eng_prompt,
            "low_engagement_videos": low_eng_prompt,
        }
    )

    # Read the CSV data and create a DataFrame
    csv_file = StringIO(response['text'])
    df       = pd.read_csv(csv_file, sep="|")   
    # df.to_csv('data.csv', index=False)  # Optional arguments explained below
    return df