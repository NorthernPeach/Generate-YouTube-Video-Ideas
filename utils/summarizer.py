import asyncio
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader
from transformers import AutoTokenizer

class Summarizer:
    def __init__(self, url, llm):
        self.url = url
        self.llm = llm
        self.doc_chunks = []
        self.metadata = []

    def _load_data(self):
        loader = YoutubeLoader.from_youtube_url(self.url, add_video_info=True)

        return loader.load()

    # def create_chunks(self):
    #     text_splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=500500,
    #         chunk_overlap=50,
    #         length_function=len,
    #     )
    #     text = self._load_data()
    #     self.doc_chunks = text_splitter.create_documents(
    #         [doc.page_content for doc in text]
    #     )
    #     self.metadata = text[0].metadata

    #     return
    

    def create_chunks(self):
        """
        This function splits text into chunks based on token count and handles metadata.

        Args:
            self: The class instance.

        Returns:
            None. Modifies self.doc_chunks and self.metadata.
        """
        # Choose an appropriate tokenizer for your use case (replace "gpt2" if needed)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        def token_length(text):
            """
            Custom function to count tokens in a text string.
            """
            return len(tokenizer(text, return_tensors="pt")["input_ids"][0])  # Count tokens in first output

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3500,  # Adjust chunk size as needed (balance between chunks and tokens)
            chunk_overlap=50,
            length_function=token_length,
        )

        text = self._load_data()
        self.doc_chunks = text_splitter.create_documents([doc.page_content for doc in text])
        self.metadata = text[0].metadata

        return 

    async def _chain_run(self, chain, docs):
        return await chain.ainvoke(docs)

    async def summarize(self):
        summarizer_chain = load_summarize_chain(llm=self.llm, chain_type="map_reduce")
        tasks = [self._chain_run(summarizer_chain, self.doc_chunks)]
        summary = await asyncio.gather(*tasks)

        return {"summary": summary[0], "metadata": self.metadata}
