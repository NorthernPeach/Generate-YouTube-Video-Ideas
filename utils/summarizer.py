import asyncio
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader

class Summarizer:
    def __init__(self, url, llm):
        self.url = url
        self.llm = llm
        self.doc_chunks = []
        self.metadata   = []

    def _load_data(self):
        loader = YoutubeLoader.from_youtube_url(self.url, add_video_info=True)
        return loader.load()

    def create_chunks(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3500,
            chunk_overlap=50,
            length_function=len,
        )
        text = self._load_data()
        self.doc_chunks = text_splitter.create_documents(
            [doc.page_content for doc in text]
        )
        self.metadata = text[0].metadata

        return
    
    async def _chain_run(self, chain, docs):
        return await chain.ainvoke(docs)

    async def summarize(self):
        summarizer_chain = load_summarize_chain(llm=self.llm, chain_type="map_reduce")
        tasks   = [self._chain_run(summarizer_chain, self.doc_chunks)]
        summary = await asyncio.gather(*tasks)

        return {"summary": summary[0], "metadata": self.metadata}
