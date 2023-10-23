import json
from typing import Optional
from langchain.tools.vectorstore.tool import VectorStoreQAWithSourcesTool
from langchain.tools.base import BaseTool
from langchain.pydantic_v1 import Field
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.chains import RetrievalQAWithSourcesChain


class MyVectorStoreQAWithSourcesTool(VectorStoreQAWithSourcesTool, BaseTool):
    search_type: Optional[str] = Field(default="similarity")
    search_keywords: Optional[dict] = Field(default=None)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        retriever = self.vectorstore.as_retriever()

        if self.search_keywords:
            retriever.search_kwargs.update(self.search_keywords)
        if self.search_type:
            retriever.search_type = self.search_type

        chain = RetrievalQAWithSourcesChain.from_chain_type(
            self.llm, retriever=retriever)

        return json.dumps(
            chain(
                {chain.question_key: query},
                return_only_outputs=True,
                callbacks=run_manager.get_child() if run_manager else None,
            )
        )
