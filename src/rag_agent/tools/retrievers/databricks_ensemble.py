from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from databricks.vector_search.client import VectorSearchClient
from langchain.retrievers import EnsembleRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool

from rag_agent.config.settings import Settings, SourceSpec


@dataclass(frozen=True, slots=True)
class _DatabricksIndexConfig:
    index_name: str
    query_type: str
    top_k: int
    text_column: str
    return_columns: tuple[str, ...]


class _DatabricksVectorSearchRetriever(BaseRetriever):
    client: VectorSearchClient
    endpoint_name: str
    index_config: _DatabricksIndexConfig

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        index = self.client.get_index(
            endpoint_name=self.endpoint_name, index_name=self.index_config.index_name
        )
        result = index.similarity_search(
            query_text=query,
            columns=list(self.index_config.return_columns),
            num_results=self.index_config.top_k,
            query_type=self.index_config.query_type,
        )
        return _results_to_documents(
            result, text_column=self.index_config.text_column, query=query
        )

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return await asyncio.to_thread(
            self._get_relevant_documents, query, run_manager=run_manager
        )


def _results_to_documents(result: Any, *, text_column: str, query: str) -> list[Document]:
    if not isinstance(result, dict):
        return []

    manifest_cols = result.get("manifest", {}).get("columns")
    if isinstance(manifest_cols, list):
        columns = [c.get("name") for c in manifest_cols if isinstance(c, dict)]
        columns = [c for c in columns if isinstance(c, str)]
    else:
        columns = []

    data_array = None
    maybe_result = result.get("result")
    if isinstance(maybe_result, dict):
        data_array = maybe_result.get("data_array") or maybe_result.get("data")
        if not columns:
            cols2 = maybe_result.get("columns")
            if isinstance(cols2, list) and all(isinstance(x, str) for x in cols2):
                columns = cols2

    if data_array is None:
        data_array = result.get("data_array") or result.get("data")

    if not isinstance(data_array, list):
        return []

    docs: list[Document] = []
    for row in data_array:
        if not isinstance(row, list):
            continue

        metadata: dict[str, Any] = {"retrieval_query": query}
        for idx, value in enumerate(row):
            if idx < len(columns) and columns[idx]:
                metadata[columns[idx]] = value
            else:
                metadata[f"col_{idx}"] = value

        page_content = ""
        if text_column in metadata and isinstance(metadata[text_column], str):
            page_content = metadata[text_column]
        elif "text" in metadata and isinstance(metadata["text"], str):
            page_content = metadata["text"]
        elif "content" in metadata and isinstance(metadata["content"], str):
            page_content = metadata["content"]
        else:
            page_content = str(row[0]) if row else ""

        docs.append(Document(page_content=page_content, metadata=metadata))

    return docs


def _build_vector_search_client(settings: Settings) -> VectorSearchClient:
    if settings.databricks.host and settings.databricks.token:
        return VectorSearchClient(
            workspace_url=settings.databricks.host,
            personal_access_token=settings.databricks.token,
        )
    return VectorSearchClient()


def build_ensemble_retriever(*, settings: Settings, source: SourceSpec) -> EnsembleRetriever:
    client = _build_vector_search_client(settings)
    endpoint = settings.databricks.vector_search_endpoint

    vector_retriever = _DatabricksVectorSearchRetriever(
        client=client,
        endpoint_name=endpoint,
        index_config=_DatabricksIndexConfig(
            index_name=source.vector_index,
            query_type=settings.retrieval.vector_query_type,
            top_k=settings.retrieval.top_k,
            text_column=settings.retrieval.text_column,
            return_columns=settings.retrieval.return_columns,
        ),
    )

    bm25_retriever = _DatabricksVectorSearchRetriever(
        client=client,
        endpoint_name=endpoint,
        index_config=_DatabricksIndexConfig(
            index_name=source.bm25_index or source.vector_index,
            query_type=settings.retrieval.bm25_query_type,
            top_k=settings.retrieval.top_k,
            text_column=settings.retrieval.text_column,
            return_columns=settings.retrieval.return_columns,
        ),
    )

    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[settings.retrieval.vector_weight, settings.retrieval.bm25_weight],
    )


class SourceEnsembleRetrieverTool(BaseTool):
    settings: Settings
    source: SourceSpec
    name: str
    description: str

    def _run(self, query: str) -> list[Document]:
        retriever = build_ensemble_retriever(settings=self.settings, source=self.source)
        docs = retriever.invoke(query)
        return list(docs) if isinstance(docs, Sequence) else []

    async def _arun(self, query: str) -> list[Document]:
        return await asyncio.to_thread(self._run, query)


def build_source_tools(*, settings: Settings) -> list[BaseTool]:
    tools: list[BaseTool] = []
    for spec in settings.sources:
        tools.append(
            SourceEnsembleRetrieverTool(
                settings=settings,
                source=spec,
                name=f"retrieve_{spec.source_id.lower()}",
                description=f"Retrieve documents from {spec.source_id}. {spec.description}",
            )
        )
    return tools

