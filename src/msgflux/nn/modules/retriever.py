from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Union

from msgflux.data.dbs.types import VectorDB
from msgflux.data.retrievers.types import (
    LexicalRetriever,
    SemanticRetriever,
    WebRetriever,
)
from msgflux.dotdict import dotdict
from msgflux.message import Message
from msgflux.nn.modules.module import Module

if TYPE_CHECKING:
    from msgflux.nn.modules.embedder import Embedder

RETRIVERS = Union[WebRetriever, LexicalRetriever, SemanticRetriever, VectorDB]


class Retriever(Module):
    """Retriever is a Module type that uses information retrivers."""

    def __init__(
        self,
        name: str,
        retriever: RETRIVERS,
        *,
        embedder: Optional["Embedder"] = None,
        message_fields: Optional[Dict[str, Any]] = None,
        response_mode: Optional[str] = "plain_response",
        response_template: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Args:
        name:
            Designer name in snake case format.
        retriever:
            Retriever client.
        embedder:
            An Embedder module instance for converting queries to embeddings.
            Optional - only needed for semantic retrieval.
        message_fields:
            Dictionary mapping Message field names to their paths in the Message object.
            Valid keys: "task_inputs"
            !!! example
                message_fields={"task_inputs": "query.user"}

            Field description:
            - task_inputs: Field path for query input (str or dict)
        response_mode:
            What the response should be.
            * `plain_response` (default): Returns the final agent response directly.
            * other: Write on field in Message object.
        response_template:
            A Jinja template to format response.
        config:
            Dictionary with configuration options. Accepts any keys without validation.
            Common options: "top_k", "threshold", "return_score", "dict_key"
            !!! example
                config={
                    "top_k": 4,
                    "threshold": 0.0,
                    "return_score": False,
                    "dict_key": "name"
                }

            Configuration options:
            - top_k: Maximum return of similar points (int)
            - threshold: Retriever threshold (float)
            - return_score: If True, return similarity score (bool)
            - dict_key: Help to extract a value from task_inputs if dict (str)
        """
        super().__init__()
        self.set_name(name)
        self._set_retriever(retriever)
        self._set_embedder(embedder)
        self._set_message_fields(message_fields)
        self._set_response_mode(response_mode)
        self._set_response_template(response_template)
        self._set_config(config)

    def forward(
        self, message: Union[str, List[str], List[Dict[str, Any]], Message], **kwargs
    ) -> Union[str, Dict[str, str], Message]:
        """Execute the retriever with the given message.

        Args:
            message: The input message, which can be:
                - str: Direct query string for retrieval
                - List[str]: List of query strings
                - List[Dict[str, Any]]: List of query dictionaries
                - Message: Message object with fields mapped via message_fields
            **kwargs: Runtime overrides for message_fields. Can include:
                - task_inputs: Override field path or direct value

        Returns:
            Retrieved results (str, dict, or Message depending on response_mode)

        Examples:
            # Direct string query
            retriever("What is machine learning?")

            # List of queries
            retriever(["query1", "query2"])

            # Using Message object with message_fields
            msg = Message(query="What is machine learning?")
            retriever(msg)

            # Runtime override
            retriever(msg, task_inputs="custom.query.path")
        """
        inputs = self._prepare_task(message, **kwargs)
        retriever_response = self._execute_retriever(**inputs)
        response = self._prepare_response(retriever_response, message)
        return response

    async def aforward(
        self, message: Union[str, List[str], List[Dict[str, Any]], Message], **kwargs
    ) -> Union[str, Dict[str, str], Message]:
        """Async version of forward. Execute the retriever asynchronously.

        Args:
            message: The input message, which can be:
                - str: Direct query string for retrieval
                - List[str]: List of query strings
                - List[Dict[str, Any]]: List of query dictionaries
                - Message: Message object with fields mapped via message_fields
            **kwargs: Runtime overrides for message_fields. Can include:
                - task_inputs: Override field path or direct value

        Returns:
            Retrieved results (str, dict, or Message depending on response_mode)

        Examples:
            # Direct string query
            await retriever.acall("What is machine learning?")

            # List of queries
            await retriever.acall(["query1", "query2"])

            # Using Message object with message_fields
            msg = Message(query="What is machine learning?")
            await retriever.acall(msg)

            # Runtime override
            await retriever.acall(msg, task_inputs="custom.query.path")
        """
        inputs = self._prepare_task(message, **kwargs)
        retriever_response = await self._aexecute_retriever(**inputs)
        response = self._prepare_response(retriever_response, message)
        return response

    def _execute_retriever(
        self, queries: List[str], model_preference: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        queries_embed = None
        if self.embedder:
            queries_embed = self.embedder(queries, model_preference=model_preference)
            # Ensure list format
            if not isinstance(queries_embed[0], list):
                queries_embed = [queries_embed]

        retriever_execution_params = self._prepare_retriever_execution(
            queries_embed or queries
        )
        retriever_response = self.retriever(**retriever_execution_params)

        results = []

        for query, query_results in zip(queries, retriever_response):
            formatted_result = {
                "results": [
                    {"data": item.get("data", None), "score": item.get("score", None)}
                    for item in query_results
                ],
            }
            if isinstance(query, str):
                formatted_result["query"] = query
            results.append(formatted_result)

        return results

    async def _aexecute_retriever(
        self, queries: List[str], model_preference: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        queries_embed = None
        if self.embedder:
            queries_embed = await self.embedder.aforward(queries, model_preference=model_preference)
            # Ensure list format
            if not isinstance(queries_embed[0], list):
                queries_embed = [queries_embed]

        retriever_execution_params = self._prepare_retriever_execution(
            queries_embed or queries
        )
        retriever_response = self.retriever(**retriever_execution_params)

        results = []

        for query, query_results in zip(queries, retriever_response):
            formatted_result = {
                "results": [
                    {"data": item.get("data", None), "score": item.get("score", None)}
                    for item in query_results
                ],
            }
            if isinstance(query, str):
                formatted_result["query"] = query
            results.append(formatted_result)

        return results

    def _prepare_retriever_execution(
        self, queries: List[Union[str, List[float]]]
    ) -> Dict[str, Any]:
        retriever_execution_params = dotdict(
            queries=queries,
            top_k=self.config.get("top_k", 4),
            return_score=self.config.get("return_score", False)
        )
        threshold = self.config.get("threshold")
        if threshold:
            retriever_execution_params.threshold = threshold
        return retriever_execution_params


    def _prepare_task(
        self, message: Union[str, List[str], List[Dict[str, Any]], Message], **kwargs
    ) -> List[str]:
        if isinstance(message, Message):
            queries = self._extract_message_values(self.task_inputs, message)
        else:
            queries = message

        if isinstance(queries, str):
            queries = [queries]
        elif isinstance(queries, list):
            if isinstance(queries[0], dict):
                queries = self._process_list_of_dict_inputs(queries)

        model_preference = kwargs.pop("model_preference", None)
        if model_preference is None and isinstance(message, Message):
            model_preference = self.get_model_preference_from_message(message)

        return {"queries": queries, "model_preference": model_preference}

    def _process_list_of_dict_inputs(self, queries: List[Dict[str, Any]]) -> List[str]:
        """Extract the query value from a dict."""
        dict_key = self.config.get("dict_key")
        if dict_key:
            queries_list = [data[dict_key] for data in queries]
            return queries_list
        else:
            raise AttributeError(
                "message that contain `List[Dict[str, Any]]` "
                "require a `dict_key` to select the key for retrieval"
            )

    def inspect_embedder_params(self, *args, **kwargs) -> Mapping[str, Any]:
        """Debug embedder input parameters.

        Returns the parameters that would be passed to the embedder module.
        """
        if self.embedder:
            inputs = self._prepare_task(*args, **kwargs)
            return {"queries": inputs["queries"], "model_preference": inputs.get("model_preference")}
        return {}

    def _set_retriever(self, retriever: RETRIVERS):
        if isinstance(
            retriever, (WebRetriever, LexicalRetriever, SemanticRetriever, VectorDB)
        ):
            self.register_buffer("retriever", retriever)
        else:
            raise TypeError(
                "`retriever` requires `HybridRetriever`, `LexicalRetriever`, "
                f"`SemanticRetriever` or `VectorDB` instance given `{type(retriever)}`"
            )

    def _set_embedder(self, embedder: Optional["Embedder"] = None):
        """Set the embedder module for semantic retrieval.

        Args:
            embedder: An Embedder module instance or None

        Raises:
            TypeError: If embedder is not an Embedder instance or None
        """
        if embedder is None:
            self.register_buffer("embedder", None)
            return

        # Import here to avoid circular dependency
        from msgflux.nn.modules.embedder import Embedder

        if not isinstance(embedder, Embedder):
            raise TypeError(
                f"`embedder` must be an Embedder instance, given `{type(embedder)}`"
            )

        self.register_buffer("embedder", embedder)

    def _set_config(self, config: Optional[Dict[str, Any]] = None):
        """Set module configuration without key validation.

        Args:
            config: Dictionary with configuration options.
                Accepts any keys - commonly used: "top_k", "threshold", "return_score", "dict_key"

        Raises:
            TypeError: If config is not a dict or None
        """
        if config is None:
            self.config = {}
            return

        if not isinstance(config, dict):
            raise TypeError(
                f"`config` must be a dict or None, given `{type(config)}`"
            )

        # Store config without validation - accepts any keys
        self.config = config.copy()
