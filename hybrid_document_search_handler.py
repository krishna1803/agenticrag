"""Entry point for the hybrid search query handler.

Designed to perform keyphrase and semantic search for documents

Typical usage example:

  handler = HybridDocumentSearchHandler()
  result = handler.handle_query(query)
"""

import logging
import re

import psycopg2
import sentence_transformers

from exceptions.search_error import SearchError
from llm.base_llm import BaseLLM
from model.api.query_response import QueryResponse
from model.api.response_citation import ResponseCitation
from model.llm.chat_request import ChatRequest
from parsers import parser_utils
from prompt_templates.generate_search_filter import (
    GenerateSearchFilter,
    SearchFilter,
    AttributeFilter,
)
from search_handlers.base_search_handler import BaseSearchHandler

logger = logging.getLogger(__name__)

equality_lookup = {"eq": "=", "ne": "!=", "gt": ">=", "lt": "<="}

# Singleton model. This can be extended to a dict for multiple embedding models
# if required in the future.
_model: sentence_transformers.SentenceTransformer = None


class HybridDocumentSearchHandler(BaseSearchHandler):
    """Class to act as the entrypoint for handling a query

    Attributes:
      db_connection: psycopg2.extensions.connection
      llm_implementation: BaseLLM
      max_results:int - Max number of results for each type of query
      deduplicate_results: bool - Flag for whether to deduplicate the results
    """

    # TODO: Should we take an implementation of the embedding model? Or
    # just create one here? We might have multiple instances of this
    # class, so I guess apply a singleton pattern? How thread-safe is that?
    db_connection: psycopg2.extensions.connection
    llm_implementation: BaseLLM
    max_results: int
    deduplicate_results: bool

    def __init__(
        self,
        db_connection: psycopg2.extensions.connection,
        llm_implementation: BaseLLM,
        max_results: int = 5,
        deduplicate_results: bool = False,
        _use_mock_model: bool = False,
    ):
        """Instantiate the Document Search Handler

        Args:
            db_connection (psycopg2.extensions.connection): A PostGreSQL DB 
                Connection
            llm_implementation (BaseLLM): An LLM implementation to use for 
                generating filters
            max_results (bool, optional): Max number of results for each type 
                of query. Defaults to 5
            deduplicate_results (bool, optional): Flag for whether to 
                deduplicate the results. Set to True when returning the result
                list. Defaults to False
            _use_mock_model (bool, optional): Don't load the model - use this 
                for unit testing. Defaults to False
        """
        self.db_connection = db_connection
        self.llm_implementation = llm_implementation
        self.max_results = max_results
        self.deduplicate_results = deduplicate_results
        global _model
        if _model is None and not _use_mock_model:
            _model = sentence_transformers.SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )

    @staticmethod
    def _document_and_metadata_to_response_citation(
        documents: list[tuple],
    ) -> list[ResponseCitation]:
        """Transform the DB response to a Response Citation

        Args:
            documents (list[tuple]): document, metadata

        Returns:
            list[ResponseCitation]: The structured object from the query
        """
        response: list[ResponseCitation] = []
        for document in documents:
            # try:
            #     metadata = json.loads(document[1])
            # except json.JSONDecodeError as e:
            #     logger.warning(
            #         "Error while parsing JSON metadata from the DB for document [%s]\n[%s]",
            #         document[0][:64],
            #         e,
            #     )
            #     continue
            # The jsonb object is automatically marshalled?
            metadata = document[1]
            # Cases, Journals and Legislation all need to be handled differently
            if metadata["type"] == "legislation":
                # Ensuring that the name is of the form 'Act Name 2001 s 123',
                # so concat the name and section number.
                # Similarly - reference is the url, and should be appended with
                # the section number i.e. <act_url>/s123.html
                response.append(
                    ResponseCitation(
                        doc_type="legislation",
                        reference=f"{metadata["url"]}s{metadata["section"].lower()}.html",
                        jurisdiction=metadata["jurisdiction"],
                        date=metadata["date"][:10],
                        name=f"{metadata["name"]} s {metadata["section"]}",
                        database=metadata["database"],
                        text=document[0],
                    )
                )
            elif metadata["type"] == "case":
                # Cases need name and citation cleanup.
                # Check for 'famous name'
                name_parts = []
                moniker = None
                for name in metadata["titles"]:
                    # If it isn't of form 'A v B'
                    if not re.match(r".+\sv\s.+", name):
                        moniker = name.strip()
                    else:
                        name_parts.append(name.strip())
                case_name = "; ".join(name_parts)
                if moniker:
                    case_name += f' ("{moniker} case")'
                clean_citations = list(
                    filter(
                        lambda c: re.match(r"^(?:(?:\[|\()?\d{4}(?:\]|\))?)", c),
                        metadata["citations"],
                    )
                )
                if len(clean_citations) != 0:
                    # We want bound citations first (i.e. [1234]), so just lexical sort
                    clean_citations = "; ".join(sorted(clean_citations, reverse=True))
                else:
                    clean_citations = metadata["citations"]
                response.append(
                    ResponseCitation(
                        doc_type="case",
                        reference=metadata["url"],
                        jurisdiction=metadata["jurisdiction"],
                        date=metadata["date"][:10],
                        name=case_name,
                        court=metadata["court"],
                        citation=clean_citations,
                        text=document[0],
                    )
                )
            else:
                # Journal
                response.append(
                    ResponseCitation(
                        doc_type="journal",
                        reference=metadata["url"],
                        date=metadata["date"][:10],
                        name=f'{metadata["author"]}; "{metadata["title"]}"',
                        citation=metadata["citation"],
                        text=document[0],
                    )
                )
        return response

    def handle_query(self, query: str) -> QueryResponse:
        """Handle a query.

        The key entrypoint for hybrid and semantic search

        Args:
            query (str): The natural language legal query.

        Returns:
            QueryResponse: The response to the query, as a specialised object to
              support the variety of possible response shapes.
        """
        logger.debug("handle_query invoked with [%s]", query)
        # Validate our query parameter.
        # Not sure what validations make sense yet - TODO.
        if not isinstance(query, str):
            raise TypeError("Query must be a string!")

        # Perform an LLM request to generate our search filter
        try:
            gen_filter_request = ChatRequest(
                message=GenerateSearchFilter.get_prompt(query), temperature=0.1
            )
            gen_filter_response = self.llm_implementation.invoke_llm(
                request=gen_filter_request
            )
            filter_response: SearchFilter = parser_utils.parse_response(
                gen_filter_response.text, GenerateSearchFilter.get_schema()
            )
        except Exception as err:
            logger.error(err)
            raise SearchError(error_code="EGENFILTER", status=500) from err
        logger.debug(filter_response)
        # Use extracted filters to run our searches
        relevant_docs = self._search_documents_with_filters(
            query=query, search_filter=filter_response
        )
        return QueryResponse(title="", text="", citations=relevant_docs)

    def extract_predicate_from_query(self, query: str) -> tuple[str, dict]:
        """Extract search filters from query and return SQL predicate.
        
        This method allows external code to get the predicate without running the full search.
        
        Args:
            query (str): The natural language legal query.
            
        Returns:
            tuple[str, dict]: The SQL predicate string and values dictionary
        """
        logger.debug("extract_predicate_from_query invoked with [%s]", query)
        
        try:
            gen_filter_request = ChatRequest(
                message=GenerateSearchFilter.get_prompt(query), temperature=0.1
            )
            gen_filter_response = self.llm_implementation.invoke_llm(
                request=gen_filter_request
            )
            filter_response: SearchFilter = parser_utils.parse_response(
                gen_filter_response.text, GenerateSearchFilter.get_schema()
            )
            
            # Generate predicate from filters
            if "filter" in filter_response and filter_response["filter"]:
                predicate, values = self._generate_search_predicate(filter_response["filter"])
                logger.debug(f"Generated predicate: {predicate}")
                return predicate, values
            else:
                logger.debug("No filters found in query")
                return "", {}
                
        except Exception as err:
            logger.error(f"Error extracting predicate from query: {err}")
            return "", {}

    def _generate_search_predicate(
        self, search_filters: list[AttributeFilter]
    ) -> tuple[str, dict]:
        """Generate a SQL predicate from the supplied filters

        Args:
            search_filters (list[dict]): The list of filters to apply

        Returns:
            str, dict: The SQL predicate and a dict with the named attributes being used
        """
        predicate = []
        values = {}
        for att_filter in search_filters:
            # Ignore unknown equalities
            if att_filter["equality"] not in equality_lookup:
                logger.warning(
                    "Search filter contained an invalid equality operator [%s]",
                    att_filter["equality"],
                )
                continue
            # Source and court have some special handling
            if att_filter["attribute"] == "source" and att_filter["equality"] == "eq":
                # Approximate match using 'ILIKE'
                # The name however is in different places depending on the
                # document type, so we do some real hackiness here to make it line up.
                # TODO: Fix the data model!
                # A year is often included in case names - that needs to be stripped.
                source_name = att_filter["value"].strip()
                year_matches = re.search(r"(?:\[|\()?(\d{4})(?:\]|\)?)$", source_name)
                if year_matches is not None:
                    # It might make sense to add a date filter based upon this.
                    # Needs evaluation.
                    source_name = source_name[:year_matches.start()].strip()
                predicate.append(
                    "(LOWER((cmetadata->>'name')) ILIKE %(source)s OR LOWER((cmetadata->>'title')) ILIKE %(source)s OR LOWER((cmetadata->>'titles')) ILIKE %(source)s)"
                )
                values["source"] = f"%{source_name.lower()}%"
                continue
            # TODO: We should handle court appropriately, but as we only have
            # one source for the POC, just ignore it.
            if att_filter["attribute"] == "court":
                continue
            if att_filter["attribute"] in GenerateSearchFilter.filter_attributes:
                values[att_filter["attribute"]] = att_filter["value"].lower()
                predicate.append(
                    f"LOWER((cmetadata->>'{att_filter["attribute"]}')) {equality_lookup[att_filter["equality"]]} %({att_filter["attribute"]})s"
                )
        return " AND ".join(predicate), values

    def _search_documents_with_filters(
        self,
        query: str,
        search_filter: SearchFilter,
    ) -> list[ResponseCitation]:
        """Search the legislation sections, using both keyphrase and semantic

        Args:
            query (str): The search query
            search_filters (list[dict]): A list of search filters

        Returns:
            list[ResponseCitation]: List of documents
        """
        # Search key phrases:
        key_phrase_docs = []
        if "key_phrases" in search_filter and len(search_filter["key_phrases"]) > 0:
            key_phrase_docs = self._search_documents_based_on_key_phrases(
                search_filter, self.max_results
            )
        # Search on semantic similarity.
        semantic_docs = self._semantic_search_documents(
            query, search_filter, self.max_results
        )
        # Merge the results - keyphrase first, since that has better scoring
        for document in semantic_docs:
            # Since we may have multiple chunks from the same source we need to
            # use the contents of the chunk - this is probably inefficient...
            if document.text not in map(
                lambda doc: doc.text, key_phrase_docs
            ):
                key_phrase_docs.append(document)
        if not self.deduplicate_results:
            return key_phrase_docs
        # Deduplicate based upon reference
        references = set()
        docs = []
        for document in key_phrase_docs:
            if document.reference not in references:
                docs.append(document)
                references.add(document.reference)
        return docs
        

    def _search_documents_based_on_key_phrases(
        self, search_filter: SearchFilter, max_results: int = 5
    ) -> list[ResponseCitation]:
        """Perform a keyphrase search on legislation sections

        Args:
            search_filters (list[dict]): A list of search filters
            max_results (int, optional): Max number of docs to return. Defaults to 5.

        Returns:
            list[ResponseCitation]: List of documents
        """
        # keyphrases to ts_query...
        ts_query = "|".join(
            [f"({"&".join(k.split(" "))})" for k in search_filter["key_phrases"]]
        )
        if len(ts_query) == 0:
            return []
        predicate = ""
        search_values = {}
        if "filter" in search_filter:
            predicate, search_values = self._generate_search_predicate(
                search_filter["filter"]
            )
        if predicate != "":
            predicate = "AND " + predicate
        search_query = f"""
        SELECT document, cmetadata, ts_rank_cd(document_fts, query) AS rank
            FROM langchain_pg_embedding, plainto_tsquery('english', %(ts_query)s) query
            WHERE document_fts @@ query {predicate}
            ORDER BY rank DESC
            LIMIT %(max_results)s;
        """
        search_values["ts_query"] = ts_query
        search_values["max_results"] = max_results
        with self.db_connection.cursor() as curs:
            curs.execute(search_query, search_values)
            results = curs.fetchall()
        return self._document_and_metadata_to_response_citation(results)

    def _semantic_search_documents(
        self, query: str, search_filter: SearchFilter, max_results: int = 10
    ) -> list[ResponseCitation]:
        """Use Vector search on the documents

        Args:
            query (str): The search query
            search_filter (SearchFilter): The filter returned from the LLM
            max_results (int, optional): Max number of docs to return. Defaults to 10.

        Returns:
            list[ResponseCitation]: List of documents
        """
        # Might be worth investigating HyDE for better search, but also, more
        # LLM == Slower...
        # Generate the predicate from the filters
        predicate = ""
        search_values = {}
        if "filter" in search_filter:
            predicate, search_values = self._generate_search_predicate(
                search_filter["filter"]
            )
        if predicate != "":
            predicate = "WHERE " + predicate
        # Generate a vector for the search query
        embedding = _model.encode(query)
        vector_query = f"""SELECT document, cmetadata, embedding <#> %(embedding)s AS vector_score
        FROM langchain_pg_embedding
        {predicate}
        ORDER BY embedding <#> %(embedding)s ASC
        LIMIT %(max_results)s
        """
        search_values["embedding"] = embedding
        search_values["max_results"] = max_results
        with self.db_connection.cursor() as curs:
            curs.execute(vector_query, search_values)
            results = curs.fetchall()
        return self._document_and_metadata_to_response_citation(results)
