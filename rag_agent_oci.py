from html import parser
from typing import List, Dict, Any, Optional
import json
import os
import argparse
import logging
import time
import asyncio
import concurrent.futures
import re
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from functools import lru_cache

# --- Added for provenance tracking (Step 1) ---
try:
    from provenance import ProvenanceStore
except ImportError:
    ProvenanceStore = None
# --- End provenance addition ---

try:
    import yaml
except ImportError:
    yaml = None

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        pass

# OCI imports
import oci
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.llms.oci_generative_ai import OCIGenAI
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_core.prompts import PromptTemplate

# Local imports
from agents.agent_factory import create_agents
try:
    from OracleDBVectorStore import OracleDBVectorStore
    ORACLE_DB_AVAILABLE = True
except ImportError:
    ORACLE_DB_AVAILABLE = False
    print("Oracle DB support not available. Install with: pip install oracledb sentence-transformers")
    


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG_PROFILE = "DEFAULT"  # Default OCI config profile

# Optimization features - Add caching for LLM and agent initialization
from functools import lru_cache
_llm_cache = {}
_agent_cache = {}

class LLMBatchProcessor:
    """Handles batch processing of LLM requests for improved performance"""
    
    def __init__(self, llm, batch_config: Dict[str, Any] = None):
        self.llm = llm
        self.logger = logging.getLogger(__name__)
        
        # Set batch processing configuration
        self.batch_config = batch_config or {}
        self.max_concurrent = self.batch_config.get("MAX_CONCURRENT", 3)
        self.request_timeout = self.batch_config.get("REQUEST_TIMEOUT", 60)
        self.batch_timeout = self.batch_config.get("BATCH_TIMEOUT", 120)
        
        self.logger.info(f"Batch processor initialized with max_concurrent={self.max_concurrent}, "
                        f"request_timeout={self.request_timeout}s, batch_timeout={self.batch_timeout}s")
    
    def batch_process_requests(self, requests: List[Dict[str, Any]], max_concurrent: int = None) -> List[str]:
        """
        Process multiple LLM requests in batches to reduce latency
        
        Args:
            requests: List of dicts with 'messages' and 'type' keys
            max_concurrent: Maximum number of concurrent requests (overrides config if provided)
            
        Returns:
            List of response strings in the same order as requests
        """
        if not requests:
            return []
        
        # Use provided max_concurrent or fall back to config
        max_workers = max_concurrent or self.max_concurrent
        
        self.logger.info(f"Processing {len(requests)} LLM requests in batch with {max_workers} workers")
        start_time = time.time()
        
        # For now, we'll use ThreadPoolExecutor for concurrent processing
        # In the future, this could be enhanced with actual batch API calls
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {}
            
            for i, request in enumerate(requests):
                future = executor.submit(self._process_single_request, request)
                future_to_index[future] = i
            
            # Collect results in order with timeout
            results = [None] * len(requests)
            from concurrent.futures import as_completed
            
            try:
                for future in as_completed(future_to_index, timeout=self.batch_timeout):
                    index = future_to_index[future]
                    try:
                        results[index] = future.result(timeout=self.request_timeout)
                    except Exception as e:
                        self.logger.error(f"Error processing request {index}: {str(e)}")
                        results[index] = f"Error processing request: {str(e)}"
            except Exception as e:
                self.logger.error(f"Batch processing timeout or error: {str(e)}")
                # Fill any remaining None results with error messages
                for i, result in enumerate(results):
                    if result is None:
                        results[i] = f"Request {i} timed out or failed"
        
        duration = time.time() - start_time
        self.logger.info(f"Batch processing completed in {duration:.2f} seconds")
        return results
    
    def _process_single_request(self, request: Dict[str, Any]) -> str:
        """Process a single LLM request"""
        try:
            messages = request['messages']
            response = self.llm.invoke(messages)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            self.logger.error(f"Error in single request: {str(e)}")
            return f"Error: {str(e)}"
    
    def batch_reasoning_requests(self, query: str, research_results: List[Dict[str, Any]]) -> List[str]:
        """Create batch reasoning requests for multiple research results - simplified version"""
        reasoning_requests = []
        
        for result in research_results:
            if result.get("findings"):
                findings = result["findings"]
                context_str = "\n\n".join([f"Context {i+1}:\n{item['content']}" for i, item in enumerate(findings)])
                
                # Create simple message structure
                prompt_text = f"""Analyze the information and draw a clear conclusion for this step.
                
                Step: {result["step"]}
                Context: {context_str}
                Query: {query}

                Conclusion: Provide the conclusion in plain text format. DO NOT use LaTeX or special formatting."""
                
                reasoning_requests.append({
                    'messages': [{"role": "user", "content": prompt_text}],
                    'type': 'reasoning',
                    'step': result["step"]
                })
        
        return self.batch_process_requests(reasoning_requests)

@lru_cache(maxsize=10)  # Priority 3: Cache agent creation
# Removed cache_config param to keep lru_cache arguments hashable (dicts are unhashable)
def get_cached_agents(model_id: str, compartment_id: str, vector_store_id: str):
    """Get cached agents to avoid repeated initialization.
    Only depends on hashable identifiers; configuration dict handled outside.
    """
    # Ensure global caches exist
    if not isinstance(model_id, str):
        model_id = str(model_id)
    if not isinstance(compartment_id, str):
        compartment_id = str(compartment_id)
    if not isinstance(vector_store_id, str):
        vector_store_id = str(vector_store_id)
    cache_key = f"{model_id}_{compartment_id}_{vector_store_id}"
    
    if cache_key not in _agent_cache:
        llm_cache_key = f"{model_id}_{compartment_id}"
        if llm_cache_key not in _llm_cache:
            logger.info(f"Creating and caching new LLM instance for {llm_cache_key}")
            config = load_oci_config()
            _llm_cache[llm_cache_key] = ChatOCIGenAI(
                auth_profile=CONFIG_PROFILE,
                model_id=model_id,
                compartment_id=compartment_id,
                service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
                is_stream=False,
                model_kwargs={"temperature": 0, "max_tokens": 1500}
            )
        else:
            logger.info(f"Using cached LLM instance for {llm_cache_key}")
        llm = _llm_cache[llm_cache_key]
        _agent_cache[cache_key] = {
            'llm': llm,
            'structure': None  # Will be created when vector_store is available
        }
    else:
        logger.info(f"Using existing cache entry for {cache_key}")
    
    return _agent_cache[cache_key]
DEBUG = False  # Set to True for detailed debug output
SERVICE_ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
MY_CONFIG_FILE_LOCATION = "~/.oci/config"
#
# load_oci_config(): load the OCI security config
#
def load_oci_config():
    # read OCI config to connect to OCI with API key
    oci_config = oci.config.from_file("~/.oci/config", CONFIG_PROFILE)

    # check the config to access to api keys
    if DEBUG:
        print("OCI Config:")
        print(oci_config)

    return oci_config

class OCIRAGAgent:
    def __init__(self, vector_store: OracleDBVectorStore, use_cot: bool = False, collection: str = None, skip_analysis: bool = False,
                 model_id: str = "cohere.command-latest", compartment_id: str = None, use_stream: bool = False,
                 config: Dict[str, Any] = None):
        """Initialize RAG agent with vector store and OCI Generative AI"""
        self.vector_store = vector_store
        if vector_store is OracleDBVectorStore:
            self.retriever = vector_store.as_retriever()
        else:
            self.retriever = None
        self.use_cot = use_cot
        self.collection = collection
        self.model_id = model_id
        self.compartment_id = compartment_id or os.getenv("OCI_COMPARTMENT_ID")
        self.use_stream = use_stream
        
        # Load configuration
        self.config = config or load_config()
        
        # --- provenance store initialization (Step 1) ---
        self.provenance = ProvenanceStore() if ProvenanceStore else None
        # -------------------------------------------------
        
        # Set similarity search parameters from config
        similarity_config = self.config.get("SIMILARITY_SEARCH", {})
        self.max_results = similarity_config.get("MAX_RESULTS", 3)
        self.max_chunks_per_step = similarity_config.get("MAX_CHUNKS_PER_STEP", 2)
        self.max_findings_per_step = similarity_config.get("MAX_FINDINGS_PER_STEP", 3)
        self.max_tokens_per_finding = similarity_config.get("MAX_TOKENS_PER_FINDING", 1000)
        
        # Set performance parameters from config
        performance_config = self.config.get("PERFORMANCE", {})
        self.batch_config = performance_config.get("BATCH_PROCESSING", {})
        self.cache_config = performance_config.get("CACHING", {})
        self.parallel_config = performance_config.get("PARALLEL_PROCESSING", {})
        self.context_config = performance_config.get("CONTEXT", {})
        
        # Set response quality parameters from config
        quality_config = self.config.get("RESPONSE_QUALITY", {})
        self.max_plan_steps = quality_config.get("MAX_PLAN_STEPS", 10)
        self.remove_latex = quality_config.get("REMOVE_LATEX_FORMATTING", True)
        self.enable_validation = quality_config.get("ENABLE_VALIDATION", True)
        self.fallback_on_error = quality_config.get("FALLBACK_ON_ERROR", True)
        
        logger.info(f"RAG Agent initialized with config: max_results={self.max_results}, "
                   f"max_chunks_per_step={self.max_chunks_per_step}, "
                   f"max_findings_per_step={self.max_findings_per_step}")

        # Priority 3: Use cached LLM and agent initialization
        vector_store_id = str(hash(str(vector_store))) if vector_store else "none"
        cached_data = get_cached_agents(model_id, self.compartment_id, vector_store_id)
        
        self.genai_client = cached_data['llm']
        
        # Initialize specialized agents with configurable caching
        if use_cot:
            # Check if agent caching is enabled in configuration
            agent_caching_enabled = self.cache_config.get("AGENT_CACHING_ENABLED", True)
            
            if agent_caching_enabled and cached_data['structure'] is None:
                logger.info("Creating and caching new agent structure")
                cached_data['structure'] = create_agents(self.genai_client, vector_store)
                self.agents = cached_data['structure']
            elif agent_caching_enabled and cached_data['structure'] is not None:
                logger.info("Using cached agent structure")
                self.agents = cached_data['structure']
            else:
                # Agent caching disabled - create fresh agents for each instance
                logger.info("Agent caching disabled - creating fresh agents")
                self.agents = create_agents(self.genai_client, vector_store)
        else:
            self.agents = None
        
        # Priority 2: Initialize batch processor for LLM operations
        if self.batch_config.get("ENABLED", True):
            self.batch_processor = LLMBatchProcessor(self.genai_client, self.batch_config)
        else:
            self.batch_processor = None
    
    @classmethod
    def warm_cache(cls, model_id: str, compartment_id: str, vector_store_types: List[str] = None):
        """
        Warm up the LLM and agent caches for faster initialization
        
        Args:
            model_id: OCI model ID to cache
            compartment_id: OCI compartment ID
            vector_store_types: List of vector store types to preload
        """
        logger.info(f"Warming cache for model {model_id} in compartment {compartment_id}")
        start_time = time.time()
        
        # Default vector store types if not specified
        if vector_store_types is None:
            vector_store_types = ["oracle", "postgres"]
        
        try:
            # Pre-create LLM instance
            llm_cache_key = f"{model_id}_{compartment_id}"
            if llm_cache_key not in _llm_cache:
                config = load_oci_config()
                _llm_cache[llm_cache_key] = ChatOCIGenAI(
                    auth_profile=CONFIG_PROFILE,
                    model_id=model_id,
                    compartment_id=compartment_id,
                    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
                    is_stream=False,
                    model_kwargs={"temperature": 0, "max_tokens": 1500}
                )
                
            # Pre-create agent structures for different vector store combinations
            for vs_type in vector_store_types:
                cache_key = f"{model_id}_{compartment_id}_{vs_type}"
                if cache_key not in _agent_cache:
                    _agent_cache[cache_key] = {
                        'llm': _llm_cache[llm_cache_key],
                        'structure': None  # Will be created when actual vector_store is available
                    }
            
            duration = time.time() - start_time
            logger.info(f"Cache warming completed in {duration:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error warming cache: {str(e)}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the current cache state"""
        return {
            "llm_cache_size": len(_llm_cache),
            "agent_cache_size": len(_agent_cache),
            "llm_cache_keys": list(_llm_cache.keys()),
            "agent_cache_keys": list(_agent_cache.keys())
        }
    
    def clear_cache(self):
        """Clear all cached LLMs and agents"""
        global _llm_cache, _agent_cache
        _llm_cache.clear()
        _agent_cache.clear()
        logger.info("All caches cleared")
    
    def clear_agent_context(self):
        """Clear any context or state from agents to prevent bleeding between queries"""
        if self.agents and isinstance(self.agents, dict):
            logger.info("Clearing agent context to prevent bleeding between queries")
            
            # If agents have a reset or clear method, call it
            for agent_name, agent in self.agents.items():
                if hasattr(agent, 'clear_context'):
                    agent.clear_context()
                    logger.debug(f"Cleared context for {agent_name} agent")
                elif hasattr(agent, 'reset'):
                    agent.reset()
                    logger.debug(f"Reset {agent_name} agent")
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Get current cache configuration"""
        return {
            "llm_caching_enabled": self.cache_config.get("LLM_CACHING_ENABLED", True),
            "agent_caching_enabled": self.cache_config.get("AGENT_CACHING_ENABLED", True),
            "cache_enabled": self.cache_config.get("ENABLED", True),
            "llm_cache_size": len(_llm_cache),
            "agent_cache_size": len(_agent_cache),
            "current_llm_cache_keys": list(_llm_cache.keys()),
            "current_agent_cache_keys": list(_agent_cache.keys())
        }
    
    def disable_caching(self):
        """Disable all caching for this instance"""
        logger.info("Disabling caching for this RAG agent instance")
        self.cache_config["ENABLED"] = False
        self.cache_config["LLM_CACHING_ENABLED"] = False
        self.cache_config["AGENT_CACHING_ENABLED"] = False
    
    def enable_caching(self):
        """Enable caching for this instance"""
        logger.info("Enabling caching for this RAG agent instance")
        self.cache_config["ENABLED"] = True
        self.cache_config["LLM_CACHING_ENABLED"] = True
        self.cache_config["AGENT_CACHING_ENABLED"] = True

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query using the agentic RAG pipeline"""
        logger.info(f"Processing query with collection: {self.collection}")
        
        # Process based on collection type and CoT setting
        if self.collection == "General Knowledge":
            # For General Knowledge, directly use general response
            if self.use_cot:
                return self._process_query_with_cot(query)
            else:
                return self._generate_general_response(query)
        else:
            # For PDF or Repository collections, use context-based processing
            if self.use_cot:
                return self._process_query_with_cot(query)
            else:
                return self._process_query_standard(query)

    def _process_query_standard(self, query: str) -> Dict[str, Any]:
        """Process query using standard approach without Chain of Thought"""
        context = []
        
        # Get context based on selected collection
        if self.collection == "PDF Collection":
            context = self.vector_store.query_pdf_collection(query)
        elif self.collection == "Repository Collection":
            context = self.vector_store.query_repo_collection(query)
        elif self.collection == "Web Knowledge Base":
            context = self.vector_store.query_web_collection(query)
        
        # Track provenance for retrieved context (Step 1)
        if self.provenance and context:
            for item in context:
                try:
                    self.provenance.add_source(item, step="standard")
                except Exception:
                    pass
        
        # Generate response using context if available, otherwise use general knowledge
        if context:
            response = self._generate_response(query, context)
        else:
            response = self._generate_general_response(query)
        
        # Attach provenance references (Step 1)
        if self.provenance:
            response.setdefault("citations", self.provenance.to_reference_list())
        return response

    def _generate_response(self, query: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a response based on the query and context using OCI Generative AI"""
        formatted_context = "\n\n".join([f"Context {i+1}:\n{item['content']}" 
                                       for i, item in enumerate(context)])
        
        prompt_template = """## Query or Task
{query}

## Retrieved Documents
{formatted_context}

## Response
Answer the question based on the context provided. If the answer is not in the context, say "I don't have enough information to answer this question." Be concise and accurate."""

        prompt = PromptTemplate.from_template(prompt_template)
        formatted_prompt = prompt.format(query=query, formatted_context=formatted_context)
        
        messages = [{"role": "user", "content": formatted_prompt}]
        response = self.genai_client.invoke(messages)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "answer": answer,
            "context": context
        }

    def _generate_general_response(self, query: str) -> Dict[str, Any]:
        """Generate a response using general knowledge when no context is available"""
        user_content = f"Query: {query}\n\nAnswer:"
        
        messages = [{"role": "user", "content": user_content}]
        response = self.genai_client.invoke(messages)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "answer": answer,
            "context": []
        }

    def _process_query_with_cot(self, query: str) -> Dict[str, Any]:
        """Process query using Chain of Thought reasoning with multiple agents"""
        logger.info("Processing query with Chain of Thought reasoning")
        
        # Get initial context based on selected collection
        initial_context = []
        if self.collection == "PDF Collection":
            initial_context = self.vector_store.query_pdf_collection(query)
        elif self.collection == "Repository Collection":
            initial_context = self.vector_store.query_repo_collection(query)
        elif self.collection == "Web Knowledge Base":
            initial_context = self.vector_store.query_web_collection(query)
        
        if not self.agents:
            logger.warning("No agents available for CoT, falling back to standard response")
            return self._generate_general_response(query)
        
        try:
            # --- Track provenance for initial context and retain ordering for citation mapping ---
            initial_context_doc_ids: List[str] = []
            if self.provenance and initial_context:
                for item in initial_context:
                    try:
                        doc_id = self.provenance.add_source(item, step="initial")
                        initial_context_doc_ids.append(doc_id)
                    except Exception:
                        initial_context_doc_ids.append("")
            
            # Step 1: Planning (structured output)
            plan_result = self.agents["planner"].plan(query, initial_context, context_doc_ids=initial_context_doc_ids)
            plan_steps = plan_result.get("steps", [])
            logger.info(f"Planner produced {len(plan_steps)} steps")
            
            # Step 2: Research for each plan step (structured outputs)
            research_results = []
            for step in plan_steps:
                research_dict = self.agents["researcher"].research(
                    query,
                    step,
                    max_chunks=self.max_chunks_per_step,
                    max_findings=self.max_findings_per_step,
                    max_tokens=self.max_tokens_per_finding,
                    max_results=self.max_results,
                    provenance=self.provenance
                )
                research_results.append({"step": step, **research_dict})
            
            # Step 3: Reasoning per research step
            reasoning_results = []
            reasoning_texts: List[str] = []
            for r in research_results:
                findings = r.get("findings", [])
                if not findings:
                    continue
                reasoning_dict = self.agents["reasoner"].reason(
                    query,
                    r["step"],
                    findings  # list of dicts with 'content' and optional 'citations'
                )
                reasoning_results.append({"step": r["step"], **reasoning_dict})
                reasoning_texts.append(reasoning_dict.get("raw", ""))
            
            # Step 4: Synthesis (pass full reasoning dicts for citation mapping)
            if reasoning_results:
                synthesis_dict = self.agents["synthesizer"].synthesize(query, reasoning_results)
                final_answer_text = synthesis_dict.get("answer") or synthesis_dict.get("raw") or ""
            else:
                synthesis_dict = {"answer": "I was unable to generate a complete answer based on the available information.", "raw": ""}
                final_answer_text = synthesis_dict["answer"]
            
            response: Dict[str, Any] = {
                "answer": final_answer_text,
                "plan": plan_result,
                "research": research_results,
                "reasoning_details": reasoning_results,
                "reasoning_steps": reasoning_texts,
                # New unified reasoning field (list of dicts with step + text + optional citations)
                "reasoning": [
                    {
                        "step": rr.get("step"),
                        "text": rr.get("raw") or rr.get("reasoning") or rr.get("conclusion") or rr.get("answer") or "",
                        "citations": rr.get("citations") or rr.get("citation_numbers")
                    }
                    for rr in reasoning_results
                ],
                "synthesis": synthesis_dict,
                "context": initial_context
            }
            # Merge citations from synthesis and prior stages (provenance store already tracks sources)
            if self.provenance:
                response["citations"] = self.provenance.to_reference_list()
            # Include answer-level citation markers for UI
            if synthesis_dict.get("citation_numbers"):
                response["answer_citation_markers"] = synthesis_dict["citation_numbers"]
            return response
            
        except Exception as e:
            logger.error(f"Error in CoT processing: {str(e)}")
            traceback.print_exc()
            response = self._generate_general_response(query)
            if self.provenance:
                response.setdefault("citations", self.provenance.to_reference_list())
            return response


def load_config() -> Dict[str, Any]:
    """Load configuration from config_oci.yaml with default values"""
    # Default configuration values
    default_config = {
        "OCI_COMPARTMENT_ID": "",
        "OCI_MODEL_ID": "meta.llama-4-maverick-17b-128e-instruct-fp8",
        "VECTOR_DB": "postgres",
        "COLLECTION": "PDF Collection",
        "USE_COT": False,
        "SIMILARITY_SEARCH": {
            "MAX_RESULTS": 3,
            "MAX_CHUNKS_PER_STEP": 2,
            "MAX_FINDINGS_PER_STEP": 3,
            "MAX_TOKENS_PER_FINDING": 1000
        },
        "PERFORMANCE": {
            "BATCH_PROCESSING": {
                "ENABLED": True,
                "MAX_CONCURRENT": 3,
                "REQUEST_TIMEOUT": 60,
                "BATCH_TIMEOUT": 120
            },
            "CACHING": {
                "ENABLED": True,
                "LLM_CACHE_SIZE": 10,
                "AGENT_CACHE_SIZE": 10,
                "WARM_CACHE_ON_STARTUP": True
            },
            "PARALLEL_PROCESSING": {
                "ENABLED": True,
                "MAX_WORKERS": 3,
                "USE_THREAD_POOL": True
            },
            "CONTEXT": {
                "MAX_TOKENS": 12000,
                "CHAR_TO_TOKEN_RATIO": 4,
                "AUTO_LIMIT_CONTEXT": True
            }
        },
        "RESPONSE_QUALITY": {
            "REMOVE_LATEX_FORMATTING": True,
            "ENABLE_VALIDATION": True,
            "MAX_PLAN_STEPS": 10,
            "FALLBACK_ON_ERROR": True
        },
        "LOGGING": {
            "LEVEL": "INFO",
            "LOG_PROMPTS": True,
            "LOG_PERFORMANCE": True,
            "LOG_CACHE_STATS": True
        }
    }
    
    try:
        config_path = Path("config_oci.yaml")
        if not config_path.exists():
            logger.info("config_oci.yaml not found, using default configuration")
            return default_config
        
        if yaml is None:
            logger.warning("PyYAML not available, using default configuration")
            return default_config
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        if not config:
            logger.warning("Empty config file, using default configuration")
            return default_config
            
        # Merge loaded config with defaults (deep merge)
        def deep_merge(default: Dict, loaded: Dict) -> Dict:
            result = default.copy()
            for key, value in loaded.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        merged_config = deep_merge(default_config, config)
        return merged_config
        
    except Exception as e:
        print(f"Warning: Error loading config: {str(e)}. Using default configuration.")
        return default_config


def process_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """Process a query request using the RAG agent"""
    try:
        # Load Postgres DB credentials from config_pg.yaml
        credentials = load_config()

        compartment_id = credentials.get("OCI_COMPARTMENT_ID", "")
        collection = credentials.get("COLLECTION", "PDF Collection")
        model_id = request["model"] or credentials.get("OCI_MODEL_ID", "meta.llama-4-maverick-17b-128e-instruct-fp8")
        vector_db = credentials.get("VECTOR_DB", "postgres")
        # The value in the request is a boolean, just pass it through
        use_cot = request.get("use_cot", False)

        # Check for OCI compartment ID
        if not compartment_id:
            return {"error": "OCI_COMPARTMENT_ID not found in config file"}

        # Initialize vector store based on configuration
        if vector_db == "oracle":
            if not ORACLE_DB_AVAILABLE:
                return {"error": "Oracle DB support is not available. Install with: pip install oracledb sentence-transformers"}
            
            # Initialize Oracle DB Vector Store
            store = OracleDBVectorStore(collection_name=collection)
        else:
            # Initialize Postgres Vector Store (assuming similar interface)
            from PostgresVectorStore import PostgresVectorStore
            store = PostgresVectorStore(collection_name=collection)

        if not store:
            return {"error": "Failed to initialize vector store"}

        # Create the agent
        agent = OCIRAGAgent(
            store,
            use_cot=use_cot,
            collection=collection,
            model_id=model_id,
            compartment_id=compartment_id,
            use_stream=False,
            config=credentials
        )

        # Process the query
        response = agent.process_query(request["query"])
        
        # Ensure citations present if provenance enabled (Step 1)
        if getattr(agent, "provenance", None) and "citations" not in response and agent.provenance:
            try:
                response["citations"] = agent.provenance.to_reference_list()
            except Exception:
                pass
        
        # Return the response dictionary
        return response
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {"error": str(e)}


def _format_pages(pages_value):
    """Normalize page number representations to a comma-separated string.
    Accepts list/tuple/set/str/int/None. Deduplicates & preserves numeric ordering."""
    if pages_value in (None, ""):
        return "n/a"
    # If already a string containing commas or digits only
    if isinstance(pages_value, str):
        # Split on non-digit separators, gather ints
        parts = re.split(r"[^0-9]+", pages_value)
        nums = [int(p) for p in parts if p.isdigit()]
    elif isinstance(pages_value, (list, tuple, set)):
        nums = []
        for p in pages_value:
            if p is None: 
                continue
            if isinstance(p, int):
                nums.append(p)
            elif isinstance(p, str) and p.strip():
                # Could be range like "3-5"; expand minimally
                if re.match(r"^\d+$", p.strip()):
                    nums.append(int(p.strip()))
                elif re.match(r"^\d+\s*[-:]\s*\d+$", p.strip()):
                    a, b = re.split(r"[-:]", p.strip())
                    try:
                        a_i, b_i = int(a), int(b)
                        if a_i <= b_i and b_i - a_i <= 50:  # guardrail
                            nums.extend(range(a_i, b_i + 1))
                    except Exception:
                        pass
            elif isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(x, int) for x in p):
                a_i, b_i = p
                if a_i <= b_i and b_i - a_i <= 50:
                    nums.extend(range(a_i, b_i + 1))
        
    elif isinstance(pages_value, int):
        nums = [pages_value]
    else:
        return str(pages_value)
    if not nums:
        return "n/a"
    # Deduplicate & sort
    nums = sorted(set(nums))
    return ",".join(str(n) for n in nums)

def main():
    import argparse
    import os
    import re
    import traceback
    from dotenv import load_dotenv
    
    parser = argparse.ArgumentParser(description="Query documents using OCI Generative AI")
    parser.add_argument("--query", required=True, help="Query to process")
    parser.add_argument("--store-path", default="chroma_db", help="Path to the vector store")
    parser.add_argument("--use-cot", action="store_true", help="Enable Chain of Thought reasoning")
    parser.add_argument("--collection", choices=["PDF Collection", "Repository Collection", "Web Knowledge Base", "General Knowledge"], 
                        help="Specify which collection to query")
    parser.add_argument("--model-id", default="cohere.command-latest", help="OCI Gen AI model ID to use")
    parser.add_argument("--compartment-id", help="OCI compartment ID")
    parser.add_argument("--verbose", action="store_true", help="Show full content of sources")
    parser.add_argument("--use-stream", action="store_true", help="Enable streaming responses from OCI Gen AI")
    parser.add_argument("--vector-db", default="oracle", choices=["postgres", "oracle"], help="Type of vector database to use")
    parser.add_argument("--max-chunks-per-step", type=int, 
                        help="Maximum chunks per research step (overrides config file, default from config: varies)")
    parser.add_argument("--max-findings-per-step", type=int, 
                        help="Maximum findings per research step (overrides config file, default from config: varies)")
    parser.add_argument("--max-tokens-per-finding", type=int, 
                        help="Maximum tokens per finding (overrides config file, default from config: varies)")
    parser.add_argument("--max-results", type=int, 
                        help="Maximum similarity search results (overrides config file, default from config: 5)")

    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Load configuration from config file
    credentials = load_config()
    
    # Override config values with command line arguments if provided
    if args.max_chunks_per_step is not None:
        credentials.setdefault("SIMILARITY_SEARCH", {})["MAX_CHUNKS_PER_STEP"] = args.max_chunks_per_step
    if args.max_findings_per_step is not None:
        credentials.setdefault("SIMILARITY_SEARCH", {})["MAX_FINDINGS_PER_STEP"] = args.max_findings_per_step
    if args.max_tokens_per_finding is not None:
        credentials.setdefault("SIMILARITY_SEARCH", {})["MAX_TOKENS_PER_FINDING"] = args.max_tokens_per_finding
    if args.max_results is not None:
        credentials.setdefault("SIMILARITY_SEARCH", {})["MAX_RESULTS"] = args.max_results

    compartment_id = args.compartment_id or credentials.get("OCI_COMPARTMENT_ID", "")
    collection = args.collection or credentials.get("COLLECTION", "PDF Collection")
    model_id = args.model_id or credentials.get("OCI_MODEL_ID", "meta.llama-4-maverick-17b-128e-instruct-fp8")
    vector_db = args.vector_db or credentials.get("VECTOR_DB", "postgres")
    use_cot = args.use_cot or credentials.get("USE_COT", "false").lower() == "true"

    # Check for OCI compartment ID
    if not compartment_id:
        print("✗ Error: OCI_COMPARTMENT_ID not found in config file or command line arguments")
        print("Please set the OCI_COMPARTMENT_ID in config file or provide --compartment-id")
        exit(1)
    
    print("\nInitializing RAG agent...")
    print("=" * 50)
    
    try:
        if args.vector_db == "oracle":
            if not ORACLE_DB_AVAILABLE:
                print("✗ Error: Oracle DB support is not available. Install with: pip install oracledb sentence-transformers")
                exit(1)
            
            # Initialize Oracle DB Vector Store
            store = OracleDBVectorStore(collection_name=args.collection)
        else:
            # Initialize Postgres Vector Store (assuming similar interface)
            from PostgresVectorStore import PostgresVectorStore
            print("Using Postgres Vector Store")
            store = PostgresVectorStore(collection_name=args.collection)
        if not store:
            print("✗ Error: Failed to initialize vector store")
            exit(1)
            
        agent = OCIRAGAgent(
            store,
            use_cot=use_cot,
            collection=collection,
            model_id=model_id,
            compartment_id=compartment_id,
            use_stream=args.use_stream,
            config=credentials  # Pass the full configuration instead of individual parameters
        )
    
        print(f"\nProcessing query: {args.query}")
        print("=" * 50)
        
        response = agent.process_query(args.query)
        
        # Ensure reasoning field populated if CoT used but missing
        if use_cot and response.get("reasoning") is None and response.get("reasoning_steps"):
            response["reasoning"] = [
                {"step": i+1, "text": txt} for i, txt in enumerate(response.get("reasoning_steps", []))
            ]
        
        # In the main function, add this check before printing the answer
        if "The final answer is:" in response.get("answer", ""):
            # Extract only what follows "The final answer is:"
            response["answer"] = re.sub(r'.*The final answer is:\s*', '', response["answer"])
            # Remove any remaining LaTeX formatting
            response["answer"] = re.sub(r'\\\\[.*?\\\\]|\\$.*?\\$|\\\\\(.*?\\\\\)', '', response["answer"])
        
        print("\nResponse:")
        print("-" * 50)
        print(response.get("answer", "(no answer)"))
        
        # Print citations if available (Step 1)
        if response.get("citations"):
            print("\nCitations:")
            print("-" * 50)
            for c in response["citations"]:
                cid = c.get("id")
                source = c.get("source")
                pages_fmt = _format_pages(c.get("pages"))
                used = ",".join(c.get("used_in_steps", []))
                print(f"[{cid}] {source} pages={pages_fmt} steps={used}")
        
        # Ensure reasoning steps are shown explicitly when CoT enabled
        if use_cot:
            steps = response.get("reasoning") or response.get("reasoning_steps") or []
            print("\nReasoning Steps:")
            print("-" * 50)
            if steps:
                # steps could be list of dicts or list of strings
                for i, step in enumerate(steps):
                    print(f"\nStep {i+1}:")
                    if isinstance(step, dict):
                        print(step.get("text", ""))
                    else:
                        print(step)
            else:
                print("(none returned)")
        elif response.get("reasoning") or response.get("reasoning_steps"):
            print("\nReasoning Steps (CoT disabled):")
            print("-" * 50)
            steps = response.get("reasoning") or response.get("reasoning_steps")
            for i, step in enumerate(steps):
                print(f"\nStep {i+1}:")
                if isinstance(step, dict):
                    print(step.get("text", ""))
                else:
                    print(step)
        
        if response.get("context"):
            print("\nSources used:")
            print("-" * 50)
            for i, ctx in enumerate(response["context"]):
                source = ctx["metadata"].get("source", "Unknown")
                if "page_numbers" in ctx["metadata"]:
                    pages_fmt = _format_pages(ctx["metadata"].get("page_numbers", []))
                    print(f"[{i+1}] {source} (pages: {pages_fmt})")
                else:
                    file_path = ctx["metadata"].get("file_path", "Unknown")
                    print(f"[{i+1}] {source} (file: {file_path})")
                if args.verbose:
                    content_preview = ctx["content"][:300] + "..." if len(ctx["content"]) > 300 else ctx["content"]
                    print(f"    Content: {content_preview}\n")
    
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()