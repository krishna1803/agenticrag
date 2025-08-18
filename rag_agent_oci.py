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
def get_cached_agents(model_id: str, compartment_id: str, vector_store_id: str, cache_config: Dict[str, Any] = None):
    """Get cached agents to avoid repeated initialization"""
    # Check if LLM caching is enabled
    if cache_config is None:
        cache_config = {}
    
    llm_caching_enabled = cache_config.get("LLM_CACHING_ENABLED", True)
    
    cache_key = f"{model_id}_{compartment_id}_{vector_store_id}"
    
    if cache_key not in _agent_cache:
        # Create LLM based on caching configuration
        llm_cache_key = f"{model_id}_{compartment_id}"
        
        if llm_caching_enabled and llm_cache_key not in _llm_cache:
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
            llm = _llm_cache[llm_cache_key]
        elif llm_caching_enabled and llm_cache_key in _llm_cache:
            logger.info(f"Using cached LLM instance for {llm_cache_key}")
            llm = _llm_cache[llm_cache_key]
        else:
            # LLM caching disabled - create fresh LLM instance
            logger.info(f"LLM caching disabled - creating fresh LLM instance for {llm_cache_key}")
            config = load_oci_config()
            llm = ChatOCIGenAI(
                auth_profile=CONFIG_PROFILE,
                model_id=model_id,
                compartment_id=compartment_id,
                service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
                is_stream=False,
                model_kwargs={"temperature": 0, "max_tokens": 1500}
            )
        
        # Create agents cache entry
        # Note: We can't cache vector_store directly due to connection state
        # So we'll cache the agents structure but create vector_store fresh
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
        cached_data = get_cached_agents(model_id, self.compartment_id, vector_store_id, self.cache_config)
        
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

    # ...existing code...
    
    @lru_cache(maxsize=10)  # Priority 3: Cache agent creation
def get_cached_agents(model_id: str, compartment_id: str, vector_store_id: str, cache_config: Dict[str, Any] = None):
    """Get cached agents to avoid repeated initialization"""
    # Check if LLM caching is enabled
    if cache_config is None:
        cache_config = {}
    
    llm_caching_enabled = cache_config.get("LLM_CACHING_ENABLED", True)
    
    cache_key = f"{model_id}_{compartment_id}_{vector_store_id}"
    
    if cache_key not in _agent_cache:
        # Create LLM based on caching configuration
        llm_cache_key = f"{model_id}_{compartment_id}"
        
        if llm_caching_enabled and llm_cache_key not in _llm_cache:
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
            llm = _llm_cache[llm_cache_key]
        elif llm_caching_enabled and llm_cache_key in _llm_cache:
            logger.info(f"Using cached LLM instance for {llm_cache_key}")
            llm = _llm_cache[llm_cache_key]
        else:
            # LLM caching disabled - create fresh LLM instance
            logger.info(f"LLM caching disabled - creating fresh LLM instance for {llm_cache_key}")
            config = load_oci_config()
            llm = ChatOCIGenAI(
                auth_profile=CONFIG_PROFILE,
                model_id=model_id,
                compartment_id=compartment_id,
                service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
                is_stream=False,
                model_kwargs={"temperature": 0, "max_tokens": 1500}
            )
        
        # Create agents cache entry
        # Note: We can't cache vector_store directly due to connection state
        # So we'll cache the agents structure but create vector_store fresh
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
        cached_data = get_cached_agents(model_id, self.compartment_id, vector_store_id, self.cache_config)
        
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

    # ...existing code...
    
    @lru_cache(maxsize=10)  # Priority 3: Cache agent creation
def get_cached_agents(model_id: str, compartment_id: str, vector_store_id: str, cache_config: Dict[str, Any] = None):
    """Get cached agents to avoid repeated initialization"""
    # Check if LLM caching is enabled
    if cache_config is None:
        cache_config = {}
    
    llm_caching_enabled = cache_config.get("LLM_CACHING_ENABLED", True)
    
    cache_key = f"{model_id}_{compartment_id}_{vector_store_id}"
    
    if cache_key not in _agent_cache:
        # Create LLM based on caching configuration
        llm_cache_key = f"{model_id}_{compartment_id}"
        
        if llm_caching_enabled and llm_cache_key not in _llm_cache:
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
            llm = _llm_cache[llm_cache_key]
        elif llm_caching_enabled and llm_cache_key in _llm_cache:
            logger.info(f"Using cached LLM instance for {llm_cache_key}")
            llm = _llm_cache[llm_cache_key]
        else:
            # LLM caching disabled - create fresh LLM instance
            logger.info(f"LLM caching disabled - creating fresh LLM instance for {llm_cache_key}")
            config = load_oci_config()
            llm = ChatOCIGenAI(
                auth_profile=CONFIG_PROFILE,
                model_id=model_id,
                compartment_id=compartment_id,
                service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
                is_stream=False,
                model_kwargs={"temperature": 0, "max_tokens": 1500}
            )
        
        # Create agents cache entry
        # Note: We can't cache vector_store directly due to connection state
        # So we'll cache the agents structure but create vector_store fresh
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
        cached_data = get_cached_agents(model_id, self.compartment_id, vector_store_id, self.cache_config)
        
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

    # ...existing code...
    
    @lru_cache(maxsize=10)  # Priority 3: Cache agent creation
def get_cached_agents(model_id: str, compartment_id: str, vector_store_id: str, cache_config: Dict[str, Any] = None):
    """Get cached agents to avoid repeated initialization"""
    # Check if LLM caching is enabled
    if cache_config is None:
        cache_config = {}
    
    llm_caching_enabled = cache_config.get("LLM_CACHING_ENABLED", True)
    
    cache_key = f"{model_id}_{compartment_id}_{vector_store_id}"
    
    if cache_key not in _agent_cache:
        # Create LLM based on caching configuration
        llm_cache_key = f"{model_id}_{compartment_id}"
        
        if llm_caching_enabled and llm_cache_key not in _llm_cache:
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
            llm = _llm_cache[llm_cache_key]
        elif llm_caching_enabled and llm_cache_key in _llm_cache:
            logger.info(f"Using cached LLM instance for {llm_cache_key}")
            llm = _llm_cache[llm_cache_key]
        else:
            # LLM caching disabled - create fresh LLM instance
            logger.info(f"LLM caching disabled - creating fresh LLM instance for {llm_cache_key}")
            config = load_oci_config()
            llm = ChatOCIGenAI(
                auth_profile=CONFIG_PROFILE,
                model_id=model_id,
                compartment_id=compartment_id,
                service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
                is_stream=False,
                model_kwargs={"temperature": 0, "max_tokens": 1500}
            )
        
        # Create agents cache entry
        # Note: We can't cache vector_store directly due to connection state
        # So we'll cache the agents structure but create vector_store fresh
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
        cached_data = get_cached_agents(model_id, self.compartment_id, vector_store_id, self.cache_config)
        
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

    # ...existing code...
    
    @lru_cache(maxsize=10)  # Priority 3: Cache agent creation
def get_cached_agents(model_id: str, compartment_id: str, vector_store_id: str, cache_config: Dict[str, Any] = None):
    """Get cached agents to avoid repeated initialization"""
    # Check if LLM caching is enabled
    if cache_config is None:
        cache_config = {}
    
    llm_caching_enabled = cache_config.get("LLM_CACHING_ENABLED", True)
    
    cache_key = f"{model_id}_{compartment_id}_{vector_store_id}"
    
    if cache_key not in _agent_cache:
        # Create LLM based on caching configuration
        llm_cache_key = f"{model_id}_{compartment_id}"
        
        if llm_caching_enabled and llm_cache_key not in _llm_cache:
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
            llm = _llm_cache[llm_cache_key]
        elif llm_caching_enabled and llm_cache_key in _llm_cache:
            logger.info(f"Using cached LLM instance for {llm_cache_key}")
            llm = _llm_cache[llm_cache_key]
        else:
            # LLM caching disabled - create fresh LLM instance
            logger.info(f"LLM caching disabled - creating fresh LLM instance for {llm_cache_key}")
            config = load_oci_config()
            llm = ChatOCIGenAI(
                auth_profile=CONFIG_PROFILE,
                model_id=model_id,
                compartment_id=compartment_id,
                service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
                is_stream=False,
                model_kwargs={"temperature": 0, "max_tokens": 1500}
            )
        
        # Create agents cache entry
        # Note: We can't cache vector_store directly due to connection state
        # So we'll cache the agents structure but create vector_store fresh
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
        cached_data = get_cached_agents(model_id, self.compartment_id, vector_store_id, self.cache_config)
        
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

    # ...existing code...
    
    @lru_cache(maxsize=10)  # Priority 3: Cache agent creation
def get_cached_agents(model_id: str, compartment_id: str, vector_store_id: str, cache_config: Dict[str, Any] = None):
    """Get cached agents to avoid repeated initialization"""
    # Check if LLM caching is enabled
    if cache_config is None:
        cache_config = {}
    
    llm_caching_enabled = cache_config.get("LLM_CACHING_ENABLED", True)
    
    cache_key = f"{model_id}_{compartment_id}_{vector_store_id}"
    
    if cache_key not in _agent_cache:
        # Create LLM based on caching configuration
        llm_cache_key = f"{model_id}_{compartment_id}"
        
        if llm_caching_enabled and llm_cache_key not in _llm_cache:
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
            llm = _llm_cache[llm_cache_key]
        elif llm_caching_enabled and llm_cache_key in _llm_cache:
            logger.info(f"Using cached LLM instance for {llm_cache_key}")
            llm = _llm_cache[llm_cache_key]
        else:
            # LLM caching disabled - create fresh LLM instance
            logger.info(f"LLM caching disabled - creating fresh LLM instance for {llm_cache_key}")
            config = load_oci_config()
            llm = ChatOCIGenAI(
                auth_profile=CONFIG_PROFILE,
                model_id=model_id,
                compartment_id=compartment_id,
                service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
                is_stream=False,
                model_kwargs={"temperature": 0, "max_tokens": 1500}
            )
        
        # Create agents cache entry
        # Note: We can't cache vector_store directly due to connection state
        # So we'll cache the agents structure but create vector_store fresh
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
        cached_data = get_cached_agents(model_id, self.compartment_id, vector_store_id, self.cache_config)
        
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

    # ...existing code...
    
    @lru_cache(maxsize=10)  # Priority 3: Cache agent creation
def get_cached_agents(model_id: str, compartment_id: str, vector_store_id: str, cache_config: Dict[str, Any] = None):
    """Get cached agents to avoid repeated initialization"""
    # Check if LLM caching is enabled
    if cache_config is None:
        cache_config = {}
    
    llm_caching_enabled = cache