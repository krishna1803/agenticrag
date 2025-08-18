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
from search_filter_extractor import SearchFilterExtractor
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
def get_cached_agents(model_id: str, compartment_id: str, vector_store_id: str):
    """Get cached agents to avoid repeated initialization"""
    cache_key = f"{model_id}_{compartment_id}_{vector_store_id}"
    
    if cache_key not in _agent_cache:
        # Create LLM if not cached
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
        
        # Create agents with cached LLM
        llm = _llm_cache[llm_cache_key]
        # Note: We can't cache vector_store directly due to connection state
        # So we'll cache the agents structure but create vector_store fresh
        _agent_cache[cache_key] = {
            'llm': llm,
            'structure': None  # Will be created when vector_store is available
        }
    
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
        cached_data = get_cached_agents(model_id, self.compartment_id, vector_store_id)
        
        self.genai_client = cached_data['llm']
        
        # Initialize specialized agents with caching
        if use_cot:
            if cached_data['structure'] is None:
                cached_data['structure'] = create_agents(self.genai_client, vector_store)
            self.agents = cached_data['structure']
        else:
            self.agents = None
        
        # Priority 2: Initialize batch processor for LLM operations
        if self.batch_config.get("ENABLED", True):
            self.batch_processor = LLMBatchProcessor(self.genai_client, self.batch_config)
        else:
            self.batch_processor = None
        
        # Initialize search filter extractor for predicate-based similarity search
        self.search_filter_extractor = SearchFilterExtractor()
    
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
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query using the agentic RAG pipeline"""
        logger.info(f"Processing query with collection: {self.collection}")
        
        try:
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
        except Exception as e:
            logger.error(f"Error processing query '{query}': {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return a valid response structure even on error
            error_message = f"I apologize, but I encountered an error while processing your query: {str(e)}"
            return {
                "answer": error_message,
                "context": [],
                "error": str(e)
            }
    
    def _process_query_with_cot(self, query: str) -> Dict[str, Any]:
        """Process query using Chain of Thought reasoning with multiple agents"""
        logger.info("Processing query with Chain of Thought reasoning")
        cot_start_time = time.time()
        
        # Get initial context based on selected collection
        initial_context = []
        try:
            # Use predicate-based context retrieval
            logger.info(f"Retrieving context from {self.collection} for query: '{query}'")
            context_chunks = self._get_context_with_predicate_filtering(query, self.collection)
            initial_context.extend(context_chunks)
            logger.info(f"Retrieved {len(context_chunks)} chunks from {self.collection}")
            
            # Log each chunk with citation number but not full content
            for i, chunk in enumerate(context_chunks):
                source = chunk["metadata"].get("source", "Unknown")
                pages = chunk["metadata"].get("page_numbers", [])
                logger.info(f"Source [{i+1}]: {source} (pages: {pages})")
                content_preview = chunk["content"][:150] + "..." if len(chunk["content"]) > 150 else chunk["content"]
                logger.info(f"Content preview for source [{i+1}]: {content_preview}")
                
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            # Fallback to standard retrieval if predicate-based fails
                
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            # Fallback to standard retrieval if predicate-based fails
            try:
                if self.collection == "PDF Collection":
                    logger.info(f"Fallback: Retrieving context from PDF Collection for query: '{query}'")
                    pdf_context = self.vector_store.query_pdf_collection(query)
                    initial_context.extend(pdf_context)
                    logger.info(f"Retrieved {len(pdf_context)} chunks from PDF Collection")
                elif self.collection == "Repository Collection":
                    logger.info(f"Fallback: Retrieving context from Repository Collection for query: '{query}'")
                    repo_context = self.vector_store.query_repo_collection(query)
                    initial_context.extend(repo_context)
                    logger.info(f"Retrieved {len(repo_context)} chunks from Repository Collection")
                elif self.collection == "Web Knowledge Base":
                    logger.info(f"Fallback: Retrieving context from Web Knowledge Base for query: '{query}'")
                    web_context = self.vector_store.query_web_collection(query)
                    initial_context.extend(web_context)
                    logger.info(f"Retrieved {len(web_context)} chunks from Web Knowledge Base")
                else:
                    logger.info("Using General Knowledge collection, no context retrieval needed")
            except Exception as fallback_error:
                logger.error(f"Error in fallback context retrieval: {str(fallback_error)}")
                initial_context = []
            
            # Apply token budget to context
            initial_context = self._limit_context(initial_context, max_tokens=40000)
            
            # Step 1: Planning - Get steps to follow
            logger.info("Step 1: Planning")
            if not self.agents or "planner" not in self.agents:
                logger.warning("No planner agent available, using direct response")
                return self._generate_general_response(query)
            
            try:
                plan = self.agents["planner"].plan(query, initial_context)
                logger.info(f"Generated plan:\n{plan}")
            except Exception as e:
                logger.error(f"Error in planning step: {str(e)}")
                logger.info("Falling back to general response")
                return self._generate_general_response(query)
            
            # Step 2: Research each step (parallel execution with optimized batching)
            logger.info("Step 2: Research (parallel execution with batch optimization)")
            research_start_time = time.time()
            
            plan_steps = [step for step in plan.split("\n") if step.strip()]
            if len(plan_steps) > self.max_plan_steps:
                logger.info(f"Limiting plan from {len(plan_steps)} steps to {self.max_plan_steps} steps (configurable)")
                plan_steps = plan_steps[:self.max_plan_steps]

            # Use improved parallel processing with ThreadPoolExecutor
            try:
                if self.parallel_config.get("ENABLED", True) and self.parallel_config.get("USE_THREAD_POOL", True):
                    logger.info("Attempting parallel research execution with ThreadPoolExecutor")
                    research_results = self._research_steps_parallel_fixed(query, plan_steps)
                else:
                    logger.info("Parallel processing disabled, using sequential research")
                    research_results = self._research_steps_sequential(query, plan_steps)
                    
                research_duration = time.time() - research_start_time
                logger.info(f"Research completed for {len(research_results)} steps in {research_duration:.2f} seconds")
            
            except Exception as e:
                logger.error(f"Error in research: {str(e)}")
                if self.fallback_on_error:
                    logger.info("Falling back to sequential research")
                    research_results = self._research_steps_sequential(query, plan_steps)
                    research_duration = time.time() - research_start_time
                    logger.info(f"Sequential research completed in {research_duration:.2f} seconds")
                else:
                    raise e
            
            # Step 3: Batch reasoning for better efficiency
            logger.info("Step 3: Reasoning (batch processing)")
            reasoning_steps = []
            
            # Try batch reasoning first for better performance
            try:
                if self.batch_processor and self.batch_config.get("ENABLED", True):
                    logger.info("Attempting batch reasoning processing")
                    batch_start_time = time.time()
                    
                    # Use the batch processor for reasoning
                    batch_reasoning_results = self.batch_processor.batch_reasoning_requests(query, research_results)
                    
                    # Convert batch results to reasoning steps
                    for i, result in enumerate(batch_reasoning_results):
                        if result and not result.startswith("Error"):
                            reasoning_steps.append(result)
                        else:
                            logger.warning(f"Batch reasoning failed for step {i+1}, using fallback")
                            # Fall back to individual reasoning for this step
                            research_result = research_results[i] if i < len(research_results) else None
                            if research_result and self.agents.get("reasoner") and research_result.get("findings"):
                                findings = research_result["findings"] or [{"content": "No specific information found.", 
                                                                       "metadata": {"source": "General Knowledge"}}]
                                if findings:
                                    step_reasoning = self.agents["reasoner"].reason(query, research_result["step"], findings)
                                    reasoning_steps.append(step_reasoning)
                            else:
                                reasoning_steps.append(f"For {research_result['step'] if research_result else 'unknown step'}, insufficient information was found.")
                    
                    batch_duration = time.time() - batch_start_time
                    logger.info(f"Batch reasoning completed in {batch_duration:.2f} seconds for {len(reasoning_steps)} steps")
                else:
                    logger.info("Batch processing disabled, using sequential reasoning")
                    # Use sequential reasoning
                    for result in research_results:
                        try:
                            if self.agents.get("reasoner") and result.get("findings"):
                                findings = result["findings"] or [{"content": "No specific information found.", 
                                                               "metadata": {"source": "General Knowledge"}}]
                                # Only process if we have findings
                                if findings:
                                    step_reasoning = self.agents["reasoner"].reason(query, result["step"], findings)
                                    reasoning_steps.append(step_reasoning)
                        except Exception as e:
                            logger.error(f"Error reasoning about step '{result['step']}': {str(e)}")
                            # Add a fallback reasoning if the step fails
                            reasoning_steps.append(f"For {result['step']}, insufficient information was found.")
                
            except Exception as e:
                logger.error(f"Error in reasoning: {str(e)}")
                if self.fallback_on_error:
                    logger.info("Falling back to sequential reasoning")
                    
                    # Fall back to sequential reasoning if batch fails
                    for result in research_results:
                        try:
                            if self.agents.get("reasoner") and result.get("findings"):  # Check if findings exist
                                findings = result["findings"] or [{"content": "No specific information found.", 
                                                               "metadata": {"source": "General Knowledge"}}]
                                # Only process if we have findings
                                if findings:
                                    step_reasoning = self.agents["reasoner"].reason(query, result["step"], findings)
                                    reasoning_steps.append(step_reasoning)
                        except Exception as e:
                            logger.error(f"Error reasoning about step '{result['step']}': {str(e)}")
                            # Add a fallback reasoning if the step fails
                            reasoning_steps.append(f"For {result['step']}, insufficient information was found.")
                else:
                    raise e

            # Check if we have any reasoning steps before synthesis
            if not reasoning_steps:
                logger.warning("No valid reasoning steps generated. Using general response.")
                return self._generate_general_response(query)
            
            # Step 4: Synthesize final answer with batch optimization
            logger.info("Step 4: Synthesis (with batch optimization)")
            # Before synthesis
            logger.info(f"Reasoning steps count: {len(reasoning_steps)}")
            if not reasoning_steps:
                logger.error("No reasoning steps available for synthesis")
                return self._generate_general_response(query)
                
            # Add debug info
            for i, step in enumerate(reasoning_steps):
                logger.info(f"Reasoning step {i+1} type: {type(step)}, length: {len(str(step))}")
            
            try:
                # Use optimized batch synthesis first
                logger.info("Attempting optimized batch synthesis")
                final_answer = self._batch_synthesis_optimize(query, reasoning_steps)
                
                # Add explicit null check
                if not final_answer:
                    logger.warning("Empty response from optimized synthesis, falling back to agent synthesis")
                    final_answer = self.agents["synthesizer"].synthesize(query, reasoning_steps)
                    
                if not final_answer:
                    logger.error("Empty response from both synthesis methods")
                    final_answer = "I was unable to generate a complete answer based on the available information."
                
                # Remove LaTeX formatting from final answer
                final_answer = self._remove_latex_formatting(final_answer)
                
                total_duration = time.time() - cot_start_time
                logger.info(f"Final synthesized answer generated - Total CoT time: {total_duration:.2f} seconds")
                
                return {
                    "answer": final_answer,
                    "context": initial_context,
                    "reasoning_steps": reasoning_steps
                }
            except Exception as e:
                logger.error(f"Error in synthesis step: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())  # Print full stack trace
                # Emergency fallback - join reasoning steps directly
                fallback_answer = "\n\n".join([
                    f"Step {i+1}: {step[:200]}..." 
                    for i, step in enumerate(reasoning_steps) if isinstance(step, str) and step.strip()
                ])
                return {
                    "answer": f"Here's what I found:\n\n{fallback_answer}",
                    "context": initial_context,
                    "reasoning_steps": reasoning_steps
                }
                
        except Exception as e:
            logger.error(f"Error in CoT processing: {str(e)}")
            return self._generate_general_response(query)
    
    def _remove_latex_formatting(self, text: str) -> str:
        """Remove LaTeX-style formatting from the text using regex for more robust cleaning"""
        if not text:
            return text
        
        import re
        
        # Use regex to clean boxed expressions and other LaTeX patterns
        patterns = [
            (r'\$\\boxed\{([^}]+)\}\$', r'\1'),  # $\boxed{text}$
            (r'\$\\boxed\{([^}]+)\}', r'\1'),    # $\boxed{text}
            (r'\\boxed\{([^}]+)\}\$', r'\1'),    # \boxed{text}$
            (r'\\boxed\{([^}]+)\}', r'\1'),      # \boxed{text}
            (r'boxed\{([^}]+)\}', r'\1'),        # boxed{text} without backslash
        ]
        
        result = text
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result)
        
        # Remove any remaining LaTeX indicators
        result = result.replace('$', '')
        result = result.replace('\\', '')
        result = result.replace('boxed{', '').replace('}', '')
        
        return result.strip()
    
    def _process_query_standard(self, query: str) -> Dict[str, Any]:
        """Process query using standard approach without Chain of Thought"""
        # Initialize context variables
        context = []
        
        # Get context based on selected collection using predicate-based retrieval
        try:
            logger.info(f"Retrieving context from {self.collection} for query: '{query}'")
            context = self._get_context_with_predicate_filtering(query, self.collection)
            logger.info(f"Retrieved {len(context)} chunks from {self.collection}")
            
            for i, chunk in enumerate(context):
                source = chunk["metadata"].get("source", "Unknown")
                pages = chunk["metadata"].get("page_numbers", [])
                logger.info(f"Source [{i+1}]: {source} (pages: {pages})")
                content_preview = chunk["content"][:150] + "..." if len(chunk["content"]) > 150 else chunk["content"]
                logger.debug(f"Content preview for source [{i+1}]: {content_preview}")
                
        except Exception as e:
            logger.error(f"Error retrieving context with predicate: {str(e)}")
            # Fallback to standard context retrieval
            try:
                if self.collection == "PDF Collection":
                    logger.info(f"Fallback: Retrieving context from PDF Collection for query: '{query}'")
                    context = self.vector_store.query_pdf_collection(query)
                elif self.collection == "Repository Collection":
                    logger.info(f"Fallback: Retrieving context from Repository Collection for query: '{query}'")
                    context = self.vector_store.query_repo_collection(query)
                elif self.collection == "Web Knowledge Base":
                    logger.info(f"Fallback: Retrieving context from Web Knowledge Base for query: '{query}'")
                    context = self.vector_store.query_web_collection(query)
                    
                logger.info(f"Retrieved {len(context)} chunks from {self.collection} (fallback)")
            except Exception as fallback_error:
                logger.error(f"Error in fallback context retrieval: {str(fallback_error)}")
                context = []
        
        # Generate response using context if available, otherwise use general knowledge
        if context:
            logger.info(f"Generating response using {len(context)} context chunks")
            response = self._generate_response(query, context)
        else:
            logger.info("No context found, using general knowledge")
            response = self._generate_general_response(query)
        
        return response
    
    def _generate_response(self, query: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a response based on the query and context using OCI Generative AI"""
        # Format context for the prompt
        formatted_context = "\n\n".join([f"Context {i+1}:\n{item['content']}" 
                                       for i, item in enumerate(context)])
        
        #system_prompt = """You are an AI assistant answering questions based on the provided context.
#Answer the question based on the context provided. If the answer is not in the context, say "I don't have enough information to answer this question." Be concise and accurate."""
        
        #user_content = f"Context:\n{formatted_context}\n\nQuestion: {query}"
        
        prompt_template = """## AustLII AI Legal Research Assistant - System Instructions

            You are a legal research assistant. Your task is to answer ONLY from the retrieved legal documents and references and citations provided below.

            **Critical Constraints:**
            - Do NOT use outside knowledge unless explicitly authorised.
            - Do NOT reference legal principles, legislation, or cases not mentioned in the provided documents.
            - Do NOT add missing details to citations or complete partial references.
            - Do not make assumptions, guesses, or inferences beyond the text.
            - Do NOT invent content.
            - If you find yourself drawing on legal knowledge beyond the documents, stop and use the fallback response.

            **If the answer is not in the provided documents, respond exactly with:**
            - "I do not have enough information to address your query."

            You must follow the Response Rules exactly.

            ---

            ## Query or Task
            {query}

            ---

            ## Retrieved Documents
            The following legal documents were retrieved from AustLII. These are your only sources. Refer to them as [#].

            {formatted_context}


            ---

            ## Response Rules

            Unless the user asks for 'explanation,' limit the response to a direct answer to the query and no need to output the explanation of each step below.

            1. **Evidence-first approach**
            - Identify and quote/paraphrase only relevant sections from the provided documents.
            - Attribute every quote/paraphrase to [#].

            2. **Summary step**
            - Summarise the key points from the identified sections.
            - Do not add interpretation beyond what is explicitly in the documents.

            3. **Final answer construction**
            - Write your final answer strictly from the summary in step 2.
            - Every factual claim must be supported by a [#] citation.

            4. **Response format**
            - Unless the user asks for 'explanation,' limit the response to a direct answer to the query subject to length limits in Response Requirements.
            - When asked for 'explanation,' provide a step-by-step breakdown.

            5. **Response Requirements**
            Length Limits:
            - Simple factual queries: 1-2 sentences maximum
            - Case summaries: 2-3 sentences maximum
            - Multi-part questions: Up to 5 sentences maximum
            - Complex analysis (explicit user request): Up to 8 sentences maximum
            - Document comparisons: Up to 6 sentences maximum

            When More Detail is Needed:
            If user requires comprehensive information, respond with available details within limits, then add: "For additional analysis, please ask specific follow-up questions about [list 2-3 specific aspects mentioned in documents]."

            Override Conditions:
            Exceed sentence limits ONLY when:
            - Documents contain extensive directly quoted relevant material on the exact query
            - User explicitly requests "detailed analysis with all available information"
            - Multiple documents provide substantial overlapping content on the same narrow topic

            6. **Style and formatting**
            - Use Australian English and formal legal language.
            - First mention of a case → full case name and citation
            - First mention of legislation → short title + jurisdiction

            7. **If no answer is found**
            - Reply: "I do not have enough information to address your query."
            - Do not guess or infer.

            ---

            ## Self-check before final output
            - Have I used ONLY the provided documents?
            - Does every factual claim have a [#] citation?
            - Have I avoided assumptions or external knowledge?
            - Have I avoided referencing cases, legislation, or legal principles not mentioned in the documents?
            - Have I avoided adding or completing citation/reference details not in the documents?
            - Have I used Australian English and the required citation format?
            """

        prompt = PromptTemplate.from_template(prompt_template)
        
        logger.info("Generating response using OCI Generative AI")
        logger.info(f"Query: {query}")
        logger.info(f"Context size: {len(formatted_context)} characters")
        logger.info(f"prompt: {prompt_template}")
        
        if self.use_stream:
            print("Generating streaming response...")
            chain = (
                prompt
                | self.genai_client
            )
            currtime = time.time()
            response = chain.invoke({"query": query,"formatted_context": formatted_context})
            logger.info(f"Response from LLM generated in {time.time() - currtime:.2f} seconds")
            # For streaming, we need to collect the tokens
            answer = ""
            for chunk in response:
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                print(content, end="", flush=True)
                answer += content
            print()  # Add newline after streaming completes
        else:
            # Non-streaming response - use direct LLM invocation
            currtime = time.time()
            formatted_prompt = prompt.format(query=query, formatted_context=formatted_context)
            messages = [{"role": "user", "content": formatted_prompt}]
            response = self.genai_client.invoke(messages)
            answer = response.content if hasattr(response, 'content') else str(response)
            logger.info(f"Response from LLM generated in {time.time() - currtime:.2f} seconds")

        # Add sources to response if available
        sources = {}
        if context:
            # Group sources by document
            for item in context:
                source = item['metadata'].get('source', 'Unknown')
                if source not in sources:
                    sources[source] = set()
                
                # Add page number if available
                if 'page' in item['metadata']:
                    sources[source].add(str(item['metadata']['page']))
                # Add file path if available for code
                if 'file_path' in item['metadata']:
                    sources[source] = item['metadata']['file_path']
            
            # Print concise source information
            print("\nSources detected:")
            for source, details in sources.items():
                if isinstance(details, set):  # PDF with pages
                    pages = ", ".join(sorted(details))
                    print(f"Document: {source} (pages: {pages})")
                else:  # Code with file path
                    print(f"Code file: {source}")
        
        return {
            "answer": answer,
            "context": context
        }

    def _generate_general_response(self, query: str) -> Dict[str, Any]:
        """Generate a response using general knowledge when no context is available"""
        user_content = f"Query: {query}\n\nAnswer:"
        
        currtime = time.time()
        messages = [{"role": "user", "content": user_content}]
        response = self.genai_client.invoke(messages)
        answer = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"General response generated in {time.time() - currtime:.2f} seconds")
        # Return a general response without context
        
        logger.info("No context available, using general knowledge response")    
        return {
            "answer": answer,
            "context": []
        }
    
    def _limit_context(self, documents: List[Dict[str, Any]], max_tokens: int = None) -> List[Dict[str, Any]]:
        """Limit context to fit within a token budget"""
        if not documents:
            return []
        
        # Use provided max_tokens or fall back to config
        if max_tokens is None:
            max_tokens = self.context_config.get("MAX_TOKENS", 12000)
        
        # Check if auto-limiting is enabled
        if not self.context_config.get("AUTO_LIMIT_CONTEXT", True):
            logger.info("Context auto-limiting disabled, returning all documents")
            return documents
        
        limited_docs = []
        token_count = 0
        
        # Use configured character to token ratio
        char_to_token_ratio = self.context_config.get("CHAR_TO_TOKEN_RATIO", 4)
        
        for doc in documents:
            # Estimate tokens in this document
            doc_tokens = len(doc["content"]) // char_to_token_ratio
            
            # If adding this would exceed budget, stop here
            if token_count + doc_tokens > max_tokens:
                # If we haven't added any documents yet, truncate this one
                if not limited_docs:
                    chars_to_keep = max_tokens * char_to_token_ratio
                    truncated_content = doc["content"][:chars_to_keep] + "..."
                    truncated_doc = doc.copy()
                    truncated_doc["content"] = truncated_content
                    limited_docs.append(truncated_doc)
                    logger.info(f"Truncated document to fit token budget ({max_tokens} tokens)")
                break
            
            limited_docs.append(doc)
            token_count += doc_tokens
        
        logger.info(f"Limited context from {len(documents)} to {len(limited_docs)} documents (~{token_count} tokens, max: {max_tokens})")
        return limited_docs

    async def _research_steps_parallel(self, query: str, plan_steps: List[str]) -> List[Dict[str, Any]]:
        """Process research steps in parallel with improved error handling"""
        # Create a single executor for all tasks
        executor = ThreadPoolExecutor(max_workers=min(len(plan_steps), 3))
    
        async def _research_single_step(step: str):
            try:
                # Use the executor passed from the parent function
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    executor,
                    self.agents["researcher"].research,
                    query, step, 
                    self.max_chunks_per_step,
                    self.max_findings_per_step,
                    self.max_tokens_per_finding
                )
                return {"step": step, "findings": result}
            except Exception as e:
                logger.error(f"Error researching step '{step}': {str(e)}")
                return {"step": step, "findings": []}
    
        try:
            # Create tasks for all steps
            tasks = [_research_single_step(step) for step in plan_steps]
        
            # Execute all research tasks concurrently with timeout
            results = await asyncio.gather(*tasks)
        
            # Process results
            research_results = []
            for result in results:
                step = result["step"]
                findings = result["findings"]
                if isinstance(findings, list):
                    research_results.append({"step": step, "findings": findings})
                else:
                    # Handle case where findings might be a dict with a 'findings' key
                    findings_list = findings.get("findings", []) if isinstance(findings, dict) else []
                    research_results.append({"step": step, "findings": findings_list})
            
                logger.info(f"Research for step: {step} - Found {len(findings) if isinstance(findings, list) else 0} findings")
        
            return research_results
    
        finally:
            # Always shut down the executor
            executor.shutdown(wait=False)

    def _batch_reasoning(self, query: str, research_results: List[Dict[str, Any]]) -> List[str]:
        """Process multiple reasoning steps in a single batch"""
        if not self.agents.get("reasoner"):
            raise ValueError("No reasoner agent available")
            
        # Format all steps and findings for a single prompt
        steps_text = []
        for i, result in enumerate(research_results):
            step = result["step"]
            findings = result["findings"]
            
            # Format findings for this step
            findings_text = ""
            for j, finding in enumerate(findings[:2]):  # Limit to 2 findings per step
                content = finding.get("content", "")[:300]  # Truncate long findings
                source = finding.get("metadata", {}).get("source", "Unknown")
                findings_text += f"Finding {j+1} ({source}): {content}\n\n"
                
            if not findings:
                findings_text = "No specific information found for this step."
                
            # Add formatted step with its findings
            steps_text.append(f"STEP {i+1}: {step}\n{findings_text}")
            
        # Join all steps
        all_steps = "\n\n".join(steps_text)
        
        # Create a prompt template for batch reasoning
        template = f"""Based on the query and research findings, provide clear reasoning for each step.

Query: {{query}}

{all_steps}

Provide your analysis for each step separately:
"""
        
        # Call the LLM directly through the reasoner's LLM
        messages = [{"role": "user", "content": template.format(query=query)}]
        
        start_time = time.time()
        response = self.agents["reasoner"].llm.invoke(messages)
        logger.info(f"Batch reasoning completed in {time.time() - start_time:.2f} seconds")
        
        # Parse the response into separate reasoning steps
        reasoning_text = response.content if hasattr(response, "content") else str(response)
        
        # Split by "STEP" markers
        step_parts = reasoning_text.split("STEP")
        reasoning_steps = []
        
        # Process each part (skip first empty part if exists)
        for part in step_parts[1:] if step_parts[0].strip() == "" else step_parts:
            if part.strip():
                # Clean up and add to results
                step_text = part.strip()
                reasoning_steps.append(step_text)
        
        # If parsing failed, use whole response as one step
        if not reasoning_steps:
            reasoning_steps = [reasoning_text]
            
        return reasoning_steps

    def _research_steps_sequential(self, query: str, plan_steps: List[str]) -> List[Dict[str, Any]]:
        """Process research steps sequentially (no asyncio)"""
        research_results = []
        
        for step in plan_steps:
            try:
                logger.info(f"Researching step: {step}")
                findings = self.agents["researcher"].research(
                    query, step, 
                    self.max_chunks_per_step,
                    self.max_findings_per_step,
                    self.max_tokens_per_finding
                )
                research_results.append({"step": step, "findings": findings})
                logger.info(f"Research for step: {step} - Found {len(findings) if isinstance(findings, list) else 0} findings")
            except Exception as e:
                logger.error(f"Error researching step '{step}': {str(e)}")
                research_results.append({"step": step, "findings": []})
        
        return research_results

    def _research_steps_parallel_fixed(self, query: str, plan_steps: List[str]) -> List[Dict[str, Any]]:
        """Process research steps in parallel using ThreadPoolExecutor without asyncio complications"""
        logger.info(f"Starting parallel research for {len(plan_steps)} steps")
        start_time = time.time()
        
        # Use configured max workers
        max_workers = min(len(plan_steps), self.parallel_config.get("MAX_WORKERS", 3))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all research tasks
            future_to_step = {}
            for step in plan_steps:
                future = executor.submit(
                    self._research_single_step,
                    query, step,
                    self.max_chunks_per_step,
                    self.max_findings_per_step,
                    self.max_tokens_per_finding
                )
                future_to_step[future] = step
            
            # Collect results as they complete
            research_results = []
            completed_count = 0
            
            # Use configured timeout
            batch_timeout = self.batch_config.get("BATCH_TIMEOUT", 120)
            request_timeout = self.batch_config.get("REQUEST_TIMEOUT", 60)
            
            for future in concurrent.futures.as_completed(future_to_step, timeout=batch_timeout):
                step = future_to_step[future]
                completed_count += 1
                
                try:
                    findings = future.result(timeout=request_timeout)
                    research_results.append({"step": step, "findings": findings})
                    
                    findings_count = len(findings) if isinstance(findings, list) else 0
                    logger.info(f"Research completed for step ({completed_count}/{len(plan_steps)}): '{step}' - Found {findings_count} findings")
                    
                except Exception as e:
                    logger.error(f"Error researching step '{step}': {str(e)}")
                    research_results.append({"step": step, "findings": []})
            
            # Sort results to maintain step order
            step_order = {step: i for i, step in enumerate(plan_steps)}
            research_results.sort(key=lambda x: step_order.get(x["step"], 999))
        
        duration = time.time() - start_time
        logger.info(f"Parallel research completed in {duration:.2f} seconds for {len(research_results)} steps with {max_workers} workers")
        
        return research_results
    
    def _research_single_step(self, query: str, step: str, max_chunks: int, max_findings: int, max_tokens: int):
        """Execute research for a single step - designed to be thread-safe"""
        try:
            logger.info(f"Researching step: {step}")
            
            # Call the researcher agent's research method with configurable max_results
            findings = self.agents["researcher"].research(
                query, step, 
                max_chunks,
                max_findings,
                max_tokens,
                max_results=self.max_results  # Pass configured max_results
            )
            
            # Ensure we return a list
            if not isinstance(findings, list):
                if isinstance(findings, dict) and "findings" in findings:
                    findings = findings["findings"]
                else:
                    findings = []
            
            return findings
            
        except Exception as e:
            logger.error(f"Error in _research_single_step for '{step}': {str(e)}")
            return []

    def _batch_synthesis_optimize(self, query: str, reasoning_steps: List[str]) -> str:
        """
        Optimized synthesis using batch processing when possible
        
        Args:
            query: The original query
            reasoning_steps: List of reasoning step results
            
        Returns:
            Final synthesized answer
        """
        logger.info("Using optimized batch synthesis")
        start_time = time.time()
        
        try:
            # For synthesis, we typically need to combine all steps in one go
            # So we'll prepare a single optimized prompt that processes all steps together
            if not reasoning_steps:
                return "I don't have enough valid analysis to provide a complete answer."
            
            # Create steps_str with optimization
            steps_str = "\n\n".join([f"Step {i+1}:\n{step}" for i, step in enumerate(reasoning_steps)])
            
            # Use a streamlined prompt for faster processing
            synthesis_prompt = f"""Combine the reasoning steps into a clear, comprehensive answer.

Query: {query}
Steps: {steps_str}

Provide a direct, clear answer in plain text format. DO NOT use LaTeX notation, mathematical symbols like \\boxed{{}}, or markdown formatting.

Answer:"""
            
            # Direct LLM call with optimized message structure
            messages = [{"role": "user", "content": synthesis_prompt}]
            response = self.genai_client.invoke(messages)
            
            # Handle different response formats
            if hasattr(response, "content"):
                result = response.content
            elif isinstance(response, dict) and "content" in response:
                result = response["content"]
            elif isinstance(response, str):
                result = response
            else:
                result = str(response)
            
            duration = time.time() - start_time
            logger.info(f"Optimized batch synthesis completed in {duration:.2f} seconds")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in batch synthesis optimization: {str(e)}")
            # Fall back to agent-based synthesis
            return self.agents["synthesizer"].synthesize(query, reasoning_steps)

    def _get_context_with_predicate_filtering(self, query: str, collection: str) -> List[Dict[str, Any]]:
        """Get context with optional predicate filtering for enhanced search precision"""
        try:
            # Extract predicate from query if available
            if not hasattr(self, 'search_filter_extractor') or self.search_filter_extractor is None:
                logger.warning("SearchFilterExtractor not available, falling back to standard search")
                return self._fallback_context_retrieval(query, collection)
            
            predicate, predicate_values = self.search_filter_extractor.extract_predicate_from_query(query)
            
            logger.info(f"Extracted predicate: '{predicate}' with values: {predicate_values}")
            
            # Ensure predicate_values is not None
            if predicate_values is None:
                predicate_values = {}
            
            if collection == "PDF Collection":
                # Check if vector store supports predicate-based querying
                if hasattr(self.vector_store, 'query_pdf_collection_with_predicate') and predicate:
                    logger.info(f"Using predicate-based search for PDF Collection")
                    return self.vector_store.query_pdf_collection_with_predicate(
                        query, 
                        n_results=self.max_results,
                        predicate=predicate,
                        predicate_values=predicate_values
                    )
                else:
                    logger.info(f"Using standard search for PDF Collection (no predicate or not supported)")
                    return self.vector_store.query_pdf_collection(query, n_results=self.max_results)
                    
            elif collection == "Repository Collection":
                # Currently no predicate support for repo collection, use standard search
                return self.vector_store.query_repo_collection(query, n_results=self.max_results)
                
            elif collection == "Web Collection":
                # Currently no predicate support for web collection, use standard search
                return self.vector_store.query_web_collection(query, n_results=self.max_results)
                
            elif collection == "General Collection":
                # Currently no predicate support for general collection, use standard search
                return self.vector_store.query_general_collection(query, n_results=self.max_results)
                
            else:
                logger.warning(f"Unknown collection: {collection}, using PDF Collection as default")
                return self.vector_store.query_pdf_collection(query, n_results=self.max_results)
                
        except Exception as e:
            logger.error(f"Error in predicate-based context retrieval: {str(e)}")
            import traceback
            traceback.print_exc()
            # Fallback to standard search
            return self._fallback_context_retrieval(query, collection)
    
    def _fallback_context_retrieval(self, query: str, collection: str) -> List[Dict[str, Any]]:
        """Fallback context retrieval method when predicate-based search fails"""
        try:
            if collection == "PDF Collection":
                return self.vector_store.query_pdf_collection(query, n_results=self.max_results)
            elif collection == "Repository Collection":
                return self.vector_store.query_repo_collection(query, n_results=self.max_results)
            elif collection == "Web Collection":
                return self.vector_store.query_web_collection(query, n_results=self.max_results)
            elif collection == "General Collection":
                return self.vector_store.query_general_collection(query, n_results=self.max_results)
            else:
                logger.warning(f"Unknown collection in fallback: {collection}")
                return []
        except Exception as e:
            logger.error(f"Error in fallback context retrieval: {str(e)}")
            return []

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
                print("Warning: config_oci.yaml not found. Using default configuration.")
                return default_config
            
            if yaml is None:
                print("Warning: yaml module not available. Cannot load config file.")
                return default_config
                
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            if not config:
                return default_config
                
            # Merge loaded config with defaults (deep merge)
            def deep_merge(default: Dict, loaded: Dict) -> Dict:
                """Deep merge loaded config with defaults"""
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

        compartment_id  = credentials.get("OCI_COMPARTMENT_ID", "")
        collection = credentials.get("COLLECTION", "PDF Collection")
        model_id = request["model"] or credentials.get("OCI_MODEL_ID", "meta.llama-4-maverick-17b-128e-instruct-fp8")
        vector_db = credentials.get("VECTOR_DB", "postgres")
        #The value in the request is a boolen, just pass it through
        use_cot = request.get("use_cot", False)

        logger.info(f"Use Chain of Thought reasoning: {use_cot}")

        # Check for OCI compartment ID
        if not compartment_id:
            print("✗ Error: OCI_COMPARTMENT_ID not found in config file or command line arguments")
            print("Please set the OCI_COMPARTMENT_ID in config file or provide --compartment-id")
            exit(1)
            
        # Priority 3: Warm cache for better initialization performance
        logger.info("Warming LLM and agent caches for optimal performance")
        OCIRAGAgent.warm_cache(model_id, compartment_id, [vector_db])
        
        #Create vector store based on vector_db type
        if vector_db == "oracle":
            if not ORACLE_DB_AVAILABLE:
                print("✗ Error: Oracle DB support is not available. Install with: pip install oracledb sentence-transformers")
                exit(1)
            # Initialize Oracle DB Vector Store
            vector_db = OracleDBVectorStore(collection_name=collection)
        else:   
            # Initialize Postgres Vector Store (assuming similar interface)
            from PostgresVectorStore import PostgresVectorStore
            print("Using Postgres Vector Store")
            vector_db = PostgresVectorStore(collection_name=collection) 
        if not vector_db:
            print("✗ Error: Failed to initialize vector store")
            exit(1)   
    
            # Use default OCI model
        logger.info(f"Using model: {model_id} for collection: {collection} and compartment: {compartment_id} and use_cot {use_cot}")
        # Initialize the RAG agent with the vector store and model (now with optimizations)    
        rag_agent = OCIRAGAgent(
                vector_store=vector_db,
                use_cot=use_cot,
                collection=collection,
                model_id=model_id,
                compartment_id=compartment_id,
                use_stream=False,
                config=credentials  # Pass the full configuration
            )
        
        # Log cache stats for monitoring optimization effectiveness
        cache_stats = rag_agent.get_cache_stats()
        logger.info(f"Cache statistics: {cache_stats}")
        
        response = rag_agent.process_query(request["query"])
        # In the main function, add this check before printing the answer
        if "The final answer is:" in response["answer"]:
            # Extract only what follows "The final answer is:"
            response["answer"] = re.sub(r'.*The final answer is:\s*', '', response["answer"])
            # Remove any remaining LaTeX formatting
            response["answer"] = rag_agent._remove_latex_formatting(response["answer"])

        print("\nResponse:")
        print("-" * 50)
        print(response["answer"])
        if response.get("reasoning_steps"):
            print("\nReasoning Steps:")
            print("-" * 50)
            # Initialize reasoning as a list, not a dictionary
            response["reasoning"] = []
            
            for i, step in enumerate(response["reasoning_steps"]):
                print(f"\nStep {i+1}:")
                print(step)
                
                # Convert step to string based on its type
                if isinstance(step, list):
                    step_text = " ".join([s.strip() for s in step if isinstance(s, str)])
                elif isinstance(step, dict):
                    step_text = " ".join([str(v).strip() for v in step.values() if isinstance(v, str)])
                elif isinstance(step, str):
                    # Handle the most common case - string
                    step_text = step.strip()
                else:
                    # Fallback for any other type
                    step_text = str(step).strip()
                
                # Add as a dictionary to the list (with step number and content)
                response["reasoning"].append({
                    "step": i+1,
                    "content": step_text
                })

        
        if response.get("context"):
            print("\nSources used:")
            print("-" * 50)
            
            # Print concise list of sources
            for i, ctx in enumerate(response["context"]):
                source = ctx["metadata"].get("source", "Unknown")
                if "page_numbers" in ctx["metadata"]:
                    pages = ctx["metadata"].get("page_numbers", [])
                    print(f"[{i+1}] {source} (pages: {pages})")
                else:
                    file_path = ctx["metadata"].get("file_path", "Unknown")
                    print(f"[{i+1}] {source} (file: {file_path})")
                
        # Return the response dictionary
        return response
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {"error": str(e)}
        logger.error(f"Error processing request: {str(e)}")
        return {"error": str(e)}

def main():
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

    compartment_id  = args.compartment_id or credentials.get("OCI_COMPARTMENT_ID", "")
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
        
        # In the main function, add this check before printing the answer
        if "The final answer is:" in response["answer"]:
            # Extract only what follows "The final answer is:"
            response["answer"] = re.sub(r'.*The final answer is:\s*', '', response["answer"])
            # Remove any remaining LaTeX formatting
            response["answer"] = agent._remove_latex_formatting(response["answer"])
        
        print("\nResponse:")
        print("-" * 50)
        print(response["answer"])
        
        if response.get("reasoning_steps"):
            print("\nReasoning Steps:")
            print("-" * 50)
            # Initialize reasoning as a list, not a dictionary
            response["reasoning"] = []
            
            for i, step in enumerate(response["reasoning_steps"]):
                print(f"\nStep {i+1}:")
                print(step)
        
        if response.get("context"):
            print("\nSources used:")
            print("-" * 50)
            
            # Print concise list of sources
            for i, ctx in enumerate(response["context"]):
                source = ctx["metadata"].get("source", "Unknown")
                if "page_numbers" in ctx["metadata"]:
                    pages = ctx["metadata"].get("page_numbers", [])
                    print(f"[{i+1}] {source} (pages: {pages})")
                else:
                    file_path = ctx["metadata"].get("file_path", "Unknown")
                    print(f"[{i+1}] {source} (file: {file_path})")
                
                # Only print content if verbose flag is set
                if args.verbose:
                    content_preview = ctx["content"][:300] + "..." if len(ctx["content"]) > 300 else ctx["content"]
                    print(f"    Content: {content_preview}\n")
    
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()