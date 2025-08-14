from html import parser
from typing import List, Dict, Any, Optional
import json
import os
import argparse
import logging
from dotenv import load_dotenv
import time
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import yaml

# OCI imports
import oci
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.llms.oci_generative_ai import OCIGenAI
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

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
                 model_id: str = "cohere.command-latest", compartment_id: str = None, use_stream:bool = False,max_chunks_per_step: int = 2,
                 max_findings_per_step: int = 3, 
                 max_tokens_per_finding: int = 1000):
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
        self.max_chunks_per_step = max_chunks_per_step
        self.max_findings_per_step = max_findings_per_step
        self.max_tokens_per_finding = max_tokens_per_finding

        # Set up OCI configuration
        config = load_oci_config()
        
        self.genai_client = ChatOCIGenAI(
            auth_profile=CONFIG_PROFILE,
            model_id=model_id,
            compartment_id=self.compartment_id,
            service_endpoint=SERVICE_ENDPOINT,
            is_stream=use_stream,  # Use streaming if enabled
            #temperature=TEMPERATURE # old endpoint
            model_kwargs={"temperature": 0, "max_tokens": 1500 }#, "stop": ["populous"]} # new endpoint
        )
        
        # Initialize specialized agents
        self.agents = create_agents(self.genai_client, vector_store) if use_cot else None
    
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
    
    def _process_query_with_cot(self, query: str) -> Dict[str, Any]:
        """Process query using Chain of Thought reasoning with multiple agents"""
        logger.info("Processing query with Chain of Thought reasoning")
        
        # Get initial context based on selected collection
        initial_context = []
        try:
            # Fetch context (collection-specific code remains the same)
            if self.collection == "PDF Collection":
                logger.info(f"Retrieving context from PDF Collection for query: '{query}'")
                pdf_context = self.vector_store.query_pdf_collection(query)
                initial_context.extend(pdf_context)
                logger.info(f"Retrieved {len(pdf_context)} chunks from PDF Collection")
                # Log each chunk with citation number but not full content
                for i, chunk in enumerate(pdf_context):
                    source = chunk["metadata"].get("source", "Unknown")
                    pages = chunk["metadata"].get("page_numbers", [])
                    logger.info(f"Source [{i+1}]: {source} (pages: {pages})")
                    content_preview = chunk["content"][:150] + "..." if len(chunk["content"]) > 150 else chunk["content"]
                    logger.info(f"Content preview for source [{i+1}]: {content_preview}")
            elif self.collection == "Repository Collection":
                logger.info(f"Retrieving context from Repository Collection for query: '{query}'")
                repo_context = self.vector_store.query_repo_collection(query)
                initial_context.extend(repo_context)
                logger.info(f"Retrieved {len(repo_context)} chunks from Repository Collection")
                for i, chunk in enumerate(repo_context):
                    source = chunk["metadata"].get("source", "Unknown")
                    file_path = chunk["metadata"].get("file_path", "Unknown")
                    logger.info(f"Source [{i+1}]: {source} (file: {file_path})")
                    content_preview = chunk["content"][:150] + "..." if len(chunk["content"]) > 150 else chunk["content"]
                    logger.info(f"Content preview for source [{i+1}]: {content_preview}")
            elif self.collection == "Web Knowledge Base":
                logger.info(f"Retrieving context from Web Knowledge Base for query: '{query}'")
                web_context = self.vector_store.query_web_collection(query)
                initial_context.extend(web_context)
                logger.info(f"Retrieved {len(web_context)} chunks from Web Knowledge Base")
                for i, chunk in enumerate(web_context):
                    source = chunk["metadata"].get("source", "Unknown")
                    title = chunk["metadata"].get("title", "Unknown")
                    logger.info(f"Source [{i+1}]: {source} (title: {title})")
                    content_preview = chunk["content"][:150] + "..." if len(chunk["content"]) > 150 else chunk["content"]
                    logger.info(f"Content preview for source [{i+1}]: {content_preview}")
            else:
                logger.info("Using General Knowledge collection, no context retrieval needed")
            
            # Apply token budget to context
            initial_context = self._limit_context(initial_context, max_tokens=12000)
            
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
            
            # Step 2: Research each step (try parallel first, fall back to sequential)
            logger.info("Step 2: Research (parallel execution)")
            plan_steps = [step for step in plan.split("\n") if step.strip()]
            if len(plan_steps) > 3:
                logger.info(f"Limiting plan from {len(plan_steps)} steps to 3 steps")
                plan_steps = plan_steps[:3]

            # Try to use parallel processing with proper safeguards
            try:
                logger.info("Attempting parallel research execution")
                
                # Check if we're already in an event loop
                try:
                    # Try to get the running loop but don't use run_coroutine_threadsafe
                    existing_loop = asyncio.get_running_loop()
                    logger.info("Existing event loop detected - switching to sequential processing for safety")
                    
                    # If we're in an event loop, use sequential processing instead
                    # This avoids potential deadlocks with nested event loops
                    research_results = self._research_steps_sequential(query, plan_steps)
                    
                except RuntimeError:
                    # No event loop running, create a new one
                    logger.info("Creating new event loop for parallel processing")
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        # Add a timeout to prevent indefinite hanging
                        research_results = loop.run_until_complete(
                            asyncio.wait_for(
                                self._research_steps_parallel(query, plan_steps),
                                timeout=120  # 2 minute timeout
                            )
                        )
                    finally:
                        loop.close()
                        logger.info("Event loop closed")
            
            except Exception as e:
                logger.error(f"Error in parallel research: {str(e)}")
                logger.info("Falling back to sequential research")
                research_results = self._research_steps_sequential(query, plan_steps)
            
            # Step 3: Batch reasoning for better efficiency
            logger.info("Step 3: Reasoning (batch processing)")
            reasoning_steps = []
            
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

            # Check if we have any reasoning steps before synthesis
            if not reasoning_steps:
                logger.warning("No valid reasoning steps generated. Using general response.")
                return self._generate_general_response(query)
            
            # Step 4: Synthesize final answer with explicit formatting instructions
            logger.info("Step 4: Synthesis")
            # Before synthesis
            logger.info(f"Reasoning steps count: {len(reasoning_steps)}")
            if not reasoning_steps:
                logger.error("No reasoning steps available for synthesis")
                return self._generate_general_response(query)
                
            # Add debug info
            for i, step in enumerate(reasoning_steps):
                logger.info(f"Reasoning step {i+1} type: {type(step)}, length: {len(str(step))}")
            
            try:
                final_answer = self.agents["synthesizer"].synthesize(query, reasoning_steps)
                
                # Add explicit null check
                if not final_answer:
                    logger.error("Empty response from synthesizer")
                    final_answer = "I was unable to generate a complete answer based on the available information."
                
                # Remove LaTeX formatting from final answer
                final_answer = self._remove_latex_formatting(final_answer)
                logger.info(f"Final synthesized answer generated")
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
        
        # Get context based on selected collection
        if self.collection == "PDF Collection":
            logger.info(f"Retrieving context from PDF Collection for query: '{query}'")
            context = self.vector_store.query_pdf_collection(query)
            logger.info(f"Retrieved {len(context)} chunks from PDF Collection")
            for i, chunk in enumerate(context):
                source = chunk["metadata"].get("source", "Unknown")
                pages = chunk["metadata"].get("page_numbers", [])
                logger.info(f"Source [{i+1}]: {source} (pages: {pages})")
                content_preview = chunk["content"][:150] + "..." if len(chunk["content"]) > 150 else chunk["content"]
                logger.debug(f"Content preview for source [{i+1}]: {content_preview}")
        elif self.collection == "Repository Collection":
            logger.info(f"Retrieving context from Repository Collection for query: '{query}'")
            context = self.vector_store.query_repo_collection(query)
            logger.info(f"Retrieved {len(context)} chunks from Repository Collection")
            for i, chunk in enumerate(context):
                source = chunk["metadata"].get("source", "Unknown")
                file_path = chunk["metadata"].get("file_path", "Unknown")
                logger.info(f"Source [{i+1}]: {source} (file: {file_path})")
                content_preview = chunk["content"][:150] + "..." if len(chunk["content"]) > 150 else chunk["content"]
                logger.debug(f"Content preview for source [{i+1}]: {content_preview}")
        elif self.collection == "Web Knowledge Base":
            logger.info(f"Retrieving context from Web Knowledge Base for query: '{query}'")
            context = self.vector_store.query_web_collection(query)
            logger.info(f"Retrieved {len(context)} chunks from Web Knowledge Base")
            for i, chunk in enumerate(context):
                source = chunk["metadata"].get("source", "Unknown")
                title = chunk["metadata"].get("title", "Unknown")
                logger.info(f"Source [{i+1}]: {source} (title: {title})")
                content_preview = chunk["content"][:150] + "..." if len(chunk["content"]) > 150 else chunk["content"]
                logger.debug(f"Content preview for source [{i+1}]: {content_preview}")
        
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
            # Non-streaming response
            chain = (
                prompt
                | self.genai_client
                | StrOutputParser()
            )
            currtime = time.time()
            answer = chain.invoke({"query": query, "formatted_context": formatted_context})
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
        system_prompt = "You are a helpful AI assistant. Answer the following query using your general knowledge."
        user_content = f"Query: {query}\n\nAnswer:"
        
        prompt = PromptTemplate.from_template(user_content)
        chain = (
            {"input": RunnablePassthrough()}
            | prompt
            | self.genai_client
            | StrOutputParser()
        )
        currtime = time.time()
        answer = chain.invoke({"query": query})
        logger.info(f"General response generated in {time.time() - currtime:.2f} seconds")
        # Return a general response without context
        
        logger.info("No context available, using general knowledge response")    
        return {
            "answer": answer,
            "context": []
        }
    
    def _limit_context(self, documents: List[Dict[str, Any]], max_tokens: int = 12000) -> List[Dict[str, Any]]:
        """Limit context to fit within a token budget"""
        if not documents:
            return []
        
        limited_docs = []
        token_count = 0
        
        # Simple token estimation (4 chars ≈ 1 token)
        char_to_token_ratio = 4
        
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
                    logger.info(f"Truncated document to fit token budget")
                break
            
            limited_docs.append(doc)
            token_count += doc_tokens
        
        logger.info(f"Limited context from {len(documents)} to {len(limited_docs)} documents (~{token_count} tokens)")
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
def load_config() -> Dict[str, str]:
        """Load configuration from config_oci.yaml"""
        try:
            config_path = Path("config_oci.yaml")
            if not config_path.exists():
                print("Warning: config_oci.yaml not found. Using empty configuration.")
                return {}
                
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config if config else {}
        except Exception as e:
            print(f"Warning: Error loading config: {str(e)}")
            return {}

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
        #use_cot = request["use_cot"] or credentials.get("USE_COT", "false").lower() == "true"

        # Check for OCI compartment ID
        if not compartment_id:
            print("✗ Error: OCI_COMPARTMENT_ID not found in config file or command line arguments")
            print("Please set the OCI_COMPARTMENT_ID in config file or provide --compartment-id")
            exit(1)
            # Determine which model to us
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
        # Initialize the RAG agent with the vector store and model    
        rag_agent = OCIRAGAgent(
                vector_store=vector_db,
                use_cot=use_cot,
                collection=collection,
                model_id=model_id,
                compartment_id=compartment_id,
                use_stream=False,
                max_chunks_per_step=2,
                max_findings_per_step=3,
                max_tokens_per_finding=1000
            )
        
        response = rag_agent.process_query(request["query"])
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
    parser.add_argument("--max-chunks-per-step", type=int, default=2, 
                        help="Maximum chunks per research step (default: 2)")
    parser.add_argument("--max-findings-per-step", type=int, default=3, 
                        help="Maximum findings per research step (default: 3)")
    parser.add_argument("--max-tokens-per-finding", type=int, default=1000, 
                        help="Maximum tokens per finding (default: 1000)")

    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
     # Load Postgres DB credentials from config_pg.yaml
    credentials = load_config()

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
            max_chunks_per_step=args.max_chunks_per_step,
            max_findings_per_step=args.max_findings_per_step,
            max_tokens_per_finding=args.max_tokens_per_finding
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