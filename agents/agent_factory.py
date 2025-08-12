from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import logging
import warnings
import time
from transformers import logging as transformers_logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress specific transformers warnings
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`")

class Agent(BaseModel):
    """Base agent class with common properties"""
    name: str
    role: str
    description: str
    llm: Any = Field(description="Language model for the agent")
    
    def log_prompt(self, prompt: str, prefix: str = ""):
        """Log a prompt being sent to the LLM"""
        # Check if the prompt contains context
        if "Context:" in prompt:
            # Split the prompt at "Context:" and keep only the first part
            parts = prompt.split("Context:")
            # Keep the first part and add a note that context is omitted
            logger.info(f"\n Full prompt passed to LLM:\n{'-'*40}\n{prompt}\n{'-'*40}")
            truncated_prompt = parts[0] + "Context: [Context omitted for brevity]"
            if len(parts) > 2 and "Key Findings:" in parts[1]:
                # For researcher prompts, keep the "Key Findings:" part
                key_findings_part = parts[1].split("Key Findings:")
                if len(key_findings_part) > 1:
                    truncated_prompt += "\nKey Findings:" + key_findings_part[1]
            logger.info(f"\n{'='*80}\n{prefix} Prompt:\n{'-'*40}\n{truncated_prompt}\n{'='*80}")
        else:
            # If no context, log the full prompt
            logger.info(f"\n{'='*80}\n{prefix} Prompt:\n{'-'*40}\n{prompt}\n{'='*80}")
        
    def log_response(self, response: str, prefix: str = ""):
        """Log a response received from the LLM"""
        # Log the response but truncate if it's too long
        if len(response) > 500:
            truncated_response = response[:500] + "... [response truncated]"
            logger.info(f"\n{'='*80}\n{prefix} Response:\n{'-'*40}\n{truncated_response}\n{'='*80}")
            logger.info(f"Full response is : \n{'-'*40}\n{response}\n{'='*40}")
        else:
            logger.info(f"\n{'='*80}\n{prefix} Response:\n{'-'*40}\n{response}\n{'='*80}")

class PlannerAgent(Agent):
    """Agent responsible for breaking down problems and planning steps"""
    def __init__(self, llm):
        super().__init__(
            name="Planner",
            role="Strategic Planner",
            description="Breaks down complex problems into manageable steps",
            llm=llm
        )
        
    def plan(self, query: str, context: List[Dict[str, Any]] = None) -> str:
        logger.info(f"\n🎯 Planning step for query: {query}")
        
        if context:
            template = """As a strategic planner, break down this problem into 3-4 clear steps.
            
            Context: {context}
            Query: {query}
            
            Steps: Provide the steps in plain text format. DO NOT use LaTeX or special formatting."""
            context_str = "\n\n".join([f"Context {i+1}:\n{item['content']}" for i, item in enumerate(context)])
            logger.info(f"Using context ({len(context)} items)")
        else:
            template = """As a strategic planner, break down this problem into 3-4 clear steps.
            
            Query: {query}

            Steps: Provide the steps in plain text format. DO NOT use LaTeX or special formatting."""
            context_str = ""
            logger.info("No context available")
            
        prompt = ChatPromptTemplate.from_template(template)
        messages = prompt.format_messages(query=query, context=context_str)
        prompt_text = "\n".join([msg.content for msg in messages])
        self.log_prompt(prompt_text, "Planner")
        
        currtime = time.time()
        logger.info(f"Generating plan using LLM...")
        # Use the LLM to generate the plan 
        response = self.llm.invoke(messages)
        logger.info(f"Plan generated in {time.time() - currtime:.2f} seconds")
        self.log_response(response.content, "Planner")
        return response.content

class ResearchAgent(Agent):
    """Agent responsible for gathering and analyzing information"""
    vector_store: Any = Field(description="Vector store for searching")
    
    def __init__(self, llm, vector_store):
        super().__init__(
            name="Researcher",
            role="Information Gatherer",
            description="Gathers and analyzes relevant information from knowledge bases",
            llm=llm,
            vector_store=vector_store
        )

    def research(self, query: str, step: str, max_chunks: int = 3, max_findings: int = 3, max_tokens: int = 1000) -> List[Dict[str, Any]]:
        logger.info(f"\n🔍 Researching for step: {step}")
        
        # Query all collections with limits
        pdf_results = self.vector_store.query_pdf_collection(query, n_results=max_chunks)
        repo_results = self.vector_store.query_repo_collection(query, n_results=max_chunks)
        
        # Combine and limit results
        all_results = (pdf_results + repo_results)[:max_chunks]
        logger.info(f"Found {len(all_results)} relevant documents (limited to {max_chunks})")
        
        if not all_results:
            logger.warning("No relevant documents found")
            return []
        
        # Create context string with length limit
        context_items = []
        total_chars = 0
        char_limit = max_tokens * 4  # Approximate chars-to-tokens ratio
        
        for i, item in enumerate(all_results):
            content = item['content']
            if total_chars + len(content) > char_limit:
                # Truncate this item to fit within limit
                remaining_chars = char_limit - total_chars
                if remaining_chars > 100:  # Only add if we can include meaningful content
                    content = content[:remaining_chars] + "..."
                    context_items.append(f"Source {i+1}:\n{content}")
                break
            
            context_items.append(f"Source {i+1}:\n{content}")
            total_chars += len(content)
        
        context_str = "\n\n".join(context_items)
        logger.info(f"Context created with {len(context_items)} items, total chars: {total_chars}")
        
        # Fixed template - use context directly, not as a variable
        template = """Extract and summarize key information relevant to this step.
    
Step: {step}
Context: {context}

Key Findings:"""
    
        # Use proper parameter name matching the template
        prompt = ChatPromptTemplate.from_template(template)
        # Fixed: Pass context_str as 'context' to match template
        messages = prompt.format_messages(step=step, context=context_str)
        
        # Log what we're sending to the LLM
        prompt_text = "\n".join([msg.content for msg in messages])
        self.log_prompt(prompt_text, "Research")
        
        # Time the execution
        start_time = time.time()
        response = self.llm.invoke(messages)
        duration = time.time() - start_time
        
        # Log what we got back
        self.log_response(response.content, f"Research ({duration:.2f}s)")
        
        # Convert findings to list format
        findings = [{
            "content": response.content,
            "metadata": {"source": "Research Summary"}
        }]
        
        return findings

class ReasoningAgent(Agent):
    """Agent responsible for logical reasoning and analysis"""
    def __init__(self, llm):
        super().__init__(
            name="Reasoner",
            role="Logic and Analysis",
            description="Applies logical reasoning to information and draws conclusions",
            llm=llm
        )
        
    def reason(self, query: str, step: str, context: List[Dict[str, Any]]) -> str:
        logger.info(f"\n🤔 Reasoning about step: {step}")
        
        template = """Analyze the information and draw a clear conclusion for this step.
        
        Step: {step}
        Context: {context}
        Query: {query}

        Conclusion: Provide the conclusion in plain text format. DO NOT use LaTeX or special formatting."""
        
        # Create context string but don't log it
        context_str = "\n\n".join([f"Context {i+1}:\n{item['content']}" for i, item in enumerate(context)])
        prompt = ChatPromptTemplate.from_template(template)
        messages = prompt.format_messages(step=step, query=query, context=context_str)
        prompt_text = "\n".join([msg.content for msg in messages])
        self.log_prompt(prompt_text, "Reasoner")

        currtime = time.time()
        logger.info(f"Generating reasoning conclusion using LLM...")
        # Use the LLM to generate the reasoning conclusion
        response = self.llm.invoke(messages)
        logger.info(f"Reasoning conclusion generated in {time.time() - currtime:.2f} seconds")
        self.log_response(response.content, "Reasoner")
        return response.content

class SynthesisAgent(Agent):
    """Agent responsible for combining information and generating final response"""
    def __init__(self, llm):
        super().__init__(
            name="Synthesizer",
            role="Information Synthesizer",
            description="Combines multiple pieces of information into a coherent response",
            llm=llm
        )
        
    def synthesize(self, query: str, reasoning_steps: List[str]) -> str:
        logger.info(f"\n📝 Synthesizing final answer from {len(reasoning_steps)} reasoning steps")
        
        # Validate reasoning steps before synthesis
        if not reasoning_steps or not all(isinstance(step, str) and step.strip() for step in reasoning_steps):
            logger.warning("Invalid reasoning steps detected. Falling back to general response.")
            return "I don't have enough valid analysis to provide a complete answer."
        
        # Create steps_str properly
        steps_str = "\n\n".join([f"Step {i+1}:\n{step}" for i, step in enumerate(reasoning_steps)])
        
        # Use a structured approach for ChatPromptTemplate
        from langchain_core.messages import SystemMessage
        
        # Define system and human messages separately
        system_message_template = SystemMessage(
            content="You are a synthesis agent that combines reasoning steps into clear answers. Provide your final answer in plain text format. DO NOT use LaTeX notation, mathematical symbols like \\boxed{{}}, or markdown formatting."
        )
        
        human_template = """Combine the reasoning steps into a clear, comprehensive answer.

Query: {query}
Steps: {steps}

Answer:"""
    
        # Create prompt template from messages
        prompt = ChatPromptTemplate.from_messages([
            system_message_template,
            ("human", human_template)
        ])
    
        # Create key-value pairs dictionary for formatting
        format_dict = {
            "query": query,
            "steps": steps_str
        }
    
        try:
            # Format the messages using the dictionary
            messages = prompt.format_messages(**format_dict)
        
            # Log what we're sending to the LLM
            prompt_text = "\n".join([msg.content for msg in messages])
            self.log_prompt(prompt_text, "Synthesizer")

            # Time the execution with robust error handling
            start_time = time.time()
            response = self.llm.invoke(messages)
            duration = time.time() - start_time
        
            # Handle different response formats
            if hasattr(response, "content"):
                result = response.content
            elif isinstance(response, dict) and "content" in response:
                result = response["content"]
            elif isinstance(response, str):
                result = response
            else:
                result = str(response)
            
            self.log_response(result, f"Synthesizer ({duration:.2f}s)")
            return result
        
        except Exception as e:
            logger.error(f"Error in synthesis template formatting: {str(e)}")
        
            # Try a simpler template as fallback
            fallback_template = ChatPromptTemplate.from_messages([
                ("system", "Synthesize the reasoning steps into a final answer."),
                ("human", f"Query: {query}\n\nSteps: {steps_str}\n\nProvide a plain text answer:")
            ])
        
            try:
                fallback_messages = fallback_template.format_messages()
                fallback_response = self.llm.invoke(fallback_messages)
                return fallback_response.content
            except Exception as e2:
                logger.error(f"Even fallback template failed: {str(e2)}")
                return f"Based on the analysis, {reasoning_steps[0][:200]}..."

def create_agents(llm, vector_store=None):
    """Create and return the set of specialized agents"""
    return {
        "planner": PlannerAgent(llm),
        "researcher": ResearchAgent(llm, vector_store) if vector_store else None,
        "reasoner": ReasoningAgent(llm),
        "synthesizer": SynthesisAgent(llm)
    }