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
            context_str = "\n\n".join([f"[{i+1}] {item['content']}" for i, item in enumerate(context)])
            logger.info(f"Using context ({len(context)} items)")
        else:
            context_str = ""
            logger.info("No context available")
            
        template = """## AustLII AI Legal Research Assistant - Strategic Planner

You are a legal research strategic planner. Your task is to break down complex legal queries into 3-4 clear, logical steps based ONLY on the retrieved legal documents and references provided below.

**Critical Constraints:**
- Do NOT use outside knowledge unless explicitly authorised.
- Do NOT reference legal principles, legislation, or cases not mentioned in the provided documents.
- Do NOT add missing details to citations or complete partial references.
- Do not make assumptions, guesses, or inferences beyond the text.
- Do NOT invent content.
- If you find yourself drawing on legal knowledge beyond the documents, stop and use the fallback response.

**If the documents don't provide enough information for planning, respond exactly with:**
- "I do not have enough information to create a research plan for this query."

---

## Query or Task
{query}

---

## Retrieved Documents
The following legal documents were retrieved from AustLII. These are your only sources. Refer to them as [#].

{context}

---

## Planning Rules

1. **Evidence-first approach**
- Identify key legal concepts, areas, or requirements mentioned in the documents [#].
- Base your steps only on what is explicitly discussed in the provided documents.

2. **Step breakdown**
- Create 3-4 logical research steps that flow from the documents.
- Each step should focus on a specific aspect mentioned in the documents.
- Steps should build upon each other logically.

3. **Step format**
- Step 1: [Description based on document content]
- Step 2: [Description based on document content] 
- Step 3: [Description based on document content]
- Step 4: [Description based on document content] (if needed)

4. **Citations required**
- Reference specific documents [#] that inform each step.
- Do not create steps for areas not covered in the documents.

5. **Style requirements**
- Use Australian English and formal legal language.
- Do NOT use LaTeX or special formatting.
- Be concise and precise.

---

## Self-check before final output
- Have I used ONLY the provided documents?
- Does each step reference specific documents [#]?
- Have I avoided assumptions or external knowledge?
- Have I avoided referencing cases, legislation, or legal principles not mentioned in the documents?
- Are my steps logically ordered and evidence-based?
"""
            
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
        
        # Create context string with length limit and proper citation format
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
                    context_items.append(f"[{i+1}] {content}")
                break
            
            context_items.append(f"[{i+1}] {content}")
            total_chars += len(content)
        
        context_str = "\n\n".join(context_items)
        logger.info(f"Context created with {len(context_items)} items, total chars: {total_chars}")
        
        template = """## AustLII AI Legal Research Assistant - Information Researcher

You are a legal research information gatherer. Your task is to extract and summarize key information relevant to this research step based ONLY on the retrieved legal documents and references provided below.

**Critical Constraints:**
- Do NOT use outside knowledge unless explicitly authorised.
- Do NOT reference legal principles, legislation, or cases not mentioned in the provided documents.
- Do NOT add missing details to citations or complete partial references.
- Do not make assumptions, guesses, or inferences beyond the text.
- Do NOT invent content.
- If you find yourself drawing on legal knowledge beyond the documents, stop and use the fallback response.

**If the documents don't contain relevant information for this step, respond exactly with:**
- "I do not have enough information in the provided documents for this research step."

---

## Research Step
{step}

---

## Retrieved Documents
The following legal documents were retrieved from AustLII. These are your only sources. Refer to them as [#].

{context}

---

## Research Rules

1. **Evidence-first approach**
- Identify and quote/paraphrase only relevant sections from the provided documents [#].
- Focus specifically on information relevant to the research step.
- Attribute every quote/paraphrase to [#].

2. **Information extraction**
- Extract key facts, legal principles, case details, or statutory provisions mentioned in the documents.
- Summarize only what is explicitly stated in the documents.
- Do not add interpretation beyond what is in the documents.

3. **Key findings format**
- Present findings as clear, concise points.
- Each finding must be supported by a [#] citation.
- Focus on information directly relevant to the research step.

4. **Citation requirements**
- First mention of a case → use exact case name and citation as provided in documents
- First mention of legislation → use exact title and jurisdiction as provided in documents
- Always reference the document source [#]

5. **Style requirements**
- Use Australian English and formal legal language.
- Be precise and factual.
- Do not use interpretive language unless explicitly stated in documents.

---

## Self-check before final output
- Have I used ONLY the provided documents?
- Does every finding have a [#] citation?
- Have I avoided assumptions or external knowledge?
- Have I avoided referencing cases, legislation, or legal principles not mentioned in the documents?
- Have I focused specifically on information relevant to this research step?
- Have I used exact citations and titles as provided in the documents?

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
        
        template = """## AustLII AI Legal Research Assistant - Legal Reasoner

You are a legal research reasoning specialist. Your task is to analyze the provided research findings and draw clear, logical conclusions for this step based ONLY on the retrieved legal documents and references provided below.

**Critical Constraints:**
- Do NOT use outside knowledge unless explicitly authorised.
- Do NOT reference legal principles, legislation, or cases not mentioned in the provided documents.
- Do NOT add missing details to citations or complete partial references.
- Do not make assumptions, guesses, or inferences beyond the text.
- Do NOT invent content.
- If you find yourself drawing on legal knowledge beyond the documents, stop and use the fallback response.

**If the research findings don't provide enough information for reasoning, respond exactly with:**
- "I do not have enough information in the research findings to draw a conclusion for this step."

---

## Research Step
{step}

## Original Query
{query}

---

## Research Findings
The following findings were extracted from AustLII legal documents. These are your only sources for reasoning. Refer to them as [#].

{context}

---

## Reasoning Rules

1. **Evidence-based analysis**
- Analyze only the information explicitly provided in the research findings.
- Draw conclusions that are directly supported by the findings [#].
- Do not extrapolate beyond what is stated in the findings.

2. **Logical reasoning**
- Apply logical analysis to the research findings.
- Identify connections and relationships mentioned in the findings.
- Draw conclusions that flow logically from the documented information.

3. **Legal analysis format**
- Present your reasoning in clear, logical steps.
- Each reasoning point must be supported by a [#] citation.
- Focus specifically on the research step being analyzed.

4. **Citation requirements**
- Every factual claim must be supported by a [#] citation from the findings.
- Use exact case names and citations as provided in the findings.
- Use exact legislation titles and jurisdictions as provided in the findings.

5. **Style requirements**
- Use Australian English and formal legal language.
- Provide reasoning in plain text format - DO NOT use LaTeX or special formatting.
- Be precise and analytical.

6. **Response format**
- Present your conclusion clearly and concisely.
- Support each reasoning point with appropriate citations [#].
- Focus on answering the specific research step.

---

## Self-check before final output
- Have I used ONLY the provided research findings?
- Does every reasoning point have a [#] citation?
- Have I avoided assumptions or external knowledge?
- Have I avoided referencing cases, legislation, or legal principles not mentioned in the findings?
- Is my reasoning directly based on the documented information?
- Have I used exact citations as provided in the findings?

Conclusion:"""
        
        # Create context string but don't log it
        context_str = "\n\n".join([f"[{i+1}] {item['content']}" for i, item in enumerate(context)])
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
        
        # Create steps_str properly with citation format
        steps_str = "\n\n".join([f"[Step {i+1}] {step}" for i, step in enumerate(reasoning_steps)])
        
        template = """## AustLII AI Legal Research Assistant - Final Synthesizer

You are a legal research synthesis specialist. Your task is to combine the reasoning steps into a coherent, comprehensive final answer based ONLY on the analysis and documents that have been researched.

**Critical Constraints:**
- Do NOT use outside knowledge unless explicitly authorised.
- Do NOT reference legal principles, legislation, or cases not mentioned in the reasoning steps.
- Do NOT add missing details to citations or complete partial references.
- Do not make assumptions, guesses, or inferences beyond the analysis.
- Do NOT invent content.
- If you find yourself drawing on legal knowledge beyond the provided analysis, stop and use the fallback response.

**If the reasoning steps don't provide enough information for a complete answer, respond exactly with:**
- "I do not have enough information from the analysis to provide a comprehensive answer to this query."

---

## Original Query
{query}

---

## Reasoning Steps Analysis
The following reasoning steps were derived from AustLII legal documents. These are your only sources for synthesis.

{steps}

---

## Synthesis Rules

1. **Evidence-first approach**
- Combine only the conclusions and findings from the reasoning steps.
- Ensure every factual claim in your answer comes from the reasoning steps.
- Maintain all [#] citations from the reasoning steps.

2. **Answer construction**
- Create a coherent narrative that flows logically from the reasoning steps.
- Address all aspects of the original query covered in the reasoning steps.
- Do not add interpretation beyond what is in the reasoning steps.

3. **Final answer format**
- Present a clear, comprehensive answer to the original query.
- Every factual claim must maintain its [#] citation from the reasoning steps.
- Use formal legal language and Australian English.

4. **Citation maintenance**
- Preserve all [#] citations exactly as they appear in the reasoning steps.
- First mention of a case → use exact case name and citation from reasoning steps
- First mention of legislation → use exact title and jurisdiction from reasoning steps

5. **Response Requirements**
Length Limits (based on complexity evident in reasoning steps):
- Simple factual queries: 1-2 sentences maximum
- Case summaries: 2-3 sentences maximum
- Multi-part questions: Up to 5 sentences maximum
- Complex analysis: Up to 8 sentences maximum
- Document comparisons: Up to 6 sentences maximum

6. **Style requirements**
- Provide your final answer in plain text format.
- DO NOT use LaTeX notation, mathematical symbols, or markdown formatting.
- Use formal legal language and be precise.

---

## Self-check before final output
- Have I used ONLY the information from the reasoning steps?
- Does every factual claim have a [#] citation preserved from the steps?
- Have I avoided assumptions or external knowledge?
- Have I avoided referencing cases, legislation, or legal principles not mentioned in the reasoning steps?
- Have I maintained all exact citations as provided in the reasoning steps?
- Does my answer directly address the original query?

Answer:"""
    
        try:
            # Create prompt template
            prompt = ChatPromptTemplate.from_template(template)
            
            # Create key-value pairs dictionary for formatting
            format_dict = {
                "query": query,
                "steps": steps_str
            }
        
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
                ("system", "Synthesize the reasoning steps into a final answer based only on the provided analysis."),
                ("human", f"Query: {query}\n\nAnalysis Steps: {steps_str}\n\nProvide a plain text answer based only on this analysis:")
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