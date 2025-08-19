from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import logging
import warnings
import time
from transformers import logging as transformers_logging
import re

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
        
    def plan(self, query: str, context: List[Dict[str, Any]] = None, context_doc_ids: List[str] = None) -> Dict[str, Any]:
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
- Only Use Australian English and formal legal language.
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
- Keep all [#] references in the final output.

3. **Step format**
- Step 1: [Description based on document content]
- Step 2: [Description based on document content] 
- Step 3: [Description based on document content]
- Step 4: [Description based on document content] etc. (if needed)
- Maintain all [#] references in the Steps.

4. **Citations required**
- Reference specific documents [#] that inform each step.
- Maintain all [#] references in the Steps.
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
        raw_plan = response.content
        # Extract steps heuristically (lines starting with 'Step')
        steps = []
        for line in raw_plan.splitlines():
            l = line.strip()
            if l.lower().startswith("step"):
                steps.append(l)
        # --- New: citation number parsing & mapping ---
        citation_numbers = []
        try:
            citation_numbers = list(dict.fromkeys(int(n) for n in re.findall(r'\[(\d+)\]', raw_plan)))
        except Exception:
            citation_numbers = []
        citations = []
        if context_doc_ids:
            for n in citation_numbers:
                if 1 <= n <= len(context_doc_ids):
                    citations.append(context_doc_ids[n-1])
        return {"raw": raw_plan, "steps": steps, "citation_numbers": citation_numbers, "citations": citations}

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

    def research(self, query: str, step: str, max_chunks: int = 3, max_findings: int = 3, max_tokens: int = 1000, max_results: int = 5, provenance=None) -> Dict[str, Any]:
        logger.info(f"\n🔍 Researching for step: {step}")
        
        # Query all collections with configurable limits
        pdf_results = self.vector_store.query_pdf_collection(query, n_results=max_results)
        repo_results = self.vector_store.query_repo_collection(query, n_results=max_results)
        
        # Combine and limit results
        all_results = (pdf_results + repo_results)[:max_chunks]
        logger.info(f"Found {len(all_results)} relevant documents (limited to {max_chunks} from {max_results} max search results)")
        # --- New: register sources in provenance & keep doc_id ordering for mapping ---
        context_doc_ids: List[str] = []
        if provenance and all_results:
            for item in all_results:
                try:
                    doc_id = provenance.add_source(item, step=f"research:{step}")
                    context_doc_ids.append(doc_id)
                except Exception:
                    context_doc_ids.append("")
        # ...existing code (no_results check)...
        if not all_results:
            logger.warning("No relevant documents found")
            return {"raw": "", "findings": [], "citations": [], "citation_numbers": [], "context_items": [], "doc_ids": []}
        # Create context string with length limit and proper citation format
        context_items = []
        total_chars = 0
        char_limit = max_tokens * 4  # Approximate chars-to-tokens ratio
        for i, item in enumerate(all_results):
            content = item['content']
            if total_chars + len(content) > char_limit:
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
- Only Use Australian English and formal legal language.

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
- Retain all [#] references of the original documents in the information. 
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
        prompt = ChatPromptTemplate.from_template(template)
        messages = prompt.format_messages(step=step, context=context_str)
        prompt_text = "\n".join([msg.content for msg in messages])
        self.log_prompt(prompt_text, "Research")
        start_time = time.time()
        response = self.llm.invoke(messages)
        duration = time.time() - start_time
        self.log_response(response.content, f"Research ({duration:.2f}s)")
        raw_findings = response.content
        # Heuristic split: split on numbered or dash list items for potential findings
        findings_list = []
        for seg in raw_findings.split('\n'):
            s = seg.strip()
            if not s:
                continue
            if len(findings_list) >= max_findings:
                break
            if s[0].isdigit() or s.startswith('-') or s.startswith('*'):
                findings_list.append(s)
        if not findings_list:
            findings_list = [raw_findings]
        # Parse citation numbers and map to doc_ids
        try:
            citation_numbers = list(dict.fromkeys(int(n) for n in re.findall(r'\[(\d+)\]', raw_findings)))
        except Exception:
            citation_numbers = []
        citations: List[str] = []
        for n in citation_numbers:
            if 1 <= n <= len(context_doc_ids):
                citations.append(context_doc_ids[n-1])
        findings = [{"content": f, "citations": citations, "metadata": {"source": "Research Summary"}} for f in findings_list]
        return {"raw": raw_findings, "findings": findings, "citation_numbers": citation_numbers, "citations": citations, "context_items": all_results, "doc_ids": context_doc_ids}

class ReasoningAgent(Agent):
    """Agent responsible for logical reasoning and analysis"""
    def __init__(self, llm):
        super().__init__(
            name="Reasoner",
            role="Logic and Analysis",
            description="Applies logical reasoning to information and draws conclusions",
            llm=llm
        )
        
    def reason(self, query: str, step: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info(f"\n🤔 Reasoning about step: {step}")
        # Build mapping of finding index -> doc_ids (citations) from context items
        index_to_doc_ids: Dict[int, List[str]] = {}
        for i, item in enumerate(context, start=1):
            cits = item.get("citations") or []
            if isinstance(cits, list):
                index_to_doc_ids[i] = cits
        template = """## AustLII AI Legal Research Assistant - Legal Reasoner

You are a legal research reasoning specialist. Your task is to analyze the provided research findings and draw clear, logical conclusions for this step based ONLY on the retrieved legal documents and references provided below.

**Critical Constraints:**
- Do NOT use outside knowledge unless explicitly authorised.
- Do NOT reference legal principles, legislation, or cases not mentioned in the provided documents.
- Do NOT add missing details to citations or complete partial references.
- Do not make assumptions, guesses, or inferences beyond the text.
- Do NOT invent content.
- Only Use Australian English and formal legal language.
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
- Keep all [#] references from the findings.
- Do not extrapolate beyond what is stated in the findings.

2. **Logical reasoning**
- Apply logical analysis to the research findings.
- Identify connections and relationships mentioned in the findings.
- Draw conclusions that flow logically from the documented information.

3. **Legal analysis format**
- Present your reasoning in clear, logical steps.
- Each reasoning point must be supported by a [#] citation.
- Retain all [#] references of the original documents in the reasoning.
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
        response = self.llm.invoke(messages)
        logger.info(f"Reasoning conclusion generated in {time.time() - currtime:.2f} seconds")
        self.log_response(response.content, "Reasoner")
        raw_reasoning = response.content
        # --- New: parse citation markers [#] referring to finding indices ---
        try:
            citation_numbers = list(dict.fromkeys(int(n) for n in re.findall(r'\[(\d+)\]', raw_reasoning)))
        except Exception:
            citation_numbers = []
        # Map to underlying provenance doc_ids (union of docs cited in those findings)
        doc_id_set = []
        seen = set()
        for n in citation_numbers:
            doc_ids = index_to_doc_ids.get(n, [])
            for d in doc_ids:
                if d and d not in seen:
                    seen.add(d)
                    doc_id_set.append(d)
        return {"raw": raw_reasoning, "citation_numbers": citation_numbers, "citations": doc_id_set, "marker_to_doc_ids": {n: index_to_doc_ids.get(n, []) for n in citation_numbers}}

class SynthesisAgent(Agent):
    """Agent responsible for combining information and generating final response"""
    def __init__(self, llm):
        super().__init__(
            name="Synthesizer",
            role="Information Synthesizer",
            description="Combines multiple pieces of information into a coherent response",
            llm=llm
        )
        
    def synthesize(self, query: str, reasoning_steps: List[Any]) -> Dict[str, Any]:
        # reasoning_steps may be list of strings (backward) or list of dicts with 'raw' & mapping
        logger.info(f"\n📝 Synthesizing final answer from {len(reasoning_steps)} reasoning steps")
        if not reasoning_steps:
            return {"raw": "", "answer": "I was unable to generate a complete answer based on the available information.", "citations": []}
        # Normalize to dicts
        norm_steps = []
        for s in reasoning_steps:
            if isinstance(s, dict):
                norm_steps.append(s)
            else:
                norm_steps.append({"raw": str(s)})
        steps_str = "\n\n".join([f"[Step {i+1}] {d.get('raw','')}" for i, d in enumerate(norm_steps)])
        # Build aggregate marker->doc_ids mapping from reasoning
        aggregate_marker_map: Dict[int, List[str]] = {}
        for d in norm_steps:
            marker_map = d.get("marker_to_doc_ids") or {}
            for k, doc_ids in marker_map.items():
                # merge unique
                existing = aggregate_marker_map.setdefault(k, [])
                for doc_id in doc_ids:
                    if doc_id and doc_id not in existing:
                        existing.append(doc_id)
        template = """## AustLII AI Legal Research Assistant - Final Synthesizer

You are a legal research synthesis specialist. Your task is to combine the reasoning steps into a coherent, comprehensive final answer based ONLY on the analysis and documents that have been researched.

**Critical Constraints:**
- Do NOT use outside knowledge unless explicitly authorised.
- Do NOT reference legal principles, legislation, or cases not mentioned in the reasoning steps.
- Do NOT add missing details to citations or complete partial references.
- Do not make assumptions, guesses, or inferences beyond the analysis.
- Do NOT invent content.
- Only Use Australian English and formal legal language.
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
            prompt = ChatPromptTemplate.from_template(template)
            messages = prompt.format_messages(query=query, steps=steps_str)
            prompt_text = "\n".join([m.content for m in messages])
            self.log_prompt(prompt_text, "Synthesizer")
            start_time = time.time()
            response = self.llm.invoke(messages)
            duration = time.time() - start_time
            if hasattr(response, "content"):
                result_text = response.content
            else:
                result_text = str(response)
            self.log_response(result_text, f"Synthesizer ({duration:.2f}s)")
            # Parse citation markers in final answer
            try:
                answer_markers = list(dict.fromkeys(int(n) for n in re.findall(r'\[(\d+)\]', result_text)))
            except Exception:
                answer_markers = []
            # Map to doc_ids (union of underlying reasoning markers)
            final_doc_ids = []
            seen = set()
            for n in answer_markers:
                for d in aggregate_marker_map.get(n, []):
                    if d and d not in seen:
                        seen.add(d)
                        final_doc_ids.append(d)
            return {"raw": result_text, "answer": result_text, "citation_numbers": answer_markers, "citations": final_doc_ids, "marker_to_doc_ids": aggregate_marker_map}
        except Exception as e:
            logger.error(f"Error in synthesis template formatting: {str(e)}")
            return {"raw": "", "answer": "I was unable to generate a complete answer due to an internal error.", "citations": []}

def create_agents(llm, vector_store=None):
    """Create and return the set of specialized agents"""
    return {
        "planner": PlannerAgent(llm),
        "researcher": ResearchAgent(llm, vector_store) if vector_store else None,
        "reasoner": ReasoningAgent(llm),
        "synthesizer": SynthesisAgent(llm)
    }