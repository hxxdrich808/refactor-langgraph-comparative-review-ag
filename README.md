# Refactor LangGraph Comparative Review Agent

## Requirements
- [high] Dynamic Criteria Generation: The plan_criteria node must invoke an LLM (OpenAI or Ollama) to produce 3–5 comparison criteria based solely on the three user‑supplied entities. No hardcoded criteria are allowed.
- [high] Iterative Research Loop: Implement a research_entity node that processes one entity‑criterion pair per iteration, updates the state with findings, and continues until all pairs are covered before moving to build_table.
- [high] Real Tavily Integration: Each research step must perform an actual web search via the Tavily SDK or API. The returned snippets should be used verbatim in the findings; fabricated links or hallucinations are disallowed.
- [normal] Markdown Table Construction: The build_table node must assemble a complete Markdown table with rows for each criterion and columns for each entity, filling all cells with the collected findings.
- [normal] Verdict Generation: A separate verdict node should use an LLM to produce 2–4 sentences recommending which entity suits specific scenarios based on the table.
- [low] CLI Interface & README: Provide a command‑line interface that accepts optional custom entities, displays the criteria plan, shows progress of research pairs, and finally prints the Markdown table and verdict. Include a README with setup instructions (pip install, .env TAVILY_API_KEY).
