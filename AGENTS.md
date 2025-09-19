# AGENTS.md

1. **Model choice — coding**  
   Default to the highest-performing available model for coding tasks (e.g., gpt-5-codex high).  
   Record the exact model/version and any fallbacks used. Never use deprecated Codex models.

2. **Determinism**  
   Use `temperature=0–0.2`. Fix `seed` if supported. Leave `top_p`, `frequency_penalty`, and `presence_penalty` at defaults unless explicitly specified.

3. **Clarity & scope**  
   - Adhere strictly to the specified environment: language, runtime, framework, style, and output format.  
   - Work only with provided files/snippets. If assumptions are made, state them explicitly. Never fabricate unseen files.

4. **Code principles**  
   Follow SOLID, DRY, KISS, 12-Factor.  
   For Python: follow PEP-8 and the Zen of Python.

5. **Output format**  
   When edits are requested, return **one of**:  
   - Unified diffs (repo-rooted paths with context lines, applicable via `git apply` or `patch`).  
   - Structured JSON describing file edits (`path`, `action`, `before`, `after`, `insert_at`).  

6. **Testing**  
   Provide minimal yet sufficient unit tests (at least one happy-path and one failure-path).  
   Include required mocks/test data and exact commands to run the tests.

7. **Validation**  
   Provide commands for static checks, formatting, and builds (e.g., `black`, `flake8`, `pytest`, `mypy`, `npm run lint`).  
   State the expected pass/fail outcome.

8. **Performance & complexity**  
   For hot paths, include algorithmic complexity (Big-O) and note practical implications.  
   Provide simple micro-benchmark steps if performance matters.

9. **Security**  
   Do not expose or invent secrets, API keys, or credentials.  
   Use safe patterns: input validation, parameterized queries, safe deserialization.

10. **Reproducibility**  
    If changes add dependencies/tools, include setup commands and version pins.  
    Where applicable, provide `requirements.txt`, `pyproject.toml`, or Dockerfile entries.

11. **Documentation**  
    Update or add docstrings, README, or CHANGELOG when behavior, configs, or public APIs change.

12. **Style & tone**  
    Follow the project’s linter/formatter.  
    No emojis unless explicitly requested.  
    Keep explanations concise and actionable (brief rationale + concrete commands/examples).
