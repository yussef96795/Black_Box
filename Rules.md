# General Coding Standards & Best Practices

## 1. Core Engineering Principles
* **KISS (Keep It Simple, Stupid):** Write simple, readable code over complex "clever" implementations.
* **DRY (Don't Repeat Yourself):** Abstract reusable logic into utility functions or shared components, but avoid premature abstraction.
* **YAGNI (You Aren't Gonna Need It):** Do not implement functionality until it is explicitly needed.
* **Single Responsibility Principle:** Every module, function, or component should do one thing and do it well.

---

## 2. Code Quality & Readability
* **Explicit Naming:** Use clear, descriptive variable and function names (e.g., `calculate_total_price` instead of `calc_tot`).
* **Functions:** Keep functions short and focused. Prefer pure functions (no side effects) where possible.
* **Comments:** Write code that self-documents. Use comments only to explain *why* something was done a certain way, not *what* the code is doing.
* **Formatting:** Enforce consistent formatting using automated tools (e.g., `ruff`/`black` for Python, `prettier` for JavaScript/TypeScript).

---

## 3. Python & FastAPI Standards
* **Type Hints:** Type annotate all function signatures, return values, and parameters explicitly using standard Python type hints or `pydantic` schemas.
* **Environment Variables:** Never hardcode secrets, keys, or endpoints. Use `.env` files managed via `pydantic-settings`.
* **Async Executions:** Use `async/await` appropriately for standard I/O bound tasks and DB calls in FastAPI routes.
* **Error Handling:** Use structured standard HTTP exception handling (`HTTPException`) with clear, informative error messages and status codes.

---

## 4. Web & Frontend Standards
* **Component Design:** Modularize UI into small, composable components.
* **State Management:** Keep state local whenever possible; lift state up only when siblings require shared access.
* **Tailwind CSS:** Keep utility classes organized logically (Layout -> Spacing -> Sizing -> Typography -> Colors).
* **API Calls:** Isolate API calls into dedicated service modules/hooks rather than inline inside UI components.

---

## 5. Security & Environment
* **Secrets Management:** Keep API keys, tokens, and database passwords out of source control. Ensure `.env` is always listed in `.gitignore`.
* **Input Validation:** Validate all incoming payload structures strictly using schema validation (e.g., `Pydantic` or `Zod`) before processing.
* **Least Privilege:** When configuring tokens, database permissions, or API integration tokens, restrict scopes strictly to required resources.

---

## 6. Git & Version Control
* **Atomic Commits:** Make small, logical commits focused on a single feature, bug fix, or refactor.
* **Commit Messages:** Follow standard conventional commit formats:
  * `feat: add user authentication endpoint`
  * `fix: handle null response in API schema`
  * `docs: update setup commands in README`
* **Clean Branches:** Keep feature branches short-lived and pull request descriptions descriptive.