import matlab.engine
import numpy as np
import json
import re
import os
import tempfile
import logging
from groq import Groq
from dotenv import load_dotenv
import base64
from io import BytesIO, StringIO
import matplotlib.pyplot as plt
import matplotlib
from dataclasses import dataclass, field
matplotlib.use('Agg')  # Non-interactive backend

load_dotenv()

logger = logging.getLogger(__name__)

client = Groq()
MODEL = "openai/gpt-oss-120b"

MAX_ITERATIONS = 5


# --- Custom Exceptions ---

class PlannerError(Exception):
    """Raised when the LLM cannot produce a valid Pipeline."""


class CycleError(Exception):
    """Raised when a cycle is detected in the dependency graph."""


class MissingArtifactError(Exception):
    """Raised when a referenced step_id is not in the artifact store."""


class SerializationError(Exception):
    """Raised when serialization of a step's output artifact fails."""


# --- Data Models ---

@dataclass
class Step:
    step_id: str                          # unique within pipeline, e.g. "step_1"
    description: str                      # natural-language description of the computation
    input_sources: list[str] = field(default_factory=list)  # CSV paths or "{step_id}.output" references
    is_terminal: bool = False             # True for the final step


@dataclass
class Pipeline:
    steps: list[Step]                     # ordered list (Planner may return any order; Runner sorts)


@dataclass
class StepResult:
    step_id: str
    description: str
    code: str  
    execution_result: dict                # {output, plots, error, step_output}
    answer: str  
    status: str                           # "done" | "failed"


def extract_matlab_code(text: str):
    """
    Extract MATLAB code from markdown code blocks.
    Handles ```matlab, ```MATLAB, and plain ``` fenced blocks.
    Returns None when no block is found.
    """
    if not text:
        return None

    # Try to find code in ```matlab or ```MATLAB blocks (case-insensitive tag)
    matlab_pattern = r"```(?:matlab|MATLAB)\s*\n(.*?)```"
    matches = re.findall(matlab_pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Fall back to generic ``` blocks
    generic_pattern = r"```\s*\n(.*?)```"
    matches = re.findall(generic_pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()

    return None


def inject_csv_context(matlab_code: str, csv_path: str) -> str:
    """
    Prepend a csv_path variable assignment header to the MATLAB code.
    The header format is:
        csv_path = '<path>';

    followed by a blank line before the original code.
    """
    header = f"csv_path = '{csv_path}';\n\n"
    return header + matlab_code


def detect_plot_intent(plan: str) -> bool:
    """
    Return True if the plan string mentions any plotting-related keywords.
    """
    keywords = [
        "plot", "figure", "graph", "chart", "visualize", "visualise",
        "draw", "bar", "histogram", "scatter", "pie", "surf", "mesh", "contour",
    ]
    plan_lower = plan.lower()
    return any(kw in plan_lower for kw in keywords)

def _build_csv_context(csv_files: list) -> str:
    """
    Build a context string describing all provided CSV files with their paths and previews.
    csv_files: list of dicts with keys 'path' and 'preview'.
    """
    if not csv_files:
        return ""
    parts = ["\n\nCSV Files available:"]
    for i, f in enumerate(csv_files, 1):
        parts.append(f"\n  File {i}: '{f['path']}'")
        if f.get("preview"):
            parts.append(f"  Preview (first 5 rows):\n{f['preview']}")
    return "\n".join(parts)


def _pipeline_planner(user_prompt: str, csv_files: list = None) -> "Pipeline":
    """
    Calls the LLM with a system prompt that instructs it to output a Pipeline JSON.
    Parses and validates the JSON into a Pipeline dataclass.
    Raises PlannerError if the LLM cannot produce a valid Pipeline.
    """
    csv_context = _build_csv_context(csv_files) if csv_files else ""
    csv_paths_hint = ""
    if csv_files:
        paths = ", ".join(f'"{f["path"]}"' for f in csv_files)
        csv_paths_hint = (
            f"\n\nAvailable CSV files: {paths}. "
            "Use these exact paths as input_sources where appropriate."
        )
    else:
        csv_paths_hint = (
            "\n\nNo CSV files were provided. All data must be derived directly from the text prompt. "
            "Do NOT reference or try to load any CSV files, and do NOT include any CSV files in input_sources."
        )

    system_msg = (
        "You are a MATLAB task planner. Given a user request, decompose it into a "
        "pipeline of steps and output ONLY a JSON object — no markdown, no extra text — "
        "matching this schema exactly:\n\n"
        '{\n'
        '  "steps": [\n'
        '    {\n'
        '      "step_id": "step_1",\n'
        '      "description": "Load the CSV and compute the mean of column A",\n'
        '      "input_sources": ["data.csv"],\n'
        '      "is_terminal": false\n'
        '    },\n'
        '    {\n'
        '      "step_id": "step_2",\n'
        '      "description": "Plot the normalized values from step_1",\n'
        '      "input_sources": ["{step_1}.output"],\n'
        '      "is_terminal": true\n'
        '    }\n'
        '  ]\n'
        '}\n\n'
        "Rules:\n"
        "- steps must be a non-empty list.\n"
        "- Each step must have: step_id (non-empty string), description (non-empty string), "
        "input_sources (list of strings), is_terminal (boolean).\n"
        "- Exactly one step must have is_terminal=true.\n"
        "- All step_id values must be unique.\n"
        "- Each input_source must be either a string ending in .csv (ONLY if present in Available CSV files) or match the pattern "
        "{word}.output (e.g. {step_1}.output).\n"
        "- Do NOT invent or hallucinate CSV file names. If there are no Available CSV files, input_sources for the first step should be empty [].\n"
        "- If data is provided in the prompt text, the generated code should hardcode or parse it directly; do not read from a file.\n"
        "- If the task can be done in a single computation, produce a pipeline with one step.\n"
        "- Output ONLY the JSON object. No markdown fences, no explanation."
    )

    user_msg = user_prompt + csv_context + csv_paths_hint

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=8000,
            stream=False,
        )
    except Exception as e:
        raise PlannerError(f"LLM call failed: {e}") from e

    raw = response.choices[0].message.content or ""

    # Strip markdown fences if present
    stripped = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL)
    if fence_match:
        stripped = fence_match.group(1).strip()

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as e:
        raise PlannerError(f"Failed to parse Pipeline JSON: {e}. LLM response: {raw!r}") from e

    # Validate top-level structure
    if not isinstance(data, dict) or "steps" not in data:
        raise PlannerError(f"Pipeline JSON missing 'steps' key. Got: {data!r}")

    raw_steps = data["steps"]
    if not isinstance(raw_steps, list) or len(raw_steps) == 0:
        raise PlannerError("Pipeline 'steps' must be a non-empty list.")

    input_source_pattern = re.compile(r"^\{[\w]+\}\.output$")
    steps: list[Step] = []
    seen_ids: set[str] = set()
    terminal_count = 0

    for i, s in enumerate(raw_steps):
        # Validate required fields
        if not isinstance(s, dict):
            raise PlannerError(f"Step {i} is not a dict: {s!r}")

        step_id = s.get("step_id")
        if not isinstance(step_id, str) or not step_id.strip():
            raise PlannerError(f"Step {i} has invalid or missing 'step_id': {step_id!r}")

        description = s.get("description")
        if not isinstance(description, str) or not description.strip():
            raise PlannerError(f"Step '{step_id}' has invalid or missing 'description': {description!r}")

        input_sources = s.get("input_sources")
        if not isinstance(input_sources, list):
            raise PlannerError(f"Step '{step_id}' 'input_sources' must be a list, got: {input_sources!r}")

        is_terminal = s.get("is_terminal")
        if not isinstance(is_terminal, bool):
            raise PlannerError(f"Step '{step_id}' 'is_terminal' must be a bool, got: {is_terminal!r}")

        # Validate uniqueness
        if step_id in seen_ids:
            raise PlannerError(f"Duplicate step_id '{step_id}' found in pipeline.")
        seen_ids.add(step_id)

        # Validate input_sources values
        for src in input_sources:
            if not isinstance(src, str):
                raise PlannerError(f"Step '{step_id}' input_source {src!r} is not a string.")
            if not (src.endswith(".csv") or input_source_pattern.match(src)):
                raise PlannerError(
                    f"Step '{step_id}' input_source {src!r} is invalid. "
                    "Must end in .csv or match {word}.output."
                )

        if is_terminal:
            terminal_count += 1

        steps.append(Step(
            step_id=step_id,
            description=description,
            input_sources=input_sources,
            is_terminal=is_terminal,
        ))

    if terminal_count != 1:
        raise PlannerError(
            f"Pipeline must have exactly one terminal step, found {terminal_count}."
        )

    return Pipeline(steps=steps)


def _topological_sort(steps: list[Step]) -> list[Step]:
    """
    Sort steps in topological order using Kahn's algorithm.
    Dependencies are inferred from input_sources: any source matching {step_id}.output
    is treated as a dependency on that step_id.
    Raises CycleError if a cycle is detected.
    """
    # Build a map from step_id -> Step for quick lookup
    step_map: dict[str, Step] = {s.step_id: s for s in steps}

    # Extract dependency pattern: {step_id}.output
    dep_pattern = re.compile(r"^\{([\w]+)\}\.output$")

    def get_deps(step: Step) -> list[str]:
        deps = []
        for src in step.input_sources:
            m = dep_pattern.match(src)
            if m:
                deps.append(m.group(1))
        return deps

    # Build adjacency list (dep -> dependents) and in-degree map
    in_degree: dict[str, int] = {s.step_id: 0 for s in steps}
    adjacency: dict[str, list[str]] = {s.step_id: [] for s in steps}

    for step in steps:
        for dep_id in get_deps(step):
            adjacency[dep_id].append(step.step_id)
            in_degree[step.step_id] += 1

    # Kahn's algorithm: start with all zero-in-degree nodes
    from collections import deque
    queue: deque[str] = deque(sid for sid, deg in in_degree.items() if deg == 0)
    sorted_ids: list[str] = []

    while queue:
        current = queue.popleft()
        sorted_ids.append(current)
        for neighbor in adjacency[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(sorted_ids) != len(steps):
        raise CycleError(
            "Cycle detected in pipeline dependency graph. "
            f"Steps involved: {[s for s in in_degree if in_degree[s] > 0]}"
        )

    return [step_map[sid] for sid in sorted_ids]


def _resolve_inputs(step: "Step", artifact_store: dict, csv_files: list) -> list:
    """
    Returns a list of resolved file paths for the step's input_sources.
    - {step_id}.output references are looked up in artifact_store.
    - Plain .csv paths are passed through unchanged.
    Raises MissingArtifactError if a referenced step_id is not in the store.
    """
    dep_pattern = re.compile(r"^\{([\w]+)\}\.output$")
    resolved = []
    for src in step.input_sources:
        m = dep_pattern.match(src)
        if m:
            ref_id = m.group(1)
            if ref_id not in artifact_store:
                raise MissingArtifactError(
                    f"Step '{step.step_id}' references '{src}', but step_id '{ref_id}' "
                    f"has no artifact in the store. Ensure '{ref_id}' ran successfully before '{step.step_id}'."
                )
            resolved.append(artifact_store[ref_id])
        else:
            # Plain CSV path — pass through unchanged
            resolved.append(src)
    return resolved


def _serialize_artifact(step_id: str, step_output) -> str:
    """
    Serializes step_output (numpy array, list, or scalar) to a temp CSV file.
    Returns the absolute file path. Raises SerializationError on failure.
    """
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp_path = tmp.name
        tmp.close()

        if isinstance(step_output, np.ndarray):
            arr = step_output.flatten() if step_output.ndim == 1 else step_output
            np.savetxt(tmp_path, arr, delimiter=",")
        elif isinstance(step_output, list):
            arr = np.array(step_output)
            arr = arr.flatten() if arr.ndim == 1 else arr
            np.savetxt(tmp_path, arr, delimiter=",")
        else:
            # scalar or any other type
            with open(tmp_path, "w") as f:
                f.write(str(step_output))

        return os.path.abspath(tmp_path)
    except Exception as e:
        raise SerializationError(
            f"Failed to serialize artifact for step '{step_id}': {e}"
        ) from e


def _cleanup_artifacts(artifact_store: dict) -> None:
    """
    Deletes all temp CSV files recorded in artifact_store.
    Logs a warning on deletion failure but does not raise.
    """
    for step_id, file_path in artifact_store.items():
        try:
            os.remove(file_path)
        except Exception as e:
            logger.warning(
                "Failed to delete temp artifact for step '%s' at '%s': %s",
                step_id, file_path, e,
            )


def _planner(user_prompt: str, csv_files: list = None) -> str:
    """
    Step 1 of the MATLAB pipeline: produce a concise numbered plan.
    csv_files: list of dicts with keys 'path' and 'preview'.
    """
    system_msg = (
        "You are a MATLAB task planner. Given a user request, output a concise "
        "numbered plan describing exactly what the MATLAB code must compute or plot. "
        "Do NOT write any MATLAB code — produce a plan only."
    )

    user_msg = user_prompt
    if csv_files:
        user_msg += _build_csv_context(csv_files)
        paths = ", ".join(f"'{f['path']}'" for f in csv_files)
        user_msg += f"\n\nThe plan must include loading data from these exact file paths: {paths}."
    else:
        user_msg += "\n\nNote: No CSV files are provided. You must plan to extract and hardcode any required data directly from this text prompt, instead of reading from a file."

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=8000,
        stream=False,
    )
    return response.choices[0].message.content


_CODE_GEN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "submit_matlab_code",
            "description": (
                "Submit the generated MATLAB code along with a flag indicating "
                "whether the code produces a plot."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "matlab_code": {
                        "type": "string",
                        "description": "The complete, executable MATLAB code."
                    },
                    "is_plot": {
                        "type": "boolean",
                        "description": (
                            "True if the code generates a plot/figure/chart; "
                            "False if it only performs calculations and prints output."
                        )
                    }
                },
                "required": ["matlab_code", "is_plot"]
            }
        }
    }
]


def _code_generator(plan: str, previous_code: str, feedback: str, csv_files: list = None, user_prompt: str = "") -> tuple[str, bool]:
    """
    Step 2 of the MATLAB pipeline: generate (or correct) MATLAB code from a plan.
    csv_files: list of dicts with keys 'path' and 'preview'.
    Returns (matlab_code, is_plot) or (None, False) if the tool call is missing.
    """
    system_msg = (
        "You are a MATLAB code generator. Generate clean, executable MATLAB code "
        "that fulfills the given plan.\n\n"
        "CRITICAL RULES:\n"
        "1. NO UI ELEMENTS: Do NOT use 'figure', 'plot', 'subplot', 'title', 'xlabel', 'ylabel', 'legend', 'grid', 'hold', or any other UI/plotting functions. The sandbox has no screen and will hang if these are called.\n"
        "2. PLOT DATA EXTRACTION: If the plan requires a plot, you MUST set is_plot=true in the submit_matlab_code tool call. When is_plot=true, you MUST store the data in workspace variables instead of using plotting functions. Use 'x_data' (for time/frequency/horizontal axis) and 'y_data' (for amplitude/magnitude/vertical axis). For multiple plots, use 'x1', 'y1', 'x2', 'y2', etc. Optionally, set 'plot_title', 'plot_xlabel', 'plot_ylabel', and 'plot_legends' (cell array of strings) for metadata.\n"
        "3. NO FILE I/O: Do NOT save files. Do NOT read files unless explicitly instructed via the CSV instruction.\n"
        "4. TERMINAL OUTPUT: Print all relevant numerical results to stdout using disp() or fprintf().\n"
        "5. STEP OUTPUT: You MUST assign the primary result of the computation to a variable named 'step_output' at the end of the script.\n"
        "6. HARDCODE DATA: If the request contains specific values or tables, hardcode them into the script.\n\n"
        "You MUST call the submit_matlab_code tool with your code and the correct is_plot flag."
    )

    csv_instruction = ""
    if csv_files:
        lines = ["\n\nIMPORTANT: You MUST load data from the following CSV files using their exact paths:"]
        for i, f in enumerate(csv_files, 1):
            lines.append(
                f"  File {i}: use readtable('{f['path']}') or readmatrix('{f['path']}') "
                f"to load '{f['path']}'. Do NOT hardcode any other path."
            )
        csv_instruction = "\n".join(lines)

    base_context = f"Original Request:\n{user_prompt}\n\nPlan:\n{plan}{csv_instruction}"
    
    if feedback is not None and previous_code is not None:
        user_msg = (
            f"{base_context}"
            f"\n\nPrevious code:\n```matlab\n{previous_code}\n```"
            f"\n\nFeedback from reviewer:\n{feedback}"
            f"\n\nGenerate corrected MATLAB code."
        )
    else:
        user_msg = f"{base_context}\n\nGenerate MATLAB code."

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        tools=_CODE_GEN_TOOLS,
        tool_choice={"type": "function", "function": {"name": "submit_matlab_code"}},
        max_tokens=8000,
        stream=False,
    )

    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        # Fallback: try to extract code from plain text response
        raw = response.choices[0].message.content or ""
        code = extract_matlab_code(raw)
        return code, False

    args = json.loads(tool_calls[0].function.arguments)
    return args.get("matlab_code"), bool(args.get("is_plot", False))


def _reviewer(plan: str, code: str, execution_result: dict, user_prompt: str = "") -> dict:
    """
    Step 4 of the MATLAB pipeline: review execution output against the plan.
    Returns a ReviewVerdict-shaped dict: {verdict: 'done'|'fix', feedback: str, answer: str}
    On JSON parse failure defaults to {verdict: 'fix', feedback: 'Reviewer response was unparseable — retry'}.
    """
    system_msg = (
        "You are a MATLAB output reviewer. Given the original request, the plan, the code, and "
        "the execution output (or error), decide: "
        "if the output satisfies the plan respond with JSON {\"verdict\":\"done\", \"answer\":\"...\"}, "
        "if not respond with JSON {\"verdict\":\"fix\", \"feedback\":\"specific correction needed\"}"
    )

    output_section = execution_result.get("output") or "(no output)"
    error_section  = execution_result.get("error")  or "(no error)"

    user_msg = (
        f"Original Request:\n{user_prompt}"
        f"\n\nPlan:\n{plan}"
        f"\n\nCode:\n```matlab\n{code}\n```"
        f"\n\nExecution Output:\n{output_section}"
        f"\n\nExecution Error:\n{error_section}"
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=8000,
        stream=False,
    )

    raw = response.choices[0].message.content
    try:
        verdict = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from the response if it's wrapped in markdown or extra text
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            try:
                verdict = json.loads(json_match.group())
            except json.JSONDecodeError:
                verdict = {"verdict": "fix", "feedback": "Reviewer response was unparseable — retry"}
        else:
            verdict = {"verdict": "fix", "feedback": "Reviewer response was unparseable — retry"}

    # Ensure required fields are present with sensible defaults
    if "verdict" not in verdict or verdict["verdict"] not in ("done", "fix"):
        verdict["verdict"] = "fix"
    if verdict["verdict"] == "done" and not verdict.get("answer"):
        verdict["answer"] = raw  # fall back to raw response as answer
    if verdict["verdict"] == "fix" and not verdict.get("feedback"):
        verdict["feedback"] = "Reviewer response was unparseable — retry"

    return verdict


def _execute(matlab_code: str, is_plot: bool = False) -> dict:
    """
    Step 3 of the MATLAB pipeline: execute MATLAB code in a matlab.engine sandbox.

    Returns a dict with keys:
        output (str)       - captured stdout
        plots  (list[str]) - base64-encoded PNG strings (populated when is_plot=True)
        error  (str|None)  - exception message, or None on success
    """
    result: dict = {"output": "", "plots": [], "error": None}

    eng = None
    try:
        eng = matlab.engine.start_matlab()
        print(matlab_code)

        output_buffer = StringIO()
        eng.eval(matlab_code, nargout=0, stdout=output_buffer)
        result["output"] = output_buffer.getvalue()

        if is_plot:
            workspace_vars = eng.eval("who", nargout=1)

            # --- metadata ---
            plot_title  = str(eng.workspace["plot_title"])  if "plot_title"  in workspace_vars else "Plot"
            plot_xlabel = str(eng.workspace["plot_xlabel"]) if "plot_xlabel" in workspace_vars else "X"
            plot_ylabel = str(eng.workspace["plot_ylabel"]) if "plot_ylabel" in workspace_vars else "Y"
            plot_legends: list[str] = []
            if "plot_legends" in workspace_vars:
                plot_legends = [str(leg) for leg in eng.workspace["plot_legends"]]

            # --- data extraction ---
            plot_data_sets: list[dict] = []

            if "x_data" in workspace_vars and "y_data" in workspace_vars:
                x = np.array(eng.workspace["x_data"]).flatten()
                y = np.array(eng.workspace["y_data"]).flatten()
                plot_data_sets.append({"x": x, "y": y, "label": "Data"})
            else:
                idx = 1
                while f"x{idx}" in workspace_vars and f"y{idx}" in workspace_vars:
                    x = np.array(eng.workspace[f"x{idx}"]).flatten()
                    y = np.array(eng.workspace[f"y{idx}"]).flatten()
                    label = plot_legends[idx - 1] if idx - 1 < len(plot_legends) else f"Plot {idx}"
                    plot_data_sets.append({"x": x, "y": y, "label": label})
                    idx += 1

            if not plot_data_sets:
                common_x = ["t", "time", "x", "freq", "w"]
                common_y = ["y", "output", "response", "mag", "magnitude"]
                x_var = next((v for v in common_x if v in workspace_vars), None)
                y_vars = [v for v in common_y if v in workspace_vars]
                if x_var and y_vars:
                    x = np.array(eng.workspace[x_var]).flatten()
                    for y_var in y_vars:
                        y = np.array(eng.workspace[y_var]).flatten()
                        plot_data_sets.append({"x": x, "y": y, "label": y_var})

            if plot_data_sets:
                plt.figure(figsize=(10, 6))
                for ds in plot_data_sets:
                    plt.plot(ds["x"], ds["y"], label=ds["label"], linewidth=2)
                plt.xlabel(plot_xlabel, fontsize=12)
                plt.ylabel(plot_ylabel, fontsize=12)
                plt.title(plot_title, fontsize=14, fontweight="bold")
                plt.grid(True, alpha=0.3)
                if len(plot_data_sets) > 1 or plot_legends:
                    plt.legend()
                buf = BytesIO()
                plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                buf.seek(0)
                result["plots"].append(base64.b64encode(buf.read()).decode("utf-8"))
                plt.close()
            else:
                if result["error"] is None:
                    result["error"] = (
                        "Could not find plot data in MATLAB workspace. "
                        "Ensure data is stored in variables like x_data/y_data or x1/y1, x2/y2, etc."
                    )

    except Exception as e:
        result["error"] = str(e)
    finally:
        if eng is not None:
            eng.quit()

    return result


def _execute_and_capture(matlab_code: str, is_plot: bool = False) -> dict:
    """
    Like _execute, but additionally extracts 'step_output' from the MATLAB workspace
    before quitting the engine.

    Returns a dict with keys:
        output      (str)       - captured stdout
        plots       (list[str]) - base64-encoded PNG strings
        error       (str|None)  - exception message, or None on success
        step_output (Any|None)  - value of the MATLAB workspace variable 'step_output',
                                  or None if absent or if an error occurred
        warnings    (list[str]) - non-fatal warnings (e.g. step_output not found)
    """
    result: dict = {"output": "", "plots": [], "error": None, "step_output": None, "warnings": []}

    eng = None
    try:
        eng = matlab.engine.start_matlab()
        print(matlab_code)

        output_buffer = StringIO()
        eng.eval(matlab_code, nargout=0, stdout=output_buffer)
        result["output"] = output_buffer.getvalue()

        # workspace_vars is needed for both plot handling and step_output extraction
        workspace_vars = eng.eval("who", nargout=1)

        if is_plot:
            # --- metadata ---
            plot_title  = str(eng.workspace["plot_title"])  if "plot_title"  in workspace_vars else "Plot"
            plot_xlabel = str(eng.workspace["plot_xlabel"]) if "plot_xlabel" in workspace_vars else "X"
            plot_ylabel = str(eng.workspace["plot_ylabel"]) if "plot_ylabel" in workspace_vars else "Y"
            plot_legends: list[str] = []
            if "plot_legends" in workspace_vars:
                plot_legends = [str(leg) for leg in eng.workspace["plot_legends"]]

            # --- data extraction ---
            plot_data_sets: list[dict] = []

            if "x_data" in workspace_vars and "y_data" in workspace_vars:
                x = np.array(eng.workspace["x_data"]).flatten()
                y = np.array(eng.workspace["y_data"]).flatten()
                plot_data_sets.append({"x": x, "y": y, "label": "Data"})
            else:
                idx = 1
                while f"x{idx}" in workspace_vars and f"y{idx}" in workspace_vars:
                    x = np.array(eng.workspace[f"x{idx}"]).flatten()
                    y = np.array(eng.workspace[f"y{idx}"]).flatten()
                    label = plot_legends[idx - 1] if idx - 1 < len(plot_legends) else f"Plot {idx}"
                    plot_data_sets.append({"x": x, "y": y, "label": label})
                    idx += 1

            if not plot_data_sets:
                common_x = ["t", "time", "x", "freq", "w"]
                common_y = ["y", "output", "response", "mag", "magnitude"]
                x_var = next((v for v in common_x if v in workspace_vars), None)
                y_vars = [v for v in common_y if v in workspace_vars]
                if x_var and y_vars:
                    x = np.array(eng.workspace[x_var]).flatten()
                    for y_var in y_vars:
                        y = np.array(eng.workspace[y_var]).flatten()
                        plot_data_sets.append({"x": x, "y": y, "label": y_var})

            if plot_data_sets:
                plt.figure(figsize=(10, 6))
                for ds in plot_data_sets:
                    plt.plot(ds["x"], ds["y"], label=ds["label"], linewidth=2)
                plt.xlabel(plot_xlabel, fontsize=12)
                plt.ylabel(plot_ylabel, fontsize=12)
                plt.title(plot_title, fontsize=14, fontweight="bold")
                plt.grid(True, alpha=0.3)
                if len(plot_data_sets) > 1 or plot_legends:
                    plt.legend()
                buf = BytesIO()
                plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                buf.seek(0)
                result["plots"].append(base64.b64encode(buf.read()).decode("utf-8"))
                plt.close()
            else:
                if result["error"] is None:
                    result["error"] = (
                        "Could not find plot data in MATLAB workspace. "
                        "Ensure data is stored in variables like x_data/y_data or x1/y1, x2/y2, etc."
                    )

        # Extract step_output from workspace (workspace_vars already fetched above)
        if "step_output" in workspace_vars:
            result["step_output"] = eng.workspace["step_output"]
        elif result["error"] is None:
            result["warnings"].append("step_output not found in MATLAB workspace")

    except Exception as e:
        result["error"] = str(e)
    finally:
        if eng is not None:
            eng.quit()

    return result


def execute_matlab_calculation(matlab_code):
    """
    Execute MATLAB code for calculations (non-plotting) and return text output
    
    Returns:
        dict with 'output' (text), 'error' (if any)
    """
    result = {
        'output': '',
        'error': None
    }
    
    try:
        print("Starting MATLAB engine for calculations...")
        eng = matlab.engine.start_matlab()
        
        # Capture output
        from io import StringIO
        output_buffer = StringIO()
        
        print("Executing MATLAB code...")
        try:
            eng.eval(matlab_code, nargout=0, stdout=output_buffer)
            result['output'] = output_buffer.getvalue()
        except Exception as eval_error:
            result['error'] = str(eval_error)
            print(f"MATLAB execution error: {eval_error}")
        
        print("Stopping MATLAB engine...")
        eng.quit()
        
    except Exception as e:
        result['error'] = f"Failed to execute MATLAB code: {str(e)}"
        print(result['error'])
    
    return result

def execute_matlab_for_plot_data(matlab_code):
    """
    Execute MATLAB code to extract plot data (x, y), then use matplotlib for plotting
    
    The MATLAB code should store plot data in variables like:
    - x_data, y_data (or x1, y1, x2, y2 for multiple plots)
    - plot_title, plot_xlabel, plot_ylabel, plot_legends (optional metadata)
    
    Returns:
        dict with 'output' (text), 'plots' (list of base64 encoded images), 'error' (if any)
    """
    result = {
        'output': '',
        'plots': [],
        'error': None
    }
    
    try:
        print("Starting MATLAB engine for plot data extraction...")
        eng = matlab.engine.start_matlab()
        
        # Capture output
        from io import StringIO
        output_buffer = StringIO()
        
        print("Executing MATLAB code to compute plot data...")
        try:
            eng.eval(matlab_code, nargout=0, stdout=output_buffer)
            result['output'] = output_buffer.getvalue()
            
            # Get workspace variables
            workspace_vars = eng.eval("who", nargout=1)
            print(f"Workspace variables: {workspace_vars}")
            
            # Extract plot metadata if available
            plot_title = "Plot"
            plot_xlabel = "X"
            plot_ylabel = "Y"
            plot_legends = []
            
            if 'plot_title' in workspace_vars:
                plot_title = str(eng.workspace['plot_title'])
            if 'plot_xlabel' in workspace_vars:
                plot_xlabel = str(eng.workspace['plot_xlabel'])
            if 'plot_ylabel' in workspace_vars:
                plot_ylabel = str(eng.workspace['plot_ylabel'])
            if 'plot_legends' in workspace_vars:
                legends_cell = eng.workspace['plot_legends']
                plot_legends = [str(leg) for leg in legends_cell]
            
            # Extract plot data - look for x_data, y_data or x1, y1, x2, y2, etc.
            plot_data_sets = []
            
            # Try single plot data first
            if 'x_data' in workspace_vars and 'y_data' in workspace_vars:
                x = np.array(eng.workspace['x_data']).flatten()
                y = np.array(eng.workspace['y_data']).flatten()
                plot_data_sets.append({'x': x, 'y': y, 'label': 'Data'})
            
            # Try numbered plot data (x1, y1, x2, y2, etc.)
            else:
                idx = 1
                while f'x{idx}' in workspace_vars and f'y{idx}' in workspace_vars:
                    x = np.array(eng.workspace[f'x{idx}']).flatten()
                    y = np.array(eng.workspace[f'y{idx}']).flatten()
                    label = plot_legends[idx-1] if idx-1 < len(plot_legends) else f'Plot {idx}'
                    plot_data_sets.append({'x': x, 'y': y, 'label': label})
                    idx += 1
            
            # If no standard variables found, try common patterns (t, y), (x, y), etc.
            if not plot_data_sets:
                common_x = ['t', 'time', 'x', 'freq', 'w']
                common_y = ['y', 'output', 'response', 'mag', 'magnitude']
                
                x_var = None
                y_vars = []
                
                for var in common_x:
                    if var in workspace_vars:
                        x_var = var
                        break
                
                for var in common_y:
                    if var in workspace_vars:
                        y_vars.append(var)
                
                if x_var and y_vars:
                    x = np.array(eng.workspace[x_var]).flatten()
                    for y_var in y_vars:
                        y = np.array(eng.workspace[y_var]).flatten()
                        plot_data_sets.append({'x': x, 'y': y, 'label': y_var})
            
            # Create matplotlib plot
            if plot_data_sets:
                plt.figure(figsize=(10, 6))
                
                for dataset in plot_data_sets:
                    plt.plot(dataset['x'], dataset['y'], label=dataset['label'], linewidth=2)
                
                plt.xlabel(plot_xlabel, fontsize=12)
                plt.ylabel(plot_ylabel, fontsize=12)
                plt.title(plot_title, fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)
                
                if len(plot_data_sets) > 1 or plot_legends:
                    plt.legend()
                
                # Convert to base64
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                result['plots'].append(plot_base64)
                plt.close()
                
                print(f"Successfully created plot with {len(plot_data_sets)} dataset(s)")
            else:
                result['error'] = "Could not find plot data in MATLAB workspace. Please ensure data is stored in variables like x_data, y_data or x1, y1, x2, y2, etc."
            
        except Exception as eval_error:
            result['error'] = str(eval_error)
            print(f"MATLAB execution error: {eval_error}")
        
        print("Stopping MATLAB engine...")
        eng.quit()
        
    except Exception as e:
        result['error'] = f"Failed to execute MATLAB code: {str(e)}"
        print(result['error'])
    
    return result

def run_matlab_executor_agent(user_prompt, csv_files: list = None):
    """
    Main pipeline loop for the MATLAB executor agent.

    csv_files: list of dicts with keys 'path' and 'preview', one per CSV file.
               e.g. [{"path": "/data/a.csv", "preview": "col1,col2\\n1,2\\n..."}, ...]
    Runs the plan→generate→execute→review cycle up to MAX_ITERATIONS times per step.
    Returns a formatted response string (with optional embedded plots).
    """
    # Step 1: Plan the pipeline
    try:
        pipeline = _pipeline_planner(user_prompt, csv_files)
    except PlannerError as e:
        return f"Pipeline planning failed: {e}"

    # Step 2: Topological sort
    try:
        ordered_steps = _topological_sort(pipeline.steps)
    except CycleError as e:
        return f"Pipeline has a dependency cycle and cannot be executed: {e}"

    # Step 3: Initialize state
    artifact_store: dict[str, str] = {}
    step_results: list[StepResult] = []
    warnings: list[str] = []

    # Step 4: Execute steps with cleanup in finally
    try:
        for step in ordered_steps:
            # 5a: Resolve inputs
            try:
                resolved_paths = _resolve_inputs(step, artifact_store, csv_files or [])
            except MissingArtifactError as e:
                step_results.append(StepResult(
                    step_id=step.step_id,
                    description=step.description,
                    code=None,
                    execution_result={"output": "", "plots": [], "error": str(e), "step_output": None},
                    answer=None,
                    status="failed",
                ))
                break

            # 5b: Build step_csv_files from resolved paths
            step_csv_files = [{"path": p, "preview": ""} for p in resolved_paths] if resolved_paths else None

            # 5c: Generate → execute → review loop
            previous_code = None
            feedback = None
            result = {"output": "", "plots": [], "error": None, "step_output": None}
            verdict: dict = {}
            code: str   = None
            step_done = False

            for _iteration in range(MAX_ITERATIONS):
                code, is_plot = _code_generator(step.description, previous_code, feedback, step_csv_files, user_prompt=user_prompt)

                if code is None:
                    feedback = "Code generation failed — no code block found"
                    continue

                result = _execute_and_capture(code, is_plot)

                # Collect warnings from this execution
                for w in result.get("warnings", []):
                    warnings.append(f"[{step.step_id}] {w}")

                verdict = _reviewer(step.description, code, result, user_prompt=user_prompt)

                if verdict.get("verdict") == "done":
                    step_done = True
                    break
                else:
                    previous_code = code
                    feedback = verdict.get("feedback", "retry")

            # 5d: After the loop
            if step_done:
                # Serialize artifact
                try:
                    artifact_path = _serialize_artifact(step.step_id, result["step_output"])
                    artifact_store[step.step_id] = artifact_path
                except SerializationError as e:
                    warnings.append(f"[{step.step_id}] Serialization failed: {e}")
                    step_results.append(StepResult(
                        step_id=step.step_id,
                        description=step.description,
                        code=code,
                        execution_result=result,
                        answer=None,
                        status="failed",
                    ))
                    break

                step_results.append(StepResult(
                    step_id=step.step_id,
                    description=step.description,
                    code=code,
                    execution_result=result,
                    answer=verdict.get("answer"),
                    status="done",
                ))
            else:
                # Exhausted iterations
                step_results.append(StepResult(
                    step_id=step.step_id,
                    description=step.description,
                    code=code,
                    execution_result=result,
                    answer=None,
                    status="failed",
                ))
                break

    finally:
        _cleanup_artifacts(artifact_store)

    # Step 6: Format and return final response
    return format_final_response_multi(step_results, warnings)

def format_final_response(answer: str, code: str, result: dict) -> str:
    """
    Format the final response combining the textual answer, MATLAB code, execution
    output, and any base64-encoded plots.

    Always returns a non-empty string.
    When result contains plots (base64 PNG strings), embeds them as:
        ![plot](data:image/png;base64,<data>)
    """
    parts = []

    if answer:
        parts.append(answer)

    if code:
        parts.append(f"\n\n**MATLAB Code:**\n```matlab\n{code}\n```")

    if result:
        if result.get("output"):
            parts.append(f"\n\n**Execution Output:**\n```\n{result['output']}\n```")

        if result.get("plots"):
            parts.append("\n\n**Generated Plot(s):**")
            for idx, plot_data in enumerate(result["plots"], 1):
                parts.append(f"\n![plot](data:image/png;base64,{plot_data})")

        if result.get("error"):
            parts.append(f"\n\n**Error:**\n```\n{result['error']}\n```")

    response = "".join(parts)
    return response if response.strip() else "No output was produced."


def format_final_response_multi(step_results: list[StepResult], warnings: list[str]) -> str:
    """
    Format the final response for a multi-step pipeline.

    For single-step pipelines, delegates to format_final_response for backward compatibility.
    For multi-step pipelines, builds a response with each step's details labeled by step_id,
    the terminal step's answer at the top, and any warnings at the bottom.
    """
    if not step_results:
        return "No output was produced."

    done_results = [r for r in step_results if r.status == "done"]

    # Single-step: delegate to existing formatter
    if len(step_results) == 1:
        sr = step_results[0]
        return format_final_response(sr.answer, sr.code, sr.execution_result)

    # Find terminal step answer (last done result that is terminal, or just last done)
    terminal_answer = None
    for sr in reversed(done_results):
        terminal_answer = sr.answer
        break

    parts = []

    # Terminal answer at the top
    if terminal_answer:
        parts.append(terminal_answer)

    # Per-step sections
    for sr in step_results:
        parts.append(f"\n\n### Step {sr.step_id}")
        parts.append(f"\n**Description:** {sr.description}")

        if sr.status == "failed":
            parts.append("\n\n**Status:** ❌ Failed")

        if sr.code:
            parts.append(f"\n\n**MATLAB Code:**\n```matlab\n{sr.code}\n```")

        if sr.execution_result:
            if sr.execution_result.get("output"):
                parts.append(f"\n\n**Execution Output:**\n```\n{sr.execution_result['output']}\n```")

            if sr.execution_result.get("plots"):
                parts.append("\n\n**Generated Plot(s):**")
                for plot_data in sr.execution_result["plots"]:
                    parts.append(f"\n![plot](data:image/png;base64,{plot_data})")

            if sr.execution_result.get("error"):
                parts.append(f"\n\n**Error:**\n```\n{sr.execution_result['error']}\n```")

    # Failed step note
    failed = [r for r in step_results if r.status == "failed"]
    if failed:
        failed_ids = ", ".join(r.step_id for r in failed)
        parts.append(f"\n\n**Note:** The following step(s) failed: {failed_ids}")

    # Warnings
    if warnings:
        parts.append("\n\n**Warnings:**")
        for w in warnings:
            parts.append(f"\n- {w}")

    response = "".join(parts)
    return response if response.strip() else "No output was produced."


if __name__ == "__main__":
    # Test 1: Plotting query
    print("\n" + "="*80)
    print("TEST 1: PLOTTING TASK")
    print("="*80)
    test_query_plot = """
    Plot the step response of the transfer function H(s) = 5 / (s^2 + 3s + 2)
    """
    result1 = run_matlab_executor_agent(test_query_plot)
    print("\nFINAL RESULT:")
    print(result1)
    
    print("\n\n" + "="*80)
    print("TEST 2: CALCULATION TASK")
    print("="*80)
    # Test 2: Calculation query
    test_query_calc = """
    Create a 3x3 matrix with values [[1,2,3],[4,5,6],[7,8,9]] and calculate its determinant and eigenvalues. Display the results.
    """
    result2 = run_matlab_executor_agent(test_query_calc)
    print("\nFINAL RESULT:")
    print(result2)

