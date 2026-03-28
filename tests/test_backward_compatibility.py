# Tests for Requirement 8: Backward Compatibility

import unittest
from unittest.mock import patch, MagicMock

from chatbot.agents.matlab_executor_agent import (
    Pipeline,
    Step,
    run_matlab_executor_agent,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SINGLE_STEP_PIPELINE = Pipeline(
    steps=[
        Step(
            step_id="step_1",
            description="compute x",
            input_sources=[],
            is_terminal=True,
        )
    ]
)

_CODE = "x = 1;"
_EXEC_RESULT = {"output": "ans = 1", "plots": [], "error": None, "step_output": 1.0, "warnings": []}
_VERDICT = {"verdict": "done", "answer": "The result is 1."}
_ARTIFACT_PATH = "/tmp/step_1.csv"

_PATCHES = {
    "chatbot.agents.matlab_executor_agent._pipeline_planner": _SINGLE_STEP_PIPELINE,
    "chatbot.agents.matlab_executor_agent._code_generator": (_CODE, False),
    "chatbot.agents.matlab_executor_agent._execute_and_capture": _EXEC_RESULT,
    "chatbot.agents.matlab_executor_agent._reviewer": _VERDICT,
    "chatbot.agents.matlab_executor_agent._serialize_artifact": _ARTIFACT_PATH,
}


def _apply_mocks(test_fn):
    """Decorator that applies all standard mocks to a test method."""
    for target, return_value in reversed(list(_PATCHES.items())):
        test_fn = patch(target, return_value=return_value)(test_fn)
    # _cleanup_artifacts is a no-op
    test_fn = patch("chatbot.agents.matlab_executor_agent._cleanup_artifacts")(test_fn)
    return test_fn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBackwardCompatibility(unittest.TestCase):

    @_apply_mocks
    def test_accepts_user_prompt_only(self, *mocks):
        """Req 8.1 — accepts user_prompt with no csv_files."""
        result = run_matlab_executor_agent("compute the mean")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result.strip()) > 0)

    @_apply_mocks
    def test_accepts_csv_files_kwarg(self, *mocks):
        """Req 8.1 — accepts csv_files as keyword argument."""
        result = run_matlab_executor_agent(
            "compute the mean",
            csv_files=[{"path": "data.csv", "preview": "a,b\n1,2"}],
        )
        self.assertIsInstance(result, str)
        self.assertTrue(len(result.strip()) > 0)

    @_apply_mocks
    def test_accepts_csv_files_positional(self, *mocks):
        """Req 8.1 — accepts csv_files as positional argument."""
        result = run_matlab_executor_agent(
            "compute the mean",
            [{"path": "data.csv", "preview": "a,b\n1,2"}],
        )
        self.assertIsInstance(result, str)
        self.assertTrue(len(result.strip()) > 0)

    @_apply_mocks
    def test_single_step_returns_string(self, *mocks):
        """Req 8.3 — return type is str."""
        result = run_matlab_executor_agent("compute the mean")
        self.assertIsInstance(result, str)

    @_apply_mocks
    def test_single_step_response_non_empty(self, *mocks):
        """Req 8.3 — response is non-empty after stripping whitespace."""
        result = run_matlab_executor_agent("compute the mean")
        self.assertGreater(len(result.strip()), 0)

    @_apply_mocks
    def test_single_step_response_contains_answer(self, *mocks):
        """Req 8.2 — reviewer answer text appears in the response."""
        result = run_matlab_executor_agent("compute the mean")
        self.assertIn("The result is 1.", result)

    @_apply_mocks
    def test_single_step_response_contains_code(self, *mocks):
        """Req 8.2 — generated code appears in the response."""
        result = run_matlab_executor_agent("compute the mean")
        self.assertIn("x = 1;", result)

    @_apply_mocks
    def test_single_step_response_contains_output(self, *mocks):
        """Req 8.2 — execution output appears in the response."""
        result = run_matlab_executor_agent("compute the mean")
        self.assertIn("ans = 1", result)

    def test_single_step_uses_format_final_response(self):
        """Req 8.2 — single-step pipeline delegates to format_final_response exactly once."""
        with patch("chatbot.agents.matlab_executor_agent._pipeline_planner",
                   return_value=_SINGLE_STEP_PIPELINE), \
             patch("chatbot.agents.matlab_executor_agent._code_generator",
                   return_value=(_CODE, False)), \
             patch("chatbot.agents.matlab_executor_agent._execute_and_capture",
                   return_value=_EXEC_RESULT), \
             patch("chatbot.agents.matlab_executor_agent._reviewer",
                   return_value=_VERDICT), \
             patch("chatbot.agents.matlab_executor_agent._serialize_artifact",
                   return_value=_ARTIFACT_PATH), \
             patch("chatbot.agents.matlab_executor_agent._cleanup_artifacts"), \
             patch("chatbot.agents.matlab_executor_agent.format_final_response",
                   wraps=__import__(
                       "chatbot.agents.matlab_executor_agent",
                       fromlist=["format_final_response"],
                   ).format_final_response) as mock_fmt:
            run_matlab_executor_agent("compute the mean")
            mock_fmt.assert_called_once()

    @_apply_mocks
    def test_csv_files_none_default(self, *mocks):
        """Req 8.1 — explicit csv_files=None is accepted and returns non-empty string."""
        result = run_matlab_executor_agent("compute the mean", csv_files=None)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result.strip()) > 0)


if __name__ == "__main__":
    unittest.main()
