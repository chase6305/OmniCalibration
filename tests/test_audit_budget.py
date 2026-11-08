import pytest

from scripts.audit_results import audit_continuations


def test_nested_over_budget_continuation_cannot_pass_audit():
    trace = {"used": True, "converged": False, "continuation_history": [{}, {}]}
    record = {"primary_initialization": {"local_continuation": trace}}
    with pytest.raises(AssertionError, match="exceeded budget"):
        audit_continuations(record, 1)
    assert audit_continuations(record, 2) == 1
    error = {
        "local_continuation": {"used": True, "converged": False, "error": "association failure"}
    }
    assert audit_continuations(error, 1) == 1
