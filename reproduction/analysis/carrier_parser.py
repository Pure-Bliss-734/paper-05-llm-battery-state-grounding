from __future__ import annotations
from dataclasses import dataclass, asdict
import json
from typing import Any
from actions import ActionContractError, expand_sparse_segments, action_energy_kwh

STATE_FIELDS = (
    "current_step",
    "soc_kwh",
    "usable_capacity_kwh",
    "charge_limit_kw",
    "discharge_limit_kw",
    "export_limit_kw",
    "reserve_kwh",
    "forecast_version",
    "active_commitments",
    "revoked_commitments",
    "sequence",
)

@dataclass(frozen=True)
class ParsedP0Output:
    ok: bool
    state: dict[str, Any] | None
    raw_action: Any
    dense_action_kw: list[float]
    diagnostics: list[str]

def parse_output(raw_text: str, interface: str, horizon: int) -> ParsedP0Output:
    diagnostics: list[str] = []
    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError:
        return ParsedP0Output(False, None, None, [], ["invalid_json"])
    if not isinstance(value, dict):
        return ParsedP0Output(False, None, None, [], ["top_level_not_object"])
    action_field = "actions" if interface in {"I2", "I3"} else "actions_kw"
    if set(value) != {"state", action_field}:
        diagnostics.append("top_level_fields_invalid")
    state = value.get("state")
    if not isinstance(state, dict) or set(state) != set(STATE_FIELDS):
        diagnostics.append("state_fields_invalid")
        parsed_state = None
    else:
        parsed_state = state
        diagnostics.extend(_state_diagnostics(state))
    raw_action = value.get(action_field)
    dense: list[float] = []
    try:
        if interface in {"I2", "I3"}:
            if not isinstance(raw_action, list):
                raise ActionContractError("segments_not_array")
            dense = expand_sparse_segments(raw_action, horizon=horizon)
        else:
            dense = _validate_dense_dynamic(raw_action, horizon)
    except ActionContractError as exc:
        diagnostics.append(str(exc))
    return ParsedP0Output(not diagnostics, parsed_state, raw_action, dense, diagnostics)

def parsed_payload(parsed: ParsedP0Output) -> dict[str, Any]:
    return {**asdict(parsed), "action_energy_kwh": action_energy_kwh(parsed.dense_action_kw)}

def _validate_dense_dynamic(value: Any, horizon: int) -> list[float]:
    if not isinstance(value, list):
        raise ActionContractError("dense_not_array")
    if len(value) != horizon:
        raise ActionContractError(f"dense_length_{len(value)}_expected_{horizon}")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ActionContractError(f"dense_{index}_not_number")
        number = float(item)
        if not (-250.0 <= number <= 250.0):
            raise ActionContractError(f"dense_{index}_out_of_range")
        result.append(number)
    return result

def _state_diagnostics(state: dict[str, Any]) -> list[str]:
    diagnostics: list[str] = []
    numeric = (
        "soc_kwh",
        "usable_capacity_kwh",
        "charge_limit_kw",
        "discharge_limit_kw",
        "export_limit_kw",
        "reserve_kwh",
    )
    if isinstance(state["current_step"], bool) or not isinstance(state["current_step"], int):
        diagnostics.append("state_current_step_invalid")
    if isinstance(state["sequence"], bool) or not isinstance(state["sequence"], int):
        diagnostics.append("state_sequence_invalid")
    for field in numeric:
        if isinstance(state[field], bool) or not isinstance(state[field], (int, float)):
            diagnostics.append(f"state_{field}_invalid")
    if not isinstance(state["forecast_version"], str) or not state["forecast_version"]:
        diagnostics.append("state_forecast_version_invalid")
    for field in ("active_commitments", "revoked_commitments"):
        if not isinstance(state[field], list) or any(not isinstance(item, dict) for item in state[field]):
            diagnostics.append(f"state_{field}_invalid")
    return diagnostics
