"""Frozen experimental prompt contracts for E2 and complete-contract schedules."""
from __future__ import annotations
import copy
from copy import deepcopy
import json
from typing import Any, cast
from pathlib import Path
from carrier_parser import STATE_FIELDS

PROMPT_TEMPLATE = Path(__file__).with_name("schedule-prompt.txt")


def visible_suffix_rows(task):
    # All displayed steps are post-activation; the active-event limits therefore
    # apply to every supplied interval. The revised PV suffix is already frozen.
    current = e2_states(task)["C"]
    out = []
    for offset,row in enumerate(task["model_visible_episode"]["stage_2"]["remaining_timeseries"]):
        out.append({
            "offset":offset,"timestamp_utc":row["timestamp_utc"],
            "load_kw":float(row["load_kw"]),"pv_forecast_kw":float(row["pv_forecast_kw"]),
            "price_usd_mwh":float(row["dam_price_usd_mwh"]),
            **{key:float(current[key]) for key in ("usable_capacity_kwh","charge_limit_kw","discharge_limit_kw","reserve_kwh","export_limit_kw")}
        })
    return out


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


PHYSICAL_FIELDS = {
    "soc_kwh",
    "usable_capacity_kwh",
    "charge_limit_kw",
    "discharge_limit_kw",
    "export_limit_kw",
    "reserve_kwh",
    "forecast_version",
    "forecast_series",
}


METADATA_FIELDS = {"active_commitments", "revoked_commitments", "sequence"}


def mutate_carrier(
    canonical: dict[str, Any],
    mutation: str,
    *,
    stale_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = deepcopy(canonical)
    stale_values = stale_values or {}
    if mutation == "C" or mutation == "C2":
        return result
    if mutation == "M":
        return deepcopy(stale_values.get("model_carrier", canonical))
    if mutation == "S":
        field = stale_values.get("physical_field")
        if not isinstance(field, str) or field not in PHYSICAL_FIELDS:
            raise ValueError("S requires one registered physical_field")
        if field not in stale_values:
            raise ValueError(f"S stale value missing for {field}")
        result[field] = deepcopy(stale_values[field])
        return result
    if mutation == "K":
        for field in METADATA_FIELDS:
            if field in stale_values:
                result[field] = deepcopy(stale_values[field])
        return result
    raise ValueError(f"unknown carrier mutation {mutation}")


def assert_mutation_scope(
    canonical: dict[str, Any], mutated: dict[str, Any], mutation: str
) -> None:
    changed = {key for key in canonical if canonical.get(key) != mutated.get(key)}
    if mutation in {"C", "C2"} and changed:
        raise AssertionError(f"canonical mutation changed fields: {sorted(changed)}")
    if mutation == "K" and not changed.issubset(METADATA_FIELDS):
        raise AssertionError(f"metadata mutation changed physical fields: {sorted(changed)}")
    if mutation == "S" and not changed.issubset(PHYSICAL_FIELDS):
        raise AssertionError(f"physical mutation changed unexpected fields: {sorted(changed)}")


INTERFACES = ("I0", "I1", "I2", "I3")


def canonical_initial_state(task: dict[str, Any]) -> dict[str, Any]:
    visible = task["model_visible_episode"]["stage_1"]["visible_initial_state"]
    return {
        "current_step": 0,
        "soc_kwh": float(visible["soc_kwh"]),
        "usable_capacity_kwh": float(visible["usable_capacity_kwh"]),
        "charge_limit_kw": float(visible["max_charge_kw"]),
        "discharge_limit_kw": float(visible["max_discharge_kw"]),
        "export_limit_kw": float(visible["export_limit_kw"]),
        "reserve_kwh": float(visible["reserve_soc_kwh"]),
        "forecast_version": str(visible["active_forecast_version"]),
        "active_commitments": copy.deepcopy(visible.get("active_commitments", [])),
        "revoked_commitments": copy.deepcopy(
            visible.get("revoked_or_superseded_items", [])
        ),
        "sequence": 0,
    }


def canonical_event_state(
    task: dict[str, Any], event_soc_kwh: float
) -> dict[str, Any]:
    initial = canonical_initial_state(task)
    update = task["visible_update"]
    field_map = {
        "usable_capacity_kwh": "usable_capacity_kwh",
        "max_charge_kw": "charge_limit_kw",
        "max_discharge_kw": "discharge_limit_kw",
        "export_limit_kw": "export_limit_kw",
        "reserve_soc_kwh": "reserve_kwh",
    }
    state = copy.deepcopy(initial)
    state["current_step"] = int(update["activation_step"])
    state["soc_kwh"] = round(float(event_soc_kwh), 6)
    for source, target in field_map.items():
        if source in update:
            state[target] = float(update[source])
    if task["event_family"] == "forecast_revision":
        state["forecast_version"] = "forecast_revision_v2"
    changed = [
        {"field": source, "value": copy.deepcopy(update[source])}
        for source in field_map
        if source in update
    ]
    state["active_commitments"] = changed
    state["revoked_commitments"] = [
        {"field": item["field"], "previous_value": initial[field_map[item["field"]]]}
        for item in changed
    ]
    state["sequence"] = 1
    return state


def compile_prompt(
    task: dict[str, Any],
    interface: str,
    stage: str,
    canonical_state: dict[str, Any],
) -> str:
    if interface not in INTERFACES:
        raise ValueError(f"unknown P0 interface: {interface}")
    if stage not in {"stage1", "stage2"}:
        raise ValueError(f"unknown P0 stage: {stage}")
    if stage == "stage1":
        timeseries = task["model_visible_episode"]["stage_1"]["visible_timeseries"]
        visible_update: dict[str, Any] | None = None
    else:
        timeseries = task["model_visible_episode"]["stage_2"]["remaining_timeseries"]
        visible_update = task["visible_update"]
    horizon = len(timeseries)
    sparse = interface in {"I2", "I3"}
    action_contract: dict[str, Any]
    if sparse:
        action_contract = {
            "field": "actions",
            "type": "array",
            "maximum_segments": 24,
            "segment_fields": ["start_step", "end_step_exclusive", "power_kw"],
            "index_domain": [0, horizon],
            "end_step_exclusive": True,
            "overlap": "forbidden",
            "uncovered_steps": "deterministically_zero",
        }
    else:
        action_contract = {
            "field": "actions_kw",
            "type": "array_of_numbers",
            "required_length": horizon,
        }
    contract = {
        "task": "PV-BESS scheduling",
        "stage": stage,
        "scenario_id": task["scenario_id"],
        "resolution_minutes": 15,
        "action_convention": "positive_charge_negative_discharge_kw",
        "terminal_soc_target_kwh": 325.0,
        "current_state": canonical_state,
        "visible_update": visible_update,
        "visible_timeseries": timeseries,
        "output_contract": {
            "top_level_fields_exactly": ["state", action_contract["field"]],
            "state_fields_exactly": list(STATE_FIELDS),
            "action": action_contract,
            "prose_or_markdown": "forbidden",
        },
        "evaluation_boundary": {
            "raw_action_scored": True,
            "projection_not_disclosed": True,
            "oracle_not_available": True,
        },
    }
    header = (
        "Return exactly one compact JSON object. Respect SOC, capacity, charge and "
        "discharge power, reserve, export, and terminal-SOC constraints."
    )
    if interface == "I0":
        header += " Use direct text JSON."
    else:
        header += " The provider JSON-output channel is active."
    if interface == "I3":
        header += (
            " Independent format example only: a 10 kW charge over local steps 0 through 3 "
            "is {\"start_step\":0,\"end_step_exclusive\":4,\"power_kw\":10}. "
            "Do not copy it as the task answer."
        )
    return f"{header}\n{canonical_json(contract)}"


def build_payload(
    *,
    model_id: str,
    provider: str,
    interface: str,
    prompt: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_id,
        "stream": False,
        "temperature": 0.0,
        "max_tokens": max_output_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if interface != "I0":
        payload["response_format"] = {"type": "json_object"}
    if provider == "deepseek":
        payload["thinking"] = {"type": "disabled"}
    elif provider == "qwen":
        payload["enable_thinking"] = False
    else:
        raise ValueError(f"unsupported provider: {provider}")
    return payload


def e2_states(task: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return frozen deterministic carriers; M is supplied from Stage 1."""
    intervention = task["v2_prefix_intervention"]
    canonical = canonical_event_state(task, float(intervention["canonical_event_soc_kwh"]))
    initial = canonical_initial_state(task)
    stale_soc = mutate_carrier(
        canonical,
        "S",
        stale_values={"physical_field": "soc_kwh", "soc_kwh": intervention["stale_event_soc_kwh"]},
    )
    stale_metadata = mutate_carrier(
        canonical,
        "K",
        stale_values={
            "active_commitments": copy.deepcopy(initial["active_commitments"]),
            "revoked_commitments": copy.deepcopy(initial["revoked_commitments"]),
            "sequence": initial["sequence"],
        },
    )
    canonical2 = mutate_carrier(canonical, "C2")
    assert_mutation_scope(canonical, canonical2, "C2")
    assert_mutation_scope(canonical, stale_soc, "S")
    assert_mutation_scope(canonical, stale_metadata, "K")
    return {
        "initial": initial,
        "C": canonical,
        "S": stale_soc,
        "K": stale_metadata,
        "C2": canonical2,
    }


def model_carrier(stage1_state: dict[str, Any] | None) -> dict[str, Any] | None:
    """Preserve the parsed model carrier exactly; do not repair missing fields."""
    if stage1_state is None:
        return None
    return copy.deepcopy(stage1_state)


TERMINAL_TARGET_KWH = 325.0


def carrier_for(task: dict[str, Any], branch: str) -> dict[str, Any]:
    if branch not in {"C1", "C2", "S1", "S2"}:
        raise ValueError(f"unknown E2b-v2 branch: {branch}")
    return cast(dict[str, Any], e2_states(task)["C" if branch.startswith("C") else "S"])


def model_visible_contract(task: dict[str, Any], branch: str) -> dict[str, Any]:
    carrier = carrier_for(task, branch)
    rows = visible_suffix_rows(task)
    return {
        "protocol_id": "energybench-e2b-stage2-only-carrier-sensitivity-v2",
        "scenario_id": str(task["scenario_id"]),
        "carrier_state": {
            "soc_kwh": float(carrier["soc_kwh"]),
            "usable_capacity_kwh": float(carrier["usable_capacity_kwh"]),
            "charge_limit_kw": float(carrier["charge_limit_kw"]),
            "discharge_limit_kw": float(carrier["discharge_limit_kw"]),
            "reserve_kwh": float(carrier["reserve_kwh"]),
            "export_limit_kw": float(carrier["export_limit_kw"]),
        },
        "suffix_horizon_intervals": len(rows),
        "terminal_target_kwh": TERMINAL_TARGET_KWH,
        "timeseries": rows,
    }


def render_prompt(task: dict[str, Any], branch: str) -> str:
    template = PROMPT_TEMPLATE.read_text(encoding="utf-8")
    return template.replace("{{TASK_CONTRACT_JSON}}", canonical_json(model_visible_contract(task, branch)))


def request_payload(model_spec: dict[str, Any], prompt: str) -> dict[str, Any]:
    provider = str(model_spec["provider"])
    payload: dict[str, Any] = {
        "model": str(model_spec["configured_model_id"]),
        "stream": False,
        "temperature": float(model_spec["temperature"]),
        "max_tokens": int(model_spec["max_output_tokens"]),
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    if provider == "deepseek":
        payload["thinking"] = {"type": "disabled"}
    elif provider == "qwen":
        payload["enable_thinking"] = False
    else:
        raise ValueError(f"unsupported frozen provider: {provider}")
    return payload
