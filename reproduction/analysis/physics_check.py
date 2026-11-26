from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np
import math

DT_HOURS = 0.25

TARGET_SOC_KWH = 325.0

POWER_TOL_KW = 1e-4

ENERGY_TOL_KWH = 1e-5

EXPORT_TOL_KW = 1e-4

ENGINEERING_TERMINAL_TOL_KWH = 1.0

@dataclass(frozen=True)
class Battery:
    capacity_kwh: float
    max_charge_kw: float
    max_discharge_kw: float
    charge_efficiency: float
    discharge_efficiency: float
    reserve_kwh: float
    export_limit_kw: float
    degradation_cost_per_kwh: float

def task_arrays(task: dict[str, Any], battery: Battery) -> dict[str, np.ndarray]:
    rows = task["model_visible_episode"]["stage_2"]["remaining_timeseries"]
    update = task["visible_update"]
    length = len(rows)
    return {
        "load_kw": np.asarray([float(row["load_kw"]) for row in rows], dtype=float),
        "pv_kw": np.asarray([float(row["pv_forecast_kw"]) for row in rows], dtype=float),
        "price_usd_mwh": np.asarray([float(row["dam_price_usd_mwh"]) for row in rows], dtype=float),
        "capacity_kwh": np.full(length, float(update.get("usable_capacity_kwh", battery.capacity_kwh))),
        "max_charge_kw": np.full(length, float(update.get("max_charge_kw", battery.max_charge_kw))),
        "max_discharge_kw": np.full(length, float(update.get("max_discharge_kw", battery.max_discharge_kw))),
        "reserve_kwh": np.full(length, float(update.get("reserve_soc_kwh", battery.reserve_kwh))),
        "export_limit_kw": np.full(length, float(update.get("export_limit_kw", battery.export_limit_kw))),
    }

def evaluate(
    task: dict[str, Any],
    action_kw: Sequence[float],
    initial_soc_kwh: float,
    battery: Battery,
    *,
    curtailment_kw: Sequence[float] | None = None,
    terminal_tolerance_kwh: float = ENGINEERING_TERMINAL_TOL_KWH,
) -> dict[str, Any]:
    arrays = task_arrays(task, battery)
    action = np.asarray(action_kw, dtype=float)
    length = len(arrays["load_kw"])
    curtailment = np.zeros(length, dtype=float) if curtailment_kw is None else np.asarray(curtailment_kw, dtype=float)
    if len(action) != length or len(curtailment) != length:
        raise ValueError("action or curtailment horizon mismatch")
    if not np.isfinite(action).all() or not np.isfinite(curtailment).all() or not math.isfinite(initial_soc_kwh):
        raise ValueError("non-finite checker input")

    soc = float(initial_soc_kwh)
    trace: list[dict[str, float]] = []
    cost = 0.0
    total_violation_energy = 0.0
    dynamic_bad_count = 0
    first_violation: int | None = None
    category_counts = {
        "charge_power": 0,
        "discharge_power": 0,
        "soc_lower": 0,
        "soc_upper": 0,
        "export": 0,
        "curtailment_lower": 0,
        "curtailment_upper": 0,
    }
    for step in range(length):
        requested = float(action[step])
        charge = max(requested, 0.0)
        discharge = max(-requested, 0.0)
        curtailed = float(curtailment[step])
        next_soc = soc + battery.charge_efficiency * charge * DT_HOURS - discharge * DT_HOURS / battery.discharge_efficiency
        charge_excess = max(charge - arrays["max_charge_kw"][step], 0.0)
        discharge_excess = max(discharge - arrays["max_discharge_kw"][step], 0.0)
        soc_lower = max(arrays["reserve_kwh"][step] - next_soc, 0.0)
        soc_upper = max(next_soc - arrays["capacity_kwh"][step], 0.0)
        curtailment_lower = max(-curtailed, 0.0)
        curtailment_upper = max(curtailed - arrays["pv_kw"][step], 0.0)
        grid = arrays["load_kw"][step] - arrays["pv_kw"][step] + curtailed + requested
        export_excess = max(-arrays["export_limit_kw"][step] - grid, 0.0)
        flags = {
            "charge_power": charge_excess > POWER_TOL_KW,
            "discharge_power": discharge_excess > POWER_TOL_KW,
            "soc_lower": soc_lower > ENERGY_TOL_KWH,
            "soc_upper": soc_upper > ENERGY_TOL_KWH,
            "export": export_excess > EXPORT_TOL_KW,
            "curtailment_lower": curtailment_lower > POWER_TOL_KW,
            "curtailment_upper": curtailment_upper > POWER_TOL_KW,
        }
        for category, active in flags.items():
            category_counts[category] += int(active)
        dynamic_bad = any(flags.values())
        if dynamic_bad and first_violation is None:
            first_violation = step
        dynamic_bad_count += int(dynamic_bad)
        total_violation_energy += (
            DT_HOURS * (charge_excess + discharge_excess + export_excess + curtailment_lower + curtailment_upper)
            + soc_lower
            + soc_upper
        )
        cost += DT_HOURS / 1000.0 * arrays["price_usd_mwh"][step] * grid
        cost += battery.degradation_cost_per_kwh * DT_HOURS * (charge + discharge)
        trace.append(
            {
                "offset": step,
                "requested_power_kw": requested,
                "requested_charge_kw": charge,
                "requested_discharge_kw": discharge,
                "curtailment_kw": curtailed,
                "soc_start_kwh": soc,
                "soc_end_kwh": next_soc,
                "charge_power_exceedance_kw": charge_excess,
                "discharge_power_exceedance_kw": discharge_excess,
                "soc_lower_violation_kwh": soc_lower,
                "soc_upper_violation_kwh": soc_upper,
                "grid_kw": grid,
                "export_violation_kw": export_excess,
                "curtailment_lower_violation_kw": curtailment_lower,
                "curtailment_upper_violation_kw": curtailment_upper,
            }
        )
        soc = next_soc
    terminal_signed = soc - TARGET_SOC_KWH
    terminal_abs = abs(terminal_signed)
    dynamic_ok = dynamic_bad_count == 0
    return {
        "initial_soc_kwh": float(initial_soc_kwh),
        "terminal_soc_kwh": float(soc),
        "terminal_signed_error_kwh": float(terminal_signed),
        "terminal_absolute_error_kwh": float(terminal_abs),
        "dynamic_constraints_ok": dynamic_ok,
        "engineering_feasible": bool(dynamic_ok and terminal_abs <= terminal_tolerance_kwh),
        "strict_feasible": bool(dynamic_ok and terminal_abs <= 1e-6),
        "violation_interval_count": int(dynamic_bad_count),
        "first_violation_timestep": first_violation,
        "category_counts": category_counts,
        "simultaneous_violation_categories": int(sum(value > 0 for value in category_counts.values())),
        "total_violation_energy": float(total_violation_energy),
        "action_energy_kwh": float(DT_HOURS * np.abs(action).sum()),
        "throughput_kwh": float(DT_HOURS * np.abs(action).sum()),
        "curtailment_kwh": float(DT_HOURS * curtailment.sum()),
        "physical_cost_usd": float(cost),
        "trace": trace,
    }

def synthetic_task(
    *,
    length: int,
    load_kw: float = 100.0,
    pv_kw: float = 0.0,
    capacity_kwh: float = 500.0,
    reserve_kwh: float = 75.0,
    max_charge_kw: float = 250.0,
    max_discharge_kw: float = 250.0,
    export_limit_kw: float = 150.0,
) -> dict[str, Any]:
    rows = [
        {"load_kw": load_kw, "pv_forecast_kw": pv_kw, "dam_price_usd_mwh": 50.0}
        for _ in range(length)
    ]
    return {
        "model_visible_episode": {"stage_2": {"remaining_timeseries": rows}},
        "visible_update": {
            "usable_capacity_kwh": capacity_kwh,
            "reserve_soc_kwh": reserve_kwh,
            "max_charge_kw": max_charge_kw,
            "max_discharge_kw": max_discharge_kw,
            "export_limit_kw": export_limit_kw,
        },
    }

def fixture_results(battery: Battery) -> list[dict[str, Any]]:
    cases: list[tuple[str, dict[str, Any], list[float], float, bool]] = [
        ("normal_feasible", synthetic_task(length=4), [0.0, 0.0, 0.0, 0.0], TARGET_SOC_KWH, True),
        ("soc_upper", synthetic_task(length=2, capacity_kwh=326.0), [10.0, 10.0], 325.0, False),
        ("soc_lower", synthetic_task(length=2, reserve_kwh=324.0), [-10.0, -10.0], 325.0, False),
        ("charge_power", synthetic_task(length=2, max_charge_kw=5.0), [6.0, 0.0], 323.575, False),
        ("discharge_power", synthetic_task(length=2, max_discharge_kw=5.0), [-6.0, 0.0], 326.57894736842104, False),
        ("export", synthetic_task(length=2, load_kw=0.0, pv_kw=200.0, export_limit_kw=150.0), [0.0, 0.0], 325.0, False),
        ("terminal", synthetic_task(length=2), [0.0, 0.0], 300.0, False),
        ("capacity_derating_edge", synthetic_task(length=2, capacity_kwh=300.0), [0.0, 0.0], 325.0, False),
        ("reserve_edge", synthetic_task(length=2, reserve_kwh=326.0), [0.0, 0.0], 325.0, False),
        ("exact_boundary", synthetic_task(length=2, capacity_kwh=325.0, reserve_kwh=325.0), [0.0, 0.0], 325.0, True),
        ("boundary_plus_tolerance", synthetic_task(length=2, capacity_kwh=325.0 + ENERGY_TOL_KWH / 2), [0.0, 0.0], 325.0, True),
        ("boundary_beyond_tolerance", synthetic_task(length=2, capacity_kwh=325.0 - 2 * ENERGY_TOL_KWH), [0.0, 0.0], 325.0, False),
    ]
    output = []
    for name, task, action, initial, expected in cases:
        result = evaluate(task, action, initial, battery)
        output.append(
            {
                "fixture": name,
                "expected_engineering_feasible": expected,
                "observed_engineering_feasible": result["engineering_feasible"],
                "pass": result["engineering_feasible"] is expected,
                "terminal_soc_kwh": result["terminal_soc_kwh"],
                "violation_interval_count": result["violation_interval_count"],
            }
        )
    return output
