from __future__ import annotations
import hashlib, json, math, time
from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

DT = 0.25

TARGET_KWH = 325.0

ENGINEERING_TERMINAL_TOL_KWH = 1.0

POWER_TOL_KW = 1e-4

ENERGY_TOL_KWH = 1e-5

EXPORT_TOL_KW = 1e-4

BOOTSTRAP_SEED = 20260716

RANDOM_BASELINE_SEED = 20260822

BOOTSTRAP_RESAMPLES = 10000

LEX_DISTANCE_TOL_KWH = 1e-4

LEX_THROUGHPUT_TOL_KWH = 1e-4

MILP_TIME_LIMIT_SECONDS = 10.0

MODE_TIE_BREAK_WEIGHT = 1e-8

@dataclass(frozen=True)
class Battery:
    capacity_kwh: float
    max_charge_kw: float
    max_discharge_kw: float
    eta_c: float
    eta_d: float
    initial_soc_kwh: float
    reserve_kwh: float
    export_limit_kw: float
    degradation_cost_per_kwh: float

@dataclass
class GateResult:
    status: str
    success: bool
    gate_type: str
    terminal_tolerance_kwh: float
    distance_kwh: float | None = None
    throughput_kwh: float | None = None
    physical_cost_usd: float | None = None
    curtailment_kwh: float | None = None
    curtailment_price_term_usd: float | None = None
    exact_retained_fraction: float | None = None
    correlation: float | None = None
    solve_seconds: float | None = None
    stage1_seconds: float | None = None
    stage2_seconds: float | None = None
    stage3_seconds: float | None = None
    action_kw: list[float] | None = None
    curtailment_kw: list[float] | None = None
    soc_kwh: list[float] | None = None
    message: str | None = None
    distance_lower_bound_kwh: float | None = None
    distance_optimality_gap_kwh: float | None = None
    global_distance_optimality_proven: bool | None = None
    lp_overlap_intervals: int | None = None
    lp_overlap_kwh: float | None = None
    selected_mode_source: str | None = None

def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def quantile_ci(values: Sequence[float], *, seed: int = BOOTSTRAP_SEED, resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float]:
    a = np.asarray(values, dtype=float)
    if len(a) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = a[rng.integers(0, len(a), size=(resamples, len(a)))].mean(axis=1)
    return (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))

def task_suffix(task: dict[str, Any], battery: Battery) -> dict[str, np.ndarray]:
    rows = task["model_visible_episode"]["stage_2"]["remaining_timeseries"]
    update = task["visible_update"]
    n = len(rows)
    return {
        "load_kw": np.asarray([float(row["load_kw"]) for row in rows], dtype=float),
        "pv_kw": np.asarray([float(row["pv_forecast_kw"]) for row in rows], dtype=float),
        "price_usd_mwh": np.asarray([float(row["dam_price_usd_mwh"]) for row in rows], dtype=float),
        "capacity_kwh": np.full(n, float(update.get("usable_capacity_kwh", battery.capacity_kwh))),
        "max_charge_kw": np.full(n, float(update.get("max_charge_kw", battery.max_charge_kw))),
        "max_discharge_kw": np.full(n, float(update.get("max_discharge_kw", battery.max_discharge_kw))),
        "reserve_kwh": np.full(n, float(update.get("reserve_soc_kwh", battery.reserve_kwh))),
        "export_limit_kw": np.full(n, float(update.get("export_limit_kw", battery.export_limit_kw))),
    }

def independent_replay(task: dict[str, Any], action_kw: Sequence[float], initial_soc_kwh: float, battery: Battery) -> dict[str, Any]:
    data = task_suffix(task, battery)
    action = np.asarray(action_kw, dtype=float)
    n = len(data["load_kw"])
    if len(action) != n or not np.isfinite(action).all():
        raise ValueError("invalid action horizon")
    soc = float(initial_soc_kwh)
    trace: list[dict[str, float]] = []
    cost = 0.0
    total_violation_energy = 0.0
    violation_intervals = 0
    for t in range(n):
        u = float(action[t])
        charge = max(u, 0.0)
        discharge = max(-u, 0.0)
        next_soc = soc + battery.eta_c * charge * DT - discharge * DT / battery.eta_d
        charge_excess = max(charge - data["max_charge_kw"][t], 0.0)
        discharge_excess = max(discharge - data["max_discharge_kw"][t], 0.0)
        soc_lower = max(data["reserve_kwh"][t] - next_soc, 0.0)
        soc_upper = max(next_soc - data["capacity_kwh"][t], 0.0)
        grid = data["load_kw"][t] - data["pv_kw"][t] + u
        export_excess = max(-data["export_limit_kw"][t] - grid, 0.0)
        dynamic_bad = (
            charge_excess > POWER_TOL_KW
            or discharge_excess > POWER_TOL_KW
            or soc_lower > ENERGY_TOL_KWH
            or soc_upper > ENERGY_TOL_KWH
            or export_excess > EXPORT_TOL_KW
        )
        violation_intervals += int(dynamic_bad)
        total_violation_energy += DT * (charge_excess + discharge_excess + export_excess) + soc_lower + soc_upper
        cost += DT / 1000.0 * data["price_usd_mwh"][t] * grid
        cost += battery.degradation_cost_per_kwh * DT * (charge + discharge)
        trace.append({
            "offset": t,
            "requested_power_kw": u,
            "requested_charge_kw": charge,
            "requested_discharge_kw": discharge,
            "soc_start_kwh": soc,
            "soc_end_kwh": next_soc,
            "charge_power_exceedance_kw": charge_excess,
            "discharge_power_exceedance_kw": discharge_excess,
            "soc_lower_violation_kwh": soc_lower,
            "soc_upper_violation_kwh": soc_upper,
            "grid_kw": grid,
            "export_violation_kw": export_excess,
        })
        soc = next_soc
    terminal_signed = soc - TARGET_KWH
    terminal_abs = abs(terminal_signed)
    dynamic_ok = violation_intervals == 0
    return {
        "initial_soc_kwh": float(initial_soc_kwh),
        "terminal_soc_kwh": float(soc),
        "terminal_signed_error_kwh": float(terminal_signed),
        "terminal_absolute_error_kwh": float(terminal_abs),
        "dynamic_constraints_ok": bool(dynamic_ok),
        "engineering_feasible": bool(dynamic_ok and terminal_abs <= ENGINEERING_TERMINAL_TOL_KWH),
        "strict_feasible": bool(dynamic_ok and terminal_abs <= 1e-6),
        "violation_interval_count": int(violation_intervals),
        "total_violation_energy": float(total_violation_energy),
        "action_energy_kwh": float(DT * np.abs(action).sum()),
        "physical_cost_usd": float(cost),
        "trace": trace,
    }

def action_distance(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if len(a) != len(b):
        raise ValueError("unequal action horizons")
    return float(DT * np.abs(a - b).sum())

def grouping_ed(actions: Sequence[np.ndarray], pair_a: tuple[int, int], pair_b: tuple[int, int]) -> float:
    cross = [action_distance(actions[i], actions[j]) for i in pair_a for j in pair_b]
    within_a = action_distance(actions[pair_a[0]], actions[pair_a[1]])
    within_b = action_distance(actions[pair_b[0]], actions[pair_b[1]])
    return float(2.0 * np.mean(cross) - within_a - within_b)

def _gate_problem_arrays(task: dict[str, Any], battery: Battery, reference: np.ndarray, initial_soc: float, allow_curtailment: bool, terminal_tolerance: float, distance_limit: float | None = None, throughput_limit: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LinearConstraint, dict[str, slice | int], dict[str, np.ndarray]]:
    data = task_suffix(task, battery)
    n = len(reference)
    if n != len(data["load_kw"]):
        raise ValueError("reference length mismatch")
    # c,d,q,e(0..n),a,z
    idx: dict[str, slice | int] = {
        "c": slice(0, n),
        "d": slice(n, 2*n),
        "q": slice(2*n, 3*n),
        "e": slice(3*n, 4*n+1),
        "a": slice(4*n+1, 5*n+1),
        "z": slice(5*n+1, 6*n+1),
    }
    N = 6*n+1
    lower = np.full(N, -np.inf)
    upper = np.full(N, np.inf)
    lower[idx["c"]] = 0.0; upper[idx["c"]] = data["max_charge_kw"]
    lower[idx["d"]] = 0.0; upper[idx["d"]] = data["max_discharge_kw"]
    lower[idx["q"]] = 0.0; upper[idx["q"]] = data["pv_kw"] if allow_curtailment else 0.0
    e0 = 3*n
    lower[e0] = upper[e0] = initial_soc
    for t in range(n):
        lower[e0+t+1] = data["reserve_kwh"][t]
        upper[e0+t+1] = data["capacity_kwh"][t]
    lower[e0+n] = max(lower[e0+n], TARGET_KWH-terminal_tolerance)
    upper[e0+n] = min(upper[e0+n], TARGET_KWH+terminal_tolerance)
    lower[idx["a"]] = 0.0
    lower[idx["z"]] = 0.0; upper[idx["z"]] = 1.0
    integrality = np.zeros(N, dtype=int)
    integrality[idx["z"]] = 1
    extra = int(distance_limit is not None) + int(throughput_limit is not None)
    M = sparse.lil_matrix((6*n+extra, N), dtype=float)
    lb = np.full(6*n+extra, -np.inf)
    ub = np.full(6*n+extra, np.inf)
    row = 0
    for t in range(n):
        c = t; d = n+t; q = 2*n+t; e_t = 3*n+t; e_next = e_t+1; a = 4*n+1+t; z = 5*n+1+t
        M[row, e_next] = 1.0; M[row, e_t] = -1.0; M[row, c] = -battery.eta_c*DT; M[row, d] = DT/battery.eta_d
        lb[row] = ub[row] = 0.0; row += 1
        M[row, c] = 1.0; M[row, d] = -1.0; M[row, q] = 1.0
        lb[row] = -data["export_limit_kw"][t] - data["load_kw"][t] + data["pv_kw"][t]; row += 1
        M[row, c] = 1.0; M[row, z] = -data["max_charge_kw"][t]; ub[row] = 0.0; row += 1
        M[row, d] = 1.0; M[row, z] = data["max_discharge_kw"][t]; ub[row] = data["max_discharge_kw"][t]; row += 1
        M[row, c] = 1.0; M[row, d] = -1.0; M[row, a] = -1.0; ub[row] = reference[t]; row += 1
        M[row, c] = -1.0; M[row, d] = 1.0; M[row, a] = -1.0; ub[row] = -reference[t]; row += 1
    if distance_limit is not None:
        M[row, idx["a"]] = DT; ub[row] = distance_limit; row += 1
    if throughput_limit is not None:
        M[row, idx["c"]] = DT; M[row, idx["d"]] = DT; ub[row] = throughput_limit; row += 1
    return lower, upper, integrality, np.zeros(N), LinearConstraint(M.tocsr(), lb, ub), idx, data

def _solve_milp(c: np.ndarray, lower: np.ndarray, upper: np.ndarray, integrality: np.ndarray, constraint: LinearConstraint, label: str) -> tuple[Any, float]:
    start = time.perf_counter()
    result = milp(
        c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraint,
        options={"time_limit": MILP_TIME_LIMIT_SECONDS, "mip_rel_gap": 1e-9, "presolve": True},
    )
    elapsed = time.perf_counter() - start
    if not result.success or result.x is None:
        raise RuntimeError(f"{label}: status={result.status}; {result.message}")
    return result, elapsed

def solve_gate(task: dict[str, Any], battery: Battery, reference_action: Sequence[float], initial_soc: float, *, allow_curtailment: bool, terminal_tolerance: float, full_lexicographic: bool = True, feasible_action: Sequence[float] | None = None) -> GateResult:
    reference = np.asarray(reference_action, dtype=float)
    gate_type = "expanded_system" if allow_curtailment else "battery_only"
    overall_start = time.perf_counter()
    try:
        distance_upper = None
        if feasible_action is not None:
            distance_upper = action_distance(reference, feasible_action) + LEX_DISTANCE_TOL_KWH
        lower, upper, integ, _, constraint, idx, data = _gate_problem_arrays(task, battery, reference, initial_soc, allow_curtailment, terminal_tolerance, distance_limit=distance_upper)
        n = len(reference); N = len(lower)
        zero_mode_mask = np.zeros(n, dtype=bool)
        anchor_mode = np.zeros(n, dtype=float)
        if feasible_action is not None:
            feasible_array = np.asarray(feasible_action, dtype=float)
            zero_mode_mask = np.abs(reference) <= 1e-9
            anchor_mode = (feasible_array > 0.0).astype(float)
            z_indices = np.arange(idx["z"].start, idx["z"].stop)
            lower[z_indices[zero_mode_mask]] = anchor_mode[zero_mode_mask]
            upper[z_indices[zero_mode_mask]] = anchor_mode[zero_mode_mask]
            integ[z_indices[zero_mode_mask]] = 0
        c1 = np.zeros(N); c1[idx["a"]] = DT
        raw_charge_mode = (reference > 0.0).astype(float)
        c1[idx["z"]] = MODE_TIE_BREAK_WEIGHT * (1.0 - 2.0 * raw_charge_mode)
        r1, t1 = _solve_milp(c1, lower, upper, integ, constraint, "gate distance")
        dopt = float(DT * np.sum(r1.x[idx["a"]]))
        if not full_lexicographic:
            x = r1.x
            c = x[idx["c"]]; d = x[idx["d"]]; q = x[idx["q"]]; e = x[idx["e"]]
            action = c-d
            return GateResult(status="optimal_distance_only", success=True, gate_type=gate_type, terminal_tolerance_kwh=terminal_tolerance, distance_kwh=float(DT*np.abs(action-reference).sum()), throughput_kwh=float(DT*(c+d).sum()), physical_cost_usd=_physical_cost(data, battery, action, q), curtailment_kwh=float(DT*q.sum()), curtailment_price_term_usd=float(DT/1000*np.sum(data["price_usd_mwh"]*q)), exact_retained_fraction=float(np.mean(np.abs(action-reference)<=1e-6)), correlation=_correlation(action,reference), solve_seconds=time.perf_counter()-overall_start, stage1_seconds=t1, action_kw=action.tolist(), curtailment_kw=q.tolist(), soc_kwh=e.tolist())
        lower2, upper2, integ2, _, constraint2, idx2, data2 = _gate_problem_arrays(task, battery, reference, initial_soc, allow_curtailment, terminal_tolerance, distance_limit=dopt+LEX_DISTANCE_TOL_KWH)
        if feasible_action is not None:
            z2_indices = np.arange(idx2["z"].start, idx2["z"].stop)
            lower2[z2_indices[zero_mode_mask]] = anchor_mode[zero_mode_mask]
            upper2[z2_indices[zero_mode_mask]] = anchor_mode[zero_mode_mask]
            integ2[z2_indices[zero_mode_mask]] = 0
        c2 = np.zeros(len(lower2)); c2[idx2["c"]] = DT; c2[idx2["d"]] = DT
        c2[idx2["z"]] = MODE_TIE_BREAK_WEIGHT * (1.0 - 2.0 * raw_charge_mode)
        r2, t2 = _solve_milp(c2, lower2, upper2, integ2, constraint2, "gate throughput")
        topt = float(DT * np.sum(r2.x[idx2["c"]] + r2.x[idx2["d"]]))
        lower3, upper3, integ3, _, constraint3, idx3, data3 = _gate_problem_arrays(task, battery, reference, initial_soc, allow_curtailment, terminal_tolerance, distance_limit=dopt+LEX_DISTANCE_TOL_KWH, throughput_limit=topt+LEX_THROUGHPUT_TOL_KWH)
        if feasible_action is not None:
            z3_indices = np.arange(idx3["z"].start, idx3["z"].stop)
            lower3[z3_indices[zero_mode_mask]] = anchor_mode[zero_mode_mask]
            upper3[z3_indices[zero_mode_mask]] = anchor_mode[zero_mode_mask]
            integ3[z3_indices[zero_mode_mask]] = 0
        # The first two stages include a deterministic raw-sign mode tie-break.
        # Fix that selected binary mode for the final cost tie-break; this avoids
        # re-opening a highly degenerate mixed-integer optimal face.
        selected_mode = np.rint(r2.x[idx2["z"]]).astype(float)
        lower3[idx3["z"]] = selected_mode
        upper3[idx3["z"]] = selected_mode
        integ3[idx3["z"]] = 0
        c3 = np.zeros(len(lower3))
        c3[idx3["c"]] = DT/1000*data3["price_usd_mwh"] + battery.degradation_cost_per_kwh*DT
        c3[idx3["d"]] = -DT/1000*data3["price_usd_mwh"] + battery.degradation_cost_per_kwh*DT
        c3[idx3["q"]] = DT/1000*data3["price_usd_mwh"]
        r3, t3 = _solve_milp(c3, lower3, upper3, integ3, constraint3, "gate cost fixed-mode")
        x = r3.x; c = x[idx3["c"]]; d = x[idx3["d"]]; q = x[idx3["q"]]; e = x[idx3["e"]]
        action = c-d
        return GateResult(
            status="optimal", success=True, gate_type=gate_type, terminal_tolerance_kwh=terminal_tolerance,
            distance_kwh=float(DT*np.abs(action-reference).sum()), throughput_kwh=float(DT*(c+d).sum()),
            physical_cost_usd=_physical_cost(data3,battery,action,q), curtailment_kwh=float(DT*q.sum()),
            curtailment_price_term_usd=float(DT/1000*np.sum(data3["price_usd_mwh"]*q)),
            exact_retained_fraction=float(np.mean(np.abs(action-reference)<=1e-6)), correlation=_correlation(action,reference),
            solve_seconds=time.perf_counter()-overall_start, stage1_seconds=t1, stage2_seconds=t2, stage3_seconds=t3,
            action_kw=action.tolist(), curtailment_kw=q.tolist(), soc_kwh=e.tolist(),
        )
    except Exception as exc:
        return GateResult(status="error", success=False, gate_type=gate_type, terminal_tolerance_kwh=terminal_tolerance, solve_seconds=time.perf_counter()-overall_start, message=f"{type(exc).__name__}: {exc}")

def _physical_cost(data: dict[str, np.ndarray], battery: Battery, action: np.ndarray, curtailment: np.ndarray) -> float:
    grid = data["load_kw"] - data["pv_kw"] + curtailment + action
    return float(DT/1000*np.sum(data["price_usd_mwh"]*grid) + battery.degradation_cost_per_kwh*DT*np.sum(np.abs(action)))

def _correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return None
    return float(np.corrcoef(a,b)[0,1])

def _solve_gate_lp_mode(
    task: dict[str, Any],
    battery: Battery,
    reference: np.ndarray,
    initial_soc: float,
    *,
    allow_curtailment: bool,
    terminal_tolerance: float,
    fixed_mode: np.ndarray | None,
    zero_mode_anchor: np.ndarray,
    mode_source: str,
) -> tuple[GateResult | None, dict[str, Any]]:
    """Solve the lexicographic gate with relaxed or fixed charge/discharge modes."""
    overall = time.perf_counter()
    lower, upper, integ, _, constraint, idx, data = _gate_problem_arrays(
        task, battery, reference, initial_soc, allow_curtailment, terminal_tolerance
    )
    integ[:] = 0
    if fixed_mode is not None:
        mode = np.asarray(fixed_mode, dtype=float)
        zidx = np.arange(idx["z"].start, idx["z"].stop)
        lower[zidx] = mode
        upper[zidx] = mode
    N = len(lower)
    c1 = np.zeros(N); c1[idx["a"]] = DT
    try:
        r1, t1 = _solve_milp(c1, lower, upper, integ, constraint, f"LP gate distance {mode_source}")
    except Exception as exc:
        return None, {"error": f"{type(exc).__name__}: {exc}"}
    dopt = float(DT * np.sum(r1.x[idx["a"]]))
    lower2, upper2, integ2, _, constraint2, idx2, data2 = _gate_problem_arrays(
        task, battery, reference, initial_soc, allow_curtailment, terminal_tolerance,
        distance_limit=dopt + LEX_DISTANCE_TOL_KWH,
    )
    integ2[:] = 0
    if fixed_mode is not None:
        zidx2 = np.arange(idx2["z"].start, idx2["z"].stop)
        lower2[zidx2] = mode; upper2[zidx2] = mode
    c2 = np.zeros(len(lower2)); c2[idx2["c"]] = DT; c2[idx2["d"]] = DT
    try:
        r2, t2 = _solve_milp(c2, lower2, upper2, integ2, constraint2, f"LP gate throughput {mode_source}")
    except Exception as exc:
        return None, {"distance_lower_bound_kwh": dopt, "error": f"{type(exc).__name__}: {exc}"}
    ch2 = np.asarray(r2.x[idx2["c"]], dtype=float)
    dis2 = np.asarray(r2.x[idx2["d"]], dtype=float)
    action2 = ch2 - dis2
    overlap = np.minimum(ch2, dis2)
    overlap_count = int(np.sum(overlap > 1e-6))
    overlap_kwh = float(DT * overlap.sum())
    topt = float(DT * np.sum(ch2 + dis2))
    if fixed_mode is None and overlap_count > 0:
        return None, {
            "distance_lower_bound_kwh": dopt,
            "throughput_lower_bound_kwh": topt,
            "lp_action_kw": action2,
            "lp_overlap_intervals": overlap_count,
            "lp_overlap_kwh": overlap_kwh,
            "stage1_seconds": t1,
            "stage2_seconds": t2,
        }
    selected_mode = np.asarray(fixed_mode, dtype=float) if fixed_mode is not None else (action2 > 1e-9).astype(float)
    zero_action = np.abs(action2) <= 1e-9
    selected_mode[zero_action] = zero_mode_anchor[zero_action]
    lower3, upper3, integ3, _, constraint3, idx3, data3 = _gate_problem_arrays(
        task, battery, reference, initial_soc, allow_curtailment, terminal_tolerance,
        distance_limit=dopt + LEX_DISTANCE_TOL_KWH,
        throughput_limit=topt + LEX_THROUGHPUT_TOL_KWH,
    )
    integ3[:] = 0
    zidx3 = np.arange(idx3["z"].start, idx3["z"].stop)
    lower3[zidx3] = selected_mode; upper3[zidx3] = selected_mode
    c3 = np.zeros(len(lower3))
    c3[idx3["c"]] = DT/1000*data3["price_usd_mwh"] + battery.degradation_cost_per_kwh*DT
    c3[idx3["d"]] = -DT/1000*data3["price_usd_mwh"] + battery.degradation_cost_per_kwh*DT
    c3[idx3["q"]] = DT/1000*data3["price_usd_mwh"]
    try:
        r3, t3 = _solve_milp(c3, lower3, upper3, integ3, constraint3, f"LP gate cost {mode_source}")
    except Exception as exc:
        return None, {"distance_lower_bound_kwh": dopt, "error": f"{type(exc).__name__}: {exc}"}
    x = r3.x
    ch = np.asarray(x[idx3["c"]], dtype=float); dis = np.asarray(x[idx3["d"]], dtype=float)
    q = np.asarray(x[idx3["q"]], dtype=float); e = np.asarray(x[idx3["e"]], dtype=float)
    action = ch - dis
    distance = float(DT * np.abs(action-reference).sum())
    result = GateResult(
        status="global_distance_optimum_lp_integral" if fixed_mode is None else "fixed_mode_feasible",
        success=True,
        gate_type="expanded_system" if allow_curtailment else "battery_only",
        terminal_tolerance_kwh=terminal_tolerance,
        distance_kwh=distance,
        throughput_kwh=float(DT*np.sum(ch+dis)),
        physical_cost_usd=_physical_cost(data3,battery,action,q),
        curtailment_kwh=float(DT*q.sum()),
        curtailment_price_term_usd=float(DT/1000*np.sum(data3["price_usd_mwh"]*q)),
        exact_retained_fraction=float(np.mean(np.abs(action-reference)<=1e-6)),
        correlation=_correlation(action,reference),
        solve_seconds=time.perf_counter()-overall,
        stage1_seconds=t1,stage2_seconds=t2,stage3_seconds=t3,
        action_kw=action.tolist(),curtailment_kw=q.tolist(),soc_kwh=e.tolist(),
        distance_lower_bound_kwh=dopt,
        distance_optimality_gap_kwh=max(distance-dopt,0.0),
        global_distance_optimality_proven=fixed_mode is None,
        lp_overlap_intervals=overlap_count,
        lp_overlap_kwh=overlap_kwh,
        selected_mode_source=mode_source,
    )
    return result, {"distance_lower_bound_kwh": dopt, "lp_overlap_intervals": overlap_count, "lp_overlap_kwh": overlap_kwh}

def solve_gate_hybrid(
    task: dict[str, Any],
    battery: Battery,
    reference_action: Sequence[float],
    initial_soc: float,
    *,
    allow_curtailment: bool,
    terminal_tolerance: float,
    full_lexicographic: bool = True,
    feasible_action: Sequence[float] | None = None,
) -> GateResult:
    """Complete-coverage projection with an LP certificate and fixed-mode upper bound.

    If the LP relaxation has no simultaneous charge/discharge, it is an exact
    globally distance-optimal solution to the mixed-integer gate. Otherwise,
    the LP distance is retained as a lower bound and the best feasible solution
    from a disclosed mode library is returned as an upper bound.
    """
    reference = np.asarray(reference_action, dtype=float)
    if feasible_action is None:
        raise ValueError("hybrid gate requires a feasible reference action")
    feasible = np.asarray(feasible_action, dtype=float)
    anchor = (feasible > 0.0).astype(float)
    relaxed, info = _solve_gate_lp_mode(
        task,battery,reference,initial_soc,allow_curtailment=allow_curtailment,
        terminal_tolerance=terminal_tolerance,fixed_mode=None,
        zero_mode_anchor=anchor,mode_source="lp_relaxation",
    )
    if relaxed is not None:
        if not full_lexicographic:
            relaxed.status = "global_distance_optimum_lp_integral_distance_focus"
        return relaxed
    lower_bound = float(info.get("distance_lower_bound_kwh", float("nan")))
    lp_action = np.asarray(info.get("lp_action_kw", feasible), dtype=float)
    heuristic = heuristic_action(task,battery,initial_soc)
    modes: dict[str,np.ndarray] = {}
    raw_mode = (reference > 0.0).astype(float)
    raw_mode[np.abs(reference)<=1e-9] = anchor[np.abs(reference)<=1e-9]
    lp_mode = (lp_action > 0.0).astype(float)
    lp_mode[np.abs(lp_action)<=1e-9] = anchor[np.abs(lp_action)<=1e-9]
    heur_mode = (heuristic > 0.0).astype(float)
    modes["raw_sign_with_mpc_zero_anchor"] = raw_mode
    modes["lp_net_sign_with_mpc_zero_anchor"] = lp_mode
    modes["direct_mpc_sign"] = anchor
    modes["terminal_heuristic_sign"] = heur_mode
    unique: dict[bytes,tuple[str,np.ndarray]] = {}
    for name,mode in modes.items(): unique.setdefault(mode.astype(np.uint8).tobytes(),(name,mode))
    candidates: list[GateResult] = []
    errors=[]
    for name,mode in unique.values():
        result, meta = _solve_gate_lp_mode(
            task,battery,reference,initial_soc,allow_curtailment=allow_curtailment,
            terminal_tolerance=terminal_tolerance,fixed_mode=mode,
            zero_mode_anchor=anchor,mode_source=name,
        )
        if result is not None:
            result.distance_lower_bound_kwh=lower_bound
            result.distance_optimality_gap_kwh=max(float(result.distance_kwh)-lower_bound,0.0)
            result.global_distance_optimality_proven=result.distance_optimality_gap_kwh<=LEX_DISTANCE_TOL_KWH
            result.lp_overlap_intervals=int(info.get("lp_overlap_intervals",0))
            result.lp_overlap_kwh=float(info.get("lp_overlap_kwh",0.0))
            candidates.append(result)
        else:
            errors.append({"mode":name,"meta":meta})
    if not candidates:
        return GateResult(status="error",success=False,gate_type="expanded_system" if allow_curtailment else "battery_only",terminal_tolerance_kwh=terminal_tolerance,distance_lower_bound_kwh=lower_bound,lp_overlap_intervals=int(info.get("lp_overlap_intervals",0)),lp_overlap_kwh=float(info.get("lp_overlap_kwh",0.0)),message=canonical_json(errors))
    candidates.sort(key=lambda r:(float(r.distance_kwh),float(r.throughput_kwh),float(r.physical_cost_usd)))
    best=candidates[0]
    best.status="feasible_mode_library_upper_bound" if not best.global_distance_optimality_proven else "global_distance_optimum_mode_library"
    if not full_lexicographic:
        best.status += "_distance_focus"
    return best

def solve_cost_optimal(task: dict[str, Any], battery: Battery, initial_soc: float, *, allow_curtailment: bool, terminal_tolerance: float = 0.0) -> GateResult:
    data = task_suffix(task,battery); n=len(data["load_kw"])
    reference=np.zeros(n)
    lower,upper,integ,_,constraint,idx,data = _gate_problem_arrays(task,battery,reference,initial_soc,allow_curtailment,terminal_tolerance)
    # Remove the absolute-value constraints and variables indirectly by zero objective; they are harmless.
    c=np.zeros(len(lower))
    c[idx["c"]]=DT/1000*data["price_usd_mwh"]+battery.degradation_cost_per_kwh*DT
    c[idx["d"]]=-DT/1000*data["price_usd_mwh"]+battery.degradation_cost_per_kwh*DT
    c[idx["q"]]=DT/1000*data["price_usd_mwh"]
    start=time.perf_counter()
    try:
        r,t=_solve_milp(c,lower,upper,integ,constraint,"cost optimum")
        x=r.x; ch=x[idx["c"]]; dis=x[idx["d"]]; q=x[idx["q"]]; e=x[idx["e"]]; action=ch-dis
        return GateResult(status="optimal",success=True,gate_type="expanded_system_optimum" if allow_curtailment else "battery_only_optimum",terminal_tolerance_kwh=terminal_tolerance,distance_kwh=None,throughput_kwh=float(DT*(ch+dis).sum()),physical_cost_usd=_physical_cost(data,battery,action,q),curtailment_kwh=float(DT*q.sum()),curtailment_price_term_usd=float(DT/1000*np.sum(data["price_usd_mwh"]*q)),solve_seconds=time.perf_counter()-start,stage1_seconds=t,action_kw=action.tolist(),curtailment_kw=q.tolist(),soc_kwh=e.tolist())
    except Exception as exc:
        return GateResult(status="error",success=False,gate_type="expanded_system_optimum" if allow_curtailment else "battery_only_optimum",terminal_tolerance_kwh=terminal_tolerance,solve_seconds=time.perf_counter()-start,message=f"{type(exc).__name__}: {exc}")

def heuristic_action(task: dict[str, Any], battery: Battery, initial_soc: float) -> np.ndarray:
    data=task_suffix(task,battery); n=len(data["load_kw"]); soc=float(initial_soc); out=np.zeros(n)
    for t in range(n):
        remaining=n-t; delta=TARGET_KWH-soc
        if delta>=0:
            desired=delta/(battery.eta_c*DT*remaining)
        else:
            desired=delta*battery.eta_d/(DT*remaining)
        export_floor=-data["export_limit_kw"][t]-data["load_kw"][t]+data["pv_kw"][t]
        max_charge_by_soc=max(data["capacity_kwh"][t]-soc,0)/(battery.eta_c*DT)
        max_discharge_by_soc=max(soc-data["reserve_kwh"][t],0)*battery.eta_d/DT
        lo=max(-data["max_discharge_kw"][t],export_floor,-max_discharge_by_soc)
        hi=min(data["max_charge_kw"][t],max_charge_by_soc)
        u=float(np.clip(desired,lo,hi)) if lo<=hi else float(hi)
        out[t]=u
        soc += battery.eta_c*max(u,0)*DT-max(-u,0)*DT/battery.eta_d
    return out

def deterministic_random_action(scenario_id: str, task: dict[str, Any], battery: Battery) -> np.ndarray:
    seed=int(hashlib.sha256(f"{RANDOM_BASELINE_SEED}:{scenario_id}".encode()).hexdigest()[:16],16)%(2**32)
    rng=np.random.default_rng(seed); data=task_suffix(task,battery)
    return rng.uniform(-battery.max_discharge_kw,battery.max_charge_kw,size=len(data["load_kw"]))
