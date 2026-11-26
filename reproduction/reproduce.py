"""Reproduce the nine manuscript tables from frozen observations."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "analysis"))
import carrier_parser, schedule_parser, physics_check, plant
from compare import compare_to_paper
from scaffold import replay_plan

DATA = ROOT / "data"
PLANT = plant.Battery(500, 250, 250, .95, .95, 250, 75, 150, .008)
CHECK = physics_check.Battery(500, 250, 250, .95, .95, 75, 150, .008)
BRANCHES = ("C1", "C2", "S1", "S2")


def digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def require(ok, why):
    if not ok:
        raise ValueError(why)


def file_rows(name):
    with (DATA / name).open() as handle:
        yield from map(json.loads, handle)


def support(x, tolerance=1e-8):
    v = np.asarray(x)
    return [int((v > tolerance).sum()), int((abs(v) <= tolerance).sum()), int((v < -tolerance).sum())]


def stats(x):
    x = np.asarray(x)
    return dict(mean=float(x.mean()), median=float(np.median(x)), support=support(x), ci=list(plant.quantile_ci(x)))


def read_calls(tasks):
    requests = {r["sha256"]:r["payload"] for r in file_rows("requests.jsonl")}
    for key, payload in requests.items():
        require(digest(payload) == key, "request body digest")
    blocks = []
    used_requests = set()
    grouped = {}
    for call in file_rows("responses.jsonl"):
        meta = {field:call.pop(field) for field in ("tier","model","scenario","block","repetition","interface")}
        group_key = meta["tier"],meta["block"]
        row = grouped.setdefault(group_key,{**meta,"branches":{}})
        require(all(meta[k]==row[k] for k in meta),"inconsistent block metadata")
        branch = call.pop("branch")
        require(branch not in row["branches"],"duplicate branch")
        row["branches"][branch]=call
    for row in grouped.values():
        t = tasks[row["scenario"]]
        h = len(t["model_visible_episode"]["stage_2"]["remaining_timeseries"])
        e2 = row["tier"] == "E2"
        actions = {}
        for name, item in row["branches"].items():
            if item["request"] is not None:
                used_requests.add(item["request"])
                require(item["request"] in requests, "request link")
            if "request_bytes_sha256" in item:
                require(item["request_bytes_sha256"] == item["request"], "request-byte correspondence")
            if item["status"] != "ok":
                actions[name] = None
                continue
            content = item["content"]
            require(hashlib.sha256(content.encode()).hexdigest() == item["content_sha256"], "returned content")
            if e2:
                parsed = carrier_parser.parse_output(content,row["interface"], t["stage1_horizon"] if name=="prefix" else h)
                obj = carrier_parser.parsed_payload(parsed)
            else:
                parsed = schedule_parser.parse_output(content,h)
                obj = asdict(parsed)
            require(digest(obj) == item["parsed_sha256"] and parsed.ok == item["parsed_ok"], "frozen parser output " + str((row["block"],name)))
            actions[name] = np.asarray(parsed.dense_action_kw) if parsed.ok else None
        row["actions"] = actions
        blocks.append(row)
    require(set(requests)==used_requests,"unlinked request")
    return blocks


def e2_tables(blocks,tasks):
    values = defaultdict(list)
    counts = Counter()
    for b in blocks:
        if b["tier"] != "E2": continue
        a = b["actions"]
        require(b["branches"]["C"]["request"] == b["branches"]["C2"]["request"], "E2 duplicate request")
        d = []
        for branch in ("S", "C2"):
            ok = a["C"] is not None and a[branch] is not None
            counts[branch] += ok
            d.append(plant.action_distance(a["C"],a[branch]) if ok else 0.)
        values[b["scenario"],b["model"]].append(d)
    cells = {key:np.median(rows,axis=0) for key,rows in values.items()}
    require(len(cells)==120 and all(len(r)==3 for r in values.values()),"E2 repetitions")
    scenarios = sorted(tasks)
    models = sorted({m for s,m in values})
    pooled = np.array([np.mean([cells[s,m] for m in models],axis=0) for s in scenarios])
    net = pooled[:,0]-pooled[:,1]
    table2={"canonical_stale":stats(pooled[:,0]),"canonical_duplicate":stats(pooled[:,1]),"adjusted":stats(net)}
    for m in models: table2[m] = stats([cells[s,m][0]-cells[s,m][1] for s in scenarios])
    table7={f:stats([n for s,n in zip(scenarios,net) if tasks[s]["event_family"]==f]) for f in sorted({t["event_family"] for t in tasks.values()})}
    return table2, table7,dict(counts)


def raw_tables(blocks,tasks,row_dir):
    blocks_by_model=defaultdict(list); dose=defaultdict(list); per_model=defaultdict(Counter)
    counts=defaultdict(Counter); witnesses=[]; same={}; temporal=defaultdict(Counter)
    checked=0; raw_actions={}; replay_rows=[]
    for b in blocks:
        tier,model,sid=b["tier"],b["model"],b["scenario"]
        if tier=="E2": continue
        task=tasks[sid]; iv=task["v2_prefix_intervention"]
        c,stale=iv["canonical_event_soc_kwh"],iv["stale_event_soc_kwh"]; gap=c-stale
        acts=[b["actions"][n] for n in BRANCHES]
        require(all(a is not None for a in acts),"complete contract parsing")
        ed=plant.grouping_ed(acts,(0,1),(2,3))
        direction=-np.sign(gap)*.25*np.sum((acts[0]+acts[1]-acts[2]-acts[3])/2)
        if tier.endswith("F1"):
            blocks_by_model[model].append((sid,ed,len(acts[0])*.25,acts))
        if tier=="F1": dose[model,abs(gap)].append((ed,direction))
        identical=[]
        for branch,a in zip(BRANCHES,acts):
            raw_actions[b["block"],branch]=a
            key=(model,sid,branch); item=b["branches"][branch]
            if tier=="F1":same[key]=(a,item["request"])
            if tier=="F0":
                a0,p=same[key];require(p==item["request"],"F0/F1 paired request")
                identical.append(np.array_equal(a,a0));temporal[model]["same_action"]+=int(identical[-1])
            carrier_soc=c if branch.startswith("C") else stale
            replay={}
            for kind,initial in (("carrier_consistent",carrier_soc),("authoritative",c)):
                r=physics_check.evaluate(task,a,initial,CHECK)
                alternative=plant.independent_replay(task,a,initial,PLANT)
                saved=item["saved_replay"][kind+"_replay"]
                for field,v in saved.items():
                    require(abs(r[field]-v)<=2e-5 and abs(alternative[field]-v)<=2e-5, "raw physical replay")
                replay[kind]=r;checked+=1
            ca,au=replay["carrier_consistent"],replay["authoritative"]
            if row_dir:
                replay_rows.append({"tier":tier,"scenario":sid,"model":model,"block":b["block"],"branch":branch,"carrier_feasible":ca["engineering_feasible"],"authoritative_feasible":au["engineering_feasible"],"carrier_terminal_error":ca["terminal_absolute_error_kwh"],"authoritative_terminal_error":au["terminal_absolute_error_kwh"]})
            counts[tier].update({"branches":1,"carrier":int(ca["engineering_feasible"]),"authoritative":int(au["engineering_feasible"])})
            if tier.endswith("F1"):
                per_model[model].update({"active":int(np.any(a!=0)),"dynamic_ok":int(au["dynamic_constraints_ok"]),"terminal_violation":int(au["terminal_absolute_error_kwh"]>1),"carrier":int(ca["engineering_feasible"]),"authoritative":int(au["engineering_feasible"]),"export_violation":int(au["category_counts"]["export"]>0),"soc_lower_violation":int(au["category_counts"]["soc_lower"]>0)})
            if ca["engineering_feasible"] and not au["engineering_feasible"]:
                witnesses.append([tier,branch,task["event_family"],task["location_id"],task["season"],gap,ca["terminal_absolute_error_kwh"],au["terminal_absolute_error_kwh"]])
        if tier=="F0":temporal[model]["same_blocks"]+=int(all(identical))
    table3={}; robustness={}; rng=np.random.default_rng(20260821)
    for model,x in sorted(blocks_by_model.items()):
        ed=[row[1] for row in x]; ascending=np.sort(ed);n=len(ed)
        table3[model]={**per_model[model],**stats(ed)}
        robustness[model]={"normalized":stats([row[1]/row[2] for row in x]),"trimmed_means":[float(np.mean(ascending[k:n-k])) for k in (6,12)],"drop_largest_means":[float(np.mean(ascending[:-k])) for k in (1,2,5)]}
        if model!="qwen37_plus_direct":
            mat=np.array([[plant.grouping_ed(row[3],a,b) for a,b in [((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2))]] for row in x])
            draws=rng.integers(0,3,size=(100000,n))
            null=mat[np.arange(n)[None,:],draws].mean(axis=1)
            robustness[model]["pairing_p"]=(1+int((null>=np.mean(ed)-1e-12).sum()))/100001
    table4={f"{m}/{g:g}":{"ED":stats([r[0] for r in rows]),"direction":support([r[1] for r in rows])} for (m,g),rows in dose.items()}
    if row_dir:
        emit_rows(row_dir/"dual-replay.jsonl",replay_rows)
    return table3,table4,dict(counts),witnesses,dict(temporal),robustness,checked,raw_actions


def gate_tables(tasks,actions,refit):
    references={}; zero={}; grouped=defaultdict(list); baseline=defaultdict(list); checked=0
    proven=0; expanded=Counter(); refits={}; paired={}; correction_by_model=defaultdict(list); regret_by_model=defaultdict(list)
    for sid,t in tasks.items():
        c=t["v2_prefix_intervention"]["canonical_event_soc_kwh"];s=t["v2_prefix_intervention"]["stale_event_soc_kwh"]
        d=np.array(t["mpc"]["action_kw"])
        sol=plant.solve_cost_optimal(t,PLANT,s,allow_curtailment=False)
        require(sol.success,"stale optimum")
        references[sid]={"direct_mpc":d,"zero":np.zeros_like(d),"heuristic":plant.heuristic_action(t,PLANT,c),"random_static_range":plant.deterministic_random_action(sid,t,PLANT),"stale_optimal":np.array(sol.action_kw)}
    for row in file_rows("solutions.jsonl"):
        sid=row["scenario_id"];t=tasks[sid];c=t["v2_prefix_intervention"]["canonical_event_soc_kwh"];v=row["result"]
        require(v["success"],"saved optimizer success")
        a=np.asarray(v["action_kw"]);q=v["curtailment_kw"]
        r=physics_check.evaluate(t,a,c,CHECK,curtailment_kw=q,terminal_tolerance_kwh=v["terminal_tolerance_kwh"])
        trace=[c,*[i["soc_end_kwh"] for i in r["trace"]]]
        require(r["dynamic_constraints_ok"] and r["terminal_absolute_error_kwh"]<=v["terminal_tolerance_kwh"]+1e-5 and np.max(np.abs(trace-np.asarray(v["soc_kwh"])))<=2e-5 and abs(r["physical_cost_usd"]-v["physical_cost_usd"])<=2e-5, "saved gate solution trajectory")
        checked+=1
        if row["kind"]=="llm": proposal=actions[row["block_id"],row["branch"]]
        else: proposal=references[sid][row["baseline"]]
        distance=plant.action_distance(a,proposal)
        require(abs(distance-v["distance_kwh"])<=2e-3,"distance from proposal")
        if refit:
            key=sid,row["gate_type"],tuple(proposal),v["terminal_tolerance_kwh"]
            if key not in refits:
                refits[key]=plant.solve_gate_hybrid(t,PLANT,proposal,c,allow_curtailment=row["gate_type"]=="expanded_system",terminal_tolerance=v["terminal_tolerance_kwh"],full_lexicographic=True,feasible_action=references[sid]["direct_mpc"])
            sol=refits[key]
            require(sol.global_distance_optimality_proven == v["global_distance_optimality_proven"],"distance lower-bound certificate")
            require(sol.success and abs(sol.distance_kwh-distance)<2e-3 and abs(sol.physical_cost_usd-r["physical_cost_usd"])<2e-3,f"gate refit {sid} {row.get('baseline',row.get('branch'))} {sol.message}")
        if row["layer"]!="primary":continue
        if row["gate_type"]!="battery_only":
            if row["kind"]=="llm":
                curtail=np.sum(np.maximum(q,0))*.25>1e-6
                expanded[row["model_condition"]]+=int(curtail)
                paired[(row["model_condition"],row["block_id"],row["branch"])]["curtail"]=bool(curtail)
            continue
        cost=r["physical_cost_usd"];regret=cost-t["mpc"]["objective_usd"]
        if row["kind"]=="llm":
            m=row["model_condition"]
            mode_changed=np.any((np.sign(proposal)!=np.sign(a)) & (np.abs(proposal)>1e-6) & (np.abs(a)>1e-6)) or np.any((np.abs(proposal)<=1e-6) & (np.abs(a)>1e-6))
            paired[(m,row["block_id"],row["branch"])]={"mode_changed":bool(mode_changed)}
            correction_by_model[m].append(distance);regret_by_model[m].append(regret)
            grouped[m].append((sid,distance,regret))
            proven+=int(v["global_distance_optimality_proven"])
        else:
            name=row["baseline"];raw=plant.independent_replay(t,proposal,c,PLANT)
            baseline[name].append((int(raw["engineering_feasible"]),distance,cost,regret))
            if name=="zero":zero[sid]=(distance,regret)
    table9={}
    for name,rows in baseline.items():
        arr=np.array(rows);table9[name]=[int(arr[:,0].sum()),*np.median(arr[:,1:],axis=0).tolist()]
    table6={}
    for m,rows in grouped.items():
        vals=[(sid,dist-zero[sid][0],reg-zero[sid][1]) for sid,dist,reg in rows]; metrics={}
        for field,key in [(1,"distance"),(2,"regret")]:
            xs=[p[field] for p in vals]
            reduced=[np.mean([p[field] for p in vals if p[0]==sid]) for sid in sorted(tasks)]
            metrics[key]={"mean":float(np.mean(xs)),"median":float(np.median(xs)),"support":support(xs,1e-5),"scenario_ci":list(plant.quantile_ci(reduced)),"branch_ci":list(plant.quantile_ci(xs))}
        table6[m]=metrics
    mechanics={m:{"expanded_curtailment":expanded[m],"battery_action_and_mode":sum(v["mode_changed"] and not v["curtail"] for (model,block,branch),v in paired.items() if model==m),"median_correction":float(np.median(correction_by_model[m])),"median_regret":float(np.median(regret_by_model[m]))} for m in grouped}
    return table6,table9,checked,proven,mechanics


def emit_rows(path, observations):
    with path.open("w") as handle:
        for row in observations: handle.write(json.dumps(row,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n")

def reproduce(refit,row_dir=None):
    tasks=json.loads((DATA/"scenarios.json").read_text());b=read_calls(tasks)
    if row_dir:
        row_dir.mkdir(parents=True,exist_ok=False)
        emit_rows(row_dir/"actions.jsonl",({"tier":r["tier"],"block":r["block"],"scenario":r["scenario"],"model":r["model"],"branch":n,"action_kw":None if a is None else a.tolist()} for r in b for n,a in r["actions"].items()))
    table1={};calls=Counter()
    for row in b:
        tier=row["tier"];calls[tier]+=len(row["branches"])
        for i in row["branches"].values():table1.setdefault(tier,Counter())[i["status"]]+=1
    t2,t7,parse=e2_tables(b,tasks)
    t3,t4,t5,t8,temporal,robust,n,actions=raw_tables(b,tasks,row_dir)
    t6,t9,ng,proven,expanded=gate_tables(tasks,actions,refit)
    witnesses=list(file_rows("witnesses.jsonl"))
    for w in witnesses:
        task=tasks[w["scenario"]];soc=task["v2_prefix_intervention"]["canonical_event_soc_kwh"]
        a=[w["power_kw"]]*len(task["model_visible_episode"]["stage_2"]["remaining_timeseries"])
        r=physics_check.evaluate(task,a,soc,CHECK,terminal_tolerance_kwh=1e-6)
        require(r["engineering_feasible"]==w["expected_feasible"] and abs(r["terminal_soc_kwh"]-w["terminal_kwh"])<=1e-5,"deterministic witness")
    fixtures=physics_check.fixture_results(CHECK);require(all(f["pass"] for f in fixtures),"analytic fixtures")
    candidates=list(file_rows("admission.jsonl"))
    require(set(tasks)<={r["scenario_id"] for r in candidates if r["admitted"]},"selected scenarios must be admitted")
    require(len(tasks)==len(set(tasks)) and all(n==12 for n in Counter(t["event_family"] for t in tasks.values()).values()),"balanced challenge families")
    expected_projection={(row["scenario"],row["block"],name) for row in b if row["tier"]=="E2" for name in row["branches"] if name!="prefix"}
    failures=Counter(); observed_projection=set()
    for row in file_rows("E2_projection.jsonl"):
        observed_projection.add((row["scenario_id"],row["source_record_id"],row["branch"]))
        failures[row["new_reason"] or row["new_status"]]+=1
    require(observed_projection==expected_projection,"E2 projection coverage")
    starts=defaultdict(list)
    for row in b:
        for entry in row["branches"].values():
            if entry["status"]!="skipped":
                require(0<=entry["start_us"]<=entry["end_us"],"relative timestamps")
                starts[row["tier"]].append(entry["start_us"])
    elapsed=round((min(starts["F0"])-max(starts["F1"]))/3_600_000_000,2)
    require(elapsed==json.loads((DATA/"service_interval.json").read_text())["last_F1_to_first_F0_request_hours"],"service interval")
    return {"table1":table1,"table2":t2,"table3":t3,"table4":t4,"table5":t5,"table6":t6,"table7":t7,"table8":t8,"table9":t9,
            "calls":calls,"E2_joint_parse":parse,"temporal":temporal,"F1_robustness":robust,"raw_checks":n,"gate_checks":ng,"deterministic_checks":len(witnesses),"fixture_checks":len(fixtures),"gate_global_optima":proven,"gate_mechanisms":expanded,"projection_results":dict(failures),"service_interval_hours":elapsed,"candidate_counts":[len(candidates),sum(c["admitted"] for c in candidates),len(tasks)]}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--refit",action="store_true",help="Also solve individual stored gate optimization problems again.")
    parser.add_argument("--rows",type=Path,help="New directory for the regenerated per-call actions and dual physical verdicts.")
    args=parser.parse_args()
    coverage=replay_plan()
    out=reproduce(args.refit,args.rows)
    out["attempt_coverage"]=coverage
    count=compare_to_paper(out)
    print(f"{count} printed values and row labels agree. Table 6 intervals agree under branch resampling; see TABLE_6.md.")
    args.output.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+"\n")
