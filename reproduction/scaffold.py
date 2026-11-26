"""Run the frozen experimental plan without contacting a model service."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"analysis"))
import carrier_parser
import prompt_protocol as protocol

DATA=ROOT/"data"
MODELS={"deepseek_formal":("deepseek","deepseek-v4-flash"),"qwen_flash":("qwen","qwen3.6-flash-2026-04-16"),"qwen37_plus_direct":("qwen","qwen3.7-plus")}
TARGETS={
    "E2":     (2160,2201,2146,55,2),
    "F1":     (480,482,480,2,0),
    "F0":     (120,120,120,0,0),
    "Q37-F1": (240,240,240,0,0),
    "Q37-F0": (60,60,60,0,0),
}

def sha(value):
    return hashlib.sha256(protocol.canonical_json(value).encode()).hexdigest()

def read_rows(name):
    with (DATA/name).open() as handle:
        yield from map(json.loads,handle)

def key(row):
    return row["tier"],row["block"],row["branch"]

def unique_rows(name):
    result={}
    for row in read_rows(name):
        k=key(row)
        if k in result:raise ValueError("duplicate position in "+name)
        result[k]=row
    return result

def require(ok,message):
    if not ok:raise ValueError(message)


def replay_plan(output=None):
    requests={r["sha256"]:r["payload"] for r in read_rows("requests.jsonl")}
    tasks=json.loads((DATA/"scenarios.json").read_text())
    calls=unique_rows("responses.jsonl")
    attempts=unique_rows("attempt_outcomes.jsonl")
    require(calls.keys()==attempts.keys(),"attempt and logical populations disagree")
    observed=set();used_requests=set();counts={tier:Counter() for tier in TARGETS}
    block_order={tier:[] for tier in TARGETS}
    complete_requests={};blocks=list(read_rows("call_plan.jsonl"))
    phase={"E2":0,"F1":1,"F0":2,"Q37-F0":3,"Q37-F1":3}
    actual_order=[(phase[r["tier"]],r["run_order"]) for r in blocks]
    require(actual_order==sorted(set(actual_order)),"phase and shared sensitivity queue order")
    with (output.open("x") if output else nullcontext(None)) as simulation:
        for plan in blocks:
            tier=plan["tier"];block=plan["block"];sid=plan["scenario"];t=tasks[sid]
            require(t["scenario_id"]==sid,"scenario identity")
            model=plan["model"];provider,model_id=MODELS[model]
            spec=dict(provider=provider,configured_model_id=model_id,temperature=0.,max_output_tokens=16000)
            stages=plan["branches"]
            if tier=="E2":
                require(stages==["prefix","C","M","S","K","C2"],"E2 prefix/fork")
                states=protocol.e2_states(t);dynamic=None
            else:
                require(len(stages)==4 and set(stages)=={"C1","C2","S1","S2"},"complete-contract plan")
            block_order[tier].append(plan["run_order"])
            for branch in stages:
                position=tier,block,branch
                require(position not in observed,"duplicate planned scientific position")
                observed.add(position)
                record=calls[position];att=attempts[position];facts=att["attempts"]
                require(record["scenario"]==sid and record["model"]==model,"plan to row mismatch")
                cc=counts[tier];cc["planned"]+=1;cc["physical_attempts"]+=len(facts)
                cc["extra_attempts"]+=max(0,len(facts)-1);cc["retried_positions"]+=int(len(facts)>1)
                cc["transport_failures"]+=facts.count("transport_failure")
                if tier=="E2":
                    if branch=="M" and dynamic is None:
                        require(record["status"]=="skipped" and record["request"] is None and record["content"] is None and not facts and 'choice_sha256' not in att,"missing-prefix handling")
                        cc["skips"]+=1
                        if simulation:
                            print(json.dumps({"tier":tier,"block":block,"branch":branch,"status":"skip_missing_prefix_state"}),file=simulation)
                        continue
                    state=(states["initial"] if branch=="prefix" else dynamic if branch=="M" else states[branch])
                    prompt=protocol.compile_prompt(t,"I3","stage1" if branch=="prefix" else "stage2",state)
                    body=protocol.build_payload(model_id=model_id,provider=provider,interface="I3",prompt=prompt,max_output_tokens=16000)
                else:
                    body=protocol.request_payload(spec,protocol.render_prompt(t,branch))
                    complete_requests[(tier,model,sid,branch)]=sha(body)
                h=sha(body)
                require(h==record["request"] and body==requests[h],"regenerated scientific request differs at "+str(position))
                used_requests.add(h)
                require(facts and set(facts)<=set(("returned","transport_failure")) and all(f=="transport_failure" for f in facts[:-1]),"attempt outcomes")
                if facts[-1]=="returned":
                    content=record["content"]
                    require(record["status"]=="ok" and isinstance(content,str),"successful return shape")
                    require(hashlib.sha256(content.encode()).hexdigest()==record["content_sha256"],"body digest")
                    choice={"index":0,"finish_reason":"stop","message":{"role":"assistant","content":content}}
                    require(sha(choice)==att["choice_sha256"],"scientific choice digest")
                    cc["choices"]+=1;cc["empty_choices"]+=int(content=="")
                    if tier=="E2" and branch=="prefix":
                        parsed=carrier_parser.parse_output(content,"I3",t["stage1_horizon"])
                        dynamic=protocol.model_carrier(parsed.state)
                else:
                    # The E2 recorder used an empty string when no body was received.
                    require(record["status"]=="transport_error" and record["content"]=="" and 'choice_sha256' not in att,"exhausted transport attempts")
                    cc["exhausted_positions"]+=1
                if simulation:
                    print(json.dumps({"tier":tier,"block":block,"branch":branch,"attempts":facts,"payload":body},separators=(",",":"),ensure_ascii=False),file=simulation)
    require(observed==set(calls) and used_requests==set(requests),"scientific coverage closure")
    for tier,order in block_order.items():
        require(order==sorted(set(order)),"frozen block queue order")
        cc=counts[tier]
        require(tuple(cc[k] for k in ("planned","physical_attempts","choices","transport_failures","skips"))==TARGETS[tier],"tier execution ledger")
    for (tier,model,sid,branch),h in complete_requests.items():
        if tier.endswith("F0"):
            require(complete_requests[(tier[:-1]+"1",model,sid,branch)]==h,"temporal repeat payload")
    return {k:dict(v) for k,v in counts.items()}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,help="Write the reconstructed scientific schedule and payloads to a new file.")
    args=parser.parse_args()
    print(json.dumps(replay_plan(args.output),indent=2,sort_keys=True))
