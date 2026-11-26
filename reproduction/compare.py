"""Compare the printed manuscript cells at their reported precision."""
from __future__ import annotations
from decimal import Decimal
import json
from pathlib import Path

def check(reference, observed, name="paper"):
    if isinstance(reference, dict):
        return sum(check(v,observed[k],name+"."+k) for k,v in reference.items())
    if isinstance(reference, list):
        if len(reference)!=len(observed):raise ValueError("row count differs: "+name)
        return sum(check(a,b,f"{name}[{i}]") for i,(a,b) in enumerate(zip(reference,observed)))
    if isinstance(reference, str) and reference.replace("-","",1).replace(".","",1).isdigit():
        target=Decimal(reference); unit=Decimal(1).scaleb(target.as_tuple().exponent)
        ok=abs(Decimal(str(observed))-target) <= unit/2
    else:
        ok=reference==observed
    if not ok: raise ValueError(f"{name}: observed {observed!r}, printed {reference!r}")
    return 1

def compare_to_paper(observed):
    reference = json.loads(Path(__file__).with_name("paper_values.json").read_text())
    observed = {**observed, "table8":sorted(observed["table8"])}
    return check(reference,observed)
