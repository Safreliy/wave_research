"""Preserve exact protocol wording at dynamic and physical campaign freeze."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).parent
current = (root / "PREDECLARED_PROTOCOL.md").read_bytes()
text = current.decode("utf-8")
old_phrase = ("trace refinement factor is 8 on L9/L10 and 16 on L11, as in the existing\n"
              "native protocol. Thus trace-spacing/grid-spacing ratios are respectively\n")
new_phrase = ("trace refinement factor is 8 on L9/L10, reproducing the historical inputs.\n"
              "The new L11 run uses factor 16 to hold the L10 trace/grid ratio fixed. Thus\n"
              "trace-spacing/grid-spacing ratios are respectively\n")
joint = ("An additional, separate initial-error experiment fixes the trace/grid ratio\n"
         "at 0.125 on all three levels (factors 8, 16 and 32 for 512, 1024 and 2048\n"
         "columns). It uses the same source nodes, exact initial phase means and\n"
         "bilinear control. It is reported separately from the 8/8/16 native series,\n"
         "regardless of whether its observed slopes are favourable. If native runtime\n"
         "permits, the factor-16 L10 and factor-32 L11 matched branches will also be\n"
         "continued to 0.1T against exact/control branches with byte-identical common\n"
         "fields. These runs test a fixed *refined trace/grid ratio* under receiver\n"
         "refinement, not convergence of boundary-source discretization.\n")
if text.count(new_phrase) != 1 or text.count(joint) != 1:
    raise ValueError("protocol changed beyond expected wording")
physical = text.replace(new_phrase, old_phrase)
dynamic = physical.replace(joint, "")
for label, source in (("physical", physical), ("dynamic", dynamic)):
    expected = json.loads((root / label / "campaign.json").read_text())[
        "predeclared_protocol_sha256"]
    data = source.encode("utf-8")
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"{label} protocol snapshot hash mismatch")
    (root / f"PROTOCOL_AT_{label.upper()}_FREEZE.md").write_bytes(data)
print(json.dumps({"dynamic_freeze": hashlib.sha256(dynamic.encode()).hexdigest(),
                  "physical_freeze": hashlib.sha256(physical.encode()).hexdigest(),
                  "current_clarified": hashlib.sha256(current).hexdigest()}, indent=2))
