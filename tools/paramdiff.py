"""Compare two params.py files by value, ignoring comments and layout.

    uv run python tools/paramdiff.py <old params.py> <new params.py>

Walks module-level scalars and dataclass instances, recursing through nested
dataclasses, dicts, lists and tuples, and prints every leaf whose value
differs plus anything present in only one file. Loads both as modules, so a
params.py that does not import cleanly on its own will fail here.

Authors: vkalva, Claude
"""
import dataclasses
import importlib.util
import math
import sys

ABSENT = "<absent>"
SCALARS = (int, float, str, bool, type(None))


def load(path, name):
    """Import a .py file under `name`. Returns the module."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def walk(val, prefix, out):
    """Flatten `val` into out[dotted name] = leaf scalar."""
    if dataclasses.is_dataclass(val) and not isinstance(val, type):
        for f in dataclasses.fields(val):
            walk(getattr(val, f.name), f"{prefix}.{f.name}", out)
    elif isinstance(val, dict):
        for k, v in val.items():
            walk(v, f"{prefix}[{k}]", out)
    elif isinstance(val, (list, tuple)):
        for i, v in enumerate(val):
            walk(v, f"{prefix}[{i}]", out)
    elif isinstance(val, SCALARS):
        out[prefix] = val


def flatten(mod):
    """{dotted name: scalar} for everything reachable from module level."""
    out = {}
    for key, val in vars(mod).items():
        if key.startswith("_") or isinstance(val, type):
            continue
        if dataclasses.is_dataclass(val) or isinstance(val, SCALARS):
            walk(val, key, out)
    return out


def same(x, y):
    if isinstance(x, float) and isinstance(y, float):
        return math.isclose(x, y, rel_tol=1e-12, abs_tol=0.0)
    return x == y


def fmt(v):
    return f"{v:.10g}" if isinstance(v, float) else repr(v)


def main(p_old, p_new):
    a, b = flatten(load(p_old, "p_old")), flatten(load(p_new, "p_new"))
    diffs = [(k, a.get(k, ABSENT), b.get(k, ABSENT)) for k in sorted(set(a) | set(b))
             if not same(a.get(k, ABSENT), b.get(k, ABSENT))]
    if not diffs:
        print("identical in value")
        return
    w = max(len(k) for k, _, _ in diffs)
    print(f"{'name':<{w}}  {'old':>18}  {'new':>18}")
    for k, x, y in diffs:
        print(f"{k:<{w}}  {fmt(x):>18}  {fmt(y):>18}")
    print(f"\n{len(diffs)} differing, {len(a)} vs {len(b)} values total")


if __name__ == "__main__":
    main(*sys.argv[1:3])
