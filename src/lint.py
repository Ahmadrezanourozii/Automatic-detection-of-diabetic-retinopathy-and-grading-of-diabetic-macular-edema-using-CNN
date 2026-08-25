"""
lint.py — catch the class of bug that costs a Kaggle run.

Specifically: a name referenced inside a function that is never bound there, at module
scope, or as a builtin. Python only raises NameError when the line actually executes, so a
bug like this survives every import and every syntax check, and surfaces twenty minutes into
a GPU session (ISSUES.md §11).

Run before every push. It needs no data, no GPU and no imports of the modules themselves.

Usage:  python src/lint.py [files...]
"""
from __future__ import annotations
import builtins, glob, os, symtable, sys

BUILTINS = set(dir(builtins))


def unbound_names(path):
    src = open(path).read()
    try:
        top = symtable.symtable(src, os.path.basename(path), "exec")
    except SyntaxError as e:
        return [(f"{path}:{e.lineno}", f"SyntaxError: {e.msg}")]
    module_names = {s.get_name() for s in top.get_symbols()}
    problems = []

    def visit(table, trail):
        for child in table.get_children():
            name = f"{trail}.{child.get_name()}" if trail else child.get_name()
            if child.get_type() == "function":
                for s in child.get_symbols():
                    if not s.is_referenced():
                        continue
                    if (s.is_assigned() or s.is_parameter() or s.is_global()
                            or s.is_free() or s.is_imported() or s.is_declared_global()):
                        continue
                    n = s.get_name()
                    if n in BUILTINS or n in module_names:
                        continue
                    problems.append((f"{path}:{name}()", n))
            visit(child, name)

    visit(top, "")
    return problems


def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "*.py")))
    bad = []
    for f in files:
        bad += unbound_names(f)
    for where, what in bad:
        print(f"UNBOUND  {where} references {what!r}")
    print(f"{len(files)} file(s) checked, {len(bad)} problem(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
