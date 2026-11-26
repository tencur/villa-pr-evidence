"""Print each failed test of a pytest log with the first E  line of its traceback."""
import re, sys
text = open(sys.argv[1]).read()
failed = set(re.findall(r"^FAILED \S+::(\S+)", text, re.M))
sections = re.split(r"\n_+ (\S.*?) _+\n", text)
for name, body in zip(sections[1::2], sections[2::2]):
    if name not in failed:
        continue
    err = next((l[1:].strip() for l in body.splitlines() if l.startswith("E ") and l[1:].strip()), "?")
    print(f"{name}: {err[:200]}")
