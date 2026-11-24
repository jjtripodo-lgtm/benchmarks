#!/usr/bin/env python3
"""Emit a bit-parallel exhaustive C++ comparison of BLIF to binary AIGER."""

import argparse
import sys
from pathlib import Path

from collections import deque
import re

def parse_blif(data):
    inputs, outputs, gates = [], [], {}
    current = None
    model = ended = False
    raw = re.sub(r"#[^\n]*", "", data.decode("ascii"))
    for line in re.sub(r"\\[ \t]*\r?\n", " ", raw).splitlines():
        fields = line.split()
        if not fields:
            continue
        op, *args = fields
        if ended:
            raise ValueError("text after .end")
        if op.startswith("."):
            current = None
            if op == ".model" and len(args) == 1 and not model:
                model = True
            elif op in (".inputs", ".outputs"):
                (inputs if op == ".inputs" else outputs).extend(args)
            elif op == ".names" and args and args[-1] not in gates:
                current = args[-1]
                gates[current] = (tuple(args[:-1]), [])
            elif op == ".end" and not args:
                ended = True
            else:
                raise ValueError(f"unsupported or repeated directive {line}")
        elif current is not None:
            pins, rows = gates[current]
            cube, phase = fields if pins else ("", *fields)
            if len(cube) != len(pins) or set(cube) - set("01-") or phase not in ("0", "1"):
                raise ValueError(f"bad cube {line}")
            if rows and phase != rows[0][1]:
                raise ValueError(f"mixed cover phases for {current}")
            rows.append((cube, phase))
        else:
            raise ValueError(f"unexpected {line}")
    if not model or not ended or not inputs or not outputs:
        raise ValueError("incomplete combinational BLIF")
    if len(set(inputs)) != len(inputs) or len(set(outputs)) != len(outputs) or gates.keys() & set(inputs):
        raise ValueError("duplicate port or driver")
    return inputs, outputs, gates


def measure(net):
    inputs, outputs, gates = net
    depth = dict.fromkeys(inputs, (0, 0))
    ordered = []
    missing = {}
    users = {name: [] for name in gates}
    for name, (pins, _) in gates.items():
        if any(pin not in depth and pin not in gates for pin in pins):
            raise ValueError(f"undefined pin near {name}")
        dependencies = [pin for pin in pins if pin in gates]
        missing[name] = len(dependencies)
        for pin in dependencies:
            users[pin].append(name)
    ready = deque(name for name, count in missing.items() if not count)
    while ready:
        name = ready.popleft()
        pins, _ = gates[name]
        depth[name] = (
            (len(pins) >= 2) + max((depth[pin][0] for pin in pins), default=0),
            1 + max((depth[pin][1] for pin in pins), default=0),
        )
        ordered.append(name)
        for user in users[name]:
            missing[user] -= 1
            if not missing[user]:
                ready.append(user)
    if len(ordered) != len(gates):
        raise ValueError("combinational cycle")
    live = set(outputs)
    for name in reversed(ordered):
        if name in live:
            live.update(gates[name][0])
    return {
        "lut": sum(len(gates[n][0]) >= 2 for n in ordered if n in live),
        "depth": max(depth[o][0] for o in outputs),
        "raw_nodes": len(gates),
        "raw_depth": max(depth[o][1] for o in outputs),
        "max_fanin": max((len(pins) for pins, _ in gates.values()), default=0),
        "inputs": len(inputs), "outputs": len(outputs),
        "unreachable": sorted(gates.keys() - live),
        "duplicate_fanin_nodes": sorted(n for n, (p, _) in gates.items() if len(p) != len(set(p))),
        "zero_fanin_nodes": sorted(n for n, (p, _) in gates.items() if not p),
        "unary_nodes": sorted(n for n, (p, _) in gates.items() if len(p) == 1),
        "input_alias_outputs": sorted(set(inputs) & set(outputs)),
    }, ordered


def aig_ports(data):
    end = data.index(b"\n")
    kind, _, ni, nl, no, na = data[:end].split()
    if kind != b"aig" or int(nl):
        raise ValueError("expected classic combinational binary AIGER")
    pos = end + 1
    for _ in range(int(no)):
        pos = data.index(b"\n", pos) + 1
    for _ in range(2 * int(na)):
        while data[pos] & 128:
            pos += 1
        pos += 1
    names = {b"i": {}, b"o": {}}
    for line in data[pos:].splitlines():
        if line == b"c":
            break
        match = re.fullmatch(rb"([io])(\d+) (.+)", line)
        if match:
            kind, index, name = match.groups()
            names[kind][int(index)] = name.decode("ascii")
    counts = [int(ni), int(no)]
    return [tuple(names[k].get(i) for i in range(c)) for k, c in zip((b"i", b"o"), counts)]


def aig_network(data):
    first, body = data.split(b"\n", 1)
    kind, total, ni, nl, no, na = first.split()
    total, ni, nl, no, na = map(int, (total, ni, nl, no, na))
    if kind != b"aig" or nl or total != ni + na:
        raise ValueError("unsupported AIGER layout")
    outputs = []
    for _ in range(no):
        line, body = body.split(b"\n", 1)
        outputs.append(int(line))
    pos = 0

    def delta():
        nonlocal pos
        value, shift = 0, 0
        while True:
            byte = body[pos]
            pos += 1
            value |= (byte & 127) << shift
            if byte < 128:
                return value
            shift += 7

    ands = []
    for gate in range(ni + 1, total + 1):
        lhs = 2 * gate
        a = lhs - delta()
        b = a - delta()
        if not 0 <= b <= a < lhs:
            raise ValueError("AIGER gate order")
        ands.append((a, b))
    if any(out < 0 or out > 2 * total + 1 for out in outputs):
        raise ValueError("invalid AIGER output")
    return ands, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("original_aig", type=Path)
    parser.add_argument("--mutant", action="store_true")
    args = parser.parse_args()
    net = parse_blif(args.candidate.read_bytes())
    inputs, outputs, gates = net
    if (len(inputs), len(outputs)) != (24, 25):
        raise ValueError("expected Sine")
    aig = args.original_aig.read_bytes()
    if aig_ports(aig) != list(map(tuple, (inputs, outputs))):
        raise ValueError("AIGER symbols absent or reordered")
    ands, aig_out = aig_network(aig)
    _, order = measure(net)
    ids = {name: i for i, name in enumerate(inputs + order)}

    def literal(x):
        return f"{'~' if x & 1 else ''}a[{x >> 1}]"

    print("#include <cinttypes>\n#include <cstdio>")
    print("int main() {")
    print(f"uint64_t a[{1 + len(inputs) + len(ands)}], b[{len(ids)}];")
    print("a[0] = 0;")
    print("for (unsigned base = 0; base < (1u << 24); base += 64) {")
    for i, name in enumerate(inputs):
        pattern = sum(1 << k for k in range(64) if (k >> i) & 1)
        expr = f"UINT64_C({pattern})" if i < 6 else f"-(uint64_t)((base >> {i}) & 1u)"
        print(f"a[{i+1}] = b[{ids[name]}] = {expr};")
    # Use a loop for the reference AIG, and compiled cube expressions for the BLIF.
    print("static constexpr unsigned ands[][2] = {")
    for a, b in ands:
        print(f"{{{a}, {b}}},")
    print("};")
    print("auto lit = [&](unsigned x) { return a[x >> 1] ^ -(uint64_t)(x & 1u); };")
    print(f"for (unsigned i = 0; i < {len(ands)}; ++i) a[i + 25] = lit(ands[i][0]) & lit(ands[i][1]);")
    for name in order:
        pins, rows = gates[name]
        terms = []
        for cube, _ in rows:
            terms.append("(" + (" & ".join(f"{'~' if c == '0' else ''}b[{ids[p]}]"
                                            for c, p in zip(cube, pins) if c != "-") or "~UINT64_C(0)") + ")")
        expr = " | ".join(terms) or "UINT64_C(0)"
        if rows and rows[0][1] == "0":
            expr = "~(" + expr + ")"
        print(f"b[{ids[name]}] = {expr};")
    for i, (name, ref) in enumerate(zip(outputs, aig_out)):
        flip = "~" if args.mutant and i == 0 else ""
        print(f'if (uint64_t diff = {literal(ref)} ^ {flip}b[{ids[name]}]) {{'
              f' std::printf("mismatch output={i} input=%u\\n", base + __builtin_ctzll(diff)); return 1; }}')
    print('} std::puts("equivalent: 16777216 assignments, all 25 outputs"); }')


if __name__ == "__main__":
    main()
