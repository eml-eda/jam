#!/usr/bin/env python3
"""
tosa_to_jam_ir.py - Transform TOSA MLIR models into JAM IR headers.

Maps TOSA to JAM IR (ops.h, tensor.h, graph.h) and now splits the
generated tensors into consts vs variables, with correct l3 pt.

Usage:
  python tosa_to_jam_ir.py <input.mlir> [-o <output.h>]
  python tosa_to_jam_ir.py residual_tiny_cnn_tosa.mlir
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def parse_tensor_type(s: str) -> Tuple[List[int], str]:
    s = s.strip()
    if "x" not in s:
        return [], s
    parts = s.split("x")
    dtype = parts[-1]
    dims_strs = parts[:-1]
    dims: List[int] = []
    for d in dims_strs:
        d = d.strip()
        if d == "":
            continue
        try:
            dims.append(int(d))
        except ValueError:
            dims.append(0)
    return dims, dtype

def dtype_to_jam(dtype: str) -> int:
    d = dtype.lower()
    if d == "f32":
        return 0
    if d == "i8":
        return 1
    if d == "i16":
        return 2
    if d == "i32":
        return 3
    if d == "f16":
        return 2
    if d == "i64":
        return 3
    return 0

def dtype_to_jam_enum(dtype: str) -> str:
    d = dtype.lower()
    if d == "f32":
        return "JAM_IR_DATA_TYPE_F32"
    if d == "i8":
        return "JAM_IR_DATA_TYPE_I8"
    if d == "i16":
        return "JAM_IR_DATA_TYPE_I16"
    if d == "i32":
        return "JAM_IR_DATA_TYPE_I32"
    if d == "f16":
        return "JAM_IR_DATA_TYPE_I16"
    if d == "i64":
        return "JAM_IR_DATA_TYPE_I32"
    return "JAM_IR_DATA_TYPE_F32"

def dtype_c_type(dtype: str) -> str:
    d = dtype.lower()
    if d == "f32":
        return "float"
    if d == "f16":
        return "int16_t"
    if d == "i8":
        return "int8_t"
    if d == "i16":
        return "int16_t"
    if d == "i32":
        return "int32_t"
    if d == "i64":
        return "int64_t"
    return "uint8_t"

def dtype_size(dtype: str) -> int:
    d = dtype.lower()
    if d == "f32":
        return 4
    if d == "f16":
        return 2
    if d == "i8":
        return 1
    if d == "i16":
        return 2
    if d == "i32":
        return 4
    if d == "i64":
        return 8
    return 4

def gemm_trans_from_tosa(operands: List[str], tensors: List[Dict], var_to_idx: Dict[str,int]) -> Tuple[int,int]:
    return (0, 0)

def extract_array(attr_str: str, name: str) -> List[int] | None:
    pat = re.compile(rf'{re.escape(name)}\s*=\s*array<[^:]+:\s*([^>]+)>')
    m = pat.search(attr_str)
    if not m:
        return None
    inner = m.group(1).strip()
    if inner == "":
        return []
    parts = [p.strip() for p in inner.split(",")]
    res = []
    for p in parts:
        try:
            res.append(int(p))
        except ValueError:
            try:
                res.append(int(float(p)))
            except:
                pass
    return res

def extract_perms(attr_str: str) -> List[int] | None:
    return extract_array(attr_str, "perms")

def is_relu_clamp(attrs: str | None, line: str) -> bool:
    text = (attrs or "") + " " + line
    m = re.search(r'min_val\s*=\s*([\-0-9\.eE\+]+)', text)
    if not m:
        return False
    try:
        v = float(m.group(1))
        return abs(v) < 1e-6
    except:
        return False

def sanitize_guard(name: str) -> str:
    s = re.sub(r'[^0-9a-zA-Z]+', '_', name).upper()
    if not s.startswith("_"):
        s = "__" + s
    if not s.endswith("_H__"):
        s = s + "_H__"
    return s

def sanitize_c_name(name: str) -> str:
    # %0 -> const_0, %arg0 -> arg0  (C ident cannot start with digit)
    s = re.sub(r'[^0-9a-zA-Z_]', '_', name).strip('_')
    if not s:
        s = "c"
    if s[0].isdigit():
        s = "c_" + s
    # also prefix % -> const_ for tosa.const like %0
    # ensure not empty and valid
    return s

def parse_hex_blob(hex_str: str) -> List[int]:
    # hex_str like "0x6D0B2C3E9A..." or "0x04000000..."
    s = hex_str.strip()
    if s.startswith("0x") or s.startswith("0X"):
        s = s[2:]
    # remove whitespace
    s = re.sub(r'\s+', '', s)
    # ensure even length
    if len(s) % 2 == 1:
        s = "0" + s
    out = []
    for i in range(0, len(s), 2):
        try:
            out.append(int(s[i:i+2], 16))
        except:
            out.append(0)
    return out

def extract_const_hex(line: str, resource_dict: Dict[str, str]) -> str | None:
    # try dense<"0x...">
    m = re.search(r'dense<"([^"]+)">', line)
    if m:
        return m.group(1)
    # try dense_resource<name>
    m = re.search(r'dense_resource<([^>]+)>', line)
    if m:
        name = m.group(1).strip()
        return resource_dict.get(name)
    # try dense<0> etc - not hex, return None
    return None

def hex_blob_to_c_initializer(hex_str: str, dtype: str, max_elems: int = 8) -> str:
    # For display, emit bytes; for float, emit hex bytes as uint8_t.
    # We emit as uint8_t array to avoid float parsing issues.
    blob = parse_hex_blob(hex_str)
    # Truncate for display in comment, but emit full for real consts? For large blobs, emit ellipsis comment and full data still needed for correctness.
    # We will emit full blob as bytes - for large weight tensors (e.g., 8*3*3*3*4=864 bytes -> 864 entries) that's okay but large for mobilenet (1280*3*3*3 etc). It's okay.
    # To keep file size manageable, we can emit as static const uint8_t with hex bytes.
    # For this implementation, emit all bytes.
    return ", ".join(f"0x{b:02X}" for b in blob)

# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def parse_mlir(path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    # build resource dict for dense_resource lookups
    resource_dict: Dict[str, str] = {}
    # dialect_resources section has lines like  torch_tensor_10_torch.float32: "0x04000000..."
    res_pat = re.compile(r'^\s*([A-Za-z0-9_\.]+)\s*:\s*"([^"]+)"')
    in_resources = False
    for line in lines:
        if "dialect_resources" in line:
            in_resources = True
        if in_resources:
            m = res_pat.search(line)
            if m:
                resource_dict[m.group(1)] = m.group(2)
        if in_resources and "#-}" in line:
            in_resources = False

    tensors: List[Dict[str, Any]] = []
    var_to_idx: Dict[str, int] = {}
    raw_ops: List[Dict[str, Any]] = []

    func_pat = re.compile(r'func\.func\s+@\w+\s*\(([^)]*)\)')
    for line in lines:
        m = func_pat.search(line)
        if m:
            args_str = m.group(1)
            args = [a.strip() for a in args_str.split(",") if a.strip()]
            for arg in args:
                mm = re.search(r'(%[\w\.]+)\s*:\s*tensor<([^>]+)>', arg)
                if mm:
                    var = mm.group(1)
                    tstr = mm.group(2)
                    dims, dtype = parse_tensor_type(tstr)
                    idx = len(tensors)
                    var_to_idx[var] = idx
                    tensors.append({
                        "name": var,
                        "dims": dims,
                        "dtype": dtype,
                        "dtype_int": dtype_to_jam(dtype),
                        "dtype_enum": dtype_to_jam_enum(dtype),
                        "line": line.strip(),
                        "is_const": False,
                    })
            break

    lhs_pat = re.compile(r'^\s*(%[\w\.]+)\s*=')
    for line_no, line in enumerate(lines):
        if "tosa." not in line and '"tosa.const"' not in line:
            continue
        if "func.func" in line:
            continue
        lhs_m = lhs_pat.search(line)
        if not lhs_m:
            continue
        lhs = lhs_m.group(1)
        op_name = None
        if '"tosa.const"' in line:
            op_name = "tosa.const"
        else:
            m = re.search(r'(tosa\.[a-z_0-9]+)', line)
            if m:
                op_name = m.group(1)
            else:
                continue

        if op_name == "tosa.const_shape":
            raw_ops.append({
                "lhs": lhs,
                "op_name": op_name,
                "operands": [],
                "attrs_str": line,
                "result_type": None,
                "dims": None,
                "dtype": None,
                "line": line.strip(),
            })
            continue

        res_m = re.search(r'->\s*tensor<([^>]+)>', line)
        if not res_m:
            res_type = None
            dims = None
            dtype = None
        else:
            res_type = res_m.group(1)
            dims, dtype = parse_tensor_type(res_type)
            if lhs not in var_to_idx:
                idx = len(tensors)
                var_to_idx[lhs] = idx
                # detect const data for tosa.const
                is_const = (op_name == "tosa.const")
                const_hex = None
                if is_const:
                    const_hex = extract_const_hex(line, resource_dict)
                tensors.append({
                    "name": lhs,
                    "dims": dims,
                    "dtype": dtype,
                    "dtype_int": dtype_to_jam(dtype),
                    "dtype_enum": dtype_to_jam_enum(dtype),
                    "line": line.strip(),
                    "op_name": op_name,
                    "is_const": is_const,
                    "const_hex": const_hex,
                })
            else:
                pass

        operands: List[str] = []
        if op_name != "tosa.const":
            op_pos = line.find(op_name)
            after_op = line[op_pos + len(op_name):] if op_pos != -1 else line
            cut_idx = len(after_op)
            brace = after_op.find("{")
            colon = after_op.find(":")
            if brace != -1 and colon != -1:
                cut_idx = min(brace, colon)
            elif brace != -1:
                cut_idx = brace
            elif colon != -1:
                cut_idx = colon
            operand_str = after_op[:cut_idx]
            operands = re.findall(r'%[\w\.]+', operand_str)

        attrs_m = re.search(r'\{([^}]*)\}', line)
        attrs_str = attrs_m.group(0) if attrs_m else ""

        raw_ops.append({
            "lhs": lhs,
            "op_name": op_name,
            "operands": operands,
            "attrs_str": attrs_str,
            "result_type": res_type,
            "dims": dims,
            "dtype": dtype,
            "line": line.strip(),
        })

    # For tensors that are args, ensure is_const=False already; for const tensors, is_const True.
    # For intermediate tensors (op outputs), is_const False.
    for t in tensors:
        if "is_const" not in t:
            t["is_const"] = False
    return tensors, raw_ops, var_to_idx

def tosa_to_jam_ops(tensors: List[Dict], raw_ops: List[Dict], var_to_idx: Dict[str, int]) -> List[Dict[str, Any]]:
    jam_ops: List[Dict[str, Any]] = []

    def resolve(v: str) -> int | None:
        if v in var_to_idx:
            return var_to_idx[v]
        return None

    def get_weight_kernel(var: str, op_name: str) -> List[int] | None:
        idx = var_to_idx.get(var)
        if idx is None:
            return None
        t = tensors[idx]
        dims = t["dims"]
        if len(dims) == 4:
            if op_name == "tosa.depthwise_conv2d":
                return [dims[0], dims[1]]
            else:
                return [dims[1], dims[2]]
        if len(dims) == 3:
            return dims[1:3]
        return None

    reshape_alias: Dict[str, str] = {}
    for rop in raw_ops:
        if rop["op_name"] == "tosa.reshape" and rop["operands"]:
            reshape_alias[rop["lhs"]] = rop["operands"][0]

    def resolve_alias(var: str) -> str:
        seen = set()
        cur = var
        while cur in reshape_alias and cur not in seen:
            seen.add(cur)
            cur = reshape_alias[cur]
        return cur

    def find_producer(var: str) -> int | None:
        resolved = resolve_alias(var)
        for idx in range(len(jam_ops)-1, -1, -1):
            out = jam_ops[idx].get("output")
            if out is None:
                continue
            if var in var_to_idx and out == var_to_idx[var]:
                return idx
            if resolved in var_to_idx and out == var_to_idx[resolved]:
                return idx
        return None

    def elem_enum_for(op_name: str) -> int:
        if op_name == "tosa.add":
            return 6
        if op_name == "tosa.mul":
            return 7
        if op_name == "tosa.sub":
            return 6
        return 6

    i = 0
    while i < len(raw_ops):
        rop = raw_ops[i]
        op = rop["op_name"]

        if op in ("tosa.const", "tosa.const_shape", "tosa.reshape"):
            i += 1
            continue

        if op == "tosa.clamp":
            i += 1
            continue

        def has_relu_fusion(next_idx: int, producer_lhs: str) -> Tuple[bool, str | None]:
            if next_idx >= len(raw_ops):
                return False, None
            nxt = raw_ops[next_idx]
            if nxt["op_name"] != "tosa.clamp":
                return False, None
            if not nxt["operands"]:
                return False, None
            if nxt["operands"][0] != producer_lhs:
                return False, None
            if is_relu_clamp(nxt["attrs_str"], nxt["line"]):
                return True, nxt["lhs"]
            return False, None

        if op in ("tosa.conv2d", "tosa.depthwise_conv2d"):
            operands = rop["operands"]
            input_var = operands[0] if len(operands) > 0 else None
            weight_var = operands[1] if len(operands) > 1 else None
            pad = extract_array(rop["attrs_str"], "pad")
            stride = extract_array(rop["attrs_str"], "stride")
            dilation = extract_array(rop["attrs_str"], "dilation")
            kernel = get_weight_kernel(weight_var, op) if weight_var else None
            if kernel is None:
                kernel = [3,3] if pad and len(pad)==4 else [1,1]
            fused, clamp_var = has_relu_fusion(i+1, rop["lhs"])
            if fused:
                output_var = clamp_var
                advance = 2
                activation = 1
            else:
                output_var = rop["lhs"]
                advance = 1
                activation = 0
            input_idx = resolve(input_var) if input_var else None
            output_idx = resolve(output_var)
            if pad is None:
                pad = [0,0,0,0]
            if len(pad) == 2:
                pad = [pad[0], pad[0], pad[1], pad[1]]
            if len(pad) != 4:
                pad = (pad + [0]*4)[:4]
            if stride is None:
                stride = [1,1]
            if len(stride) == 1:
                stride = [stride[0], stride[0]]
            stride = (stride + [1,1])[:2]
            if kernel is None:
                kernel = [1,1]
            if len(kernel) == 1:
                kernel = [kernel[0], kernel[0]]
            kernel = (kernel + [1,1])[:2]
            inputs = []
            if input_idx is not None:
                inputs.append(input_idx)
            jam_ops.append({
                "type": "JAM_IR_OP_CONV",
                "type_enum": "JAM_IR_OP_CONV",
                "spatial": {"kernel": kernel, "stride": stride, "pad": pad, "num_dims_spat": 2},
                "activation": activation,
                "inputs": inputs,
                "output": output_idx,
                "elementwise_type": [],
                "elementwise_count": 0,
                "dilation": dilation,
                "line": rop["line"],
                "fused": fused,
                "raw": rop,
            })
            i += advance
            continue

        if op == "tosa.matmul":
            operands = rop["operands"]
            act_var = None
            for v in operands:
                idx = var_to_idx.get(v)
                if idx is None:
                    continue
                t = tensors[idx]
                if t["dims"] == [1]:
                    continue
                act_var = v
                break
            if act_var is None and operands:
                act_var = operands[0]
            fused, clamp_var = has_relu_fusion(i+1, rop["lhs"])
            if fused:
                output_var = clamp_var
                advance = 2
                activation = 1
            else:
                output_var = rop["lhs"]
                advance = 1
                activation = 0
            inputs = []
            act_idx = resolve(act_var) if act_var else None
            if act_idx is not None:
                inputs.append(act_idx)
            output_idx = resolve(output_var)
            trans_a, trans_b = gemm_trans_from_tosa(operands, tensors, var_to_idx)
            jam_ops.append({
                "type": "JAM_IR_OP_MATMUL",
                "type_enum": "JAM_IR_OP_MATMUL",
                "gemm": {"trans_a": trans_a, "trans_b": trans_b},
                "activation": activation,
                "inputs": inputs[:2],
                "output": output_idx,
                "elementwise_type": [],
                "elementwise_count": 0,
                "line": rop["line"],
                "fused": fused,
                "raw": rop,
            })
            i += advance
            continue

        if op in ("tosa.add", "tosa.sub", "tosa.mul"):
            operands = rop["operands"][:2]
            has_clamp, clamp_var = has_relu_fusion(i+1, rop["lhs"])
            if has_clamp:
                fused_output_var = clamp_var
                advance = 2
                fused_activation = 1
            else:
                fused_output_var = rop["lhs"]
                advance = 1
                fused_activation = 0
            prod_idx = None
            other_var = None
            cand0 = find_producer(operands[0]) if len(operands) > 0 else None
            cand1 = find_producer(operands[1]) if len(operands) > 1 else None
            if cand0 is not None:
                prod_idx = cand0
                other_var = operands[1] if len(operands) > 1 else None
            elif cand1 is not None:
                prod_idx = cand1
                other_var = operands[0] if len(operands) > 0 else None
            elem_enum = elem_enum_for(op)
            if prod_idx is not None:
                producer = jam_ops[prod_idx]
                cur_cnt = producer.get("elementwise_count", 0)
                other_idx = resolve(other_var) if other_var else None
                can_fuse_inputs = True
                if other_idx is not None:
                    if other_idx not in producer.get("inputs", []) and len(producer.get("inputs", [])) >= 2:
                        can_fuse_inputs = False
                if cur_cnt < 2 and can_fuse_inputs:
                    if "elementwise_type" not in producer:
                        producer["elementwise_type"] = []
                    producer["elementwise_type"].append(elem_enum)
                    producer["elementwise_count"] = len(producer["elementwise_type"])
                    if fused_activation:
                        producer["activation"] = 1
                    if other_idx is not None and other_idx not in producer["inputs"]:
                        producer["inputs"].append(other_idx)
                        producer["inputs"] = producer["inputs"][:2]
                    new_out_idx = resolve(fused_output_var)
                    if new_out_idx is not None:
                        producer["output"] = new_out_idx
                    producer.setdefault("fused_from", []).append(rop["lhs"])
                    i += advance
                    continue
                sys.stderr.write(f"warning: cannot fuse {op} {rop['lhs']} into producer {prod_idx} (elementwise full or inputs full) — emitting standalone\n")
            if op == "tosa.add":
                jam_type = "JAM_IR_OP_ELEMENTWISE_ADD"
            elif op == "tosa.mul":
                jam_type = "JAM_IR_OP_ELEMENTWISE_MUL"
            else:
                jam_type = "JAM_IR_OP_ELEMENTWISE_ADD"
            inputs = [resolve(v) for v in operands if resolve(v) is not None]
            output_idx = resolve(fused_output_var)
            jam_ops.append({
                "type": jam_type,
                "type_enum": jam_type,
                "activation": fused_activation,
                "inputs": inputs[:2],
                "output": output_idx,
                "elementwise_type": [],
                "elementwise_count": 0,
                "line": rop["line"],
                "fused": bool(has_clamp),
                "raw": rop,
            })
            i += advance
            continue

        if op == "tosa.transpose":
            operands = rop["operands"][:1]
            perms = extract_perms(rop["attrs_str"])
            if perms is None:
                perms = [0, 1, 2, 3][:len(tensors[var_to_idx[rop["lhs"]]]["dims"]) if rop["lhs"] in var_to_idx else 2]
            axis = perms[-1] if perms else 0
            inputs = [resolve(v) for v in operands if resolve(v) is not None]
            output_idx = resolve(rop["lhs"])
            jam_ops.append({
                "type": "JAM_IR_OP_TRANSPOSE",
                "type_enum": "JAM_IR_OP_TRANSPOSE",
                "perms": perms,
                "axis": axis,
                "inputs": inputs[:2],
                "output": output_idx,
                "elementwise_type": [],
                "elementwise_count": 0,
                "activation": 0,
                "line": rop["line"],
                "raw": rop,
            })
            i += 1
            continue

        if op in ("tosa.avg_pool2d", "tosa.max_pool2d"):
            operands = rop["operands"]
            input_var = operands[0] if operands else None
            kernel = extract_array(rop["attrs_str"], "kernel")
            pad = extract_array(rop["attrs_str"], "pad")
            stride = extract_array(rop["attrs_str"], "stride")
            if kernel is None:
                kernel = [2,2]
            if pad is None:
                pad = [0,0,0,0]
            if len(pad)==2:
                pad = [pad[0],pad[0],pad[1],pad[1]]
            pad = (pad + [0]*4)[:4]
            if stride is None:
                stride = kernel[:2] if len(kernel)>=2 else [1,1]
            stride = (stride + [1,1])[:2]
            kernel = (kernel + [1,1])[:2]
            pool_output_var = rop["lhs"]
            if i+1 < len(raw_ops) and raw_ops[i+1]["op_name"] == "tosa.reshape" and raw_ops[i+1]["operands"] and raw_ops[i+1]["operands"][0] == pool_output_var:
                fused_reshape_var = raw_ops[i+1]["lhs"]
                output_idx = resolve(fused_reshape_var)
                advance = 1
                fused_reshape = True
            else:
                output_idx = resolve(pool_output_var)
                advance = 1
                fused_reshape = False
            if i+1 < len(raw_ops) and raw_ops[i+1]["op_name"] == "tosa.clamp" and raw_ops[i+1]["operands"][0] == pool_output_var and is_relu_clamp(raw_ops[i+1]["attrs_str"], raw_ops[i+1]["line"]):
                activation = 1
                if not fused_reshape:
                    output_idx = resolve(raw_ops[i+1]["lhs"])
                    advance = 2
                else:
                    pass
            else:
                activation = 0
            inputs = [resolve(input_var)] if input_var and resolve(input_var) is not None else []
            jam_type = "JAM_IR_OP_AVG_POOL" if op == "tosa.avg_pool2d" else "JAM_IR_OP_MAX_POOL"
            jam_ops.append({
                "type": jam_type,
                "type_enum": jam_type,
                "spatial": {"kernel": kernel, "stride": stride, "pad": pad, "num_dims_spat": 2},
                "activation": activation,
                "inputs": inputs[:2],
                "output": output_idx,
                "elementwise_type": [],
                "elementwise_count": 0,
                "line": rop["line"],
                "raw": rop,
                "fused_reshape": fused_reshape if 'fused_reshape' in locals() else False,
            })
            i += advance
            continue

        sys.stderr.write(f"warning: unsupported TOSA op {op} at line: {rop['line'][:120]} — skipped\n")
        i += 1
        continue

    return jam_ops

# ---------------------------------------------------------------------------
# header generation helpers
# ---------------------------------------------------------------------------

def _tensor_c_initializer(t: Dict[str, Any], l3_pt: str, valid_l3: int = 1) -> str:
    dims = t["dims"]
    ndims = len(dims)
    shape_init = ", ".join(str(d) for d in dims) if dims else "1"
    dtype_enum = t.get("dtype_enum", "JAM_IR_DATA_TYPE_F32")
    # l3 is canonical, l2/l1 tiling disabled initially
    return (
        f"        .l3_tile = {{ .shape = {{ .num_dims = {ndims}, .shape = {{{shape_init}}} }}, .pt = {l3_pt}, .valid = {valid_l3} }},\n"
        f"        .l2_tile = {{ .shape = {{ .num_dims = {ndims}, .shape = {{{shape_init}}} }}, .pt = 0, .valid = 0 }},\n"
        f"        .l1_tile = {{ .shape = {{ .num_dims = {ndims}, .shape = {{{shape_init}}} }}, .pt = 0, .valid = 0 }},\n"
        f"        .data_type = {dtype_enum}"
    )

# ---------------------------------------------------------------------------
# header generation
# ---------------------------------------------------------------------------

def generate_header(tensors: List[Dict], jam_ops: List[Dict], output_path: Path, input_path: Path):
    # Single-file mode: all tensors in one header, l3 pt correctly set.
    # Const vs variable distinction is kept for pt computation but emitted together.
    const_tensors = [t for t in tensors if t.get("is_const")]
    variable_tensors = [t for t in tensors if not t.get("is_const")]

    # Compute l3 pt for variables: bump allocator in L3
    var_pt_map: Dict[str, int] = {}
    offset = 0
    for t in variable_tensors:
        name = t["name"]
        size = 1
        for d in t["dims"]:
            size *= d if d != 0 else 1
        size *= dtype_size(t["dtype"])
        offset = (offset + 3) & ~3
        var_pt_map[name] = offset
        offset += size
        offset = (offset + 3) & ~3
    total_var_size = offset

    # Compute l3 pt for consts as offset in const pool
    const_pt_map: Dict[str, int] = {}
    const_offset = 0
    for t in const_tensors:
        const_offset = (const_offset + 3) & ~3
        const_pt_map[t["name"]] = const_offset
        nelems = 1
        for d in t["dims"]:
            nelems *= d if d != 0 else 1
        const_offset += nelems * dtype_size(t["dtype"])
        const_offset = (const_offset + 3) & ~3
    const_total_size = const_offset

    stem = output_path.stem
    guard = sanitize_guard(stem + "_JAM_IR")
    header: List[str] = []
    header.append(f"#ifndef {guard}")
    header.append(f"#define {guard}")
    header.append("")
    header.append(f"/* Auto-generated by tosa_to_jam_ir.py")
    header.append(f" * Source: {input_path.name}")
    header.append(f" * Tensors: {len(tensors)} (const {len(const_tensors)} + var {len(variable_tensors)})  Ops: {len(jam_ops)}")
    header.append(f" * Const pool {const_total_size} bytes, var pool {total_var_size} bytes")
    header.append(f" * See notes/dialect_changes.md for IR notes.")
    header.append(f" */")
    header.append("")
    header.append("#include <stdint.h>")
    header.append("#include <stddef.h>")
    header.append("")
    header.append('#include "jam/ir/ops_config.h"')
    header.append('#include "jam/ir/tensor.h"')
    header.append('#include "jam/ir/ops.h"')
    header.append('#include "jam/ir/graph.h"')
    header.append("")

    # Contiguous L3 pools: const pool (read-only) and variable pool (read-write)
    # pt for const tensors is offset in const pool, for var tensors offset in var pool + const pool size (unified view) or separate.
    # We define both pools as single contiguous arrays so pt offsets point into defined memory.
    header.append(f"#define {stem.upper()}_CONSTS_L3_SIZE {const_total_size}")
    header.append(f"#define {stem.upper()}_VARIABLES_L3_SIZE {total_var_size}")
    header.append(f"#define {stem.upper()}_L3_SIZE {const_total_size + total_var_size}")
    header.append("")
    if const_tensors:
        header.append("/* Const pool: concatenated const blobs, 4-byte aligned */")
        header.append(f"static const uint8_t {stem}_l3_const_pool[{const_total_size if const_total_size else 1}] = {{")
        # Build concatenated bytes with padding
        const_pool_bytes: List[int] = []
        for t in const_tensors:
            # align
            while len(const_pool_bytes) % 4 != 0:
                const_pool_bytes.append(0)
            hex_blob = t.get("const_hex")
            if hex_blob:
                blob_bytes = parse_hex_blob(hex_blob)
                const_pool_bytes.extend(blob_bytes)
                # pad to dtype size already handled by outer alignment, but ensure blob itself is padded to 4
                while len(const_pool_bytes) % 4 != 0:
                    const_pool_bytes.append(0)
            else:
                # For non-hex consts (float literals), emit zeros as placeholder (size already accounted)
                # We still need to reserve space: nelems * dtype_size bytes of zeros
                dims = t["dims"]
                nelems = 1
                for d in dims:
                    nelems *= d if d != 0 else 1
                const_pool_bytes.extend([0] * (nelems * dtype_size(t["dtype"])))
                while len(const_pool_bytes) % 4 != 0:
                    const_pool_bytes.append(0)
        # Trim or pad to const_total_size
        const_pool_bytes = const_pool_bytes[:const_total_size]
        while len(const_pool_bytes) < const_total_size:
            const_pool_bytes.append(0)
        if const_total_size:
            elems = ", ".join(f"0x{b:02X}" for b in const_pool_bytes)
            header.append(f"    {elems}")
        header.append("};")
        header.append("")
        # Also emit per-const views for debugging (optional, not needed for pt)
        header.append("/* Per-const views (for debugging, pt is offset in above pool) */")
        for t in const_tensors:
            cname = sanitize_c_name(t["name"])
            pt = const_pt_map[t["name"]]
            dims = t["dims"]
            header.append(f"/* {cname}: {t['name']} {dims} x {t['dtype']} pt={pt} */")
        header.append("")
    else:
        header.append(f"static const uint8_t {stem}_l3_const_pool[1] = {{0}};")
        header.append("")

    header.append(f"/* Variable pool: unified L3 for activations (placed after const pool) */")
    header.append(f"static uint8_t {stem}_l3_var_pool[{total_var_size if total_var_size else 1}];")
    header.append("")
    header.append(f"/* Unified L3 view: const pool [0..{const_total_size}) + var pool [{const_total_size}..{const_total_size+total_var_size}) */")
    header.append("")

    # Combined tensors array with correct l3 pt (base + offset)
    header.append(f"/* Tensors: {len(tensors)} (const {len(const_tensors)} at const pool, var {len(variable_tensors)} at var pool) */")
    header.append(f"static jam_ir_tensor_t {stem}_tensors[{len(tensors)}] = {{")
    for idx, t in enumerate(tensors):
        dims = t["dims"]
        is_const = t.get("is_const", False)
        if is_const:
            pt = const_pt_map[t["name"]]
            l3_pt = f"(uintptr_t)&{stem}_l3_const_pool[{pt}]"
        else:
            pt = var_pt_map[t["name"]]
            l3_pt = f"(uintptr_t)&{stem}_l3_var_pool[{pt}]"
        header.append(f"    /* [{idx}] {t['name']} : {dims} x {t['dtype']} {'const' if is_const else 'var'} pt={l3_pt} */")
        header.append(f"    {{")
        header.append(_tensor_c_initializer(t, l3_pt, valid_l3=1))
        header.append(f"    }}{',' if idx < len(tensors)-1 else ''}")
    header.append("};")
    header.append("")

    header.append(f"/* Ops: num_ops = {len(jam_ops)} */")
    header.append(f"static jam_ir_op_t {stem}_ops[{len(jam_ops) if jam_ops else 1}] = {{")
    if not jam_ops:
        header.append("    {0}")
    else:
        for idx, op in enumerate(jam_ops):
            jam_type = op["type_enum"]
            activation = op.get("activation", 0)
            inputs = op.get("inputs", [])
            output = op.get("output")
            inp0 = inputs[0] if len(inputs) > 0 else 0
            inp1 = inputs[1] if len(inputs) > 1 else 0
            num_inputs = len(inputs)
            out0 = output if output is not None else 0
            num_outputs = 1 if output is not None else 0
            et = op.get("elementwise_type", [])
            et0 = et[0] if len(et) > 0 else 0
            et1 = et[1] if len(et) > 1 else 0
            fused_note = ""
            if op.get("fused_from"):
                fused_note = f" fused:{op['fused_from']}"
            raw_line = op.get("line", "")[:80].replace("*/", "* /")
            header.append(f"    /* [{idx}] {jam_type}{fused_note} ({raw_line}) */")
            header.append(f"    {{")
            header.append(f"        .type = {jam_type},")
            if jam_type == "JAM_IR_OP_CONV":
                sp = op["spatial"]
                k0, k1 = sp["kernel"][0], sp["kernel"][1]
                s0, s1 = sp["stride"][0], sp["stride"][1]
                p = sp["pad"]
                header.append(f"        .attributes = {{ .conv = {{ .spatial = {{ .kernel_size = {{{k0}, {k1}}}, .stride = {{{s0}, {s1}}}, .padding = {{{{{p[0]}, {p[1]}}}, {{{p[2]}, {p[3]}}}}}, .num_dims_spat = {sp['num_dims_spat']} }} }} }},")
            elif jam_type in ("JAM_IR_OP_AVG_POOL", "JAM_IR_OP_MAX_POOL"):
                sp = op["spatial"]
                k0, k1 = sp["kernel"][0], sp["kernel"][1]
                s0, s1 = sp["stride"][0], sp["stride"][1]
                p = sp["pad"]
                header.append(f"        .attributes = {{ .pool = {{ .spatial = {{ .kernel_size = {{{k0}, {k1}}}, .stride = {{{s0}, {s1}}}, .padding = {{{{{p[0]}, {p[1]}}}, {{{p[2]}, {p[3]}}}}}, .num_dims_spat = {sp['num_dims_spat']} }} }} }},")
            elif jam_type == "JAM_IR_OP_TRANSPOSE":
                axis = op.get("axis", 0)
                perms = op.get("perms", [])
                header.append(f"        /* perms = {perms} */")
                header.append(f"        .attributes = {{ .transpose = {{ .axis = {axis} }} }},")
            elif jam_type == "JAM_IR_OP_GEMM":
                gemm = op.get("gemm", {"trans_a": 0, "trans_b": 0})
                header.append(f"        .attributes = {{ .gemm = {{ .trans_a = {gemm['trans_a']}, .trans_b = {gemm['trans_b']} }} }},")
            elif jam_type == "JAM_IR_OP_MATMUL":
                gemm = op.get("gemm", {"trans_a": 0, "trans_b": 0})
                if gemm["trans_a"] or gemm["trans_b"]:
                    header.append(f"        /* MATMUL trans_a={gemm['trans_a']} trans_b={gemm['trans_b']} – handled via TRANSPOSE ops, union uses gemm */")
                header.append(f"        .attributes = {{ .gemm = {{ .trans_a = {gemm['trans_a']}, .trans_b = {gemm['trans_b']} }} }},")
            else:
                header.append(f"        .attributes = {{ .conv = {{ .spatial = {{ .kernel_size = {{0,0}}, .stride = {{0,0}}, .padding = {{{{0,0}}, {{0,0}}}}, .num_dims_spat = 0 }} }} }},")
            header.append(f"        .elementwise_type = {{{et0}, {et1}}},")
            header.append(f"        .activation_type = {activation},")
            header.append(f"        .input_refs = {{{inp0}, {inp1}}},")
            header.append(f"        .num_inputs = {num_inputs},")
            header.append(f"        .output_refs = {{{out0}}},")
            header.append(f"        .num_outputs = {num_outputs}")
            header.append(f"    }}{',' if idx < len(jam_ops)-1 else ''}")

    header.append("};")
    header.append("")

    var = stem
    header.append(f"static graph_t {var}_graph = {{")
    header.append(f"    .ops = {var}_ops,")
    header.append(f"    .num_ops = {len(jam_ops)},")
    header.append(f"    .tensors = {var}_tensors,")
    header.append(f"    .num_tensors = {len(tensors)}")
    header.append("};")
    header.append("")

    header.append(f"#endif /* {guard} */")
    header.append("")

    output_path.write_text("\n".join(header), encoding="utf-8")
    print(f"[tosa_to_jam_ir] wrote {output_path} : {len(tensors)} tensors ({len(const_tensors)} const + {len(variable_tensors)} var), {len(jam_ops)} ops (from {input_path})")
    if len(jam_ops) == 0:
        print("warning: no JAM ops generated; check TOSA op coverage", file=sys.stderr)

def main():
    p = argparse.ArgumentParser(description="Transform TOSA MLIR into JAM IR C header")
    p.add_argument("input", nargs="?", type=Path, help="Input TOSA MLIR file (e.g., mlp_tosa.mlir)")
    p.add_argument("-o", "--output", type=Path, default=None, help="Output header path (default: <input_basename>_jam.h)")
    p.add_argument("--input", dest="input2", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()

    inp = args.input or args.input2
    if inp is None:
        p.print_help()
        sys.exit(1)
    if not inp.exists():
        print(f"error: input file {inp} not found", file=sys.stderr)
        sys.exit(2)

    out = args.output
    if out is None:
        stem = inp.stem
        if stem.endswith("_tosa"):
            stem = stem + "_jam"
        else:
            stem = stem + "_jam"
        out = inp.with_name(stem + ".h")

    tensors, raw_ops, var_to_idx = parse_mlir(inp)
    print(f"[tosa_to_jam_ir] parsed {inp}: {len(tensors)} tensors, {len(raw_ops)} raw TOSA ops", file=sys.stderr)
    jam_ops = tosa_to_jam_ops(tensors, raw_ops, var_to_idx)
    print(f"[tosa_to_jam_ir] lowered to {len(jam_ops)} JAM ops", file=sys.stderr)

    generate_header(tensors, jam_ops, out, inp)

if __name__ == "__main__":
    main()
