#!/usr/bin/env python3

import json
import re
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ortools.sat.python import cp_model

###############################################################################
# 1) PARSE STM32 XML
###############################################################################


def parse_signal_name(signal_name):
    """
    Map ST-style signal names (e.g. 'I2C1_SCL') to (peripheral, signal).
    Trimmed for brevity; identical logic as your code.
    """
    m = re.match(r"I2C(\d+)_(SCL|SDA|SMBA)", signal_name)
    if m:
        return (f"i2c{m.group(1)}", m.group(2).lower())

    m = re.match(r"USART(\d+)_(TX|RX)", signal_name)
    if m:
        return (f"uart{m.group(1)}", m.group(2).lower())
    m = re.match(r"UART(\d+)_(TX|RX)", signal_name)
    if m:
        return (f"uart{m.group(1)}", m.group(2).lower())
    m = re.match(r"LPUART(\d+)_(TX|RX)", signal_name)
    if m:
        return (f"uart{m.group(1)}", m.group(2).lower())

    m = re.match(r"SPI(\d+)_(MOSI|MISO|SCK|CLK|NSS)", signal_name)
    if m:
        inst = m.group(1)
        sub = m.group(2).lower()
        if sub in ("sck", "clk"):
            sub = "clk"
        return (f"spi{inst}", sub)

    m = re.match(r"FDCAN(\d+)_(TX|RX)", signal_name)
    if m:
        return (f"fdcan{m.group(1)}", m.group(2).lower())

    if signal_name.startswith("ETH_"):
        sub = signal_name[4:].lower()
        eth_allowed = {
            "ref_clk",
            "tx_en",
            "txd0",
            "txd1",
            "crs_dv",
            "rxd0",
            "rxd1",
            "mdc",
            "mdio",
        }
        return ("eth", sub) if sub in eth_allowed else None

    m = re.match(r"SDMMC(\d+)_(.*)", signal_name)
    if m:
        inst = m.group(1)
        sub = m.group(2).lower()
        sdmmc_allowed = {"ck", "cmd", "d0", "d1", "d2", "d3"}
        return (f"sdmmc{inst}", sub) if sub in sdmmc_allowed else None

    m = re.match(r"USB_OTG_FS_(.*)", signal_name)
    if m:
        sub = m.group(1).lower()
        usbfs_allowed = {"dm", "dp", "sof", "vbus"}
        return ("usbfs", sub) if sub in usbfs_allowed else None

    m = re.match(r"USB_OTG_HS_(.*)", signal_name)
    if m:
        sub = m.group(1).lower()
        if sub in ("sof", "vbus"):
            return ("usbhs", sub)
        if sub.startswith("ulpi_"):
            return ("usbhs", sub)
        return None

    m = re.match(r"TIM(\d+)_(.*)", signal_name)
    if m:
        inst = m.group(1)
        remainder = m.group(2).lower()
        if remainder in ("ch1", "ch2", "ch3", "ch4"):
            return (f"tim{inst}", remainder)
        return None

    if signal_name == "DEBUG_JTCK-SWCLK":
        return ("swd", "swclk")
    if signal_name == "DEBUG_JTMS-SWDIO":
        return ("swd", "swdio")

    return None


def parse_stm32_pins(xml_path):
    """
    Reads the ST XML for the MCU, returning:
       pin_functions[pin_idx] = [ (peripheral, signal), ... ]
       pin_names[pin_idx] = e.g. "PA10"
       pin_positions[pin_idx] = e.g. "R16" (BGA position)
    Only pins of Type="I/O" or Type="MonoIO" are considered.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    namespace = "{http://dummy.com}"  # Adjust if your file truly uses a namespace
    valid_pin_types = {"I/O", "MonoIO"}

    pin_functions = {}
    pin_names = {}
    pin_positions = {}
    pin_index = 0

    for pin_el in root.findall(f"{namespace}Pin"):
        pin_type = pin_el.get("Type")
        if pin_type not in valid_pin_types:
            continue

        pin_name = pin_el.get("Name")
        pin_position = pin_el.get("Position")

        recognized_signals = []
        for sig_el in pin_el.findall(f"{namespace}Signal"):
            name_raw = sig_el.get("Name")
            parsed = parse_signal_name(name_raw)
            if parsed:
                recognized_signals.append(parsed)

        pin_functions[pin_index] = recognized_signals
        pin_names[pin_index] = pin_name
        pin_positions[pin_index] = pin_position
        pin_index += 1

    return pin_functions, pin_names, pin_positions


###############################################################################
# 2) THE OR-TOOLS CP-SAT SOLVER WITH 4 COST TERMS (equally weighted)
###############################################################################


def solve_pin_assignment(
    peripheral_requirements,
    pin_functions,
    pin_names,
    pin_positions,
    solution_limit=5,
    max_time_s=60.0,
):
    """
    We'll use 4 cost terms, all with weight=1:
      1) # of distinct ports used
      2) bounding-box spread
      3) max pairwise distance
      4) sum of bounding-box overlap areas among used instances

    final cost = (#ports) + (bboxSpread) + (maxDist) + (overlapArea).
    """

    model = cp_model.CpModel()

    # 1) define signals for each p_type
    peripheral_signals = {
        "i2c": ["scl", "sda"],
        "uart": ["tx", "rx"],
        "spi": ["clk", "miso", "mosi", "nss"],
        "fdcan": ["tx", "rx"],
        "eth": [
            "ref_clk",
            "tx_en",
            "txd0",
            "txd1",
            "crs_dv",
            "rxd0",
            "rxd1",
            "mdc",
            "mdio",
        ],
        "usbfs": ["dm", "dp", "sof", "vbus"],
        "usbhs": [
            "sof",
            "ulpi_ck",
            "ulpi_d0",
            "ulpi_d1",
            "ulpi_d2",
            "ulpi_d3",
            "ulpi_d4",
            "ulpi_d5",
            "ulpi_d6",
            "ulpi_d7",
            "ulpi_dir",
            "ulpi_stp",
            "ulpi_nxt",
        ],
        "tim": ["ch1", "ch2", "ch3", "ch4"],
        "swd": ["swclk", "swdio"],
        "sdmmc": ["ck", "cmd", "d0", "d1", "d2", "d3"],
    }

    def separate_peripheral_and_instance(full_name):
        for p_type in peripheral_signals:
            if full_name.startswith(p_type):
                return p_type, full_name[len(p_type) :]
        return None, None

    # 2) identify all (p_type, instance)
    available_peripherals = {pt: set() for pt in peripheral_signals}
    for pin_idx, funcs in pin_functions.items():
        for fullname, sig in funcs:
            p_type, suffix = separate_peripheral_and_instance(fullname)
            if p_type is not None and suffix is not None:
                m = re.match(r"(\d+)$", suffix)
                if m:
                    available_peripherals[p_type].add(m.group(1))
                else:
                    available_peripherals[p_type].add(suffix)

    # 3) filter "complete" instances
    complete_peripherals = {}
    for p_type, sigs in peripheral_signals.items():
        inst_list = []
        for suffix in sorted(available_peripherals[p_type]):
            name = p_type + suffix
            if all(
                any((name, s) in pin_functions[p] for p in pin_functions) for s in sigs
            ):
                inst_list.append(suffix)
        complete_peripherals[p_type] = inst_list

    # 4) feasibility check
    for p_type, needed_count in peripheral_requirements.items():
        if p_type not in complete_peripherals:
            print(f"[ERROR] {p_type} not recognized.")
            return None
        have = len(complete_peripherals[p_type])
        if have < needed_count:
            print(f"[ERROR] Not enough {p_type}. Need {needed_count}, found {have}.")
            return None

    # 5) create "used" bool
    peripheral_vars = {}
    for p_type, inst_list in complete_peripherals.items():
        needed_count = peripheral_requirements.get(p_type, 0)
        peripheral_vars[p_type] = {}
        for inst in inst_list:
            ubv = model.NewBoolVar(f"{p_type}{inst}_used")
            peripheral_vars[p_type][inst] = ubv
        if needed_count == 0:
            for inst in inst_list:
                model.Add(peripheral_vars[p_type][inst] == 0)
        else:
            model.Add(
                sum(peripheral_vars[p_type][inst] for inst in inst_list) == needed_count
            )

    # 6) for each (instance, signal) => pin or -1
    pin_vars = {}

    def valid_pins_for(fn, sig):
        return [p for p in pin_functions if (fn, sig) in pin_functions[p]]

    for p_type, inst_map in peripheral_vars.items():
        for inst, ubv in inst_map.items():
            fn = p_type + inst
            for sig in peripheral_signals[p_type]:
                feas = valid_pins_for(fn, sig)
                dvals = feas + [-1]
                xv = model.NewIntVarFromDomain(
                    cp_model.Domain.FromValues(dvals), f"{fn}_{sig}_pin"
                )
                pin_vars[(fn, sig)] = xv
                model.Add(xv != -1).OnlyEnforceIf(ubv)
                model.Add(xv == -1).OnlyEnforceIf(ubv.Not())

    # 7) "at most one signal per pin"
    assigned_bvar = {}
    for (fn, sig), xv in pin_vars.items():
        for pin_idx in pin_functions:
            b = model.NewBoolVar(f"pin{pin_idx}_{fn}_{sig}")
            assigned_bvar[(fn, sig, pin_idx)] = b
            model.Add(xv == pin_idx).OnlyEnforceIf(b)
            model.Add(xv != pin_idx).OnlyEnforceIf(b.Not())

    for pin_idx in pin_functions:
        these_b = []
        for fn, sig in pin_vars:
            these_b.append(assigned_bvar[(fn, sig, pin_idx)])
        model.AddAtMostOne(these_b)

    # 8) cost #1 => #ports
    def extract_port(pn):
        m = re.match(r"^P([A-Z])\d+", pn)
        return m.group(1) if m else "?"

    port_of_pin = {p: extract_port(pin_names[p]) for p in pin_functions}
    all_ports = sorted(set(port_of_pin.values()))

    instance_uses_port = {}
    for p_type, inst_map in peripheral_vars.items():
        for inst, ubv in inst_map.items():
            fn = p_type + inst
            for port in all_ports:
                iup = model.NewBoolVar(f"{fn}_uses_port_{port}")
                instance_uses_port[(p_type, inst, port)] = iup
                model.Add(iup == 0).OnlyEnforceIf(ubv.Not())

            for port in all_ports:
                iup = instance_uses_port[(p_type, inst, port)]
                or_bvars = []
                for sig in peripheral_signals[p_type]:
                    feas = valid_pins_for(fn, sig)
                    p_b = [
                        assigned_bvar[(fn, sig, pidx)]
                        for pidx in feas
                        if port_of_pin[pidx] == port
                    ]
                    if p_b:
                        tmp_b = model.NewBoolVar(f"{fn}_{sig}_port_{port}")
                        model.AddBoolOr(p_b).OnlyEnforceIf(tmp_b)
                        for pb_ in p_b:
                            model.Add(pb_ == 0).OnlyEnforceIf(tmp_b.Not())
                        or_bvars.append(tmp_b)
                if or_bvars:
                    model.AddBoolOr(or_bvars).OnlyEnforceIf(iup)
                    for ob_ in or_bvars:
                        model.Add(ob_ == 0).OnlyEnforceIf(iup.Not())
                else:
                    model.Add(iup == 0)

    total_port_usage = model.NewIntVar(0, 999999, "port_usage")
    model.Add(
        total_port_usage == sum(instance_uses_port[k] for k in instance_uses_port)
    )

    # 9) cost #2 => bounding-box spread
    bga_rows = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6,
        "H": 7,
        "J": 8,
        "K": 9,
        "L": 10,
        "M": 11,
        "N": 12,
        "P": 13,
        "R": 14,
        "T": 15,
        "U": 16,
    }

    def parse_bga_position(pos):
        m = re.match(r"^([A-Z])(\d+)$", pos)
        if not m:
            return (0, 0)
        rr = bga_rows.get(m.group(1), 0)
        cc = int(m.group(2))
        return (rr, cc)

    pin_row = {}
    pin_col = {}
    for pidx in pin_functions:
        (r, c) = parse_bga_position(pin_positions[pidx])
        pin_row[pidx] = r
        pin_col[pidx] = c

    row_vars = {}
    col_vars = {}
    for (fn, sig), xv in pin_vars.items():
        rv = model.NewIntVar(0, 500, f"{fn}_{sig}_row")
        cv = model.NewIntVar(0, 500, f"{fn}_{sig}_col")
        row_vars[(fn, sig)] = rv
        col_vars[(fn, sig)] = cv
        model.Add(
            rv
            == sum(
                assigned_bvar[(fn, sig, pidx)] * pin_row[pidx] for pidx in pin_functions
            )
        )
        model.Add(
            cv
            == sum(
                assigned_bvar[(fn, sig, pidx)] * pin_col[pidx] for pidx in pin_functions
            )
        )

    instance_bbox_spread = {}
    # [NEW OVERLAP LOGIC START]
    # We'll store rMin, rMaxPlus1, cMin, cMaxPlus1 for later overlap computations.
    instance_rMin = {}
    instance_rMaxPlus1 = {}
    instance_cMin = {}
    instance_cMaxPlus1 = {}
    # [NEW OVERLAP LOGIC END]

    for p_type, inst_map in peripheral_vars.items():
        for inst, ubv in inst_map.items():
            fn = p_type + inst
            rMin = model.NewIntVar(0, 500, f"{fn}_rMin")
            rMax = model.NewIntVar(0, 500, f"{fn}_rMax")
            cMin = model.NewIntVar(0, 500, f"{fn}_cMin")
            cMax = model.NewIntVar(0, 500, f"{fn}_cMax")

            sigs = peripheral_signals[p_type]
            for s in sigs:
                rv = row_vars[(fn, s)]
                cv = col_vars[(fn, s)]
                model.Add(rv >= rMin)
                model.Add(rv <= rMax)
                model.Add(cv >= cMin)
                model.Add(cv <= cMax)

            spr = model.NewIntVar(0, 1000, f"{fn}_bboxSpread")
            model.Add(spr == (rMax - rMin) + (cMax - cMin))

            final_spr = model.NewIntVar(0, 1000, f"{fn}_bboxSpreadUsed")
            instance_bbox_spread[(p_type, inst)] = final_spr
            model.Add(final_spr == spr).OnlyEnforceIf(ubv)
            model.Add(final_spr == 0).OnlyEnforceIf(ubv.Not())

            # [NEW OVERLAP LOGIC START] - define rMaxPlus1, cMaxPlus1
            rMaxP1 = model.NewIntVar(0, 501, f"{fn}_rMaxPlus1")
            cMaxP1 = model.NewIntVar(0, 501, f"{fn}_cMaxPlus1")
            model.Add(rMaxP1 == rMax + 1)
            model.Add(cMaxP1 == cMax + 1)
            instance_rMin[(p_type, inst)] = rMin
            instance_rMaxPlus1[(p_type, inst)] = rMaxP1
            instance_cMin[(p_type, inst)] = cMin
            instance_cMaxPlus1[(p_type, inst)] = cMaxP1
            # [NEW OVERLAP LOGIC END]

    total_bbox_spread = model.NewIntVar(0, 999999, "bbox_spread")
    model.Add(
        total_bbox_spread == sum(instance_bbox_spread[k] for k in instance_bbox_spread)
    )

    # 10) cost #3 => max pairwise distance
    instance_max_dist = {}
    for p_type, inst_map in peripheral_vars.items():
        for inst, ubv in inst_map.items():
            fn = p_type + inst
            sigs = peripheral_signals[p_type]
            maxD = model.NewIntVar(0, 1000, f"{fn}_maxDist")
            instance_max_dist[(p_type, inst)] = maxD
            distvars = []
            for i in range(len(sigs)):
                for j in range(i + 1, len(sigs)):
                    sA = sigs[i]
                    sB = sigs[j]
                    rA = row_vars[(fn, sA)]
                    cA = col_vars[(fn, sA)]
                    rB = row_vars[(fn, sB)]
                    cB = col_vars[(fn, sB)]
                    rd = model.NewIntVar(0, 500, f"{fn}_{sA}_{sB}_rdiff")
                    cd = model.NewIntVar(0, 500, f"{fn}_{sA}_{sB}_cdiff")
                    dd = model.NewIntVar(0, 1000, f"{fn}_{sA}_{sB}_dist")
                    model.Add(rd >= rA - rB)
                    model.Add(rd >= rB - rA)
                    model.Add(cd >= cA - cB)
                    model.Add(cd >= cB - cA)
                    model.Add(dd == rd + cd)
                    distvars.append(dd)
            for dv in distvars:
                model.Add(maxD >= dv)
            model.Add(maxD == 0).OnlyEnforceIf(ubv.Not())

    total_max_dist = model.NewIntVar(0, 999999, "max_dist_sum")
    model.Add(total_max_dist == sum(instance_max_dist[k] for k in instance_max_dist))

    # [NEW OVERLAP LOGIC START]
    # 11) cost #4 => bounding-box overlap area
    instance_list = []
    used_bool = {}
    for p_type, inst_map in peripheral_vars.items():
        for inst, ubv in inst_map.items():
            instance_list.append((p_type, inst))
            used_bool[(p_type, inst)] = ubv

    overlap_terms = []
    for i in range(len(instance_list)):
        for j in range(i + 1, len(instance_list)):
            i_type, i_inst = instance_list[i]
            j_type, j_inst = instance_list[j]
            i_used = used_bool[(i_type, i_inst)]
            j_used = used_bool[(j_type, j_inst)]

            # bothUsed = i_used AND j_used
            bothUsed = model.NewBoolVar(f"bothUsed_{i_type}{i_inst}_{j_type}{j_inst}")
            # i_used=1 and j_used=1 => bothUsed=1
            # We'll do a standard linear "AND" trick:
            # bothUsed <= i_used, bothUsed <= j_used
            model.Add(bothUsed <= i_used)
            model.Add(bothUsed <= j_used)
            # bothUsed >= i_used + j_used - 1
            model.Add(bothUsed >= i_used + j_used - 1)

            # define the overlap rectangle
            i_rMin = instance_rMin[(i_type, i_inst)]
            i_rMaxP1 = instance_rMaxPlus1[(i_type, i_inst)]
            j_rMin = instance_rMin[(j_type, j_inst)]
            j_rMaxP1 = instance_rMaxPlus1[(j_type, j_inst)]

            i_cMin = instance_cMin[(i_type, i_inst)]
            i_cMaxP1 = instance_cMaxPlus1[(i_type, i_inst)]
            j_cMin = instance_cMin[(j_type, j_inst)]
            j_cMaxP1 = instance_cMaxPlus1[(j_type, j_inst)]

            # overlapRowMin = max(rMin_i, rMin_j)
            overlapRowMin = model.NewIntVar(
                0, 501, f"rowMin_{i_type}{i_inst}_{j_type}{j_inst}"
            )
            model.AddMaxEquality(overlapRowMin, [i_rMin, j_rMin])

            # overlapRowMax = min(rMaxP1_i, rMaxP1_j)
            overlapRowMax = model.NewIntVar(
                0, 501, f"rowMax_{i_type}{i_inst}_{j_type}{j_inst}"
            )
            model.AddMinEquality(overlapRowMax, [i_rMaxP1, j_rMaxP1])

            # overlapColMin = max(cMin_i, cMin_j)
            overlapColMin = model.NewIntVar(
                0, 501, f"colMin_{i_type}{i_inst}_{j_type}{j_inst}"
            )
            model.AddMaxEquality(overlapColMin, [i_cMin, j_cMin])

            # overlapColMax = min(cMaxP1_i, cMaxP1_j)
            overlapColMax = model.NewIntVar(
                0, 501, f"colMax_{i_type}{i_inst}_{j_type}{j_inst}"
            )
            model.AddMinEquality(overlapColMax, [i_cMaxP1, j_cMaxP1])

            # overlapHeightCandidate = overlapRowMax - overlapRowMin
            overlapHeightCandidate = model.NewIntVar(
                -500, 501, f"overlapHcand_{i_type}{i_inst}_{j_type}{j_inst}"
            )
            model.Add(overlapHeightCandidate == overlapRowMax - overlapRowMin)

            overlapHeight = model.NewIntVar(
                0, 501, f"overlapH_{i_type}{i_inst}_{j_type}{j_inst}"
            )
            model.Add(overlapHeight >= overlapHeightCandidate)
            model.Add(overlapHeight >= 0)

            # overlapWidthCandidate = overlapColMax - overlapColMin
            overlapWidthCandidate = model.NewIntVar(
                -500, 501, f"overlapWcand_{i_type}{i_inst}_{j_type}{j_inst}"
            )
            model.Add(overlapWidthCandidate == overlapColMax - overlapColMin)

            overlapWidth = model.NewIntVar(
                0, 501, f"overlapW_{i_type}{i_inst}_{j_type}{j_inst}"
            )
            model.Add(overlapWidth >= overlapWidthCandidate)
            model.Add(overlapWidth >= 0)

            # overlapAreaCandidate = overlapHeight * overlapWidth
            overlapAreaCandidate = model.NewIntVar(
                0, 501 * 501, f"overlapArea_{i_type}{i_inst}_{j_type}{j_inst}"
            )
            model.AddMultiplicationEquality(
                overlapAreaCandidate, [overlapHeight, overlapWidth]
            )

            # finalOverlap = overlapAreaCandidate if bothUsed=1 else 0
            finalOverlap = model.NewIntVar(
                0, 501 * 501, f"finalOverlap_{i_type}{i_inst}_{j_type}{j_inst}"
            )
            model.Add(finalOverlap == overlapAreaCandidate).OnlyEnforceIf(bothUsed)
            model.Add(finalOverlap == 0).OnlyEnforceIf(bothUsed.Not())

            overlap_terms.append(finalOverlap)

    total_overlap_area = model.NewIntVar(0, 501 * 501 * 100, "totalOverlapArea")
    model.Add(total_overlap_area == sum(overlap_terms))
    # [NEW OVERLAP LOGIC END]

    # 12) final cost => #ports + bboxSpread + maxDist + overlapArea
    cost_var = model.NewIntVar(0, 9999999, "total_cost")
    model.Add(
        cost_var
        == (
            total_port_usage
            + total_bbox_spread * 2
            + total_max_dist * 2
            + total_overlap_area * 5  # [NEW]
        )
    )
    model.Minimize(cost_var)

    # 13) Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time_s
    status = solver.Solve(model)
    print(f"Solved with status={status}, cost={solver.Value(cost_var)}")

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print("No feasible solution found.")
        return None

    # Build a structured result
    solution_data = {
        "status": status,
        "cost": solver.Value(cost_var),
        "ports_used": solver.Value(total_port_usage),
        "bbox_spread": solver.Value(total_bbox_spread),
        "max_dist_sum": solver.Value(total_max_dist),
        "overlap_area": solver.Value(total_overlap_area),  # [NEW]
        "instances": {},
    }

    # gather instance data
    for p_type, inst_map in peripheral_vars.items():
        for inst, ubv in inst_map.items():
            if solver.Value(ubv) == 1:
                fn = p_type + inst
                # ports used
                used_ports = []
                for port in all_ports:
                    if solver.Value(instance_uses_port[(p_type, inst, port)]) == 1:
                        used_ports.append(port)
                bb_sp = solver.Value(instance_bbox_spread[(p_type, inst)])
                mx_d = solver.Value(instance_max_dist[(p_type, inst)])
                # pin assignments
                pin_assign = {}
                for sig in peripheral_signals[p_type]:
                    xv = pin_vars[(fn, sig)]
                    val = solver.Value(xv)
                    if val >= 0:
                        pin_assign[sig] = {
                            "pin_name": pin_names[val],
                            "position": pin_positions[val],
                        }

                solution_data["instances"][fn] = {
                    "ports_used": used_ports,
                    "bounding_box_spread": bb_sp,
                    "max_pairwise_distance": mx_d,
                    "pin_assignments": pin_assign,
                }

    return solution_data


###############################################################################
# 3) VISUALIZATION
###############################################################################


def visualize_bga(solution_data, pin_names, pin_positions):
    """
    Identical to your existing visualize code. We just optionally display
    'overlap_area' from solution_data in the title or so.
    """

    if not solution_data or "instances" not in solution_data:
        print("No solution to visualize.")
        return

    bga_rows = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6,
        "H": 7,
        "J": 8,
        "K": 9,
        "L": 10,
        "M": 11,
        "N": 12,
        "P": 13,
        "R": 14,
        "T": 15,
        "U": 16,
    }

    def parse_bga_position(pos):
        m = re.match(r"^([A-Z])(\d+)$", pos)
        if not m:
            return (0, 0)
        rr = bga_rows.get(m.group(1), 0)
        cc = int(m.group(2))
        return (rr, cc)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect="equal")

    color_cycle = plt.cm.get_cmap("tab10")
    instance_list = list(solution_data["instances"].keys())

    inst_colors = {}
    for i, inst_name in enumerate(instance_list):
        color = color_cycle(i % 10)
        inst_colors[inst_name] = color

    all_rows, all_cols = [], []
    bounding_boxes = {}

    for inst_name in instance_list:
        inst_data = solution_data["instances"][inst_name]
        assigned_pins = inst_data["pin_assignments"]
        if not assigned_pins:
            continue

        rows = []
        cols = []
        for sig, info in assigned_pins.items():
            pos = info["position"]
            (rr, cc) = parse_bga_position(pos)
            rows.append(rr)
            cols.append(cc)

        if rows and cols:
            rMin, rMax = min(rows), max(rows)
            cMin, cMax = min(cols), max(cols)
        else:
            rMin, rMax, cMin, cMax = 0, 0, 0, 0

        width = max(0, (cMax - cMin))
        height = max(0, (rMax - rMin))
        bounding_boxes[inst_name] = (cMin, rMin, width, height)

        for sig, info in assigned_pins.items():
            pos = info["position"]
            (rr, cc) = parse_bga_position(pos)
            all_rows.append(rr)
            all_cols.append(cc)

    # Plot
    for inst_name in instance_list:
        color = inst_colors[inst_name]
        inst_data = solution_data["instances"][inst_name]
        assigned_pins = inst_data["pin_assignments"]
        if not assigned_pins:
            continue

        # scatter pins
        for sig, info in assigned_pins.items():
            pos = info["position"]
            (rr, cc) = parse_bga_position(pos)
            ax.scatter(cc, rr, color=color)
            ax.text(cc + 0.2, rr + 0.2, f"{inst_name}:{sig}", fontsize=8, color=color)

        # bounding box
        (x0, y0, w, h) = bounding_boxes[inst_name]
        rect = Rectangle(
            (x0, y0),
            w,
            h,
            fill=True,
            alpha=0.2,
            edgecolor=color,
            facecolor=color,
            linewidth=1.5,
        )
        ax.add_patch(rect)
        rect2 = Rectangle(
            (x0, y0), w, h, fill=False, alpha=0.8, edgecolor=color, linewidth=1.0
        )
        ax.add_patch(rect2)

    if all_rows and all_cols:
        ax.set_xlim(min(all_cols) - 1, max(all_cols) + 2)
        ax.set_ylim(min(all_rows) - 1, max(all_rows) + 2)
    else:
        ax.set_xlim(-1, 10)
        ax.set_ylim(-1, 10)

    ax.invert_yaxis()
    ax.set_xlabel("BGA Column")
    ax.set_ylabel("BGA Row")
    # Show overlap area in title, if you like:
    overlap_val = solution_data.get("overlap_area", 0)
    ax.set_title(f"BGA Pin Assignments (overlapArea={overlap_val})")

    bga_row_letters = list(bga_rows.keys())
    ax.set_yticks(range(17), labels=bga_row_letters)
    ax.set_xticks(range(1, 18), labels=[str(i) for i in range(1, 18)])

    plt.show()


###############################################################################
# 4) EXAMPLE MAIN
###############################################################################


def main():
    xml_path = "pin_data/STM32H747XIHx.xml"
    pin_funcs, pin_names, pin_positions = parse_stm32_pins(xml_path)

    print("\nParsed pins with recognized signals:")
    for idx, sigs in pin_funcs.items():
        if sigs:
            print(f"  Pin {pin_names[idx]} ({pin_positions[idx]}): {sigs}")

    reqs = {
        "eth": 1,
        "usbfs": 1,
        "sdmmc": 1,
        "swd": 1,
        "fdcan": 2,
        "uart": 6,
        "spi": 2,
        "tim": 4,
        "i2c": 3,
    }

    solution_data = solve_pin_assignment(
        reqs,
        pin_funcs,
        pin_names,
        pin_positions,
        solution_limit=50,
        max_time_s=60.0,
    )
    if solution_data:
        print("\nSolution found!\n")
        print(json.dumps(solution_data, indent=2))
        visualize_bga(solution_data, pin_names, pin_positions)


if __name__ == "__main__":
    main()
