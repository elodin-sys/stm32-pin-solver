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
    Adapted from your original.
    """
    # I2C
    m = re.match(r"I2C(\d+)_(SCL|SDA|SMBA)", signal_name)
    if m:
        return (f"i2c{m.group(1)}", m.group(2).lower())

    # UART/USART/LPUART
    m = re.match(r"USART(\d+)_(TX|RX)", signal_name)
    if m:
        return (f"uart{m.group(1)}", m.group(2).lower())
    m = re.match(r"UART(\d+)_(TX|RX)", signal_name)
    if m:
        return (f"uart{m.group(1)}", m.group(2).lower())
    m = re.match(r"LPUART(\d+)_(TX|RX)", signal_name)
    if m:
        return (f"uart{m.group(1)}", m.group(2).lower())

    # SPI
    m = re.match(r"SPI(\d+)_(MOSI|MISO|SCK|CLK|NSS)", signal_name)
    if m:
        inst = m.group(1)
        sub = m.group(2).lower()
        if sub in ("sck", "clk"):
            sub = "clk"
        return (f"spi{inst}", sub)

    # FDCAN
    m = re.match(r"FDCAN(\d+)_(TX|RX)", signal_name)
    if m:
        return (f"fdcan{m.group(1)}", m.group(2).lower())

    # Ethernet
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

    # SDMMC
    m = re.match(r"SDMMC(\d+)_(.*)", signal_name)
    if m:
        inst = m.group(1)
        sub = m.group(2).lower()
        sdmmc_allowed = {"ck", "cmd", "d0", "d1", "d2", "d3"}
        return (f"sdmmc{m.group(1)}", sub) if sub in sdmmc_allowed else None

    # USB FS
    m = re.match(r"USB_OTG_FS_(.*)", signal_name)
    if m:
        sub = m.group(1).lower()
        usbfs_allowed = {"dm", "dp", "sof", "vbus"}
        return ("usbfs", sub) if sub in usbfs_allowed else None

    # USB HS
    m = re.match(r"USB_OTG_HS_(.*)", signal_name)
    if m:
        sub = m.group(1).lower()
        if sub in ("sof", "vbus"):
            return ("usbhs", sub)
        if sub.startswith("ulpi_"):
            return ("usbhs", sub)
        return None

    # TIM
    m = re.match(r"TIM(\d+)_(.*)", signal_name)
    if m:
        inst = m.group(1)
        remainder = m.group(2).lower()
        if remainder in ("ch1", "ch2", "ch3", "ch4"):
            return (f"tim{inst}", remainder)
        return None

    # SWD
    if signal_name == "DEBUG_JTCK-SWCLK":
        return ("swd", "swclk")
    if signal_name == "DEBUG_JTMS-SWDIO":
        return ("swd", "swdio")

    return None


def parse_stm32_pins(xml_path):
    """
    Reads the ST XML for the MCU, returning three dictionaries:
       pin_functions[pin_idx] = [ (peripheral, signal), ... ]
       pin_names[pin_idx] = "pin name" (e.g., "PA10")
       pin_positions[pin_idx] = "position" (e.g., "R16")
    Only pins of Type="I/O" or Type="MonoIO" are considered.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # If your XML uses a namespace, update this accordingly
    namespace = "{http://dummy.com}"
    valid_pin_types = {"I/O", "MonoIO"}

    pin_functions = {}
    pin_names = {}
    pin_positions = {}
    pin_index = 0

    for pin_el in root.findall(f"{namespace}Pin"):
        pin_type = pin_el.get("Type")
        if pin_type not in valid_pin_types:
            continue  # skip power, reset, etc.

        pin_name = pin_el.get("Name")
        pin_position = pin_el.get("Position")  # e.g., "R16" in a BGA

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
# 2) THE OR-TOOLS CP-SAT SOLVER WITH 3 COST TERMS
###############################################################################


def solve_pin_assignment(
    peripheral_requirements,
    pin_functions,
    pin_names,
    pin_positions,
    solution_limit=5,
    max_time_s=60.0,
    weights=None,
):
    """
    We incorporate three cost terms:
     1) # of distinct ports used by each instance
     2) bounding-box spread (rowMax-rowMin + colMax-colMin)
     3) max pairwise distance for each instance
    combined with weights: weights['ports'], weights['bbox'], weights['maxd'].

    Returns a structured result dict with full assignment info.
    """

    # Default all weights=1 if not specified
    if weights is None:
        weights = {"ports": 1, "bbox": 1, "maxd": 1}

    w_ports = weights.get("ports", 1)
    w_bbox = weights.get("bbox", 1)
    w_maxd = weights.get("maxd", 1)

    # 1) Known signals
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

    model = cp_model.CpModel()

    def separate_peripheral_and_instance(full_name):
        for p_type in peripheral_signals:
            if full_name.startswith(p_type):
                return p_type, full_name[len(p_type) :]
        return None, None

    # 2) Identify all (p_type, instance)
    available_peripherals = {pt: set() for pt in peripheral_signals}
    for pin_idx, funcs in pin_functions.items():
        for fullname, sig in funcs:
            p_type, suffix = separate_peripheral_and_instance(fullname)
            if p_type is not None and suffix is not None:
                # parse trailing digits if any
                m = re.match(r"(\d+)$", suffix)
                if m:
                    available_peripherals[p_type].add(m.group(1))
                else:
                    available_peripherals[p_type].add(suffix)

    # 3) Keep only "complete" instances with all signals
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

    # 4) Feasibility check
    for p_type, needed_count in peripheral_requirements.items():
        if p_type not in complete_peripherals:
            print(f"[ERROR] Peripheral '{p_type}' not recognized.")
            return None
        have = len(complete_peripherals[p_type])
        if have < needed_count:
            print(
                f"[ERROR] Not enough {p_type} instances. Need {needed_count}, found {have}."
            )
            return None

    # 5) Create "used" bool var for each instance
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

    # 6) For each instance+signal => IntVar pin assignment (or -1)
    pin_vars = {}

    def valid_pins_for(full_pname, sig):
        return [p for p in pin_functions if (full_pname, sig) in pin_functions[p]]

    for p_type, inst_map in peripheral_vars.items():
        for inst, used_bv in inst_map.items():
            fullname = p_type + inst
            for sig in peripheral_signals[p_type]:
                feas = valid_pins_for(fullname, sig)
                domain_vals = feas + [-1]
                xvar = model.NewIntVarFromDomain(
                    cp_model.Domain.FromValues(domain_vals), f"{fullname}_{sig}_pin"
                )
                pin_vars[(fullname, sig)] = xvar
                model.Add(xvar != -1).OnlyEnforceIf(used_bv)
                model.Add(xvar == -1).OnlyEnforceIf(used_bv.Not())

    # 7) At most one signal per pin
    assigned_bvar = {}
    for (fullname, sig), xvar in pin_vars.items():
        for pin_idx in pin_functions:
            b = model.NewBoolVar(f"pin{pin_idx}_{fullname}_{sig}")
            assigned_bvar[(fullname, sig, pin_idx)] = b
            model.Add(xvar == pin_idx).OnlyEnforceIf(b)
            model.Add(xvar != pin_idx).OnlyEnforceIf(b.Not())

    for pin_idx in pin_functions:
        bvars_this_pin = []
        for fp, s in pin_vars:
            bvars_this_pin.append(assigned_bvar[(fp, s, pin_idx)])
        model.AddAtMostOne(bvars_this_pin)

    # -----------------------------------------------------------------------
    # 8) COST #1: # of distinct ports used
    # -----------------------------------------------------------------------
    def extract_port(pin_name):
        m = re.match(r"^P([A-Z])\d+", pin_name)
        return m.group(1) if m else "?"

    port_of_pin = {p: extract_port(pin_names[p]) for p in pin_functions}
    all_ports = sorted(set(port_of_pin.values()))

    instance_uses_port = {}
    for p_type, inst_map in peripheral_vars.items():
        for inst, used_bv in inst_map.items():
            fullname = p_type + inst
            for port in all_ports:
                iup = model.NewBoolVar(f"{fullname}_uses_port_{port}")
                instance_uses_port[(p_type, inst, port)] = iup
                model.Add(iup == 0).OnlyEnforceIf(used_bv.Not())

            # iup=1 if any signal is pinned in that port
            for port in all_ports:
                iup = instance_uses_port[(p_type, inst, port)]
                or_bvars = []
                for sig in peripheral_signals[p_type]:
                    feas_pins = valid_pins_for(fullname, sig)
                    p_bvars = [
                        assigned_bvar[(fullname, sig, pidx)]
                        for pidx in feas_pins
                        if port_of_pin[pidx] == port
                    ]
                    if p_bvars:
                        sbv = model.NewBoolVar(f"{fullname}_{sig}_port_{port}")
                        model.AddBoolOr(p_bvars).OnlyEnforceIf(sbv)
                        for pb in p_bvars:
                            model.Add(pb == 0).OnlyEnforceIf(sbv.Not())
                        or_bvars.append(sbv)
                if or_bvars:
                    model.AddBoolOr(or_bvars).OnlyEnforceIf(iup)
                    for ob in or_bvars:
                        model.Add(ob == 0).OnlyEnforceIf(iup.Not())
                else:
                    model.Add(iup == 0)

    total_port_usage = model.NewIntVar(0, 999999, "total_port_usage")
    model.Add(
        total_port_usage == sum(instance_uses_port[k] for k in instance_uses_port)
    )

    # -----------------------------------------------------------------------
    # 9) COST #2: bounding-box "spread"
    # -----------------------------------------------------------------------
    # parse row,col from pin_positions
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
        # e.g. 'R16' => row=14, col=16
        m = re.match(r"^([A-Z])(\d+)$", pos)
        if not m:
            return (0, 0)
        row_let = m.group(1)
        col_str = m.group(2)
        rr = bga_rows.get(row_let, 0)
        cc = int(col_str)
        return (rr, cc)

    pin_row = {}
    pin_col = {}
    for pidx in pin_functions:
        (r, c) = parse_bga_position(pin_positions[pidx])
        pin_row[pidx] = r
        pin_col[pidx] = c

    # rowVar/colVar for each signal
    row_vars = {}
    col_vars = {}
    for (fullname, sig), xvar in pin_vars.items():
        rv = model.NewIntVar(0, 500, f"{fullname}_{sig}_row")
        cv = model.NewIntVar(0, 500, f"{fullname}_{sig}_col")
        row_vars[(fullname, sig)] = rv
        col_vars[(fullname, sig)] = cv

        model.Add(
            rv
            == sum(
                assigned_bvar[(fullname, sig, pidx)] * pin_row[pidx]
                for pidx in pin_functions
            )
        )
        model.Add(
            cv
            == sum(
                assigned_bvar[(fullname, sig, pidx)] * pin_col[pidx]
                for pidx in pin_functions
            )
        )

    # bounding box
    instance_bbox_spread = {}
    for p_type, inst_map in peripheral_vars.items():
        for inst, used_bv in inst_map.items():
            fn = p_type + inst
            rMin = model.NewIntVar(0, 500, f"{fn}_rMin")
            rMax = model.NewIntVar(0, 500, f"{fn}_rMax")
            cMin = model.NewIntVar(0, 500, f"{fn}_cMin")
            cMax = model.NewIntVar(0, 500, f"{fn}_cMax")

            # link each signal's row/col to (rMin,rMax,cMin,cMax)
            sigs = peripheral_signals[p_type]
            for s in sigs:
                rv = row_vars[(fn, s)]
                cv = col_vars[(fn, s)]
                model.Add(rv >= rMin)
                model.Add(rv <= rMax)
                model.Add(cv >= cMin)
                model.Add(cv <= cMax)

            # spread = (rMax-rMin)+(cMax-cMin)
            spr = model.NewIntVar(0, 1000, f"{fn}_bboxSpread")
            model.Add(spr == (rMax - rMin) + (cMax - cMin))

            # final spread if used, else 0
            final_spr = model.NewIntVar(0, 1000, f"{fn}_bboxSpreadUsed")
            instance_bbox_spread[(p_type, inst)] = final_spr
            model.Add(final_spr == spr).OnlyEnforceIf(used_bv)
            model.Add(final_spr == 0).OnlyEnforceIf(used_bv.Not())

    total_bbox_spread = model.NewIntVar(0, 999999, "total_bbox_spread")
    model.Add(
        total_bbox_spread == sum(instance_bbox_spread[k] for k in instance_bbox_spread)
    )

    # -----------------------------------------------------------------------
    # 10) COST #3: max pairwise distance
    # -----------------------------------------------------------------------
    instance_max_dist = {}
    for p_type, inst_map in peripheral_vars.items():
        for inst, used_bv in inst_map.items():
            fn = p_type + inst
            sigs = peripheral_signals[p_type]
            maxD = model.NewIntVar(0, 1000, f"{fn}_maxDist")
            instance_max_dist[(p_type, inst)] = maxD

            pair_dists = []
            for i in range(len(sigs)):
                for j in range(i + 1, len(sigs)):
                    sA = sigs[i]
                    sB = sigs[j]
                    rA = row_vars[(fn, sA)]
                    cA = col_vars[(fn, sA)]
                    rB = row_vars[(fn, sB)]
                    cB = col_vars[(fn, sB)]
                    row_diff = model.NewIntVar(0, 500, f"{fn}_{sA}_{sB}_rdiff")
                    col_diff = model.NewIntVar(0, 500, f"{fn}_{sA}_{sB}_cdiff")
                    dist = model.NewIntVar(0, 1000, f"{fn}_{sA}_{sB}_dist")
                    model.Add(row_diff >= rA - rB)
                    model.Add(row_diff >= rB - rA)
                    model.Add(col_diff >= cA - cB)
                    model.Add(col_diff >= cB - cA)
                    model.Add(dist == row_diff + col_diff)
                    pair_dists.append(dist)

            # maxD >= each dist
            for dvar in pair_dists:
                model.Add(maxD >= dvar)

            # if not used => maxD=0
            model.Add(maxD == 0).OnlyEnforceIf(used_bv.Not())

    total_max_dist = model.NewIntVar(0, 999999, "total_max_dist")
    model.Add(total_max_dist == sum(instance_max_dist[k] for k in instance_max_dist))

    # -----------------------------------------------------------------------
    # 11) Combine cost
    # -----------------------------------------------------------------------
    cost_var = model.NewIntVar(0, 9999999, "cost")
    model.Add(
        cost_var
        == w_ports * total_port_usage
        + w_bbox * total_bbox_spread
        + w_maxd * total_max_dist
    )
    model.Minimize(cost_var)

    # -----------------------------------------------------------------------
    # 12) Solve
    # -----------------------------------------------------------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time_s

    print(f"\nSolving with weights={weights}")
    status = solver.Solve(model)
    print(
        f"Solver status={status} (OPTIMAL={cp_model.OPTIMAL}, FEASIBLE={cp_model.FEASIBLE})"
    )
    print(f"Solved in {solver.WallTime():.3f}s, cost={solver.Value(cost_var)}")

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print("No feasible solution found.")
        return None

    # Gather structured results
    # e.g. a dictionary:
    solution_data = {
        "status": status,
        "cost": solver.Value(cost_var),
        "ports_used": solver.Value(total_port_usage),
        "bbox_spread": solver.Value(total_bbox_spread),
        "max_dist_sum": solver.Value(total_max_dist),
        "instances": {},
    }

    # Collect info per used instance
    for p_type, inst_map in peripheral_vars.items():
        for inst, ubv in inst_map.items():
            if solver.Value(ubv) == 1:
                fn = p_type + inst
                used_ports = []
                for port in all_ports:
                    if solver.Value(instance_uses_port[(p_type, inst, port)]) == 1:
                        used_ports.append(port)
                bb_spread = solver.Value(instance_bbox_spread[(p_type, inst)])
                max_d = solver.Value(instance_max_dist[(p_type, inst)])

                # Pin assignments
                pin_assignments = {}
                for sig in peripheral_signals[p_type]:
                    xvar = pin_vars[(fn, sig)]
                    assigned_pin = solver.Value(xvar)
                    if assigned_pin >= 0:
                        pin_assignments[sig] = {
                            "pin_name": pin_names[assigned_pin],
                            "position": pin_positions[assigned_pin],
                        }

                solution_data["instances"][fn] = {
                    "ports_used": used_ports,
                    "bounding_box_spread": bb_spread,
                    "max_pairwise_distance": max_d,
                    "pin_assignments": pin_assignments,
                }

    return solution_data


###############################################################################
# 3) VISUALIZATION
###############################################################################


def visualize_bga(solution_data, pin_names, pin_positions):
    """
    Create a matplotlib scatterplot of the BGA layout, labeling assigned pins,
    and drawing bounding boxes for each used peripheral.
    """
    if not solution_data or "instances" not in solution_data:
        print("No solution data to visualize.")
        return

    # We'll parse row/col from each assigned pin's position
    # also track (rowMin, colMin, rowMax, colMax) for bounding boxes
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
        # e.g. 'R16'
        m = re.match(r"^([A-Z])(\d+)$", pos)
        if not m:
            return (0, 0)
        rr = bga_rows.get(m.group(1), 0)
        cc = int(m.group(2))
        return (rr, cc)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # We'll store all assigned pin coords so we can auto-scale
    all_rows = []
    all_cols = []

    # For bounding boxes: we'll store (rMin, cMin, rMax, cMax)
    # We'll have to *actually* recompute them from assigned pins, or trust the solution_data's bounding_box_spread.
    # But bounding_box_spread is just the size, not the corners.
    # We'll just recalc from the assigned pins for each instance.
    instance_boxes = {}

    for inst_name, inst_data in solution_data["instances"].items():
        # gather row/col from pin_assignments
        assigned_pins = inst_data.get("pin_assignments", {})
        if not assigned_pins:
            continue
        rows = []
        cols = []
        for sig, info in assigned_pins.items():
            pos = info["position"]
            (r, c) = parse_bga_position(pos)
            rows.append(r)
            cols.append(c)

        rMin, rMax = min(rows), max(rows)
        cMin, cMax = min(cols), max(cols)
        instance_boxes[inst_name] = (rMin, cMin, rMax, cMax)

    # Now we also want to scatter each assigned pin across all instances
    # We'll combine them in one pass
    used_pin_coords = []  # (row,col,label)
    for inst_name, inst_data in solution_data["instances"].items():
        assigned_pins = inst_data["pin_assignments"]
        for sig, info in assigned_pins.items():
            pos = info["position"]
            (r, c) = parse_bga_position(pos)
            label = f"{inst_name}:{sig}"
            used_pin_coords.append((r, c, label))

    # We'll scatter them
    for r, c, label in used_pin_coords:
        ax.scatter(c, r)  # x=c, y=r
        ax.text(c + 0.2, r + 0.2, label, fontsize=8)
        all_rows.append(r)
        all_cols.append(c)

    # Draw bounding boxes
    for inst_name, (rMin, cMin, rMax, cMax) in instance_boxes.items():
        # rectangle with x=cMin, y=rMin, width=(cMax-cMin), height=(rMax-rMin)
        width = cMax - cMin
        height = rMax - rMin
        rect = Rectangle((cMin, rMin), width, height, fill=False)
        ax.add_patch(rect)
        # We'll put a small label near top-left corner
        ax.text(cMin, rMin - 0.5, inst_name, fontsize=9)

    if all_rows and all_cols:
        # set axis limits
        ax.set_xlim(min(all_cols) - 1, max(all_cols) + 2)
        ax.set_ylim(min(all_rows) - 1, max(all_rows) + 2)
    else:
        ax.set_xlim(-1, 10)
        ax.set_ylim(-1, 10)

    ax.set_xlabel("BGA Column")
    ax.set_ylabel("BGA Row")
    ax.set_title("Assigned Pins with Bounding Boxes")

    plt.gca().invert_yaxis()  # if you want row=0 at top
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

    # Example requirements
    reqs = {
        "eth": 1,
        "usbfs": 1,
        "sdmmc": 1,
        "swd": 1,
        "fdcan": 1,
        "uart": 4,
        "spi": 2,
        "tim": 4,
        "i2c": 3,
    }

    # We'll pass equal weights for all cost functions
    wts = {"ports": 1, "bbox": 1, "maxd": 1}

    solution_data = solve_pin_assignment(
        reqs,
        pin_funcs,
        pin_names,
        pin_positions,
        solution_limit=5,
        max_time_s=60.0,
        weights=wts,
    )

    if solution_data:
        print("\n\nStructured Solution Data (as JSON-ish):")
        print(json.dumps(solution_data, indent=2))

        # Visualize
        visualize_bga(solution_data, pin_names, pin_positions)


if __name__ == "__main__":
    main()
