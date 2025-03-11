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

    # If your XML has a namespace, adapt this accordingly
    namespace = "{http://dummy.com}"
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
# 2) THE OR-TOOLS CP-SAT SOLVER WITH 3 COST TERMS (equally weighted)
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
    We'll use 3 cost terms with equal weights (1,1,1):
      1) # of distinct ports used
      2) bounding-box spread
      3) max pairwise distance
    Returns a structured dictionary with the solution info.
    """

    # We'll keep them all = 1
    weights = {"ports": 1, "bbox": 1, "maxd": 1}

    #######################################################################
    # The model building is unchanged from the prior example:
    #######################################################################
    model = cp_model.CpModel()

    # 1) define signals
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

    # 2) Identify all (p_type, instance)
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
            # must have all signals
            if all(
                any((name, s) in pin_functions[p] for p in pin_functions) for s in sigs
            ):
                inst_list.append(suffix)
        complete_peripherals[p_type] = inst_list

    # 4) check feasibility
    for p_type, needed_count in peripheral_requirements.items():
        if p_type not in complete_peripherals:
            print(f"[ERROR] {p_type} not recognized.")
            return None
        have = len(complete_peripherals[p_type])
        if have < needed_count:
            print(f"[ERROR] Not enough {p_type}. Need {needed_count}, found {have}.")
            return None

    # 5) create used bools
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

    # 6) for each (instance, signal) => IntVar
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

    # 7) at most one signal per pin
    assigned_bvar = {}
    for (fn, sig), xv in pin_vars.items():
        for pin_idx in pin_functions:
            b = model.NewBoolVar(f"pin{pin_idx}_{fn}_{sig}")
            assigned_bvar[(fn, sig, pin_idx)] = b
            model.Add(xv == pin_idx).OnlyEnforceIf(b)
            model.Add(xv != pin_idx).OnlyEnforceIf(b.Not())

    for pin_idx in pin_functions:
        all_b = []
        for fn, sig in pin_vars:
            all_b.append(assigned_bvar[(fn, sig, pin_idx)])
        model.AddAtMostOne(all_b)

    # 8) cost #1 => ports
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

    # 11) final cost = #ports + bboxSpread + maxDist
    cost_var = model.NewIntVar(0, 9999999, "total_cost")
    model.Add(cost_var == total_port_usage + total_bbox_spread + total_max_dist)
    model.Minimize(cost_var)

    # 12) Solve
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
    Plot a BGA grid:
      - Each instance gets a unique color
      - All pins for that instance are that color
      - The bounding box is drawn translucent in the same color
      - For zero-width or zero-height boxes, we artificially enlarge them for visibility
      - If multiple boxes overlap exactly, we can apply a small offset or different lines
    """

    if not solution_data or "instances" not in solution_data:
        print("No solution to visualize.")
        return

    # We'll parse row/col from each pin
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

    fig = plt.figure(figsize=(10, 10))
    # set size of the plot
    ax = fig.add_subplot(111, aspect="equal")

    # We'll cycle colors so each instance has a distinct color
    color_cycle = plt.cm.get_cmap("tab10")  # or any other colormap
    instance_list = list(solution_data["instances"].keys())

    # We can also do a small offset for each bounding box if needed:
    # e.g. offset_index = 0,1,2,...  => offset = offset_index * 0.2, etc.
    # That helps if bounding boxes align exactly.
    # We'll store the offset in a dictionary below if we want to apply it.

    inst_colors = {}
    inst_offsets = {}

    for i, inst_name in enumerate(instance_list):
        color = color_cycle(i % 10)  # repeat if >10
        inst_colors[inst_name] = color
        inst_offsets[inst_name] = i * 0.0  # set to 0 if you don't want a shift

    # We'll gather all pin coords
    all_rows, all_cols = [], []

    # We'll also need bounding box corners
    bounding_boxes = {}

    for i, inst_name in enumerate(instance_list):
        inst_data = solution_data["instances"][inst_name]
        assigned_pins = inst_data["pin_assignments"]
        if not assigned_pins:
            continue

        # gather row/col
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

        # if width=0 or height=0 => enlarge a bit
        width = max(0, (cMax - cMin))
        height = max(0, (rMax - rMin))

        # optionally shift them by a small offset if you want to avoid perfect overlap
        shift = inst_offsets[inst_name]
        bounding_boxes[inst_name] = (
            cMin - shift,
            rMin - shift,  # x,y for rectangle
            width,
            height,
        )

        # keep track for scatter points
        for sig, info in assigned_pins.items():
            pos = info["position"]
            (rr, cc) = parse_bga_position(pos)
            all_rows.append(rr)
            all_cols.append(cc)

    # Now let's actually draw
    for inst_name in instance_list:
        color = inst_colors[inst_name]
        inst_data = solution_data["instances"][inst_name]
        assigned_pins = inst_data["pin_assignments"]
        if not assigned_pins:
            continue

        # scatter each pin
        for sig, info in assigned_pins.items():
            pos = info["position"]
            (rr, cc) = parse_bga_position(pos)
            ax.scatter(cc, rr, color=color)
            ax.text(cc + 0.2, rr + 0.2, f"{inst_name}:{sig}", fontsize=8, color=color)

        # bounding box
        (x0, y0, w, h) = bounding_boxes[inst_name]
        # We'll do a translucent fill
        rect = Rectangle(
            (x0, y0),
            w,
            h,
            fill=True,
            alpha=0.2,  # translucent
            edgecolor=color,
            facecolor=color,
            linewidth=1.5,
        )
        ax.add_patch(rect)
        # Also optionally draw an outline that’s less translucent:
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

    ax.invert_yaxis()  # so row=0 is near top if you prefer
    ax.set_xlabel("BGA Column")
    ax.set_ylabel("BGA Row")
    ax.set_title("BGA Pin Assignments")

    bga_row_letters = list(bga_rows.keys())
    ax.set_yticks(range(17), labels=bga_row_letters)
    ax.set_xticks(range(1, 18), labels=[str(i) for i in range(1, 18)])

    plt.show()


###############################################################################
# 4) EXAMPLE MAIN
###############################################################################


def main():
    # Example usage
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

    # Solve with each cost factor = 1
    solution_data = solve_pin_assignment(reqs, pin_funcs, pin_names, pin_positions)
    if solution_data:
        print("\nSolution found!\n")
        # Print structured data
        print(json.dumps(solution_data, indent=2))
        # Visualize
        visualize_bga(solution_data, pin_names, pin_positions)


if __name__ == "__main__":
    main()
