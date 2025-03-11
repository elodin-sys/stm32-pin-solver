#!/usr/bin/env python3

import re
import xml.etree.ElementTree as ET

from ortools.sat.python import cp_model

###############################################################################
# 1) PARSE STM32 XML
###############################################################################


def parse_signal_name(signal_name):
    """Map ST-style signal names (e.g. 'I2C1_SCL') to (peripheral, signal)."""

    # I2C (I2C1_SCL, I2C2_SDA, I2C4_SMBA, etc.)
    m = re.match(r"I2C(\d+)_(SCL|SDA|SMBA)", signal_name)
    if m:
        return (f"i2c{m.group(1)}", m.group(2).lower())

    # UART/USART/LPUART (USART1_TX, UART4_RX, LPUART1_TX, etc.)
    m = re.match(r"USART(\d+)_(TX|RX)", signal_name)
    if m:
        return (f"uart{m.group(1)}", m.group(2).lower())

    m = re.match(r"UART(\d+)_(TX|RX)", signal_name)
    if m:
        return (f"uart{m.group(1)}", m.group(2).lower())

    m = re.match(r"LPUART(\d+)_(TX|RX)", signal_name)
    if m:
        return (f"uart{m.group(1)}", m.group(2).lower())

    # SPI (SPI1_MISO, SPI2_NSS, etc.)
    m = re.match(r"SPI(\d+)_(MOSI|MISO|SCK|CLK|NSS)", signal_name)
    if m:
        inst = m.group(1)
        sub = m.group(2).lower()
        if sub in ("sck", "clk"):
            sub = "clk"
        return (f"spi{inst}", sub)

    # FDCAN (FDCAN2_TX, FDCAN1_RX)
    m = re.match(r"FDCAN(\d+)_(TX|RX)", signal_name)
    if m:
        return (f"fdcan{m.group(1)}", m.group(2).lower())

    # Ethernet (ETH_TX_EN, ETH_MDC, etc.)
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

    # SDMMC (SDMMC1_D0, SDMMC2_CMD, etc.)
    m = re.match(r"SDMMC(\d+)_(.*)", signal_name)
    if m:
        inst = m.group(1)
        sub = m.group(2).lower()
        sdmmc_allowed = {"ck", "cmd", "d0", "d1", "d2", "d3"}
        return (f"sdmmc{inst}", sub) if sub in sdmmc_allowed else None

    # USB FS (USB_OTG_FS_DP, USB_OTG_FS_SOF, etc.)
    m = re.match(r"USB_OTG_FS_(.*)", signal_name)
    if m:
        sub = m.group(1).lower()
        usbfs_allowed = {"dm", "dp", "sof", "vbus"}
        return ("usbfs", sub) if sub in usbfs_allowed else None

    # USB HS (USB_OTG_HS_DM, USB_OTG_HS_ULPI_D3, etc.)
    m = re.match(r"USB_OTG_HS_(.*)", signal_name)
    if m:
        sub = m.group(1).lower()
        if sub in ("sof", "vbus"):
            return ("usbhs", sub)
        if sub.startswith("ulpi_"):
            return ("usbhs", sub)
        return None

    # TIM (TIM2_CH1..CH4, TIM17_CH3..CH4 not used, etc.)
    m = re.match(r"TIM(\d+)_(.*)", signal_name)
    if m:
        inst = m.group(1)
        remainder = m.group(2).lower()
        if remainder in ("ch1", "ch2", "ch3", "ch4"):
            return (f"tim{inst}", remainder)
        return None

    # SWD (DEBUG_JTCK-SWCLK, DEBUG_JTMS-SWDIO)
    if signal_name == "DEBUG_JTCK-SWCLK":
        return ("swd", "swclk")
    if signal_name == "DEBUG_JTMS-SWDIO":
        return ("swd", "swdio")

    # If nothing matched
    return None


def parse_stm32_pins(xml_path):
    """
    Reads the ST XML for the MCU, returning a dictionary:
       pin_functions[pin_idx] = [ (peripheral, signal), ... ]
    Only pins of Type="I/O" or Type="MonoIO" are considered.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    namespace = "{http://dummy.com}"  # Adjust if your <Mcu> has a different namespace
    valid_pin_types = {"I/O", "MonoIO"}

    pin_functions = {}
    pin_index = 0

    for pin_el in root.findall(f"{namespace}Pin"):
        pin_type = pin_el.get("Type")
        if pin_type not in valid_pin_types:
            continue  # skip power, reset, etc.

        recognized_signals = []
        for sig_el in pin_el.findall(f"{namespace}Signal"):
            name_raw = sig_el.get("Name")
            parsed = parse_signal_name(name_raw)
            if parsed:
                recognized_signals.append(parsed)

        pin_functions[pin_index] = recognized_signals
        pin_index += 1

    return pin_functions


###############################################################################
# 2) THE OR-TOOLS CP-SAT SOLVER
###############################################################################


class PinAssignmentSolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(
        self, pin_vars, peripheral_vars, peripheral_signals, pin_functions, limit=5
    ):
        super().__init__()
        self._pin_vars = pin_vars
        self._peripheral_vars = peripheral_vars
        self._peripheral_signals = peripheral_signals
        self._pin_functions = pin_functions
        self._solution_limit = limit
        self._solution_count = 0

    def on_solution_callback(self):
        self._solution_count += 1
        print(f"\nSolution {self._solution_count}:")

        # Which peripheral instances are used?
        active_peripherals = {}
        for p_type, inst_map in self._peripheral_vars.items():
            used_insts = []
            for inst, used_var in inst_map.items():
                if self.Value(used_var) == 1:
                    used_insts.append(inst)
            active_peripherals[p_type] = used_insts

        # Print pin assignments
        for p_type, inst_list in active_peripherals.items():
            for inst in sorted(inst_list):
                full_name = p_type + inst if inst else p_type
                print(f"  {full_name}:")
                for sig in self._peripheral_signals[p_type]:
                    pin_var = self._pin_vars[(full_name, sig)]
                    assigned_pin = self.Value(pin_var)
                    if assigned_pin >= 0:
                        recognized = self._pin_functions[assigned_pin]
                        print(
                            f"    {sig} -> Pin {assigned_pin} (Pin has: {recognized})"
                        )

        if self._solution_count >= self._solution_limit:
            print(f"Reached solution limit ({self._solution_limit}). Stopping search.")
            self.StopSearch()

    def solution_count(self):
        return self._solution_count


def solve_pin_assignment(peripheral_requirements, pin_functions, solution_limit=5):
    """Builds and solves a CP-SAT model to assign pins for the requested peripherals."""

    # 1) Define signals for each peripheral type
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
            # "vbus",
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

    # 2) Identify all possible (p_type, instance) in pin_functions
    available_peripherals = {pt: set() for pt in peripheral_signals}

    def separate_peripheral_and_instance(full_name):
        """Separate the base prefix (p_type) from trailing digits (instance)."""
        for p_type in peripheral_signals:
            if full_name.startswith(p_type):
                suffix = full_name[len(p_type) :]
                return p_type, suffix
        return None, None

    for pin_idx, funcs in pin_functions.items():
        for fullname, sig in funcs:
            p_type, suffix = separate_peripheral_and_instance(fullname)
            if p_type is not None:
                if suffix.isdigit():
                    available_peripherals[p_type].add(suffix)
                else:
                    # handle empty or partial
                    if suffix == "":
                        available_peripherals[p_type].add("")
                    else:
                        digits = re.search(r"(\d+)$", suffix)
                        if digits:
                            available_peripherals[p_type].add(digits.group(1))
                        else:
                            available_peripherals[p_type].add(suffix)

    # 3) Determine which are "complete" (all signals present on at least one pin)
    complete_peripherals = {}
    for p_type, sigs in peripheral_signals.items():
        inst_list = []
        for suffix in sorted(available_peripherals[p_type]):
            name = p_type + suffix if suffix else p_type
            # Check if this instance has all required signals
            if all(
                any((name, s) in pin_functions[p] for p in pin_functions) for s in sigs
            ):
                inst_list.append(suffix)
        complete_peripherals[p_type] = inst_list

    # 4) Check if we can meet user requirements
    for p_type, needed_count in peripheral_requirements.items():
        if p_type not in complete_peripherals:
            print(f"Peripheral '{p_type}' not recognized.")
            return None
        have = len(complete_peripherals[p_type])
        if have < needed_count:
            print(f"Not enough {p_type}. Need {needed_count}, found {have}.")
            return None

    # 5) Create a BoolVar for each complete instance: "this instance is used or not"
    peripheral_vars = {}
    for p_type, inst_list in complete_peripherals.items():
        peripheral_vars[p_type] = {}
        for inst in inst_list:
            var_name = f"{p_type}{inst if inst else '_noinst'}_used"
            used_var = model.NewBoolVar(var_name)
            peripheral_vars[p_type][inst] = used_var

        # Must use exactly 'needed_count'
        needed_count = peripheral_requirements.get(p_type, 0)
        if needed_count == 0:
            for inst in inst_list:
                model.Add(peripheral_vars[p_type][inst] == 0)
        else:
            model.Add(
                sum(peripheral_vars[p_type][inst] for inst in inst_list) == needed_count
            )

    # 6) For each (peripheral_name, signal), create an IntVar domain=[feasible pins]+[-1]
    pin_vars = {}

    def valid_pins_for(name, signal):
        return [pin for pin in pin_functions if (name, signal) in pin_functions[pin]]

    for p_type, inst_list in peripheral_vars.items():
        for inst in inst_list:
            used_var = peripheral_vars[p_type][inst]
            pname = p_type + inst if inst else p_type
            for sig in peripheral_signals[p_type]:
                feasible_pins = valid_pins_for(pname, sig)
                domain_vals = feasible_pins + [-1]
                xvar = model.NewIntVarFromDomain(
                    cp_model.Domain.FromValues(domain_vals), f"{pname}_{sig}_pin"
                )
                pin_vars[(pname, sig)] = xvar

                # If used => xvar != -1, if not used => xvar == -1
                model.Add(xvar != -1).OnlyEnforceIf(used_var)
                model.Add(xvar == -1).OnlyEnforceIf(used_var.Not())

    # 7) "At most one signal per pin"
    bools_by_pin = {}
    for (pname, sig), xvar in pin_vars.items():
        for pin_idx in pin_functions:
            if pin_idx not in bools_by_pin:
                bools_by_pin[pin_idx] = []
            bvar = model.NewBoolVar(f"pin{pin_idx}_{pname}_{sig}")
            bools_by_pin[pin_idx].append(bvar)
            model.Add(xvar == pin_idx).OnlyEnforceIf(bvar)
            model.Add(xvar != pin_idx).OnlyEnforceIf(bvar.Not())

    for pin_idx, blist in bools_by_pin.items():
        model.AddAtMostOne(blist)

    # 8) Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    solver.parameters.enumerate_all_solutions = True

    solution_printer = PinAssignmentSolutionPrinter(
        pin_vars, peripheral_vars, peripheral_signals, pin_functions, solution_limit
    )

    print("\n--- Solving Pin Assignment Problem ---")
    status = solver.Solve(model, solution_printer)

    # Print summary
    print("\nSolver results:")
    if status == cp_model.OPTIMAL:
        print("  Status: OPTIMAL (exhaustive search done).")
    elif status == cp_model.FEASIBLE:
        print("  Status: FEASIBLE (found solutions, not exhaustive).")
    elif status == cp_model.INFEASIBLE:
        print("  Status: INFEASIBLE (no solution).")
    elif status == cp_model.MODEL_INVALID:
        print("  Status: MODEL INVALID.")
    else:
        print(f"  Status: UNKNOWN (code={status}).")

    print(f"  Solutions found: {solution_printer.solution_count()}")
    print(f"  Search time: {solver.WallTime():.3f} seconds")

    return status


def main():
    xml_path = "pin_data/STM32H747XIHx.xml"  # Adjust as needed
    pin_funcs = parse_stm32_pins(xml_path)

    print("\nParsed pins with recognized signals:")
    for idx, sigs in pin_funcs.items():
        if sigs:
            print(f"  Pin {idx}: {sigs}")

    # Example requirement:
    reqs = {
        "eth": 1,
        "usbhs": 1,
        "sdmmc": 1,
        "swd": 1,
        "fdcan": 1,
        "uart": 4,
        "spi": 2,
        "tim": 4,
        "i2c": 3,
    }

    solve_pin_assignment(reqs, pin_funcs, solution_limit=5)


if __name__ == "__main__":
    main()
