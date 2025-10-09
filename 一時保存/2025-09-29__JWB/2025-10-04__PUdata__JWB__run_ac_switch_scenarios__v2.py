# -*- coding: utf-8 -*-
"""Automate LTspice AC sweeps and export results to CSV files."""
from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
import re
from shutil import copy2
from typing import Callable, Optional

import numpy as np
import pandas as pd
from PyLTSpice import RawRead, SimCommander
from spicelib.editor.asc_editor import AscEditor
from spicelib.editor.spice_editor import SpiceEditor

BASE_DIR = Path(__file__).parent.resolve()  # スクリプトのあるディレクトリ

# ===== User configuration =====
INPUT_PATH = BASE_DIR / "asc" / "JWB_Analysis.asc"
LTSPICE_EXE = r"C:\Users\***\AppData\Local\Programs\ADI\LTspice\LTspice.exe"
TARGET_NODE = "Amp-In"
PU_Name = "JWB"
ANALYSIS_TEMPLATE = BASE_DIR / "Template" / "Analysis_Template.xlsm"
# ==============================

PREFERRED_ENC = "cp932"
FALLBACK_ENC = "utf-8"

TextTransform = Callable[[str], str]

def read_text_auto(path: Path) -> str:
    try:
        return path.read_text(encoding=PREFERRED_ENC)
    except Exception:
        return path.read_text(encoding=FALLBACK_ENC, errors="ignore")


def write_text_cp932(path: Path, text: str) -> None:
    path.write_text(text, encoding=PREFERRED_ENC, errors="replace")


def normalize_micro_symbols(text: str) -> str:
    return text.replace("\u00B5", "u").replace("\u03BC", "u")


def tone_param_transform(text: str) -> str:
    """Swap pickup parameter definitions for tone sweep."""
    text = text.replace(";param RT=Rx*(x**k-1)/(x-1)+50m", ".param RT=Rx*(x**k-1)/(x-1)+50m")
    text = text.replace(".param RT=250k", ";param RT=250k")
    text = text.replace(".param RVa=Rx*(x**k-1)/(x-1)+50m", ";param RVa=Rx*(x**k-1)/(x-1)+50m")
    text = re.sub(r";param RVa=Rx(?!\*)", ".param RVa=Rx", text)
    return text
def make_outdir(base: Path) -> Path:
    stamp = datetime.now().strftime(f"%y-%m-%d__{PU_Name}__%H-%M-%S")
    outdir = base / stamp
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def detect_format(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".asc":
        return "asc"
    text = read_text_auto(path)
    stripped = text.lstrip()
    if stripped.startswith("*") or stripped.startswith(".title") or stripped.startswith(".include"):
        return "spice"
    if "ExpressPCB Netlist" in text:
        return "expresspcb"
    return "unknown"


def asc_save_compat(editor: AscEditor, path: Path) -> None:
    if hasattr(editor, "save") and callable(getattr(editor, "save")):
        editor.save()
        return
    if hasattr(editor, "save_netlist") and callable(getattr(editor, "save_netlist")):
        editor.save_netlist(str(path))
        return
    if hasattr(editor, "write_netlist") and callable(getattr(editor, "write_netlist")):
        editor.write_netlist(str(path))
        return
    if hasattr(editor, "save_as") and callable(getattr(editor, "save_as")):
        editor.save_as(str(path))
        return
    raise AttributeError("AscEditor does not expose a supported save API")


def prepare_editor(
    input_path: Path,
    work_dir: Path,
    text_transform: Optional[Callable[[str], str]] = None,
) -> tuple[object, Path, str, str]:
    kind = detect_format(input_path)
    if kind == "expresspcb":
        raise RuntimeError(
            "ExpressPCB netlists are not supported. Export a SPICE netlist or use the .asc schematic."
        )
    if kind == "unknown":
        raise RuntimeError(
            "Could not detect the input file format. Please provide an .asc or SPICE netlist."
        )

    work_dir.mkdir(parents=True, exist_ok=True)

    normalized = normalize_micro_symbols(read_text_auto(input_path))
    transformed = text_transform(normalized) if text_transform else normalized

    if kind == "asc":
        edited = work_dir / input_path.name
        write_text_cp932(edited, transformed)
        editor = AscEditor(str(edited))
        restore_text = normalized
        return editor, edited, kind, restore_text

    edited = work_dir / (input_path.stem + ".cir")
    restore_text = normalized
    transformed_output = transformed
    if not normalized.lstrip().startswith("*"):
        header = "* converted for SpiceEditor\r\n"
        restore_text = header + restore_text
        transformed_output = header + transformed_output
    write_text_cp932(edited, transformed_output)
    editor = SpiceEditor(str(edited))
    return editor, edited, kind, restore_text

def remove_existing_directives(editor: object) -> None:
    """Remove existing .wrdata directives that would interfere with exports."""
    if hasattr(editor, "remove_Xinstruction"):
        editor.remove_Xinstruction(r"\.wrdata")
        return
    if hasattr(editor, "remove_instruction"):
        try:
            editor.remove_instruction(".wrdata")
        except Exception:
            pass


def write_editor(editor: object, path: Path, kind: str) -> None:
    if kind == "asc":
        asc_save_compat(editor, path)
        refreshed = normalize_micro_symbols(read_text_auto(path))
        write_text_cp932(path, refreshed)
    else:
        if hasattr(editor, "write_netlist"):
            editor.write_netlist(str(path))
        else:
            asc_save_compat(editor, path)
        refreshed = normalize_micro_symbols(read_text_auto(path))
        write_text_cp932(path, refreshed)


def run_ltspice_batch(executable: str, input_file: Path) -> None:
    exe_path = Path(executable)
    if not exe_path.exists():
        raise FileNotFoundError(f"LTspice executable not found: {executable}")

    candidates = [
        [executable, "-b", str(input_file)],
        [executable, "-Run", "-b", str(input_file)],
        [executable, "-b", "-Run", str(input_file)],
    ]

    last_error = None
    for cmd in candidates:
        try:
            result = subprocess.run(
                cmd,
                cwd=str(input_file.parent),
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return
            last_error = (
                f"returncode={result.returncode}"
                f"\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        except Exception as exc:
            last_error = str(exc)
    raise RuntimeError(
        "LTspice batch execution failed.\n"
        f"Tried: {candidates}\nLastError: {last_error}"
    )


def run_simulation(kind: str, editor_path: Path) -> None:
    if kind == "asc":
        run_ltspice_batch(LTSPICE_EXE, editor_path)
        return

    sim = SimCommander(str(editor_path))
    try:
        if LTSPICE_EXE:
            sim.run(executable=LTSPICE_EXE)
        else:
            sim.run()
    except TypeError:
        if LTSPICE_EXE:
            sim.run(ltspice_path=LTSPICE_EXE)
        else:
            sim.run()


def data_from_raw(raw_path: Path) -> pd.DataFrame:
    raw = RawRead(str(raw_path), verbose=False)

    target_lower = f"v({TARGET_NODE.lower()})"
    trace_name = next(
        (name for name in raw.get_trace_names() if name.lower() == target_lower),
        None,
    )
    if trace_name is None:
        available = ", ".join(raw.get_trace_names())
        raise RuntimeError(
            f"Trace for node '{TARGET_NODE}' not found in RAW file. Available traces: {available}"
        )

    trace = raw.get_trace(trace_name)
    plot = raw._plots[0] if raw._plots else None
    steps_info = plot.steps if plot and getattr(plot, "steps", None) else None

    frames: list[pd.DataFrame] = []
    steps = list(raw.get_steps())
    if not steps:
        steps = [0]

    for step_idx in steps:
        freq = np.asarray(raw.get_axis(step_idx)).real.astype(float)
        wave = np.asarray(trace.get_wave(step_idx))
        mag = np.abs(wave)
        mag_db = 20.0 * np.log10(np.where(mag > 0.0, mag, np.finfo(float).tiny))
        phase = np.degrees(np.angle(wave))

        df = pd.DataFrame(
            {
                "frequency_Hz": freq,
                "mag_dB": mag_db,
                "phase_deg": phase,
                "step_index": step_idx,
            }
        )

        if steps_info and step_idx < len(steps_info):
            for key, value in steps_info[step_idx].items():
                df[f"step_{key}"] = value

        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    ordered = ["frequency_Hz", "mag_dB", "phase_deg"]
    extras = [col for col in result.columns if col not in ordered]
    return result[ordered + extras]


def run_case(
    input_path: Path,
    out_csv: Path,
    v2: str,
    v3: str,
    v4: str,
    text_transform: Optional[TextTransform] = None,
) -> None:
    work_dir = out_csv.parent
    editor, edited_file, kind, restore_text = prepare_editor(
        input_path, work_dir, text_transform=text_transform
    )

    try:
        editor.set_component_value("V2", v2)
        editor.set_component_value("V3", v3)
        editor.set_component_value("V4", v4)

        remove_existing_directives(editor)

        write_editor(editor, edited_file, kind)

        run_simulation(kind, edited_file)

        raw_path = edited_file.with_suffix(".raw")
        if not raw_path.exists():
            log_path = edited_file.with_suffix(".log")
            if log_path.exists():
                tail = "\n".join(log_path.read_text(errors="ignore").splitlines()[-120:])
                raise RuntimeError(
                    "Simulation finished without producing a RAW file.\n"
                    f"Log: {log_path}\n---- log tail ----\n{tail}\n------------------",
                )
            raise FileNotFoundError(f"Expected RAW file not found: {raw_path}")

        df = data_from_raw(raw_path)
        df.to_csv(out_csv, index=False, encoding=PREFERRED_ENC)
    finally:
        write_text_cp932(edited_file, restore_text)

def main() -> None:
    base_dir = Path(__file__).parent.resolve()
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    if not ANALYSIS_TEMPLATE.exists():
        raise FileNotFoundError(f"Template workbook not found: {ANALYSIS_TEMPLATE}")

    outdir = make_outdir(base_dir)
    dest_template = outdir / f"{PU_Name}_Analysis{ANALYSIS_TEMPLATE.suffix}"
    copy2(ANALYSIS_TEMPLATE, dest_template)

    cases = [
        ("Neck", {"V2": "5", "V3": "0", "V4": "0"}),
        ("Middle", {"V2": "0", "V3": "5", "V4": "0"}),
        ("Bridge", {"V2": "0", "V3": "0", "V4": "5"}),
        ("Neck-Middle", {"V2": "5", "V3": "5", "V4": "0"}),
        ("Middle-Bridge", {"V2": "0", "V3": "5", "V4": "5"}),
        ("Bridge-Neck", {"V2": "5", "V3": "0", "V4": "5"}),
    ]

    variants: list[tuple[str, Optional[TextTransform]]] = [
        ("Vol", None),
        ("Tone", tone_param_transform),
    ]

    for suffix, transform in variants:
        for name, values in cases:
            out_csv = outdir / f"{PU_Name}__{name}_{suffix}.csv"
            run_case(
                INPUT_PATH,
                out_csv,
                values["V2"],
                values["V3"],
                values["V4"],
                text_transform=transform,
            )
            print(f"saved: {out_csv}")

    print("\nDone.")
    print(f"Output folder: {outdir}")

if __name__ == "__main__":
    main()










