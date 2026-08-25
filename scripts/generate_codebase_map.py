"""Generate a factual static architecture map for the SMFS Catalog package.

This intentionally derives its inventory from Python's AST.  It does not use
comments or docstrings as evidence about behaviour.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "smfs_catalog"
OUT = ROOT / "docs" / "CODEBASE_ARCHITECTURE.md"


GROUPS = {
    "Entry and orchestration": {"run_dashboard", "dashboard_window", "analysis_worker", "navigator_bar", "widgets"},
    "Ingest and catalog persistence": {"add_data_dialog", "bulk_metadata_dialog", "remove_files_dialog", "scanner", "db"},
    "L1 loading and qualification": {"curve_loader"},
    "L1 to L2 signal/event pipeline": {"curve_analysis", "signal_processing", "roi_detection", "roi_events", "roi_assembly", "roi_pipeline"},
    "L2 exploration and gating": {"variables", "criteria_gate", "criteria_dialog", "variable_window", "categorical_window", "scatter_window", "event_summary_window", "clustering", "pca_window"},
    "L3/L4 models, fits, and products": {"models", "regression", "histogram_binning", "event_processor", "dist_fit_core", "dist_fit_window", "gmm_fit_core", "gmm_fit_window", "base_2dh_window", "normalized_2dh_window", "physical_2dh_window", "mean_curve_window", "wlc_view_window", "isoforce_window", "export_utils", "ledger"},
    "Inspection and diagnostics UI": {"rawcurve_window", "decomposition_window", "display_roi", "fft_window", "class_lineplot_window", "trace_overlay_panel"},
    "Cross-cutting infrastructure": {"bandwidth_warning", "quantities", "style", "sample_marks", "qt_utils", "scope", "date_picker_dialog", "crashlog", "utils", "__init__"},
}


def module_name(path: Path) -> str:
    return path.stem


def dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = dotted(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    return None


def package_imports(tree: ast.Module) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            if node.module:
                found.add(node.module.split(".")[0])
            else:
                found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("smfs_catalog."):
                    found.add(alias.name.split(".")[1])
        elif isinstance(node, ast.ImportFrom) and node.module == "smfs_catalog":
            found.update(a.name.split(".")[0] for a in node.names)
    return found


def callable_rows(tree: ast.Module):
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    rows = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        owner = None
        cur = parents.get(node)
        while cur is not None:
            if isinstance(cur, ast.ClassDef):
                owner = cur.name
                break
            cur = parents.get(cur)
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                name = dotted(child.func)
                if name:
                    calls.append(name)
        internal = sorted({c for c in calls if c.startswith(("_db.", "db.", "_scanner.", "scanner.", "_roi", "roi_", "analyse", "compute_", "populate_", "assemble_", "variables.", "clustering."))})
        kind = "method" if owner else "function"
        if node.name.startswith("_on_") or node.name.endswith("Event"):
            role = "UI/event handler"
        elif node.name.startswith(("get_", "list_", "read_", "load_", "resolve_", "describe_", "available", "coverage")):
            role = "read/query/resolve"
        elif node.name.startswith(("set_", "write_", "save_", "upsert_", "add_", "remove_", "delete_", "drop_", "clear_", "enqueue_", "dequeue_", "import_", "mark_")):
            role = "state mutation"
        elif node.name.startswith(("compute_", "analyse", "fit_", "find_", "detect_", "build_", "assemble_", "project_", "segment_", "decompose_", "correlate", "linear_fit")):
            role = "computation/transformation"
        elif owner and node.name == "__init__":
            role = "construction/wiring"
        elif node.name.startswith(("_build", "_draw", "_render", "_refresh", "_update", "_populate", "_load", "_rebuild")):
            role = "UI/view coordination"
        else:
            role = "helper/control flow"
        qual = f"{owner}.{node.name}" if owner else node.name
        rows.append((node.lineno, getattr(node, "end_lineno", node.lineno), qual, kind, role, internal))
    return sorted(rows)


def main() -> None:
    paths = [ROOT / "run_dashboard.py", *sorted(PKG.glob("*.py"))]
    trees = {module_name(p): ast.parse(p.read_text(encoding="utf-8-sig")) for p in paths}
    lines = [
        "# SMFS Catalog codebase architecture",
        "",
        "> Generated from executable Python syntax. Comments and docstrings are not used as evidence. "
        "Function roles are name/structure classifications; direct dependencies come from call expressions.",
        "",
        "## Confirmed top-level flow",
        "",
        "```mermaid",
        "flowchart LR",
        "  RAW[IBW files / L1] --> SCAN[scanner + curve_loader qualification]",
        "  SCAN --> CAT[(SQLite catalog)]",
        "  CAT --> QUEUE[dashboard queue]",
        "  QUEUE --> WORKER[analysis_worker]",
        "  WORKER --> ANA[curve_analysis]",
        "  ANA --> SIG[signal_processing]",
        "  SIG --> DET[roi_detection]",
        "  DET --> EVENTS[roi_events + roi_assembly]",
        "  EVENTS --> L2[(analysis_results + event_map / L2)]",
        "  L2 --> EXP[event summary + variables + criteria]",
        "  EXP --> PCA[PCA + k-means]",
        "  PCA -->|global clustering registry| EXP",
        "  EXP --> FIT[distribution/GMM/WLC/2DH products]",
        "  FIT --> EXPORT[manifested exports / L3-L4 work products]",
        "  CAT -->|open/requeue| QUEUE",
        "  EXP -->|reveal raw/ROI| QUEUE",
        "```",
        "",
        "The straight path exists, but persistence is shared rather than layered: `db.py` is used by ingest, analysis, exploration, fitting, and export-adjacent code. Reentry is mostly path/file-ID based through the dashboard and navigator. The PCA/k-means return edge is an in-memory publish/subscribe registry in `clustering.py`, consumed by exploration windows.",
        "",
        "## Organization by responsibility",
        "",
    ]
    assigned = set()
    for group, mods in GROUPS.items():
        present = sorted(mods & trees.keys())
        assigned.update(present)
        lines += [f"### {group}", "", ", ".join(f"`{m}.py`" for m in present), ""]
    missing = sorted(set(trees) - assigned)
    if missing:
        lines += ["### Unclassified", "", ", ".join(f"`{m}.py`" for m in missing), ""]

    lines += ["## Static module dependency graph", "", "```mermaid", "graph TD"]
    for mod, tree in trees.items():
        for dep in sorted(package_imports(tree)):
            if dep in trees and dep != mod:
                lines.append(f"  {mod} --> {dep}")
    lines += ["```", "", "## Module and callable inventory", "", "Line ranges refer to the current source. `Direct app calls` lists statically visible calls into major app services; an empty cell does not mean the callable has no effect.", ""]

    for path in paths:
        mod = module_name(path)
        tree = trees[mod]
        imports = sorted(package_imports(tree))
        nlines = len(path.read_text(encoding="utf-8-sig").splitlines())
        classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
        lines += [f"### `{path.name}` ({nlines} lines)", "", f"Imports app modules: {', '.join(f'`{x}`' for x in imports) or 'none'}. Top-level classes: {', '.join(f'`{x}`' for x in classes) or 'none'}.", "", "| Callable | Lines | Structural role | Direct app calls |", "|---|---:|---|---|"]
        rows = callable_rows(tree)
        if not rows:
            lines.append("| *(no callables)* | — | package metadata/constants | — |")
        for start, end, qual, _kind, role, calls in rows:
            dep = ", ".join(f"`{c}`" for c in calls) if calls else "—"
            lines.append(f"| `{qual}` | {start}–{end} | {role} | {dep} |")
        lines.append("")

    lines += [
        "## Structural observations for red-team follow-up",
        "",
        "1. `db.py` and `dashboard_window.py` are concentration risks by size and fan-in/fan-out, independent of whether their current behaviour is correct.",
        "2. SQLite is simultaneously catalog, queue, parameter/profile store, analysis cache, event-map store, and fit-history store. Those are distinct lifecycles behind one module and one database.",
        "3. The core numerical path is comparatively separable: loader → signal processing → detection → event construction/assembly. Persistence enters heavily in `curve_analysis.py` and `roi_pipeline.py`.",
        "4. Exploration state is mixed: criteria/settings and computed maps persist in SQLite, while clustering is process-global in memory. Reproducibility therefore depends on exports capturing the clustering provenance before process exit.",
        "5. UI modules commonly coordinate reads, computation, persistence, and child-window lifecycle directly; there is no distinct application-service layer.",
        "6. The current full test command aborts during collection in five modules with missing fixture rows/IDs. This map records that fact without assigning a cause.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
