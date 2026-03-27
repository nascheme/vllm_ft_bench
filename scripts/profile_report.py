#!/usr/bin/env python3
"""Parse Firefox profiler JSON into Python objects and report costly call chains."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Lib:
    name: str
    path: str
    debug_name: str
    debug_path: str
    breakpad_id: str
    code_id: str | None
    arch: str | None


@dataclass
class Category:
    name: str
    color: str
    subcategories: list[str]


@dataclass
class Sample:
    stack_index: int  # -> StackTable
    time: float  # ms since epoch
    weight: int
    thread_cpu_delta: int | None  # µs
    wall_delta: float | None = (
        None  # ms, computed from consecutive sample times
    )


@dataclass
class StackEntry:
    index: int
    frame_index: int  # -> FrameTable
    prefix_index: int | None  # -> StackTable (parent), None = root
    category: int
    subcategory: int


@dataclass
class Frame:
    index: int
    func_index: int  # -> FuncTable
    address: int
    inline_depth: int
    category: int
    subcategory: int
    line: int | None
    column: int | None


@dataclass
class Func:
    index: int
    name: str  # resolved from stringArray
    is_js: bool
    resource_index: int
    file_name: str | None
    line_number: int | None
    column_number: int | None


@dataclass
class Resource:
    index: int
    lib_index: int  # -> top-level libs
    name: str  # resolved from stringArray
    type: int


@dataclass
class NativeSymbol:
    index: int
    name: str  # resolved from stringArray
    address: int
    function_size: int
    lib_index: int


@dataclass
class Marker:
    name: str  # resolved from stringArray
    start_time: float
    end_time: float
    category: int
    phase: int
    data: dict | None


# Prefixes of CPython interpreter functions that carry no semantic value.
_CPYTHON_NOISE_PREFIXES = (
    "PyEval_",
    "py_trampoline_",
    "PyObject_",
    "method_vectorcall",
    "_PyEval_",
    "_PyObject_",
    "_PyFunction_",
    "PyFunction_",
    "PyVectorcall_",
    "PyRun_",
    "PyRun_",
    "Py_RunMain",
    "Py_BytesMain",
    "pymain_",
    "call_function",
    "cfunction_",
    "slot_",
    "wrap_",
    "vectorcall_",
    "method_call",
    "builtin_",
    "type_call",
    "object_",
    "py::",
    "TAIL_CALL_",
    "run_mod",
    "run_eval_code_obj",
    "context_run",
    "thread_run",
)

# Regex-like patterns matched as substrings for additional noise.
_CPYTHON_NOISE_EXACT = frozenset(
    {
        "main",
        "start",
        "_libc_start_main_impl",
        "_libc_start_call_main",
        "clone3",
        "start_thread",
        "pythread_wrapper",
        "<module>:<string>",
    }
)


def _is_interesting_frame(name: str, is_js: bool) -> bool:
    """Decide whether a frame should be kept in ``smart`` filter mode.

    Keep if:
    - It's a Python/script frame (is_js=True), OR
    - It has a real symbol name AND is not CPython interpreter machinery.

    Drop: hex addresses (0x...), unresolved ``fun_XXXX`` stubs, CPython
    eval/call/trampoline noise, C runtime entry points.
    """
    if is_js:
        return True
    # Drop unresolved hex addresses.
    if name.startswith("0x"):
        return False
    # Drop samply-style unresolved stubs like "fun_1a2b3c".
    if name.startswith("fun_") and all(
        c in "0123456789abcdef" for c in name[4:]
    ):
        return False
    # Drop exact matches.
    if name in _CPYTHON_NOISE_EXACT:
        return False
    # Drop CPython interpreter noise.
    for prefix in _CPYTHON_NOISE_PREFIXES:
        if name.startswith(prefix):
            return False
    return True


@dataclass
class Thread:
    name: str
    pid: str | int
    tid: str | int
    is_main_thread: bool
    process_name: str | None
    process_type: str | int | None
    register_time: float | None
    unregister_time: float | None

    samples: list[Sample] = field(default_factory=list)
    stack_table: list[StackEntry] = field(default_factory=list)
    frames: list[Frame] = field(default_factory=list)
    funcs: list[Func] = field(default_factory=list)
    resources: list[Resource] = field(default_factory=list)
    native_symbols: list[NativeSymbol] = field(default_factory=list)
    markers: list[Marker] = field(default_factory=list)
    string_array: list[str] = field(default_factory=list)

    def resolve_stack(self, stack_index: int) -> list[tuple[str, int, bool]]:
        """Walk the stack linked list, return (func_name, category, is_js) leaf-to-root."""
        result = []
        idx: int | None = stack_index
        while idx is not None:
            entry = self.stack_table[idx]
            frame = self.frames[entry.frame_index]
            func = self.funcs[frame.func_index]
            result.append((func.name, frame.category, func.is_js))
            idx = entry.prefix_index
        return result

    def resolve_stack_names(
        self,
        stack_index: int,
        frame_filter: str | None = None,
        script_categories: frozenset[int] | None = None,
    ) -> tuple[str, ...]:
        """Return tuple of function names from root to leaf.

        *frame_filter*: ``"script"`` keeps only frames whose category is in
        *script_categories*; ``"native"`` keeps everything else; ``"smart"``
        keeps Python frames + interesting native frames (drops CPython
        interpreter noise and hex addresses); ``None``/``"all"`` keeps all.
        """
        triples = self.resolve_stack(stack_index)
        if (
            frame_filter
            and frame_filter != "all"
            and script_categories is not None
        ):
            if frame_filter == "script":
                triples = [
                    (n, c, j) for n, c, j in triples if c in script_categories
                ]
            elif frame_filter == "smart":
                triples = [
                    (n, c, j)
                    for n, c, j in triples
                    if _is_interesting_frame(n, j)
                ]
            else:  # native
                triples = [
                    (n, c, j)
                    for n, c, j in triples
                    if c not in script_categories
                ]
        return tuple(n for n, _c, _j in reversed(triples))


@dataclass
class Profile:
    meta: dict
    libs: list[Lib]
    categories: list[Category]
    threads: list[Thread]
    pages: list[dict]

    @property
    def interval(self) -> float:
        return self.meta.get("interval", 1.0)

    @property
    def start_time(self) -> float:
        return self.meta.get("startTime", 0.0)

    @property
    def product(self) -> str:
        return self.meta.get("product", "unknown")

    @property
    def script_categories(self) -> frozenset[int]:
        """Category indices that represent script/interpreted frames (e.g. 'Python')."""
        return frozenset(
            i
            for i, c in enumerate(self.categories)
            if c.name.lower() in ("python", "javascript", "js")
        )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@dataclass
class _SharedTables:
    """Pre-parsed tables from the ``shared`` section of a processed profile."""

    string_array: list[str]
    funcs: list[Func]
    resources: list[Resource]
    frames: list[Frame]
    stack_table: list[StackEntry]
    native_symbols: list[NativeSymbol]


def _parse_tables(
    tables: dict,
    sa: list[str],
    sources: dict | None = None,
) -> tuple[
    list[Func],
    list[Resource],
    list[Frame],
    list[StackEntry],
    list[NativeSymbol],
]:
    """Parse the core tables (funcTable, resourceTable, frameTable,
    stackTable, nativeSymbols) that may live per-thread or in ``shared``.

    *sources*: optional ``shared.sources`` table used in processed profiles
    where ``funcTable`` has ``source`` instead of ``fileName``.
    """

    # --- funcTable ---
    ft = tables["funcTable"]
    # In processed profiles, fileName is replaced by source (index into sources table).
    has_file_name = ft.get("fileName") is not None
    has_source = ft.get("source") is not None
    funcs: list[Func] = []
    for i in range(ft["length"]):
        file_name: str | None = None
        if has_file_name:
            fn_idx = ft["fileName"][i]
            file_name = sa[fn_idx] if fn_idx is not None else None
        elif has_source and sources is not None:
            src_idx = ft["source"][i]
            if src_idx is not None:
                fn_str_idx = sources["filename"][src_idx]
                file_name = sa[fn_str_idx] if fn_str_idx is not None else None
        funcs.append(
            Func(
                index=i,
                name=sa[ft["name"][i]],
                is_js=ft["isJS"][i],
                resource_index=ft["resource"][i],
                file_name=file_name,
                line_number=ft["lineNumber"][i]
                if ft.get("lineNumber")
                else None,
                column_number=ft["columnNumber"][i]
                if ft.get("columnNumber")
                else None,
            )
        )

    # --- resourceTable ---
    rt = tables["resourceTable"]
    resources = [
        Resource(
            index=i,
            lib_index=rt["lib"][i] if rt["lib"][i] is not None else -1,
            name=sa[rt["name"][i]],
            type=rt["type"][i],
        )
        for i in range(rt["length"])
    ]

    # --- frameTable ---
    frt = tables["frameTable"]
    frames = [
        Frame(
            index=i,
            func_index=frt["func"][i],
            address=frt["address"][i],
            inline_depth=frt["inlineDepth"][i]
            if frt.get("inlineDepth")
            else 0,
            category=frt["category"][i]
            if frt["category"][i] is not None
            else 0,
            subcategory=frt["subcategory"][i]
            if frt["subcategory"][i] is not None
            else 0,
            line=frt["line"][i] if frt.get("line") else None,
            column=frt["column"][i] if frt.get("column") else None,
        )
        for i in range(frt["length"])
    ]

    # --- stackTable ---
    # Processed profiles omit category/subcategory from stackTable; derive from frame.
    st = tables["stackTable"]
    has_stack_cat = "category" in st
    stack_table: list[StackEntry] = []
    for i in range(st["length"]):
        frame_idx = st["frame"][i]
        if has_stack_cat:
            cat = st["category"][i]
            subcat = st["subcategory"][i]
        else:
            cat = frames[frame_idx].category
            subcat = frames[frame_idx].subcategory
        stack_table.append(
            StackEntry(
                index=i,
                frame_index=frame_idx,
                prefix_index=st["prefix"][i],
                category=cat,
                subcategory=subcat,
            )
        )

    # --- nativeSymbols ---
    ns = tables.get("nativeSymbols", {"length": 0})
    native_symbols = [
        NativeSymbol(
            index=i,
            name=sa[ns["name"][i]],
            address=ns["address"][i],
            function_size=ns["functionSize"][i],
            lib_index=ns["libIndex"][i],
        )
        for i in range(ns["length"])
    ]

    return funcs, resources, frames, stack_table, native_symbols


def _parse_samples(raw_samples: dict) -> list[Sample]:
    """Parse a thread's samples table."""
    s = raw_samples
    cpu_deltas = s.get("threadCPUDelta")
    return [
        Sample(
            stack_index=s["stack"][i],
            time=s["time"][i],
            weight=s["weight"][i] if s.get("weight") else 1,
            thread_cpu_delta=cpu_deltas[i] if cpu_deltas else None,
        )
        for i in range(s["length"])
    ]


def _parse_markers(raw_markers: dict, sa: list[str]) -> list[Marker]:
    """Parse a thread's markers table."""
    m = raw_markers
    return [
        Marker(
            name=sa[m["name"][i]],
            start_time=m["startTime"][i],
            end_time=m["endTime"][i],
            category=m["category"][i],
            phase=m["phase"][i],
            data=m["data"][i] if m.get("data") else None,
        )
        for i in range(m["length"])
    ]


def _make_thread(
    raw: dict,
    *,
    samples: list[Sample],
    stack_table: list[StackEntry],
    frames: list[Frame],
    funcs: list[Func],
    resources: list[Resource],
    native_symbols: list[NativeSymbol],
    markers: list[Marker],
    string_array: list[str],
) -> Thread:
    return Thread(
        name=raw["name"],
        pid=raw["pid"],
        tid=raw["tid"],
        is_main_thread=raw.get("isMainThread", False),
        process_name=raw.get("processName"),
        process_type=raw.get("processType"),
        register_time=raw.get("registerTime"),
        unregister_time=raw.get("unregisterTime"),
        samples=samples,
        stack_table=stack_table,
        frames=frames,
        funcs=funcs,
        resources=resources,
        native_symbols=native_symbols,
        markers=markers,
        string_array=string_array,
    )


def parse_thread(raw: dict) -> Thread:
    """Parse a thread from the original (non-shared) profile format."""
    sa = raw["stringArray"]
    funcs, resources, frames, stack_table, native_symbols = _parse_tables(
        raw, sa
    )
    samples = _parse_samples(raw["samples"])
    markers = _parse_markers(raw.get("markers", {"length": 0}), sa)
    return _make_thread(
        raw,
        samples=samples,
        stack_table=stack_table,
        frames=frames,
        funcs=funcs,
        resources=resources,
        native_symbols=native_symbols,
        markers=markers,
        string_array=sa,
    )


def _parse_shared_tables(shared: dict) -> _SharedTables:
    """Parse the ``shared`` section of a processed profile."""
    sa = shared["stringArray"]
    sources = shared.get("sources")
    funcs, resources, frames, stack_table, native_symbols = _parse_tables(
        shared, sa, sources=sources
    )
    return _SharedTables(
        string_array=sa,
        funcs=funcs,
        resources=resources,
        frames=frames,
        stack_table=stack_table,
        native_symbols=native_symbols,
    )


def _parse_thread_shared(raw: dict, shared: _SharedTables) -> Thread:
    """Parse a thread that references shared tables (processed profile)."""
    sa = shared.string_array
    samples = _parse_samples(raw["samples"])
    markers = _parse_markers(raw.get("markers", {"length": 0}), sa)
    return _make_thread(
        raw,
        samples=samples,
        stack_table=shared.stack_table,
        frames=shared.frames,
        funcs=shared.funcs,
        resources=shared.resources,
        native_symbols=shared.native_symbols,
        markers=markers,
        string_array=sa,
    )


def parse_profile(path: str) -> Profile:
    """Load a Firefox profiler JSON (or .json.gz) file.

    Supports both the original per-thread format and the processed/preprocessed
    format where tables live in a top-level ``shared`` section.
    """
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)

    meta = raw["meta"]

    libs = [
        Lib(
            name=lib["name"],
            path=lib["path"],
            debug_name=lib["debugName"],
            debug_path=lib["debugPath"],
            breakpad_id=lib["breakpadId"],
            code_id=lib.get("codeId"),
            arch=lib.get("arch"),
        )
        for lib in raw.get("libs", [])
    ]

    categories = [
        Category(
            name=cat["name"],
            color=cat["color"],
            subcategories=cat.get("subcategories", []),
        )
        for cat in meta.get("categories", [])
    ]

    if "shared" in raw:
        shared = _parse_shared_tables(raw["shared"])
        threads = [
            _parse_thread_shared(t, shared) for t in raw.get("threads", [])
        ]
    else:
        threads = [parse_thread(t) for t in raw.get("threads", [])]

    return Profile(
        meta=meta,
        libs=libs,
        categories=categories,
        threads=threads,
        pages=raw.get("pages", []),
    )


# ---------------------------------------------------------------------------
# Path shortening
# ---------------------------------------------------------------------------


class PathShortener:
    """Discovers common path prefixes across function names and replaces them
    with short numbered labels like ``//1/relative.py``."""

    def __init__(self, profile: Profile) -> None:
        # Collect unique file paths from "func_name:/abs/path" names.
        # Inline extraction + dedup to avoid per-func method call overhead.
        names = set(func.name for t in profile.threads for func in t.funcs)
        seen_paths: set[str] = set()
        for name in names:
            idx = name.find(":/")
            if idx != -1:
                seen_paths.add(name[idx + 1 :])

        self.prefixes: list[tuple[str, str]] = []  # (prefix, label)
        if seen_paths:
            self.prefixes = self._find_prefixes(seen_paths)

        # Cache for already-shortened names.
        self._cache: dict[str, str] = {}

    @staticmethod
    def _find_prefixes(paths: set[str]) -> list[tuple[str, str]]:
        """Find a small set of meaningful common prefixes.

        Strategy:
        1. Greedily pick the prefix covering the most uncovered files.
        2. After the greedy pass, split: for each selected prefix, check if a
           child subtree accounts for >=15% of that prefix's files.  If so,
           promote the child to its own label (saves characters on every line
           that uses it).
        """
        total = len(paths)
        threshold = max(total * 0.02, 3)  # at least 2% of files or 3 files

        # Map paths to integer indices for faster set operations.
        path_list = sorted(paths)
        path_idx = {p: i for i, p in enumerate(path_list)}

        # Count files per directory prefix (using indices).
        prefix_files: dict[str, set[int]] = defaultdict(set)
        for p in path_list:
            idx = path_idx[p]
            parts = os.path.dirname(p).split("/")
            # Build prefixes incrementally instead of repeated slicing+joining.
            if len(parts) >= 3:
                cur = "/".join(parts[:3])
                prefix_files[cur].add(idx)
                for j in range(3, len(parts)):
                    cur = cur + "/" + parts[j]
                    prefix_files[cur].add(idx)

        # Pre-filter prefixes that can never meet the threshold.
        prefix_files = {
            p: f for p, f in prefix_files.items() if len(f) >= threshold
        }

        uncovered = set(range(total))
        selected: list[str] = []
        max_labels = 15

        # --- greedy pass ---
        while uncovered and len(selected) < max_labels:
            best_prefix = ""
            best_count = 0
            for prefix, files in prefix_files.items():
                covered_count = len(files & uncovered)
                if covered_count > best_count or (
                    covered_count >= best_count * 0.9
                    and len(prefix) > len(best_prefix)
                ):
                    best_prefix = prefix
                    best_count = covered_count

            if best_count < threshold:
                break

            selected.append(best_prefix)
            uncovered -= prefix_files[best_prefix]

        # --- split pass: promote large subtrees within each selected prefix ---
        split_threshold = 0.15  # child must cover >=15% of parent's files
        extra: list[str] = []
        for parent in list(selected):
            parent_files = prefix_files[parent]
            parent_count = len(parent_files)
            # Find children that are strictly longer than parent.
            children = [
                (p, f)
                for p, f in prefix_files.items()
                if p.startswith(parent + "/")
                and len(f) >= parent_count * split_threshold
            ]
            # Pick the longest (most specific) children that don't overlap.
            children.sort(key=lambda pf: -len(pf[0]))
            child_covered: set[int] = set()
            for child_prefix, child_files in children:
                newly = child_files - child_covered
                if len(newly) >= parent_count * split_threshold:
                    extra.append(child_prefix)
                    child_covered |= child_files
                if len(extra) + len(selected) >= max_labels:
                    break

        selected.extend(extra)

        # Sort longest-first so replacement picks the most specific match.
        selected.sort(key=lambda p: -len(p))

        return [(prefix, str(i)) for i, prefix in enumerate(selected, 1)]

    @staticmethod
    def _extract_path(name: str) -> str | None:
        """Extract the absolute file path from 'func_name:/abs/path.py'."""
        idx = name.find(":/")
        if idx == -1:
            return None
        return name[idx + 1 :]

    def shorten(self, name: str) -> str:
        """Shorten and reorder: 'func:/long/path' -> '//1/short:func'."""
        cached = self._cache.get(name)
        if cached is not None:
            return cached

        idx = name.find(":/")
        if idx == -1:
            self._cache[name] = name
            return name

        func_part = name[:idx]  # "func_name"
        path_part = name[idx + 1 :]  # "/abs/path/to/file.py"

        for prefix, label in self.prefixes:
            if path_part.startswith(prefix + "/"):
                rest = path_part[len(prefix) + 1 :]
                result = f"//{label}/{rest}:{func_part}"
                self._cache[name] = result
                return result
            if path_part.startswith(prefix):
                result = f"//{label}:{func_part}"
                self._cache[name] = result
                return result

        # No prefix matched — still swap to path:func order.
        result = f"{path_part}:{func_part}"
        self._cache[name] = result
        return result

    def print_legend(self) -> None:
        """Print the prefix legend table."""
        if not self.prefixes:
            return
        print("Path prefixes:")
        for prefix, label in self.prefixes:
            print(f"  //{label}  =  {prefix}")
        print()


# An identity shortener when --full-paths is on.
class _NoShorten:
    def shorten(self, name: str) -> str:
        return name

    def print_legend(self) -> None:
        pass


_NO_SHORTEN = _NoShorten()


# ---------------------------------------------------------------------------
# Wall-delta computation
# ---------------------------------------------------------------------------


def compute_wall_deltas(threads: list[Thread]) -> None:
    """Set wall_delta on each sample from consecutive time differences within a thread."""
    for t in threads:
        for i, s in enumerate(t.samples):
            if i == 0:
                s.wall_delta = None
            else:
                s.wall_delta = s.time - t.samples[i - 1].time


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def report_summary(profile: Profile) -> None:
    """Print a brief summary of the profile."""
    print(f"Product:    {profile.product}")
    print(f"Interval:   {profile.interval} ms")
    print(f"Libs:       {len(profile.libs)}")
    print(f"Categories: {', '.join(c.name for c in profile.categories)}")
    print(f"Threads:    {len(profile.threads)}")
    total_samples = sum(len(t.samples) for t in profile.threads)
    print(f"Samples:    {total_samples}")
    print()

    # Detect duplicate thread names for disambiguation.
    name_counts: Counter[str] = Counter(
        t.name for t in profile.threads if t.samples
    )
    dup_names = {n for n, c in name_counts.items() if c > 1}

    for t in profile.threads:
        if not t.samples:
            continue
        label = f"{t.name}:{t.tid}" if t.name in dup_names else t.name
        cpu_total = None
        if t.samples[0].thread_cpu_delta is not None:
            cpu_total = sum(s.thread_cpu_delta or 0 for s in t.samples)
        wall_total = sum(
            s.wall_delta for s in t.samples if s.wall_delta is not None
        )
        cpu_str = (
            f"  CPU: {cpu_total / 1000:.1f}ms"
            if cpu_total is not None
            else ""
        )
        wall_str = f"  Wall: {wall_total:.1f}ms" if wall_total > 0 else ""
        wait_str = ""
        if cpu_total is not None and wall_total > 0:
            wait_ms = wall_total - cpu_total / 1000
            wait_str = f"  Wait: {wait_ms:.1f}ms"
        print(
            f"  {label:40s}  samples={len(t.samples):6d}{cpu_str}{wall_str}{wait_str}"
        )


def report_costly_stacks(
    profile: Profile,
    *,
    top_n: int = 25,
    min_samples: int = 1,
    thread_filter: str | None = None,
    frame_filter: str | None = None,
    sort_by: str = "wall",
    shortener: PathShortener | _NoShorten = _NO_SHORTEN,
) -> None:
    """Aggregate samples by full call chain and print the most frequent."""
    script_cats = profile.script_categories
    # Key: (thread_label, tid, stack_names) — tid keeps same-named threads distinct
    _StackKey = tuple[str, str | int, tuple[str, ...]]
    stack_counts: Counter[_StackKey] = Counter()
    stack_cpu: dict[_StackKey, int] = {}
    stack_wall: dict[_StackKey, float] = {}

    # Detect duplicate thread names so we can disambiguate in display.
    name_counts: Counter[str] = Counter()
    for t in profile.threads:
        if thread_filter and thread_filter.lower() not in t.name.lower():
            continue
        name_counts[t.name] += 1
    dup_names = {n for n, c in name_counts.items() if c > 1}

    for t in profile.threads:
        if not t.samples:
            continue
        if thread_filter and thread_filter.lower() not in t.name.lower():
            continue

        label = f"{t.name}:{t.tid}" if t.name in dup_names else t.name
        for sample in t.samples:
            names = t.resolve_stack_names(
                sample.stack_index, frame_filter, script_cats
            )
            if not names:
                continue
            key: _StackKey = (label, t.tid, names)
            stack_counts[key] += sample.weight
            if sample.thread_cpu_delta is not None:
                stack_cpu[key] = (
                    stack_cpu.get(key, 0) + sample.thread_cpu_delta
                )
            if sample.wall_delta is not None:
                stack_wall[key] = stack_wall.get(key, 0.0) + sample.wall_delta

    total_samples = sum(stack_counts.values())
    if total_samples == 0:
        print("No samples found.")
        return

    # Sort by the requested metric.
    if sort_by == "cpu":
        ranked = sorted(
            stack_counts.items(),
            key=lambda kv: stack_cpu.get(kv[0], 0),
            reverse=True,
        )
    elif sort_by == "wait":
        ranked = sorted(
            stack_counts.items(),
            key=lambda kv: stack_wall.get(kv[0], 0.0)
            - stack_cpu.get(kv[0], 0) / 1000,
            reverse=True,
        )
    else:  # wall (default) — sort by sample count as before
        ranked = stack_counts.most_common()

    print(f"Total samples: {total_samples}  (sorted by {sort_by})")
    print(f"Showing top {top_n} call chains (min {min_samples} samples):\n")

    shown = 0
    for rank, (key, count) in enumerate(ranked, 1):
        if count < min_samples:
            if sort_by == "wall":
                break  # counts are monotonically decreasing
            continue
        label, _tid, names = key

        pct = count / total_samples * 100
        cpu_us = stack_cpu.get(key)
        wall_ms = stack_wall.get(key)
        cpu_str = (
            f"  CPU: {cpu_us / 1000:.1f}ms" if cpu_us is not None else ""
        )
        wait_str = ""
        if cpu_us is not None and wall_ms is not None:
            wait_ms = wall_ms - cpu_us / 1000
            wait_str = f"  Wait: {wait_ms:.1f}ms"

        print(
            f"#{rank}  {count} samples ({pct:.1f}%){cpu_str}{wait_str}  [{label}]"
        )
        for name in reversed(names):
            print(f"    {shortener.shorten(name)}")
        print()

        shown += 1
        if shown >= top_n:
            break


def report_hotspots(
    profile: Profile,
    *,
    top_n: int = 25,
    thread_filter: str | None = None,
    frame_filter: str | None = None,
    sort_by: str = "wall",
    shortener: PathShortener | _NoShorten = _NO_SHORTEN,
) -> None:
    """Show functions with the most self-time (leaf of stack)."""
    script_cats = profile.script_categories
    self_counts: Counter[str] = Counter()
    self_cpu: dict[str, int] = {}
    self_wall: dict[str, float] = {}
    total_counts: Counter[str] = Counter()

    for t in profile.threads:
        if not t.samples:
            continue
        if thread_filter and thread_filter.lower() not in t.name.lower():
            continue

        for sample in t.samples:
            names = t.resolve_stack_names(
                sample.stack_index, frame_filter, script_cats
            )
            if not names:
                continue
            leaf = names[-1] if names else "<empty>"
            self_counts[leaf] += sample.weight
            if sample.thread_cpu_delta is not None:
                self_cpu[leaf] = (
                    self_cpu.get(leaf, 0) + sample.thread_cpu_delta
                )
            if sample.wall_delta is not None:
                self_wall[leaf] = self_wall.get(leaf, 0.0) + sample.wall_delta
            for name in set(names):
                total_counts[name] += sample.weight

    total_samples = sum(self_counts.values())
    if total_samples == 0:
        print("No samples found.")
        return

    # Sort by the requested metric.
    if sort_by == "cpu":
        ranked = sorted(
            self_counts.items(),
            key=lambda kv: self_cpu.get(kv[0], 0),
            reverse=True,
        )[:top_n]
    elif sort_by == "wait":
        ranked = sorted(
            self_counts.items(),
            key=lambda kv: self_wall.get(kv[0], 0.0)
            - self_cpu.get(kv[0], 0) / 1000,
            reverse=True,
        )[:top_n]
    else:  # wall (default)
        ranked = self_counts.most_common(top_n)

    print(f"Sorted by: {sort_by}")
    print(
        f"{'Function':60s} {'Self':>8s} {'Self%':>6s} {'Total':>8s} {'Total%':>6s} {'CPU(ms)':>8s} {'Wait(ms)':>9s}"
    )
    print("-" * 111)

    for name, count in ranked:
        pct = count / total_samples * 100
        tot = total_counts[name]
        tot_pct = tot / total_samples * 100
        cpu = self_cpu.get(name)
        wall = self_wall.get(name)
        cpu_str = f"{cpu / 1000:.1f}" if cpu is not None else ""
        wait_str = ""
        if cpu is not None and wall is not None:
            wait_str = f"{wall - cpu / 1000:.1f}"
        name = shortener.shorten(name)
        display = name if len(name) <= 60 else "…" + name[-(59):]
        print(
            f"{display:60s} {count:8d} {pct:5.1f}% {tot:8d} {tot_pct:5.1f}% {cpu_str:>8s} {wait_str:>9s}"
        )


# ---------------------------------------------------------------------------
# Call tree report
# ---------------------------------------------------------------------------


@dataclass
class TreeNode:
    name: str
    self_samples: int = 0
    total_samples: int = 0
    self_cpu: int = 0  # µs
    total_cpu: int = 0  # µs
    self_wall: float = 0.0  # ms
    total_wall: float = 0.0  # ms
    children: dict[str, "TreeNode"] = field(default_factory=dict)


def _build_calltree(
    profile: Profile,
    *,
    thread_filter: str | None = None,
    frame_filter: str | None = None,
) -> TreeNode:
    """Build a merged call tree from all matching threads."""
    script_cats = profile.script_categories
    root = TreeNode(name="[all threads]")

    for t in profile.threads:
        if not t.samples:
            continue
        if thread_filter and thread_filter.lower() not in t.name.lower():
            continue

        for sample in t.samples:
            names = t.resolve_stack_names(
                sample.stack_index, frame_filter, script_cats
            )
            if not names:
                continue

            weight = sample.weight
            cpu = sample.thread_cpu_delta or 0
            wall = sample.wall_delta or 0.0

            root.total_samples += weight
            root.total_cpu += cpu
            root.total_wall += wall

            node = root
            for i, name in enumerate(names):
                child = node.children.get(name)
                if child is None:
                    child = TreeNode(name=name)
                    node.children[name] = child
                child.total_samples += weight
                child.total_cpu += cpu
                child.total_wall += wall
                if i == len(names) - 1:
                    child.self_samples += weight
                    child.self_cpu += cpu
                    child.self_wall += wall
                node = child

    return root


def _sort_key(node: TreeNode, sort_by: str) -> float:
    if sort_by == "cpu":
        return node.total_cpu
    elif sort_by == "wait":
        return node.total_wall - node.total_cpu / 1000
    return node.total_samples  # wall


def _render_calltree(
    root: TreeNode,
    *,
    min_pct: float = 0.5,
    max_depth: int = 20,
    sort_by: str = "wall",
    shortener: "PathShortener | _NoShorten" = _NO_SHORTEN,
) -> None:
    """Render the call tree with box-drawing characters."""
    total = root.total_samples
    if total == 0:
        print("No samples found.")
        return

    total_cpu = root.total_cpu
    total_wall = root.total_wall

    def _format_node(node: TreeNode) -> str:
        pct = node.total_samples / total * 100
        parts = [f"{pct:5.1f}% ({node.total_samples})"]
        if total_cpu > 0:
            cpu_ms = node.total_cpu / 1000
            parts.append(f"CPU:{cpu_ms:.0f}ms")
        if total_cpu > 0 and total_wall > 0:
            wait_ms = node.total_wall - node.total_cpu / 1000
            if wait_ms > 0:
                parts.append(f"Wait:{wait_ms:.0f}ms")
        return "  ".join(parts)

    def _collapse_chain(node: TreeNode) -> tuple[list[str], TreeNode]:
        """Collapse single-child chains with no self-time into one line."""
        chain = [shortener.shorten(node.name)]
        while len(node.children) == 1 and node.self_samples == 0:
            child = next(iter(node.children.values()))
            chain.append(shortener.shorten(child.name))
            node = child
        return chain, node

    def _render(node: TreeNode, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return

        children = sorted(
            node.children.values(),
            key=lambda c: _sort_key(c, sort_by),
            reverse=True,
        )

        # Filter children below min_pct.
        min_count = total * min_pct / 100
        visible = [c for c in children if c.total_samples >= min_count]
        hidden = len(children) - len(visible)

        for i, child in enumerate(visible):
            is_last = i == len(visible) - 1 and hidden == 0
            connector = "└── " if is_last else "├── "
            child_prefix = prefix + ("    " if is_last else "│   ")

            chain, end_node = _collapse_chain(child)
            label = " -> ".join(chain)
            stats = _format_node(child)
            print(f"{prefix}{connector}{stats}  {label}")

            _render(end_node, child_prefix, depth + 1)

        if hidden > 0:
            connector = "└── " if True else "├── "
            # Sum up hidden children.
            hidden_samples = sum(
                c.total_samples
                for c in children
                if c.total_samples < min_count
            )
            hidden_pct = hidden_samples / total * 100
            print(
                f"{prefix}└── {hidden_pct:5.1f}% ({hidden_samples})  [{hidden} other children]"
            )

    # Print root.
    stats = _format_node(root)
    print(f"{stats}  [all threads]")
    _render(root, "", 0)


def report_calltree(
    profile: Profile,
    *,
    thread_filter: str | None = None,
    frame_filter: str | None = None,
    sort_by: str = "wall",
    min_pct: float = 0.5,
    max_depth: int = 20,
    shortener: PathShortener | _NoShorten = _NO_SHORTEN,
) -> None:
    """Print a top-down merged call tree."""
    root = _build_calltree(
        profile,
        thread_filter=thread_filter,
        frame_filter=frame_filter,
    )
    _render_calltree(
        root,
        min_pct=min_pct,
        max_depth=max_depth,
        sort_by=sort_by,
        shortener=shortener,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse Firefox profiler JSON and report costly call chains."
    )
    parser.add_argument(
        "profile", help="Path to profile .json or .json.gz file"
    )
    parser.add_argument(
        "--report",
        choices=["summary", "stacks", "hotspots", "calltree", "all"],
        default="hotspots",
        help="Report type (default: hotspots)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Number of entries to show (default: 30)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="Minimum samples for a call chain to be shown (default: 2)",
    )
    parser.add_argument(
        "--thread",
        type=str,
        default=None,
        help="Filter to threads whose name contains this string",
    )
    parser.add_argument(
        "--frames",
        choices=["all", "native", "script", "smart"],
        default="all",
        help="Frame filter: all, native, script, or smart (Python + interesting native) (default: all)",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="Only include samples starting this many seconds after profiling began",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="Exclude samples from this many seconds after profiling began onwards",
    )
    parser.add_argument(
        "--sort",
        choices=["wall", "cpu", "wait"],
        default="wall",
        help="Sort hotspots/stacks by: wall (sample count), cpu, or wait time (default: wall)",
    )
    parser.add_argument(
        "--full-paths",
        action="store_true",
        default=False,
        help="Disable the replacement of common path prefixes with short //N/ labels",
    )
    parser.add_argument(
        "--min-pct",
        type=float,
        default=0.5,
        help="Calltree: prune nodes below this %% of total (default: 0.5)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=20,
        help="Calltree: max render depth (default: 20)",
    )
    args = parser.parse_args()

    profile = parse_profile(args.profile)

    # Filter samples to the requested time window.
    # Sample times use a monotonic clock unrelated to meta.startTime,
    # so we derive the origin from the earliest sample across all threads.
    if args.start is not None or args.end is not None:
        origin = min(
            (s.time for t in profile.threads for s in t.samples),
            default=0.0,
        )
        lo_ms = origin + args.start * 1000 if args.start is not None else None
        hi_ms = origin + args.end * 1000 if args.end is not None else None
        for t in profile.threads:
            t.samples = [
                s
                for s in t.samples
                if (lo_ms is None or s.time >= lo_ms)
                and (hi_ms is None or s.time < hi_ms)
            ]

    compute_wall_deltas(profile.threads)

    shortener: PathShortener | _NoShorten = _NO_SHORTEN
    if not args.full_paths:
        shortener = PathShortener(profile)

    reports = {
        "summary": lambda: report_summary(profile),
        "stacks": lambda: report_costly_stacks(
            profile,
            top_n=args.top,
            min_samples=args.min_samples,
            thread_filter=args.thread,
            frame_filter=args.frames,
            sort_by=args.sort,
            shortener=shortener,
        ),
        "hotspots": lambda: report_hotspots(
            profile,
            top_n=args.top,
            thread_filter=args.thread,
            frame_filter=args.frames,
            sort_by=args.sort,
            shortener=shortener,
        ),
        "calltree": lambda: report_calltree(
            profile,
            thread_filter=args.thread,
            frame_filter=args.frames,
            sort_by=args.sort,
            min_pct=args.min_pct,
            max_depth=args.max_depth,
            shortener=shortener,
        ),
    }

    if args.report == "all":
        for name, fn in reports.items():
            print(f"{'=' * 40}")
            print(f" {name.upper()}")
            print(f"{'=' * 40}")
            fn()
            print()
    else:
        reports[args.report]()

    # Print legend at the end so it's visible after the report.
    shortener.print_legend()


if __name__ == "__main__":
    main()
