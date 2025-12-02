"""
Dynamic Memory Management Visualizer (Paging, Segmentation, FIFO & LRU)
Single-file Streamlit app.

Run:
    pip install streamlit plotly
    streamlit run app.py

This app implements:
- Paging simulation with page table + frame table
- Segmentation (simple base+limit mapping)
- Page replacement policies: FIFO, LRU
- User inputs: memory size, frame size, sequence of page accesses, segments
- Visualizations: frames, page table, segment table, page faults/hits plot, event log

"""

from collections import deque, OrderedDict
import streamlit as st
import plotly.graph_objects as go
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time

# ---------------------------
# Simulation Engine
# ---------------------------

@dataclass
class PageTableEntry:
    page_no: int
    frame_no: Optional[int] = None
    valid: bool = False
    last_used: float = 0.0  # for LRU

@dataclass
class Frame:
    frame_no: int
    occupied: bool = False
    page_no: Optional[int] = None
    pid: Optional[int] = None  # process id if multi-process (optional)

@dataclass
class Segment:
    seg_id: int
    base: int
    limit: int

class ReplacementPolicy:
    FIFO = "FIFO"
    LRU = "LRU"

class MemoryManager:
    def __init__(self, physical_mem_kb: int, page_size_kb: int):
        # sizes in KB
        self.physical_mem_kb = physical_mem_kb
        self.page_size_kb = page_size_kb
        self.frame_count = max(1, physical_mem_kb // page_size_kb)

        # frame table
        self.frames: List[Frame] = [Frame(i) for i in range(self.frame_count)]

        # page table: maps virtual page -> PageTableEntry
        self.page_table: Dict[int, PageTableEntry] = {}

        # replacement structures
        self.fifo_queue: deque = deque()  # store frame indexes in load order
        self.time_counter = 0.0

        # stats
        self.hits = 0
        self.faults = 0
        self.event_log: List[str] = []

        # default policy
        self.policy = ReplacementPolicy.FIFO

        # segmentation
        self.segments: Dict[int, Segment] = {}

    # ---------------------
    # Configuration
    # ---------------------
    def set_policy(self, policy: str):
        assert policy in (ReplacementPolicy.FIFO, ReplacementPolicy.LRU)
        self.policy = policy

    def reset(self):
        self.frames = [Frame(i) for i in range(self.frame_count)]
        self.page_table = {}
        self.fifo_queue = deque()
        self.hits = 0
        self.faults = 0
        self.event_log = []
        self.time_counter = 0.0
        self.segments = {}

    # ---------------------
    # Segmentation APIs
    # ---------------------
    def create_segment(self, seg_id: int, base_kb: int, limit_kb: int):
        # base and limit presume virtual address space units in KB
        if seg_id in self.segments:
            raise ValueError("Segment already exists")
        self.segments[seg_id] = Segment(seg_id, base_kb, limit_kb)
        self.event_log.append(f"Segment {seg_id} created: base={base_kb}KB limit={limit_kb}KB")

    def translate_segment(self, seg_id: int, offset_kb: int) -> Tuple[int, int]:
        # returns (virtual_page_no, page_offset)
        if seg_id not in self.segments:
            raise ValueError("Segment not found")
        seg = self.segments[seg_id]
        if offset_kb < 0 or offset_kb >= seg.limit:
            raise IndexError("Segmentation Fault: offset out of bound")
        virt_addr_kb = seg.base + offset_kb
        page_no = virt_addr_kb // self.page_size_kb
        page_offset = virt_addr_kb % self.page_size_kb
        return page_no, page_offset

    # ---------------------
    # Paging / VM APIs
    # ---------------------
    def access_page(self, page_no: int) -> Tuple[bool, Optional[int]]:
        """
        Access a virtual page number. Returns (hit, frame_no_or_None)
        Updates stats, handles page fault and replacement as needed.
        """
        self.time_counter += 1.0
        pte = self.page_table.get(page_no)

        # if PTE missing, create an entry
        if pte is None:
            pte = PageTableEntry(page_no)
            self.page_table[page_no] = pte

        # HIT
        if pte.valid:
            pte.last_used = self.time_counter
            self.hits += 1
            self.event_log.append(f"Hit: Page {page_no} in Frame {pte.frame_no}")
            # For LRU we may update ordering elsewhere
            return True, pte.frame_no

        # FAULT
        self.faults += 1
        self.event_log.append(f"Fault: Page {page_no} not in memory")

        # find a free frame
        free_frame = next((f for f in self.frames if not f.occupied), None)
        if free_frame is not None:
            self._load_page_into_frame(page_no, free_frame.frame_no)
            return False, free_frame.frame_no

        # no free frame -> replacement
        replaced_frame_no = self._replace_frame(page_no)
        return False, replaced_frame_no

    def _load_page_into_frame(self, page_no: int, frame_no: int):
        frame = self.frames[frame_no]
        frame.occupied = True
        frame.page_no = page_no

        # set PTE
        pte = self.page_table[page_no]
        pte.frame_no = frame_no
        pte.valid = True
        pte.last_used = self.time_counter

        # enqueue FIFO
        self.fifo_queue.append(frame_no)
        self.event_log.append(f"Loaded: Page {page_no} -> Frame {frame_no}")

    def _replace_frame(self, page_no: int) -> int:
        if self.policy == ReplacementPolicy.FIFO:
            # pop oldest frame
            victim_frame_no = self.fifo_queue.popleft()
        else:  # LRU
            # choose frame whose page has smallest last_used
            min_time = float('inf')
            victim_frame_no = None
            for f in self.frames:
                if f.occupied and f.page_no is not None:
                    p = self.page_table.get(f.page_no)
                    if p and p.last_used < min_time:
                        min_time = p.last_used
                        victim_frame_no = f.frame_no
            if victim_frame_no is None:
                # fallback
                victim_frame_no = 0

        # evict
        victim = self.frames[victim_frame_no]
        evicted_page = victim.page_no
        self.event_log.append(f"Evicting: Page {evicted_page} from Frame {victim_frame_no}")

        # invalidate old PTE
        if evicted_page is not None:
            old_pte = self.page_table.get(evicted_page)
            if old_pte:
                old_pte.valid = False
                old_pte.frame_no = None

        # load new page here
        victim.page_no = page_no
        self.event_log.append(f"Loaded: Page {page_no} -> Frame {victim_frame_no} (replaced)")

        # update new PTE
        new_pte = self.page_table[page_no]
        new_pte.frame_no = victim_frame_no
        new_pte.valid = True
        new_pte.last_used = self.time_counter

        # FIFO: append this frame as newest
        if self.policy == ReplacementPolicy.FIFO:
            self.fifo_queue.append(victim_frame_no)
        return victim_frame_no

    # ---------------------
    # Utilities
    # ---------------------
    def get_frame_table(self) -> List[Frame]:
        return self.frames

    def get_page_table_snapshot(self) -> Dict[int, PageTableEntry]:
        return dict(self.page_table)

    def get_stats(self) -> Dict[str, float]:
        total_refs = self.hits + self.faults
        hit_ratio = (self.hits / total_refs) if total_refs > 0 else 0.0
        fault_rate = (self.faults / total_refs) if total_refs > 0 else 0.0
        return {
            "hits": self.hits,
            "faults": self.faults,
            "hit_ratio": round(hit_ratio, 4),
            "fault_rate": round(fault_rate, 4),
            "total_refs": total_refs,
        }

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Memory Management Visualizer", layout="wide")
st.title("Memory Management Visualizer — Paging, Segmentation & Replacement")

# Sidebar inputs
st.sidebar.header("Simulation Settings")
physical_mem_kb = st.sidebar.number_input("Physical memory (KB)", min_value=4, max_value=65536, value=32, step=4)
page_size_kb = st.sidebar.selectbox("Page size (KB)", options=[1,2,4,8,16,32], index=2)
policy = st.sidebar.selectbox("Replacement Policy", options=[ReplacementPolicy.FIFO, ReplacementPolicy.LRU])

# initialize manager in session state
if 'manager' not in st.session_state:
    st.session_state.manager = MemoryManager(physical_mem_kb, page_size_kb)
else:
    # if sizes changed, re-create manager
    mgr = st.session_state.manager
    if mgr.physical_mem_kb != physical_mem_kb or mgr.page_size_kb != page_size_kb:
        st.session_state.manager = MemoryManager(physical_mem_kb, page_size_kb)

manager: MemoryManager = st.session_state.manager
manager.set_policy(policy)

st.sidebar.markdown("---")

# Controls for access sequence
st.sidebar.header("Access / Workload")
access_input = st.sidebar.text_area("Page access sequence (comma separated page numbers)", value="0,1,2,3,2,1,4,0,1,5")
run_speed = st.sidebar.slider("Playback speed (ops/sec)", min_value=0.5, max_value=5.0, value=1.0)

if st.sidebar.button("Reset Simulation"):
    manager.reset()
    st.sidebar.success("Simulation reset")

st.sidebar.markdown("---")
st.sidebar.header("Segmentation")
seg_id = st.sidebar.number_input("Segment ID (int)", min_value=0, value=0)
seg_base = st.sidebar.number_input("Segment base (KB)", min_value=0, value=0)
seg_limit = st.sidebar.number_input("Segment limit (KB)", min_value=1, value=8)
if st.sidebar.button("Create Segment"):
    try:
        manager.create_segment(seg_id, seg_base, seg_limit)
        st.sidebar.success(f"Created segment {seg_id}")
    except Exception as e:
        st.sidebar.error(str(e))

st.sidebar.markdown("---")

# Layout: left (controls & logs), right (visuals)
col1, col2 = st.columns([1, 2])

# Left column: controls
with col1:
    st.subheader("Controls")
    if st.button("Step Once"):
        # perform one step from access_input sequence
        seq = [int(x.strip()) for x in access_input.split(',') if x.strip()!='']
        if len(seq) > 0:
            page = seq[0]
            # mutate input by removing first element
            remaining = seq[1:]
            st.sidebar.text_area("Page access sequence (comma separated page numbers)", value=','.join(map(str, remaining)))
            hit, frame = manager.access_page(page)
            st.success(f"Accessed page {page} -> {'HIT' if hit else 'FAULT'} (frame={frame})")

    if st.button("Run Sequence"):
        seq = [int(x.strip()) for x in access_input.split(',') if x.strip()!='']
        if len(seq)==0:
            st.warning("No pages to run")
        else:
            for p in seq:
                hit, frame = manager.access_page(p)
                # redraw UI incrementally
                time.sleep(1.0 / run_speed)
                # streamlit rerender will show updates after loop finishes; small sleep just for pacing
            st.success("Sequence run finished")

    st.subheader("Event Log")
    # show last 20 events
    for ev in manager.event_log[-20:][::-1]:
        st.write(ev)

# Right column: visualizations
with col2:
    st.subheader("Physical Frames")
    frames = manager.get_frame_table()
    # draw frames as horizontal rectangles using plotly
    fig = go.Figure()

    x = []
    y = []
    text = []
    colors = []
    for f in frames:
        label = f"F{f.frame_no}: " + (f"P{f.page_no}" if f.page_no is not None else "Free")
        text.append(label)
        colors.append("lightgreen" if f.occupied else "lightgray")
        x.append(f.frame_no)
        y.append(1)

    fig.add_trace(go.Bar(x=x, y=y, text=text, marker_color=colors, hovertext=text, hoverinfo='text'))
    fig.update_layout(height=150, showlegend=False, yaxis=dict(showticklabels=False))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Page Table (snapshot)")
    ptable = manager.get_page_table_snapshot()
    if len(ptable)==0:
        st.write("Page table empty — no pages referenced yet")
    else:
        # show table with columns: page_no, valid, frame_no, last_used
        rows = []
        for pno, pte in sorted(ptable.items()):
            rows.append({
                "page": pno,
                "valid": pte.valid,
                "frame": pte.frame_no,
                "last_used": pte.last_used
            })
        st.table(rows)

    st.subheader("Segmentation Table")
    if len(manager.segments)==0:
        st.write("No segments defined")
    else:
        seg_rows = []
        for sid, s in manager.segments.items():
            seg_rows.append({"seg_id": sid, "base_kb": s.base, "limit_kb": s.limit})
        st.table(seg_rows)

    st.subheader("Statistics")
    stats = manager.get_stats()
    st.metric("Page Accesses", stats['total_refs'])
    st.metric("Page Faults", stats['faults'])
    st.metric("Hit Ratio", stats['hit_ratio'])

    # Plot page faults/hits pie or bar
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=["Hits", "Faults"], y=[stats['hits'], stats['faults']]))
    fig2.update_layout(height=300, title="Hits vs Faults")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Replacement Queue (FIFO order)")
    st.write(list(manager.fifo_queue))

# Footer: help and examples
st.markdown("---")
st.markdown("**Usage tips**:\n- Enter a comma separated page access sequence and click **Run Sequence**.\n- Create segments to test segmentation translation (then translate offsets using the engine methods).\n- Switch replacement policy between FIFO and LRU.\n- Adjust physical memory or page size to change frame count.")

st.markdown("---")
st.markdown("**Instructor examples**:\n1) Small memory: Physical=8KB, Page=4KB → 2 frames. Sequence: `0,1,2,0,1,3,0` (show replacements)\n2) LRU demo: set policy to LRU and run `0,1,2,0,3,0` to see LRU behavior.")

# Optional; provide helper debug API (visible in app) to translate segment offset
st.sidebar.markdown("---")
st.sidebar.header("Debug: Translate Segment")
trans_seg = st.sidebar.number_input("Translate seg id", min_value=0, value=0)
trans_off = st.sidebar.number_input("Offset (KB)", min_value=0, value=0)
if st.sidebar.button("Translate"):
    try:
        page_no, page_off = manager.translate_segment(trans_seg, trans_off)
        st.sidebar.success(f"Virtual page {page_no}, offset {page_off}KB")
    except Exception as e:
        st.sidebar.error(str(e))

# End of file
