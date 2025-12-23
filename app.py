"""
Memory Management Visualizer — Paging, Segmentation & Replacement

This application provides an interactive simulation and visualization of core
Operating System memory management concepts including:
    - Paging and Virtual Memory
    - Page Tables and Frame Tables
    - Page Replacement Algorithms (FIFO, LRU)
    - Segmentation with base/limit addressing

Built with Streamlit for the web interface and Plotly for visualizations.

Author: OS Course Project
Date: 2025
"""

# =============================================================================
# IMPORTS
# =============================================================================

from collections import deque, OrderedDict  # deque for FIFO queue implementation
import streamlit as st                       # Web application framework
import plotly.graph_objects as go            # Interactive plotting library
from dataclasses import dataclass, field     # For clean data class definitions
from typing import Dict, List, Optional, Tuple  # Type hints for better code clarity
import time                                  # For timing/pacing the simulation


# =============================================================================
# SIMULATION ENGINE - Core Data Structures
# =============================================================================

@dataclass
class PageTableEntry:
    """
    Represents a single entry in the Page Table.
    
    Each virtual page has an associated entry that tracks:
    - Which physical frame it maps to (if any)
    - Whether the page is currently in RAM (valid bit)
    - When it was last accessed (for LRU replacement)
    
    Attributes:
        page_no (int): The virtual page number this entry represents
        frame_no (Optional[int]): Physical frame number, None if not in memory
        valid (bool): True if page is currently loaded in RAM
        last_used (float): Timestamp of last access (for LRU algorithm)
    """
    page_no: int
    frame_no: Optional[int] = None
    valid: bool = False
    last_used: float = 0.0  # Timestamp for LRU replacement tracking


@dataclass
class Frame:
    """
    Represents a physical memory frame in RAM.
    
    Physical memory is divided into fixed-size frames that can each
    hold one page of virtual memory.
    
    Attributes:
        frame_no (int): The frame's index in physical memory
        occupied (bool): True if a page is currently loaded here
        page_no (Optional[int]): The virtual page stored here, None if free
        pid (Optional[int]): Process ID (for multi-process support, optional)
    """
    frame_no: int
    occupied: bool = False
    page_no: Optional[int] = None
    pid: Optional[int] = None  # Process ID for multi-process scenarios


@dataclass
class Segment:
    """
    Represents a memory segment for segmentation-based addressing.
    
    Segmentation divides memory into logical units (code, data, stack, etc.)
    with each segment having a base address and size limit.
    
    Attributes:
        seg_id (int): Unique identifier for this segment
        base (int): Starting virtual address of the segment (in KB)
        limit (int): Maximum size/offset allowed in segment (in KB)
    """
    seg_id: int
    base: int    # Base address in KB
    limit: int   # Segment size limit in KB


class ReplacementPolicy:
    """
    Enumeration of available page replacement algorithms.
    
    FIFO: First-In-First-Out - replaces the oldest page in memory
    LRU:  Least Recently Used - replaces the page not used for longest time
    """
    FIFO = "FIFO"
    LRU = "LRU"


# =============================================================================
# MEMORY MANAGER - Core Simulation Engine
# =============================================================================

class MemoryManager:
    """
    Core simulation engine for memory management operations.
    
    This class simulates virtual memory management including:
    - Physical memory allocation using frames
    - Virtual-to-physical address translation via page tables
    - Page fault handling and page loading
    - Page replacement using FIFO or LRU algorithms
    - Segmentation with base/limit protection
    
    Attributes:
        physical_mem_kb (int): Total physical memory size in KB
        page_size_kb (int): Size of each page/frame in KB
        frame_count (int): Number of frames in physical memory
        frames (List[Frame]): The frame table tracking physical memory
        page_table (Dict[int, PageTableEntry]): Maps virtual pages to PTEs
        fifo_queue (deque): Queue tracking frame load order for FIFO
        time_counter (float): Logical clock for LRU timestamp tracking
        hits (int): Count of page hits (page found in memory)
        faults (int): Count of page faults (page not in memory)
        event_log (List[str]): Log of all memory access events
        policy (str): Current replacement policy (FIFO or LRU)
        segments (Dict[int, Segment]): Defined memory segments
    """
    
    def __init__(self, physical_mem_kb: int, page_size_kb: int):
        """
        Initialize the Memory Manager with given memory configuration.
        
        Args:
            physical_mem_kb (int): Total physical memory size in kilobytes
            page_size_kb (int): Size of each page/frame in kilobytes
        """
        # Store memory configuration (sizes in KB)
        self.physical_mem_kb = physical_mem_kb
        self.page_size_kb = page_size_kb
        
        # Calculate number of frames: physical_memory / page_size
        # Ensure at least 1 frame exists
        self.frame_count = max(1, physical_mem_kb // page_size_kb)

        # Initialize the frame table with empty frames
        self.frames: List[Frame] = [Frame(i) for i in range(self.frame_count)]

        # Page table: maps virtual page number -> PageTableEntry
        self.page_table: Dict[int, PageTableEntry] = {}

        # FIFO replacement: queue stores frame indices in load order
        self.fifo_queue: deque = deque()
        
        # Logical clock for LRU timestamp tracking
        self.time_counter = 0.0

        # Statistics counters
        self.hits = 0      # Number of page hits
        self.faults = 0    # Number of page faults
        
        # Event log for tracking all memory operations
        self.event_log: List[str] = []

        # Default replacement policy
        self.policy = ReplacementPolicy.FIFO

        # Segmentation table: maps segment ID -> Segment
        self.segments: Dict[int, Segment] = {}

    # =========================================================================
    # CONFIGURATION METHODS
    # =========================================================================
    
    def set_policy(self, policy: str):
        """
        Set the page replacement policy.
        
        Args:
            policy (str): Either 'FIFO' or 'LRU'
            
        Raises:
            AssertionError: If policy is not FIFO or LRU
        """
        assert policy in (ReplacementPolicy.FIFO, ReplacementPolicy.LRU)
        self.policy = policy

    def reset(self):
        """
        Reset the simulation to initial state.
        
        Clears all frames, page table entries, statistics, and logs.
        Use this to start a fresh simulation without changing memory config.
        """
        self.frames = [Frame(i) for i in range(self.frame_count)]
        self.page_table = {}
        self.fifo_queue = deque()
        self.hits = 0
        self.faults = 0
        self.event_log = []
        self.time_counter = 0.0
        self.segments = {}

    # =========================================================================
    # SEGMENTATION APIs
    # =========================================================================
    
    def create_segment(self, seg_id: int, base_kb: int, limit_kb: int):
        """
        Create a new memory segment.
        
        Segments provide logical division of memory (e.g., code, data, stack)
        with bounds checking via base and limit values.
        
        Args:
            seg_id (int): Unique identifier for the segment
            base_kb (int): Starting virtual address in KB
            limit_kb (int): Maximum size of segment in KB
            
        Raises:
            ValueError: If segment with given ID already exists
        """
        if seg_id in self.segments:
            raise ValueError("Segment already exists")
        self.segments[seg_id] = Segment(seg_id, base_kb, limit_kb)
        self.event_log.append(f"Segment {seg_id} created: base={base_kb}KB limit={limit_kb}KB")

    def translate_segment(self, seg_id: int, offset_kb: int) -> Tuple[int, int]:
        """
        Translate a segment:offset address to virtual page number and offset.
        
        This performs segmentation address translation:
        1. Validates segment exists
        2. Checks offset is within segment bounds
        3. Calculates virtual address and breaks into page + offset
        
        Args:
            seg_id (int): The segment identifier
            offset_kb (int): Offset within the segment in KB
            
        Returns:
            Tuple[int, int]: (virtual_page_number, page_offset_in_KB)
            
        Raises:
            ValueError: If segment doesn't exist
            IndexError: If offset exceeds segment limit (segmentation fault)
        """
        if seg_id not in self.segments:
            raise ValueError("Segment not found")
        
        seg = self.segments[seg_id]
        
        # Bounds check: offset must be within [0, limit)
        if offset_kb < 0 or offset_kb >= seg.limit:
            raise IndexError("Segmentation Fault: offset out of bound")
        
        # Calculate virtual address: base + offset
        virt_addr_kb = seg.base + offset_kb
        
        # Convert to page number and page offset
        page_no = virt_addr_kb // self.page_size_kb
        page_offset = virt_addr_kb % self.page_size_kb
        
        return page_no, page_offset

    # =========================================================================
    # PAGING / VIRTUAL MEMORY APIs
    # =========================================================================
    
    def access_page(self, page_no: int) -> Tuple[bool, Optional[int]]:
        """
        Access a virtual page, handling hits, faults, and replacement.
        
        This is the core paging operation that:
        1. Checks if page is in memory (hit) or not (fault)
        2. On fault: finds a free frame or triggers replacement
        3. Updates statistics and event log
        
        Args:
            page_no (int): Virtual page number to access
            
        Returns:
            Tuple[bool, Optional[int]]: 
                - bool: True if hit, False if fault
                - int: Frame number where page is located
        """
        # Increment logical clock for LRU tracking
        self.time_counter += 1.0
        
        # Look up page in page table
        pte = self.page_table.get(page_no)

        # If no PTE exists, create one (page has never been referenced)
        if pte is None:
            pte = PageTableEntry(page_no)
            self.page_table[page_no] = pte

        # ----- PAGE HIT -----
        # Page is valid (in memory) - return immediately
        if pte.valid:
            pte.last_used = self.time_counter  # Update LRU timestamp
            self.hits += 1
            self.event_log.append(f"Hit: Page {page_no} in Frame {pte.frame_no}")
            return True, pte.frame_no

        # ----- PAGE FAULT -----
        # Page not in memory - must load it
        self.faults += 1
        self.event_log.append(f"Fault: Page {page_no} not in memory")

        # First, try to find a free (unoccupied) frame
        free_frame = next((f for f in self.frames if not f.occupied), None)
        
        if free_frame is not None:
            # Free frame available - load page directly
            self._load_page_into_frame(page_no, free_frame.frame_no)
            return False, free_frame.frame_no

        # No free frame available - must replace an existing page
        replaced_frame_no = self._replace_frame(page_no)
        return False, replaced_frame_no

    def _load_page_into_frame(self, page_no: int, frame_no: int):
        """
        Load a page into a specific frame (internal helper).
        
        This handles the bookkeeping when loading a page into a free frame:
        - Updates frame table entry
        - Updates page table entry
        - Adds frame to FIFO queue
        
        Args:
            page_no (int): Virtual page to load
            frame_no (int): Target frame number
        """
        # Update frame table
        frame = self.frames[frame_no]
        frame.occupied = True
        frame.page_no = page_no

        # Update page table entry
        pte = self.page_table[page_no]
        pte.frame_no = frame_no
        pte.valid = True                    # Page is now in memory
        pte.last_used = self.time_counter   # Set access timestamp

        # Add to FIFO queue (newest entry at end)
        self.fifo_queue.append(frame_no)
        
        self.event_log.append(f"Loaded: Page {page_no} -> Frame {frame_no}")

    def _replace_frame(self, page_no: int) -> int:
        """
        Select a victim frame and replace its contents (internal helper).
        
        Implements page replacement when all frames are occupied:
        - FIFO: Replace oldest loaded page (front of queue)
        - LRU: Replace page with smallest last_used timestamp
        
        Args:
            page_no (int): New page to load after replacement
            
        Returns:
            int: Frame number where new page was loaded
        """
        # Select victim frame based on replacement policy
        if self.policy == ReplacementPolicy.FIFO:
            # FIFO: Remove oldest frame from front of queue
            victim_frame_no = self.fifo_queue.popleft()
        else:  # LRU
            # LRU: Find frame with minimum last_used timestamp
            min_time = float('inf')
            victim_frame_no = None
            
            for f in self.frames:
                if f.occupied and f.page_no is not None:
                    p = self.page_table.get(f.page_no)
                    if p and p.last_used < min_time:
                        min_time = p.last_used
                        victim_frame_no = f.frame_no
            
            # Fallback to frame 0 if no victim found (shouldn't happen)
            if victim_frame_no is None:
                victim_frame_no = 0

        # ----- EVICT VICTIM PAGE -----
        victim = self.frames[victim_frame_no]
        evicted_page = victim.page_no
        self.event_log.append(f"Evicting: Page {evicted_page} from Frame {victim_frame_no}")

        # Invalidate the evicted page's table entry
        if evicted_page is not None:
            old_pte = self.page_table.get(evicted_page)
            if old_pte:
                old_pte.valid = False      # Page no longer in memory
                old_pte.frame_no = None    # Clear frame reference

        # ----- LOAD NEW PAGE -----
        victim.page_no = page_no
        self.event_log.append(f"Loaded: Page {page_no} -> Frame {victim_frame_no} (replaced)")

        # Update new page's table entry
        new_pte = self.page_table[page_no]
        new_pte.frame_no = victim_frame_no
        new_pte.valid = True
        new_pte.last_used = self.time_counter

        # FIFO: Add this frame to back of queue (now newest)
        if self.policy == ReplacementPolicy.FIFO:
            self.fifo_queue.append(victim_frame_no)
            
        return victim_frame_no

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_frame_table(self) -> List[Frame]:
        """
        Get the current frame table state.
        
        Returns:
            List[Frame]: List of all frames in physical memory
        """
        return self.frames

    def get_page_table_snapshot(self) -> Dict[int, PageTableEntry]:
        """
        Get a snapshot of the current page table.
        
        Returns:
            Dict[int, PageTableEntry]: Copy of page table mapping
        """
        return dict(self.page_table)

    def get_stats(self) -> Dict[str, float]:
        """
        Calculate and return simulation statistics.
        
        Returns:
            Dict[str, float]: Statistics including:
                - hits: Total page hits
                - faults: Total page faults
                - hit_ratio: Hits / Total accesses
                - fault_rate: Faults / Total accesses
                - total_refs: Total memory references
        """
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


# =============================================================================
# STREAMLIT UI - Web Application Interface
# =============================================================================

# Configure the Streamlit page
st.set_page_config(page_title="Memory Management Visualizer", layout="wide")

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------

# Page selector for switching between Simulator and Concepts views
page = st.sidebar.radio("Choose View", ["Simulator", "Concepts"])

# Main application title
st.title("Memory Management Visualizer — Paging, Segmentation & Replacement")

# =============================================================================
# CONCEPTS PAGE - Educational Content
# =============================================================================

if page == "Concepts":
    # ----- Display educational content about OS memory concepts -----
    st.header("Operating System Concepts Used in This Project")
    st.markdown(
        """
        ## 📘 Key Concepts

        ### **1. Paging**
        - Memory is divided into fixed-size units called *pages* (virtual memory) and *frames* (physical memory).
        - Paging avoids external fragmentation.
        - Virtual pages are mapped to physical frames using a **Page Table**.

        ### **2. Page Table**
        - Maps each virtual page to a physical frame.
        - Contains:
            - **Valid Bit**: Whether the page is in RAM.
            - **Frame Number**: The physical frame storing the page.
            - **Last Used Time**: For LRU replacement.

        ### **3. Virtual Memory**
        - Allows programs to use more memory than physically available.
        - OS loads only required pages into RAM.
        - Non-loaded pages cause **Page Faults**.

        ### **4. Page Fault**
        - Occurs when a referenced page is not in RAM.
        - OS fetches page from disk and loads it into a frame.

        ### **5. Frame Table**
        - Stores which physical frames are free/occupied.
        - Helps OS know where to load pages.

        ### **6. Page Replacement Algorithms**
        When RAM is full, OS must choose a page to remove:

        #### **FIFO (First In First Out)**
        - Replace the page that entered memory earliest.

        #### **LRU (Least Recently Used)**
        - Replace the page that hasn't been used for the longest time.

        ### **7. Segmentation**
        - Divides memory logically into segments like:
            - Code
            - Data
            - Stack
        - Each segment has *base* and *limit*.
        - Prevents out-of-bound memory access.

        ### **8. Internal & External Fragmentation**
        - **Internal**: Waste inside allocated block.
        - **External**: Total free memory exists, but not contiguously.

        ### **9. Translation (Segmentation + Paging)**
        - Virtual address → (segment, offset) → virtual page → frame.

        ---
        ### ✔ This page helps users understand the OS foundations behind your simulator.
        """
    )
    st.stop()  # Stop rendering - don't show simulator on Concepts page

# =============================================================================
# SIMULATOR PAGE - Main Interactive Interface
# =============================================================================

# Duplicate title for simulator page (appears after Concepts check)
("Memory Management Visualizer — Paging, Segmentation & Replacement")

# -----------------------------------------------------------------------------
# SIDEBAR - Simulation Settings
# -----------------------------------------------------------------------------

st.sidebar.header("Simulation Settings")

# Physical memory size input (4KB to 64MB)
physical_mem_kb = st.sidebar.number_input(
    "Physical memory (KB)", 
    min_value=4, 
    max_value=65536, 
    value=32, 
    step=4
)

# Page size selection (power of 2 sizes)
page_size_kb = st.sidebar.selectbox(
    "Page size (KB)", 
    options=[1, 2, 4, 8, 16, 32], 
    index=2  # Default: 4KB
)

# Replacement policy selection
policy = st.sidebar.selectbox(
    "Replacement Policy", 
    options=[ReplacementPolicy.FIFO, ReplacementPolicy.LRU]
)

# -----------------------------------------------------------------------------
# SESSION STATE - Memory Manager Persistence
# -----------------------------------------------------------------------------

# Initialize manager in session state (persists across Streamlit reruns)
if 'manager' not in st.session_state:
    st.session_state.manager = MemoryManager(physical_mem_kb, page_size_kb)
else:
    # If memory configuration changed, create a new manager
    mgr = st.session_state.manager
    if mgr.physical_mem_kb != physical_mem_kb or mgr.page_size_kb != page_size_kb:
        st.session_state.manager = MemoryManager(physical_mem_kb, page_size_kb)

# Get reference to the memory manager
manager: MemoryManager = st.session_state.manager
manager.set_policy(policy)  # Apply selected replacement policy

st.sidebar.markdown("---")

# -----------------------------------------------------------------------------
# SIDEBAR - Access Sequence Controls
# -----------------------------------------------------------------------------

st.sidebar.header("Access / Workload")

# Text input for page access sequence
access_input = st.sidebar.text_area(
    "Page access sequence (comma separated page numbers)", 
    value="0,1,2,3,2,1,4,0,1,5"
)

# Playback speed control for animation
run_speed = st.sidebar.slider(
    "Playback speed (ops/sec)", 
    min_value=0.5, 
    max_value=5.0, 
    value=1.0
)

# Reset button to clear simulation state
if st.sidebar.button("Reset Simulation"):
    manager.reset()
    st.sidebar.success("Simulation reset")

st.sidebar.markdown("---")

# -----------------------------------------------------------------------------
# SIDEBAR - Segmentation Configuration
# -----------------------------------------------------------------------------

st.sidebar.header("Segmentation")

# Segment creation inputs
seg_id = st.sidebar.number_input("Segment ID (int)", min_value=0, value=0)
seg_base = st.sidebar.number_input("Segment base (KB)", min_value=0, value=0)
seg_limit = st.sidebar.number_input("Segment limit (KB)", min_value=1, value=8)

# Create segment button
if st.sidebar.button("Create Segment"):
    try:
        manager.create_segment(seg_id, seg_base, seg_limit)
        st.sidebar.success(f"Created segment {seg_id}")
    except Exception as e:
        st.sidebar.error(str(e))

st.sidebar.markdown("---")

# =============================================================================
# MAIN CONTENT AREA - Two Column Layout
# =============================================================================

# Create two columns: left for controls, right for visualizations
col1, col2 = st.columns([1, 2])

# -----------------------------------------------------------------------------
# LEFT COLUMN - Controls and Event Log
# -----------------------------------------------------------------------------

with col1:
    st.subheader("Controls")
    
    # Step Once: Execute single page access
    if st.button("Step Once"):
        # Parse the page access sequence
        seq = [int(x.strip()) for x in access_input.split(',') if x.strip() != '']
        if len(seq) > 0:
            page = seq[0]  # Get first page in sequence
            # Note: Updating the text area programmatically doesn't work in Streamlit
            remaining = seq[1:]
            st.sidebar.text_area(
                "Page access sequence (comma separated page numbers)", 
                value=','.join(map(str, remaining))
            )
            # Access the page and show result
            hit, frame = manager.access_page(page)
            st.success(f"Accessed page {page} -> {'HIT' if hit else 'FAULT'} (frame={frame})")

    # Run Sequence: Execute all pages in sequence
    if st.button("Run Sequence"):
        # Parse the full sequence
        seq = [int(x.strip()) for x in access_input.split(',') if x.strip() != '']
        if len(seq) == 0:
            st.warning("No pages to run")
        else:
            # Process each page in sequence
            for p in seq:
                hit, frame = manager.access_page(p)
                # Sleep for pacing (visual feedback)
                # Note: Streamlit only updates after loop completes
                time.sleep(1.0 / run_speed)
            st.success("Sequence run finished")

    # Display event log (most recent 20 events, newest first)
    st.subheader("Event Log")
    for ev in manager.event_log[-20:][::-1]:
        st.write(ev)

# -----------------------------------------------------------------------------
# RIGHT COLUMN - Visualizations
# -----------------------------------------------------------------------------

with col2:
    # ----- Physical Frames Visualization -----
    st.subheader("Physical Frames")
    frames = manager.get_frame_table()
    
    # Create bar chart showing frame status
    fig = go.Figure()

    x = []      # Frame indices
    y = []      # Bar heights (all 1 for uniform display)
    text = []   # Labels for each frame
    colors = [] # Color coding: green=occupied, gray=free
    
    for f in frames:
        # Create label showing frame number and page (if any)
        label = f"F{f.frame_no}: " + (f"P{f.page_no}" if f.page_no is not None else "Free")
        text.append(label)
        colors.append("lightgreen" if f.occupied else "lightgray")
        x.append(f.frame_no)
        y.append(1)

    # Add bar trace
    fig.add_trace(go.Bar(
        x=x, 
        y=y, 
        text=text, 
        marker_color=colors, 
        hovertext=text, 
        hoverinfo='text'
    ))
    
    # Configure layout
    fig.update_layout(
        height=150, 
        showlegend=False, 
        yaxis=dict(showticklabels=False)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ----- Page Table Display -----
    st.subheader("Page Table (snapshot)")
    ptable = manager.get_page_table_snapshot()
    
    if len(ptable) == 0:
        st.write("Page table empty — no pages referenced yet")
    else:
        # Build table rows with page info
        rows = []
        for pno, pte in sorted(ptable.items()):
            rows.append({
                "page": pno,
                "valid": pte.valid,
                "frame": pte.frame_no,
                "last_used": pte.last_used
            })
        st.table(rows)

    # ----- Segmentation Table Display -----
    st.subheader("Segmentation Table")
    
    if len(manager.segments) == 0:
        st.write("No segments defined")
    else:
        seg_rows = []
        for sid, s in manager.segments.items():
            seg_rows.append({
                "seg_id": sid, 
                "base_kb": s.base, 
                "limit_kb": s.limit
            })
        st.table(seg_rows)

    # ----- Statistics Display -----
    st.subheader("Statistics")
    stats = manager.get_stats()
    
    # Display key metrics
    st.metric("Page Accesses", stats['total_refs'])
    st.metric("Page Faults", stats['faults'])
    st.metric("Hit Ratio", stats['hit_ratio'])

    # ----- Hits vs Faults Bar Chart -----
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=["Hits", "Faults"], 
        y=[stats['hits'], stats['faults']]
    ))
    fig2.update_layout(height=300, title="Hits vs Faults")
    st.plotly_chart(fig2, use_container_width=True)

    # ----- FIFO Queue Display -----
    st.subheader("Replacement Queue (FIFO order)")
    st.write(list(manager.fifo_queue))

# =============================================================================
# FOOTER - Usage Tips and Examples
# =============================================================================

st.markdown("---")
st.markdown(
    "**Usage tips**:\n"
    "- Enter a comma separated page access sequence and click **Run Sequence**.\n"
    "- Create segments to test segmentation translation (then translate offsets using the engine methods).\n"
    "- Switch replacement policy between FIFO and LRU.\n"
    "- Adjust physical memory or page size to change frame count."
)

st.markdown("---")
st.markdown(
    "**Instructor examples**:\n"
    "1) Small memory: Physical=8KB, Page=4KB → 2 frames. "
    "Sequence: `0,1,2,0,1,3,0` (show replacements)\n"
    "2) LRU demo: set policy to LRU and run `0,1,2,0,3,0` to see LRU behavior."
)

# -----------------------------------------------------------------------------
# SIDEBAR - Debug Tools: Segment Translation
# -----------------------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.header("Debug: Translate Segment")

# Inputs for testing segment translation
trans_seg = st.sidebar.number_input("Translate seg id", min_value=0, value=0)
trans_off = st.sidebar.number_input("Offset (KB)", min_value=0, value=0)

# Translate button
if st.sidebar.button("Translate"):
    try:
        # Perform segment:offset -> page:offset translation
        page_no, page_off = manager.translate_segment(trans_seg, trans_off)
        st.sidebar.success(f"Virtual page {page_no}, offset {page_off}KB")
    except Exception as e:
        st.sidebar.error(str(e))


