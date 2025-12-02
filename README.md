# Memory Management Visualizer

An interactive educational tool for visualizing memory management concepts including **paging**, **segmentation**, and **page replacement algorithms**. Built with Python and Streamlit for easy use and visual understanding.

## 🎯Features

### Paging Simulation
- **Page Table Management**: View and track virtual-to-physical page mappings
- **Frame Table Visualization**: See physical memory frames and their contents
- **Hit/Fault Tracking**: Monitor page hits and page faults in real-time

### Page Replacement Policies
- **FIFO (First-In-First-Out)**: Replace the oldest page in memory
- **LRU (Least Recently Used)**: Replace the page that hasn't been used for the longest time

### Segmentation
- Create and manage memory segments with base and limit addresses
- Translate segment offsets to virtual page numbers
- Detect segmentation faults for out-of-bound accesses

### Memory Allocation Algorithms (via engine.py)
- **First-Fit**: Allocate to the first suitable block
- **Best-Fit**: Allocate to the smallest suitable block
- **Worst-Fit**: Allocate to the largest suitable block
- **Block Coalescing**: Automatic merging of adjacent free blocks
- **Fragmentation Metrics**: Track external fragmentation and memory utilization

### Visualizations
- Interactive frame table with color-coded allocation status
- Page table snapshots showing validity, frame mappings, and LRU timestamps
- Segment table display
- Hits vs Faults bar chart
- Real-time statistics dashboard
- Event log for tracking all memory operations

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Sohamactive/memory-management-visualizer.git
cd memory-management-visualizer
```

2. Install dependencies:
```bash
pip install streamlit plotly
```

Or create a virtual environment first (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install streamlit plotly
```

## 💻 Usage



### Running the Application

Start the Streamlit application:
```bash
streamlit run app.py
```

This will open the visualizer in your default web browser (typically at `http://localhost:8501`).

### Configuration Options

Use the sidebar to configure:
- **Physical Memory (KB)**: Set the total physical memory size (4-65536 KB)
- **Page Size (KB)**: Choose from 1, 2, 4, 8, 16, or 32 KB
- **Replacement Policy**: Select FIFO or LRU

### Running Simulations

1. **Page Access Sequence**: Enter a comma-separated list of page numbers (e.g., `0,1,2,3,2,1,4,0,1,5`)
2. **Step Once**: Process one page access at a time
3. **Run Sequence**: Process the entire sequence with visual playback
4. **Playback Speed**: Adjust the speed of sequence playback (0.5-5.0 ops/sec)

### Segmentation

1. Enter a Segment ID, base address (KB), and limit (KB)
2. Click "Create Segment" to add it to the segment table
3. Use the "Debug: Translate Segment" section to translate segment offsets to virtual pages

### Example Scenarios

**Small Memory Demonstration:**
- Physical Memory: 8KB
- Page Size: 4KB (results in 2 frames)
- Sequence: `0,1,2,0,1,3,0`
- This demonstrates page replacement behavior

**LRU Policy Demo:**
- Set policy to LRU
- Run sequence: `0,1,2,0,3,0`
- Observe LRU replacement decisions

## 📁 Project Structure

```
memory-management-visualizer/
├── app.py          # Main Streamlit application with paging/segmentation simulation
├── engine.py       # Memory allocation engine (First-Fit, Best-Fit, Worst-Fit)
├── utils.py        # Utility functions (color generation)
├── README.md       # This file
└── .gitignore      # Git ignore configuration
```

### Component Details

- **app.py**: Contains the `MemoryManager` class for paging/segmentation simulation and the Streamlit UI
- **engine.py**: Contains the `MemoryEngine` class for contiguous memory allocation with various fit algorithms
- **utils.py**: Helper functions for visualization

## 🔧 Technical Details

### Memory Manager (app.py)
- Implements virtual memory simulation with page tables and frame tables
- Supports configurable physical memory size and page size
- Tracks page hits, faults, and maintains event logs
- Implements FIFO queue and LRU timestamps for replacement decisions

### Memory Engine (engine.py)
- Manages contiguous memory blocks
- Implements allocation strategies (First-Fit, Best-Fit, Worst-Fit)
- Handles block splitting and coalescing
- Calculates fragmentation metrics

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Built for educational purposes to help students understand memory management concepts
- Uses [Streamlit](https://streamlit.io/) for the web interface
- Uses [Plotly](https://plotly.com/python/) for interactive visualizations
