# engine.py

class Block:
    def __init__(self, start, size, allocated=False, block_id=None):
        self.start = start
        self.size = size
        self.allocated = allocated
        self.block_id = block_id

    def __repr__(self):
        state = "A" if self.allocated else "F"
        return f"[{state}|{self.start}|{self.size}]"


class MemoryEngine:
    def __init__(self, total_size=100):
        self.total_size = total_size
        self.algorithm = "First-Fit"  # default
        self.reset()

    def reset(self):
        self.blocks = [Block(0, self.total_size, allocated=False, block_id=None)]
        self.next_id = 1

    def set_algorithm(self, algo):
        self.algorithm = algo

    # -----------------------------
    # Allocate Dispatcher
    # -----------------------------
    def allocate(self, req_size):
        if self.algorithm == "First-Fit":
            return self._first_fit(req_size)
        elif self.algorithm == "Best-Fit":
            return self._best_fit(req_size)
        elif self.algorithm == "Worst-Fit":
            return self._worst_fit(req_size)
        return None

    # -----------------------------
    # Algorithms
    # -----------------------------
    def _first_fit(self, req):
        for i, block in enumerate(self.blocks):
            if not block.allocated and block.size >= req:
                return self._split_block(i, req)
        return None

    def _best_fit(self, req):
        best_index = None
        best_size = float('inf')

        for i, block in enumerate(self.blocks):
            if not block.allocated and block.size >= req and block.size < best_size:
                best_size = block.size
                best_index = i

        if best_index is not None:
            return self._split_block(best_index, req)
        return None

    def _worst_fit(self, req):
        worst_index = None
        worst_size = -1

        for i, block in enumerate(self.blocks):
            if not block.allocated and block.size >= req and block.size > worst_size:
                worst_size = block.size
                worst_index = i

        if worst_index is not None:
            return self._split_block(worst_index, req)
        return None

    # -----------------------------
    # Helpers
    # -----------------------------
    def _split_block(self, index, req_size):
        block = self.blocks[index]
        block_id = self.next_id
        self.next_id += 1

        # Perfect fit
        if block.size == req_size:
            block.allocated = True
            block.block_id = block_id
            return block_id

        # Split into allocated + free
        allocated_block = Block(block.start, req_size, True, block_id)
        free_block = Block(block.start + req_size, block.size - req_size, False, None)

        self.blocks[index] = allocated_block
        self.blocks.insert(index + 1, free_block)
        return block_id

    def free(self, block_id):
        for block in self.blocks:
            if block.block_id == block_id:
                block.allocated = False
                block.block_id = None
                break
        self._coalesce()

    def _coalesce(self):
        merged = []
        i = 0

        while i < len(self.blocks):
            current = self.blocks[i]

            if not current.allocated:
                total = current.size
                start = current.start

                j = i + 1
                while j < len(self.blocks) and not self.blocks[j].allocated:
                    total += self.blocks[j].size
                    j += 1

                merged.append(Block(start, total, False))
                i = j
            else:
                merged.append(current)
                i += 1

        self.blocks = merged

    def get_state(self):
        return self.blocks

        # --------------------------------------
    # Fragmentation Metrics
    # --------------------------------------
    def get_fragmentation_metrics(self):
        free_blocks = [b.size for b in self.blocks if not b.allocated]
        allocated_blocks = [b.size for b in self.blocks if b.allocated]

        total_free = sum(free_blocks)
        total_alloc = sum(allocated_blocks)
        total_memory = self.total_size

        # External fragmentation:
        # sum of free blocks smaller than the largest requested block area
        # For simplicity, we use:
        # external = 1 - (largest_free_block / total_free)
        if total_free == 0:
            external_frag = 0
        else:
            largest_free = max(free_blocks) if free_blocks else 0
            external_frag = 1 - (largest_free / total_free)

        # Internal fragmentation = wasted inside allocated blocks
        # (We assume each allocation is perfect; set internal frag to 0)
        internal_frag = 0

        utilization = total_alloc / total_memory

        return {
            "external": round(external_frag, 4),
            "internal": round(internal_frag, 4),
            "utilization": round(utilization, 4)
        }
