"""Mojo kernels for minimap2-style minimizer sketching and chaining."""

comptime U8Ptr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime U64Ptr = UnsafePointer[UInt64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int, AnyOrigin[mut=True]]


def nt4(c: UInt8) -> Int:
    if c == UInt8(65) or c == UInt8(97):
        return 0
    if c == UInt8(67) or c == UInt8(99):
        return 1
    if c == UInt8(71) or c == UInt8(103):
        return 2
    if c == UInt8(84) or c == UInt8(116) or c == UInt8(85) or c == UInt8(117):
        return 3
    return 4


def hash64(key_in: UInt64, mask: UInt64) -> UInt64:
    var key = (~key_in + (key_in << 21)) & mask
    key = key ^ (key >> 24)
    key = ((key + (key << 3)) + (key << 8)) & mask
    key = key ^ (key >> 14)
    key = ((key + (key << 2)) + (key << 4)) & mask
    key = key ^ (key >> 28)
    return (key + (key << 31)) & mask


@export("mmp_sketch")
def mmp_sketch(
    seq_addr: Int,
    length: Int,
    k: Int,
    w: Int,
    hashes_addr: Int,
    positions_addr: Int,
    strands_addr: Int,
    scratch_hash_addr: Int,
    scratch_meta_addr: Int,
    output_capacity: Int,
) abi("C") -> Int:
    if length <= 0 or output_capacity < length or k <= 0 or k > 28 or w <= 0 or w > 255:
        return -1
    if seq_addr == 0 or hashes_addr == 0 or positions_addr == 0 or strands_addr == 0:
        return -1
    if scratch_hash_addr == 0 or scratch_meta_addr == 0:
        return -1
    var seq = U8Ptr(unsafe_from_address=seq_addr)
    var hashes = U64Ptr(unsafe_from_address=hashes_addr)
    var positions = IPtr(unsafe_from_address=positions_addr)
    var strands = IPtr(unsafe_from_address=strands_addr)
    var ring_hash = U64Ptr(unsafe_from_address=scratch_hash_addr)
    var ring_meta = U64Ptr(unsafe_from_address=scratch_meta_addr)
    var invalid = ~UInt64(0)
    var mask = (UInt64(1) << UInt64(2 * k)) - UInt64(1)
    var shift1 = 2 * (k - 1)
    var forward = UInt64(0)
    var reverse = UInt64(0)
    var valid = 0
    var ring_pos = 0
    var min_hash = invalid
    var min_meta = invalid
    var min_pos = 0
    var count = 0
    var i = 0
    while i < length:
        var c = nt4(seq.load(i))
        var info_hash = invalid
        var info_meta = invalid
        if c < 4:
            forward = ((forward << 2) | UInt64(c)) & mask
            reverse = (reverse >> 2) | (UInt64(3 ^ c) << UInt64(shift1))
            if forward == reverse:
                i += 1
                continue
            var strand = 0 if forward < reverse else 1
            var selected = forward if strand == 0 else reverse
            valid += 1
            if valid >= k:
                info_hash = hash64(selected, mask)
                info_meta = (UInt64(i) << 1) | UInt64(strand)
        else:
            valid = 0
        ring_hash.store(ring_pos, info_hash)
        ring_meta.store(ring_pos, info_meta)
        if valid == w + k - 1 and min_hash != invalid:
            var j = ring_pos + 1
            while j < w:
                if min_hash == ring_hash.load(j) and min_meta != ring_meta.load(j):
                    if count >= output_capacity:
                        return -1
                    hashes.store(count, min_hash)
                    positions.store(count, Int(ring_meta.load(j) >> 1))
                    strands.store(count, Int(ring_meta.load(j) & UInt64(1)))
                    count += 1
                j += 1
            j = 0
            while j < ring_pos:
                if min_hash == ring_hash.load(j) and min_meta != ring_meta.load(j):
                    hashes.store(count, min_hash)
                    positions.store(count, Int(ring_meta.load(j) >> 1))
                    strands.store(count, Int(ring_meta.load(j) & UInt64(1)))
                    count += 1
                j += 1
        if info_hash <= min_hash:
            if valid >= w + k and min_hash != invalid:
                if count >= output_capacity:
                    return -1
                hashes.store(count, min_hash)
                positions.store(count, Int(min_meta >> 1))
                strands.store(count, Int(min_meta & UInt64(1)))
                count += 1
            min_hash = info_hash
            min_meta = info_meta
            min_pos = ring_pos
        elif ring_pos == min_pos:
            if valid >= w + k - 1 and min_hash != invalid:
                if count >= output_capacity:
                    return -1
                hashes.store(count, min_hash)
                positions.store(count, Int(min_meta >> 1))
                strands.store(count, Int(min_meta & UInt64(1)))
                count += 1
            min_hash = invalid
            var j = ring_pos + 1
            while j < w:
                if min_hash >= ring_hash.load(j):
                    min_hash = ring_hash.load(j)
                    min_meta = ring_meta.load(j)
                    min_pos = j
                j += 1
            j = 0
            while j <= ring_pos:
                if min_hash >= ring_hash.load(j):
                    min_hash = ring_hash.load(j)
                    min_meta = ring_meta.load(j)
                    min_pos = j
                j += 1
            if valid >= w + k - 1 and min_hash != invalid:
                j = ring_pos + 1
                while j < w:
                    if min_hash == ring_hash.load(j) and min_meta != ring_meta.load(j):
                        if count >= output_capacity:
                            return -1
                        hashes.store(count, min_hash)
                        positions.store(count, Int(ring_meta.load(j) >> 1))
                        strands.store(count, Int(ring_meta.load(j) & UInt64(1)))
                        count += 1
                    j += 1
                j = 0
                while j <= ring_pos:
                    if min_hash == ring_hash.load(j) and min_meta != ring_meta.load(j):
                        hashes.store(count, min_hash)
                        positions.store(count, Int(ring_meta.load(j) >> 1))
                        strands.store(count, Int(ring_meta.load(j) & UInt64(1)))
                        count += 1
                    j += 1
        ring_pos += 1
        if ring_pos == w:
            ring_pos = 0
        i += 1
    if min_hash != invalid:
        if count >= output_capacity:
            return -1
        hashes.store(count, min_hash)
        positions.store(count, Int(min_meta >> 1))
        strands.store(count, Int(min_meta & UInt64(1)))
        count += 1
    return count


def ilog2(x: Int) -> Int:
    var value = x
    var result = 0
    while value > 1:
        value >>= 1
        result += 1
    return result


@export("mmp_chain_dp")
def mmp_chain_dp(
    reference_addr: Int,
    query_addr: Int,
    spans_addr: Int,
    n: Int,
    max_gap: Int,
    bandwidth: Int,
    max_skip: Int,
    penalty_scale: Int,
    scores_addr: Int,
    predecessors_addr: Int,
) abi("C") -> Int:
    if n <= 0 or reference_addr == 0 or query_addr == 0 or spans_addr == 0:
        return -1
    if scores_addr == 0 or predecessors_addr == 0:
        return -1
    var reference = IPtr(unsafe_from_address=reference_addr)
    var query = IPtr(unsafe_from_address=query_addr)
    var spans = IPtr(unsafe_from_address=spans_addr)
    var scores = IPtr(unsafe_from_address=scores_addr)
    var predecessors = IPtr(unsafe_from_address=predecessors_addr)
    var best_index = 0
    var i = 0
    while i < n:
        var qi = query.load(i)
        var ri = reference.load(i)
        var best_score = spans.load(i)
        var best_prev = -1
        var skips = 0
        var j = i - 1
        while j >= 0:
            var dq = qi - query.load(j)
            var dr = ri - reference.load(j)
            if dr > max_gap:
                break
            if dq > 0 and dr > 0 and dq <= max_gap:
                var diagonal = dr - dq if dr > dq else dq - dr
                if diagonal <= bandwidth:
                    var gap = dr if dr < dq else dq
                    var span = spans.load(j)
                    var contribution = span if span < gap else gap
                    if diagonal > 0 or gap > span:
                        var linear = penalty_scale * diagonal
                        var logarithmic = ilog2(diagonal + 1) if diagonal > 0 else 0
                        contribution -= (linear + 50 * logarithmic) / 100
                    var candidate = scores.load(j) + contribution
                    if candidate > best_score:
                        best_score = candidate
                        best_prev = j
                        if skips > 0:
                            skips -= 1
                    else:
                        skips += 1
                        if skips > max_skip:
                            break
            j -= 1
        scores.store(i, best_score)
        predecessors.store(i, best_prev)
        if best_score > scores.load(best_index):
            best_index = i
        i += 1
    return best_index
