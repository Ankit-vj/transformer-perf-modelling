# Save as: C:\Users\ankit\transformer-perf-model\transformer_perf\mapping\tiling.py

"""
Loop Tiling Strategies for Matrix Operations.
Determines optimal tile sizes to maximize cache utilization
when executing transformer operations on RISC-V cores.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class TileConfig:
    """Configuration for a single tiling dimension"""
    tile_m: int = 64       # Tile size for M dimension
    tile_n: int = 64       # Tile size for N dimension
    tile_k: int = 64       # Tile size for K dimension

    @property
    def tile_volume(self) -> int:
        """Total elements in tile"""
        return self.tile_m * self.tile_n * self.tile_k

    def summary(self) -> Dict:
        return {
            "tile_m": self.tile_m,
            "tile_n": self.tile_n,
            "tile_k": self.tile_k,
        }


@dataclass
class TilingResult:
    """Complete tiling analysis result"""
    tile_config: TileConfig
    num_tiles_m: int = 0
    num_tiles_n: int = 0
    num_tiles_k: int = 0
    total_tiles: int = 0
    data_per_tile_bytes: int = 0
    target_cache_level: str = "L1D"
    cache_utilization: float = 0.0
    estimated_cache_misses: int = 0
    reuse_ratio: float = 1.0

    def summary(self) -> Dict:
        return {
            "tile_config": self.tile_config.summary(),
            "num_tiles": {
                "m": self.num_tiles_m,
                "n": self.num_tiles_n,
                "k": self.num_tiles_k,
                "total": self.total_tiles,
            },
            "data_per_tile_bytes": self.data_per_tile_bytes,
            "target_cache": self.target_cache_level,
            "cache_utilization": f"{self.cache_utilization:.1%}",
            "reuse_ratio": self.reuse_ratio,
        }


class TilingStrategy:
    """
    Compute optimal loop tiling for matrix operations
    to maximize cache utilization on RISC-V cores.

    For GEMM: C[M,N] = A[M,K] × B[K,N]
    Tiling creates sub-problems that fit in cache:
        C_tile[tm,tn] = A_tile[tm,tk] × B_tile[tk,tn]
    """

    def __init__(self, memory_hierarchy=None):
        """
        Args:
            memory_hierarchy: MemoryHierarchy object from hardware module
        """
        self.memory = memory_hierarchy

        # Default cache sizes if no memory hierarchy provided
        self.default_cache_sizes = {
            "L1D": 32 * 1024,       # 32 KB
            "L2": 256 * 1024,       # 256 KB
            "L3": 4 * 1024 * 1024,  # 4 MB
        }

    def _get_cache_size(self, level: str) -> int:
        """Get cache size for a given level"""
        if self.memory is not None:
            level_map = {
                "L1D": self.memory.l1d,
                "L2": self.memory.l2,
                "L3": self.memory.l3,
            }
            cache = level_map.get(level)
            if cache:
                return cache.size_bytes
        return self.default_cache_sizes.get(level, 32 * 1024)

    def compute_gemm_tiling(
        self,
        M: int,
        N: int,
        K: int,
        bytes_per_element: int = 4,
        target_cache: str = "L1D",
        cache_usage_fraction: float = 0.75,
        vlen_elements: int = 8
    ) -> TilingResult:
        """
        Compute optimal tile sizes for GEMM operation.

        For C[M,N] = A[M,K] × B[K,N], each tile needs:
        - A_tile: tm × tk elements
        - B_tile: tk × tn elements
        - C_tile: tm × tn elements
        Total: (tm*tk + tk*tn + tm*tn) × bytes_per_element

        Args:
            M: Rows of output matrix
            N: Columns of output matrix
            K: Inner dimension
            bytes_per_element: Bytes per data element
            target_cache: Target cache level ("L1D", "L2", "L3")
            cache_usage_fraction: Fraction of cache to use (leave room)
            vlen_elements: Vector register width in elements

        Returns:
            TilingResult with optimal configuration
        """
        cache_size = self._get_cache_size(target_cache)
        usable_cache = int(cache_size * cache_usage_fraction)

        # Find tile sizes that fit in cache
        # Constraint: (tm*tk + tk*tn + tm*tn) * bytes <= usable_cache
        tile_config = self._find_optimal_tiles(
            M, N, K, bytes_per_element, usable_cache, vlen_elements
        )

        # Calculate number of tiles
        num_tiles_m = math.ceil(M / tile_config.tile_m)
        num_tiles_n = math.ceil(N / tile_config.tile_n)
        num_tiles_k = math.ceil(K / tile_config.tile_k)
        total_tiles = num_tiles_m * num_tiles_n * num_tiles_k

        # Data per tile
        tm, tn, tk = tile_config.tile_m, tile_config.tile_n, tile_config.tile_k
        data_per_tile = (tm * tk + tk * tn + tm * tn) * bytes_per_element

        # Cache utilization
        cache_utilization = data_per_tile / cache_size

        # Estimate data reuse ratio
        # B tiles are reused across M tiles, A tiles reused across N tiles
        reuse_ratio = self._estimate_reuse_ratio(
            M, N, K, tm, tn, tk, bytes_per_element, cache_size
        )

        return TilingResult(
            tile_config=tile_config,
            num_tiles_m=num_tiles_m,
            num_tiles_n=num_tiles_n,
            num_tiles_k=num_tiles_k,
            total_tiles=total_tiles,
            data_per_tile_bytes=data_per_tile,
            target_cache_level=target_cache,
            cache_utilization=cache_utilization,
            reuse_ratio=reuse_ratio,
        )

    def _find_optimal_tiles(
        self,
        M: int,
        N: int,
        K: int,
        bpe: int,
        cache_bytes: int,
        vlen: int
    ) -> TileConfig:
        """
        Find optimal tile sizes using heuristic search.

        Strategy:
        1. Align tile sizes to vector length for efficient vectorization
        2. Maximize tile size while fitting in cache
        3. Prioritize N dimension (output columns) for vectorization
        """
        # Start with maximum possible tiles
        # Constraint: (tm*tk + tk*tn + tm*tn) * bpe <= cache_bytes
        max_elements = cache_bytes // bpe

        # Heuristic: try to make tiles roughly cubic
        # but aligned to vector length
        cubic_side = int(math.pow(max_elements / 3, 1.0 / 2))

        # Align to vector length
        def align(val, alignment):
            return max(alignment, (val // alignment) * alignment)

        # Start with aligned cubic tiles
        tn = align(min(cubic_side, N), vlen)
        tm = align(min(cubic_side, M), vlen)

        # Calculate maximum tk that fits
        # tm*tk + tk*tn + tm*tn <= max_elements
        # tk*(tm + tn) <= max_elements - tm*tn
        remaining = max_elements - tm * tn
        if (tm + tn) > 0:
            tk = min(remaining // (tm + tn), K)
        else:
            tk = min(K, vlen)

        tk = max(align(tk, vlen), vlen)

        # Verify it fits
        total = (tm * tk + tk * tn + tm * tn) * bpe
        while total > cache_bytes and tm > vlen:
            tm = tm - vlen
            total = (tm * tk + tk * tn + tm * tn) * bpe

        while total > cache_bytes and tk > vlen:
            tk = tk - vlen
            total = (tm * tk + tk * tn + tm * tn) * bpe

        # Ensure minimum tile sizes
        tm = max(tm, min(vlen, M))
        tn = max(tn, min(vlen, N))
        tk = max(tk, min(vlen, K))

        return TileConfig(tile_m=tm, tile_n=tn, tile_k=tk)

    def _estimate_reuse_ratio(
        self,
        M: int, N: int, K: int,
        tm: int, tn: int, tk: int,
        bpe: int, cache_size: int
    ) -> float:
        """
        Estimate data reuse ratio.
        Higher ratio = less data movement from memory.
        """
        # Total data without tiling
        total_data_no_tile = (M * K + K * N + M * N) * bpe

        # With tiling: each B tile loaded num_tiles_m times
        # each A tile loaded num_tiles_n times
        num_tm = math.ceil(M / tm)
        num_tn = math.ceil(N / tn)
        num_tk = math.ceil(K / tk)

        # Data loaded with tiling (simplified model)
        a_loads = num_tn * M * K * bpe      # A loaded once per N-tile
        b_loads = num_tm * K * N * bpe      # B loaded once per M-tile
        c_loads = num_tk * M * N * bpe * 2  # C read+write per K-tile

        total_data_tiled = a_loads + b_loads + c_loads

        if total_data_tiled == 0:
            return 1.0

        # Reuse ratio: how much less data we load compared to naive
        return max(1.0, total_data_no_tile / total_data_tiled * num_tk)

    def compute_attention_tiling(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        bytes_per_element: int = 4,
        target_cache: str = "L2"
    ) -> Dict[str, TilingResult]:
        """
        Compute tiling for all attention sub-operations.

        Attention:
            1. QK = Q[S,D] @ K^T[D,S]  → [S,S]
            2. AV = softmax(QK)[S,S] @ V[S,D]  → [S,D]
        """
        results = {}

        # QK^T: [seq_len, head_dim] @ [head_dim, seq_len]
        results["QK_matmul"] = self.compute_gemm_tiling(
            M=seq_len, N=seq_len, K=head_dim,
            bytes_per_element=bytes_per_element,
            target_cache=target_cache,
        )

        # Attn @ V: [seq_len, seq_len] @ [seq_len, head_dim]
        results["AttnV_matmul"] = self.compute_gemm_tiling(
            M=seq_len, N=head_dim, K=seq_len,
            bytes_per_element=bytes_per_element,
            target_cache=target_cache,
        )

        return results

    def compute_mlp_tiling(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        intermediate_dim: int,
        bytes_per_element: int = 4,
        target_cache: str = "L2"
    ) -> Dict[str, TilingResult]:
        """
        Compute tiling for MLP layers.

        MLP:
            1. Up: [B*S, H] @ [H, I] → [B*S, I]
            2. Down: [B*S, I] @ [I, H] → [B*S, H]
        """
        M = batch_size * seq_len

        results = {}

        results["up_projection"] = self.compute_gemm_tiling(
            M=M, N=intermediate_dim, K=hidden_dim,
            bytes_per_element=bytes_per_element,
            target_cache=target_cache,
        )

        results["down_projection"] = self.compute_gemm_tiling(
            M=M, N=hidden_dim, K=intermediate_dim,
            bytes_per_element=bytes_per_element,
            target_cache=target_cache,
        )

        return results


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    tiler = TilingStrategy()

    print("=" * 60)
    print("GEMM Tiling Analysis: [128, 768] × [768, 3072]")
    print("=" * 60)

    result = tiler.compute_gemm_tiling(
        M=128, N=3072, K=768,
        bytes_per_element=4,
        target_cache="L1D"
    )

    for key, val in result.summary().items():
        print(f"  {key}: {val}")

    print("\n" + "=" * 60)
    print("Attention Tiling Analysis")
    print("=" * 60)

    attn_results = tiler.compute_attention_tiling(
        batch_size=1, num_heads=12,
        seq_len=128, head_dim=64
    )

    for name, res in attn_results.items():
        print(f"\n  {name}:")
        for key, val in res.summary().items():
            print(f"    {key}: {val}")