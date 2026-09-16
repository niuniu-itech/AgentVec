"""H — the Ascend310P1 hardware abstraction used by P-AIR's Align(I, H).

These are the real platform facts (from the CANN platform_config Ascend310P1.ini on
the target). P-AIR uses them as the *hardware-derived legality boundary*: every
schedule decision that can be settled deductively from the intent and these facts is
settled symbolically (tile <= UB, lmul/blockDim ceilings, packing, roofline regime),
never by the model. Cf. AgentVec Table "Division of labor in Align".
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class AscendHW:
    soc: str = "Ascend310P1"
    aic_arch: str = "dav-m200"
    ai_core_cnt: int = 10          # AI Cores (cube+vector)
    vector_core_cnt: int = 8       # dedicated vector cores (AIV)
    cube_m: int = 16               # cube tile dims (fp16 16x16x16)
    cube_n: int = 16
    cube_k: int = 16
    cube_freq_mhz: int = 1060
    vec_freq_mhz: int = 1000
    ub_bytes: int = 262144         # Unified Buffer per core (256 KB)
    ub_align: int = 32             # ubblock_size — UB access granularity (bytes)
    l0a_bytes: int = 65536
    l0b_bytes: int = 65536
    l0c_bytes: int = 262144
    l1_bytes: int = 1048576
    l2_bytes: int = 16777216
    # relative bandwidths (bytes/cycle units from the .ini); used for roofline regime
    ddr_rate: int = 17
    l2_rate: int = 114
    ub_to_l1_rate: int = 256

    def dtype_bytes(self, dtype: str) -> int:
        return {"f16": 2, "f32": 4, "i32": 4}[dtype]

    def vector_lane_bytes(self) -> int:
        # AI Core vector unit processes 256 B / cycle (128 fp16 or 64 fp32).
        return 256

    def peak_cube_flops(self) -> float:
        # 2 flops/MAC * m*n*k MACs/cycle * cores * freq
        return 2.0 * self.cube_m * self.cube_n * self.cube_k * self.ai_core_cnt * self.cube_freq_mhz * 1e6

    def roofline_ridge_flop_per_byte(self) -> float:
        # Compute peak / (relative DDR byte rate scaled by vector clock); a *relative*
        # ridge that classifies an operator as memory- or compute-bound. The constant
        # only needs to be consistent across operators for the regime decision.
        peak = 2.0 * self.vector_lane_bytes() / 4 * self.vector_core_cnt * self.vec_freq_mhz * 1e6  # ~fp32 vec flops
        bw = self.ddr_rate * self.vec_freq_mhz * 1e6 * 8  # relative bytes/s, 8 vec engines
        return peak / bw

    def max_map_tile_elems(self, dtype: str, n_buffers: int, double_buffer: bool = True) -> int:
        """Largest UB tile (elements) that fits `n_buffers` live tensors (x double buffer)."""
        eb = self.dtype_bytes(dtype)
        factor = (2 if double_buffer else 1) * n_buffers
        raw = self.ub_bytes // (factor * eb)
        # align down to the UB granularity (in elements), keep a small safety margin
        per = self.ub_align // eb if self.ub_align >= eb else 1
        raw = int(raw * 0.9)
        return max(per, (raw // per) * per)


HW = AscendHW()
