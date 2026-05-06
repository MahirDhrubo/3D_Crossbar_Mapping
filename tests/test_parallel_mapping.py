import random
import unittest

from mem3dmapper.mapping.parallel_mapping import (
    _check_parallel_gate_placement,
    _check_parallel_gate_placement_parallel,
)
from mem3dmapper.mapping.state import MappingState
from mem3dmapper.mapping.types import MappingConfig
from mem3dmapper.netlist.types import Gate, GateType


class ParallelPlacementSearchTests(unittest.TestCase):
    def test_parallel_search_matches_sequential_search(self) -> None:
        state = MappingState(MappingConfig(total_rows=3, total_columns=4))
        state.init_params([])
        state.uses_left = {"a": 1, "b": 1}
        state.write_net("a", (0, 0))
        state.write_net("b", (1, 0))
        gate = Gate(0, GateType.NOR, ["a", "b"], "out")

        sequential = _check_parallel_gate_placement(state, gate, blocked_rows=[1])
        parallel = _check_parallel_gate_placement_parallel(
            state,
            gate,
            blocked_rows=[1],
            max_workers=2,
        )

        self.assertEqual(parallel, sequential)

    def test_parallel_search_handles_no_usable_rows(self) -> None:
        state = MappingState(MappingConfig(total_rows=2, total_columns=4))
        state.init_params([])
        gate = Gate(0, GateType.NOR, ["a", "b"], "out")

        placement, cost = _check_parallel_gate_placement_parallel(
            state,
            gate,
            blocked_rows=[0, 1],
        )

        self.assertIsNone(placement)
        self.assertEqual(cost, float("inf"))

    def test_fast_search_matches_exhaustive_search_on_random_states(self) -> None:
        rng = random.Random(42)

        for case in range(200):
            rows = rng.randint(1, 4)
            columns = rng.randint(2, 8)
            state = MappingState(
                MappingConfig(
                    total_rows=rows,
                    total_columns=columns,
                    alpha=rng.choice([0.0, 0.5, 1.0]),
                    beta=rng.choice([0.0, 0.5, 1.0]),
                    gamma=0.0,
                )
            )
            state.init_params([])
            state.uses_left = {
                "a": rng.randint(0, 2),
                "b": rng.randint(0, 2),
                "dead": 0,
                "live": 1,
            }

            for row in range(rows):
                for column in range(columns):
                    if rng.random() < 0.55:
                        state._put_net(
                            rng.choice(["a", "b", "dead", "live"]),
                            (column, row),
                        )

            gate_type = rng.choice([GateType.NOR, GateType.NOT])
            gate_inputs = ["a", "b"] if gate_type == GateType.NOR else ["a"]
            gate = Gate(case, gate_type, gate_inputs, f"out{case}")
            blocked_rows = [row for row in range(rows) if rng.random() < 0.25]

            exhaustive = _check_parallel_gate_placement(
                state,
                gate,
                blocked_rows,
            )
            fast = _check_parallel_gate_placement_parallel(
                state,
                gate,
                blocked_rows,
            )
            self.assertEqual(fast, exhaustive, f"Mismatch in random case {case}")


if __name__ == "__main__":
    unittest.main()
