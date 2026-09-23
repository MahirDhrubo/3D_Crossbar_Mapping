from copy import deepcopy
import unittest

from mem3dmapper.mapping.state import MappingState
from mem3dmapper.mapping.types import MappingConfig, Operation, Operation_type
from mem3dmapper.mapping.validator import MappingValidationError, validate_mapping
from mem3dmapper.netlist.types import Gate, GateType, Netlist


def make_valid_mapping() -> tuple[Netlist, MappingState]:
    """Create a small valid horizontal-then-vertical MAGIC schedule."""

    netlist = Netlist(
        name="validator_fixture",
        primary_inputs=["a", "b", "c"],
        primary_outputs=["y"],
        gates=[
            Gate(0, GateType.NOR, ["a", "b"], "n0"),
            Gate(1, GateType.AND, ["n0", "c"], "y"),
        ],
    )
    netlist.build_gate_map()

    state = MappingState(MappingConfig(total_rows=4, total_columns=4))
    state.init_params(netlist.primary_outputs)
    state.uses_left = {"a": 1, "b": 1, "c": 1, "n0": 1}

    state.write_net("a", (0, 0))
    state.primary_input_locations["a"] = (0, 0)
    state.write_net("b", (1, 0))
    state.primary_input_locations["b"] = (1, 0)
    state.write_net("c", (0, 2))
    state.primary_input_locations["c"] = (0, 2)

    state.execute_net("n0", (2, 0), GateType.NOR, ("a", "b"), [(0, 0), (1, 0)])
    state.uses_left["a"] -= 1
    state.uses_left["b"] -= 1

    state.copy_net("n0", (2, 0), (0, 1))

    state.execute_net("y", (0, 3), GateType.AND, ("n0", "c"), [(0, 1), (0, 2)])
    state.uses_left["n0"] -= 1
    state.uses_left["c"] -= 1
    state.primary_output_locations["y"] = (0, 3)

    state.tail_x = [3, 1, 1, 1]
    return netlist, state


class ValidatorTests(unittest.TestCase):
    def assert_invalid(self, netlist: Netlist, state: MappingState, text: str) -> None:
        with self.assertRaisesRegex(MappingValidationError, text):
            validate_mapping(netlist, state)

    def test_accepts_valid_mapping(self) -> None:
        netlist, state = make_valid_mapping()
        validate_mapping(netlist, state)

    def test_accepts_parallel_gates_separated_by_one_row(self) -> None:
        netlist = Netlist(
            name="valid_concurrency",
            primary_inputs=["a", "b", "c", "d"],
            primary_outputs=["x", "y"],
            gates=[
                Gate(0, GateType.NOR, ["a", "b"], "x"),
                Gate(1, GateType.NOR, ["c", "d"], "y"),
            ],
        )
        netlist.build_gate_map()
        state = MappingState(MappingConfig(4, 4))
        state.init_params(netlist.primary_outputs)
        state.uses_left = {net: 1 for net in netlist.primary_inputs}
        writes = {
            "a": (0, 0),
            "b": (1, 0),
            "c": (0, 2),
            "d": (1, 2),
        }
        for net, cell in writes.items():
            state.write_net(net, cell)
            state.primary_input_locations[net] = cell

        state.execute_concurrent_nets(
            netlist.gates,
            [(2, 0), (2, 2)],
            {0: [(0, 0), (1, 0)], 1: [(0, 2), (1, 2)]},
        )
        for net in netlist.primary_inputs:
            state.uses_left[net] -= 1
        state.primary_output_locations = {"x": (2, 0), "y": (2, 2)}
        state.tail_x = [3, 0, 3, 0]

        validate_mapping(netlist, state)

    def test_accepts_reused_series_window_with_matching_direction(self) -> None:
        netlist = Netlist(
            name="valid_series_reuse",
            primary_inputs=["a", "b", "c", "d"],
            primary_outputs=["x", "y"],
            gates=[
                Gate(0, GateType.AND, ["a", "b"], "x"),
                Gate(1, GateType.AND, ["c", "d"], "y"),
            ],
        )
        netlist.build_gate_map()
        state = MappingState(MappingConfig(4, 4))
        state.init_params(netlist.primary_outputs)
        state.uses_left = {net: 1 for net in netlist.primary_inputs}
        writes = {
            "a": (0, 1),
            "b": (0, 2),
            "c": (1, 1),
            "d": (1, 2),
        }
        for net, cell in writes.items():
            state.write_net(net, cell)
            state.primary_input_locations[net] = cell

        state.execute_concurrent_nets(
            netlist.gates,
            [(0, 0), (1, 0)],
            {0: [(0, 1), (0, 2)], 1: [(1, 1), (1, 2)]},
        )
        for net in netlist.primary_inputs:
            state.uses_left[net] -= 1
        state.primary_output_locations = {"x": (0, 0), "y": (1, 0)}
        state.tail_x = [2, 2, 2, 0]

        validate_mapping(netlist, state)

    def test_rejects_missing_gate_input_cell(self) -> None:
        netlist, state = make_valid_mapping()
        state.ops[3] = Operation(
            Operation_type.NOR,
            "n0",
            1,
            (2, 0),
            inputs=("a", "b"),
            src=[(0, 0)],
        )
        self.assert_invalid(netlist, state, "needs 2 input cells")

    def test_rejects_gate_before_input_is_available(self) -> None:
        netlist, state = make_valid_mapping()
        write_c = Operation(Operation_type.WRITE, "c", 3, (0, 2))
        state.ops = state.ops[:2] + state.ops[3:5] + [write_c, state.ops[5]]
        self.assert_invalid(netlist, state, "input cells contain")

    def test_rejects_bad_parallel_geometry(self) -> None:
        netlist, state = make_valid_mapping()
        state.ops[3] = Operation(
            Operation_type.NOR,
            "n0",
            1,
            (2, 1),
            inputs=("a", "b"),
            src=[(0, 0), (1, 0)],
        )
        self.assert_invalid(netlist, state, "must occupy one row")

    def test_rejects_bad_series_geometry(self) -> None:
        netlist, state = make_valid_mapping()
        state.ops[-1] = Operation(
            Operation_type.AND,
            "y",
            3,
            (1, 3),
            inputs=("n0", "c"),
            src=[(0, 1), (0, 2)],
        )
        self.assert_invalid(netlist, state, "one column")

    def test_rejects_copy_from_wrong_net(self) -> None:
        netlist, state = make_valid_mapping()
        state.ops[4] = Operation(Operation_type.COPY, "n0", 2, (0, 1), src=[(0, 0)])
        self.assert_invalid(netlist, state, "COPY source .* contains 'a', not 'n0'")

    def test_rejects_overwriting_live_cell(self) -> None:
        netlist, state = make_valid_mapping()
        state.ops[4] = Operation(Operation_type.COPY, "c", 2, (2, 0), src=[(0, 2)])
        self.assert_invalid(netlist, state, "overwrites live net 'n0'")

    def test_live_copy_can_be_reused_when_another_copy_survives(self) -> None:
        netlist = Netlist(
            name="live_copy_reuse",
            primary_inputs=["a", "b", "c"],
            primary_outputs=["x", "y"],
            gates=[
                Gate(0, GateType.NOR, ["b", "c"], "n0"),
                Gate(1, GateType.NOR, ["a", "b"], "x"),
                Gate(2, GateType.NOR, ["a", "c"], "y"),
            ],
        )
        netlist.build_gate_map()
        state = MappingState(MappingConfig(3, 6))
        state.init_params(netlist.primary_outputs)
        state.uses_left = {"a": 2, "b": 2, "c": 2, "n0": 0}

        for net, cell in {"a": (0, 0), "b": (1, 0), "c": (2, 0)}.items():
            state.write_net(net, cell)
            state.primary_input_locations[net] = cell

        state.copy_net("a", (0, 0), (3, 0))
        state.copy_net("c", (2, 0), (3, 0))

        state.execute_net("n0", (4, 0), GateType.NOR, ("b", "c"), [(1, 0), (2, 0)])
        state.uses_left["b"] -= 1
        state.uses_left["c"] -= 1

        state.execute_net("x", (5, 0), GateType.NOR, ("a", "b"), [(0, 0), (1, 0)])
        state.uses_left["a"] -= 1
        state.uses_left["b"] -= 1
        state.primary_output_locations["x"] = (5, 0)

        state.execute_net("y", (4, 0), GateType.NOR, ("a", "c"), [(0, 0), (2, 0)])
        state.uses_left["a"] -= 1
        state.uses_left["c"] -= 1
        state.primary_output_locations["y"] = (4, 0)
        state.tail_x = [6, 0, 0]

        validate_mapping(netlist, state)

    def test_live_copy_can_be_reused_while_net_participates(self) -> None:
        """A copy of a live net may be overwritten by an operation that also
        reads that same net this cycle (from a different cell), as long as
        another copy survives elsewhere. The read uses the pre-cycle
        snapshot, so it is unaffected by the destination cell being
        overwritten within the same cycle."""

        netlist = Netlist(
            name="participating_live_copy",
            primary_inputs=["a", "b"],
            primary_outputs=["x", "y"],
            gates=[
                Gate(0, GateType.NOR, ["a", "b"], "x"),
                Gate(1, GateType.NOR, ["a", "b"], "y"),
            ],
        )
        netlist.build_gate_map()
        state = MappingState(MappingConfig(2, 4))
        state.init_params(netlist.primary_outputs)
        state.uses_left = {"a": 2, "b": 2}
        for net, cell in {"a": (0, 0), "b": (1, 0)}.items():
            state.write_net(net, cell)
            state.primary_input_locations[net] = cell

        state.copy_net("a", (0, 0), (2, 0))

        # Overwrites the copy of 'a' at (2, 0) while also reading 'a' from
        # (0, 0) as an input in this same operation -- allowed, since (0, 0)
        # still holds 'a' afterward.
        state.execute_net("x", (2, 0), GateType.NOR, ("a", "b"), [(0, 0), (1, 0)])
        state.uses_left["a"] -= 1
        state.uses_left["b"] -= 1
        state.primary_output_locations["x"] = (2, 0)

        state.execute_net("y", (3, 0), GateType.NOR, ("a", "b"), [(0, 0), (1, 0)])
        state.uses_left["a"] -= 1
        state.uses_left["b"] -= 1
        state.primary_output_locations["y"] = (3, 0)
        state.tail_x = [4, 0]

        validate_mapping(netlist, state)

    def test_rejects_reported_state_that_disagrees_with_replay(self) -> None:
        netlist, state = make_valid_mapping()
        state.location_to_net[(3, 3)] = "ghost"
        self.assert_invalid(netlist, state, "location_to_net does not match")

    def test_primary_output_anchor_ignores_copied_locations(self) -> None:
        netlist, state = make_valid_mapping()
        state.copy_net("y", (0, 3), (1, 3))
        state.tail_x[3] = 2

        validate_mapping(netlist, state)
        self.assertEqual(state.primary_output_locations, {"y": (0, 3)})
        self.assertEqual(state.net_location["y"], {(0, 3), (1, 3)})

    def test_primary_output_copy_may_be_reclaimed_but_anchor_may_not(self) -> None:
        netlist, state = make_valid_mapping()
        state.copy_net("y", (0, 3), (1, 3))
        state.copy_net("n0", (2, 0), (1, 3))
        state.tail_x[3] = 2

        validate_mapping(netlist, state)
        self.assertEqual(state.primary_output_locations, {"y": (0, 3)})
        self.assertEqual(state.net_location["y"], {(0, 3)})

        invalid_state = deepcopy(state)
        invalid_state.copy_net("n0", (2, 0), (0, 3))
        self.assert_invalid(netlist, invalid_state, "retained primary-output cell")

    def test_primary_input_copy_may_be_reclaimed_but_anchor_may_not(self) -> None:
        netlist, state = make_valid_mapping()
        state.copy_net("a", (0, 0), (1, 3))
        state.copy_net("n0", (2, 0), (1, 3))
        state.tail_x[3] = 2

        validate_mapping(netlist, state)
        self.assertEqual(state.primary_input_locations, {"a": (0, 0), "b": (1, 0), "c": (0, 2)})
        self.assertEqual(state.net_location["a"], {(0, 0)})

        invalid_state = deepcopy(state)
        invalid_state.copy_net("n0", (2, 0), (0, 0))
        self.assert_invalid(netlist, invalid_state, "reserved primary-input cell")

    def test_rejects_same_or_adjacent_parallel_rows(self) -> None:
        netlist = Netlist(
            name="blocked_rows",
            primary_inputs=["a", "b", "c", "d"],
            primary_outputs=["x", "y"],
            gates=[
                Gate(0, GateType.NOR, ["a", "b"], "x"),
                Gate(1, GateType.NOR, ["c", "d"], "y"),
            ],
        )
        netlist.build_gate_map()
        state = MappingState(MappingConfig(4, 4))
        state.init_params(netlist.primary_outputs)
        state.uses_left = {net: 0 for net in netlist.primary_inputs}
        writes = {
            "a": (0, 0),
            "b": (1, 0),
            "c": (0, 1),
            "d": (1, 1),
        }
        for net, cell in writes.items():
            state.write_net(net, cell)
            state.primary_input_locations[net] = cell
        state.ops.extend(
            [
                Operation(Operation_type.NOR, "x", 1, (2, 0), ("a", "b"), [(0, 0), (1, 0)]),
                Operation(Operation_type.NOR, "y", 1, (2, 1), ("c", "d"), [(0, 1), (1, 1)]),
            ]
        )
        state.cycle_count = 1
        self.assert_invalid(netlist, state, "same or adjacent rows")


if __name__ == "__main__":
    unittest.main()
