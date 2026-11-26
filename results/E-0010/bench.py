"""Физические прогоны. Показания среды для разбора записываются после решения."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import json
from pathlib import Path
import sys
import importlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from maze.datasets import load_split
from maze.environment import Environment, Status
from maze.excursion_automaton import Excursion, RangeExcursion
from maze.frontier_reserve import ReserveFrontier
from maze.interface import Move
from maze.leg_automaton import AnchorOrder
from maze.levels import Level
from maze.passage_automaton import JointPassage, Passage

AGENTS = {"reserve": ReserveFrontier, "anchor": AnchorOrder, "passage": Passage,
          "joint": JointPassage, "excursion": Excursion, "range": RangeExcursion,
          "roads": "road_hypotheses.RoadHypotheses", "range-roads": "road_hypotheses.RangeRoads",
          "bridge": "bridge_automaton.BridgePassage", "relay": "relay_automaton.Relay",
          "corridor": "bridge_automaton.CorridorBridge", "cells": "cell_hypotheses.CellRoads",
          "fork-budget": "fork_budget.ForkBudget", "outer": "outer_automaton.OuterRange",
          "outer-service": "outer_automaton.OuterService", "coupled": "road_hypotheses.CoupledRoads",
          "return": "road_hypotheses.ReturnEvidence", "witness": "route_witness.WitnessRoads",
          "streets": "pattern_roads.PatternRoads", "last-unit": "last_unit.LastUnit",
          "shared": "shared_approach.SharedApproach", "consensus": "shared_approach.Consensus",
          "homing": "homing_automaton.Homing", "probe-order": "probe_order.ProbeOrder",
          "mission": "mission_automaton.MissionRoads", "marked-streets": "pattern_roads.MarkedStreets",
          "many-streets": "pattern_roads.ManyStreets", "mixed-streets": "pattern_roads.MixedStreets",
          "trunk": "shared_trunk.SharedTrunk", "trunk-reserve": "shared_trunk.TrunkReserve",
          "street-trunk": "street_trunk.StreetTrunk","exact-worlds":"exact_worlds.ExactWorlds",
          "street-consensus":"calibrated_routes.StreetConsensus","calibrated":"calibrated_routes.CalibratedRoutes",
          "all-approaches":"pattern_roads.AllApproaches","insured-approaches":"pattern_roads.InsuredApproaches",
          "survey-streets":"sortie_automaton.SurveyStreets","scouting-sortie":"sortie_automaton.ScoutingSortie",
          "terminal-streets":"sortie_automaton.TerminalStreets","texture-cells":"texture_roads.TextureCells",
          "adaptive-trunk":"street_trunk.AdaptiveTrunk","stable-streets":"pattern_roads.StableStreets",
          "preview":"preview_roads.PreviewRoads","symmetric-streets":"symmetric_roads.SymmetricStreets",
          "last-mission":"last_mission.LastMission","cycle-gate":"street_trunk.CycleGate",
          "margin-gate":"street_trunk.MarginGate","wall-lines":"wall_lines.WallLines",
          "proven-approach":"proven_approach.ProvenApproach","individual-stops":"last_mission.IndividualStops",
          "deadline-mission":"last_mission.DeadlineMission","deadline-goal":"deadline_goal.DeadlineGoal",
          "paired-witness":"paired_witness.PairedWitness","common-vote":"pattern_roads.CommonVote",
          "street-trunk-reserve":"street_trunk.StreetTrunkReserve","compact-trunk":"compact_trunk.CompactTrunk",
          "deduced-roads":"deduced_roads.DeducedRoads","conditioned-streets":"conditioned_streets.ConditionedStreets",
          "room-passages":"room_roads.RoomPassages","weighted-rooms":"room_roads.WeightedRooms",
          "implied-turn":"implied_turn.ImpliedTurn","renewable-routes":"renewable_routes.RenewableRoutes",
          "passage-flow":"passage_flow.PassageFlow","mission-order":"mission_order.MissionOrder",
          "remote-station":"remote_station.RemoteStation","mandatory-stop":"remote_station.MandatoryStop",
          "viability-vote":"last_mission.ViabilityVote","independent-exits":"street_trunk.IndependentExits",
          "stretch-sortie":"stretch_sortie.StretchSortie","conditional-mission":"conditional_mission.ConditionalMission",
          "window-bounds":"window_bounds.WindowBounds","uniform-streets":"uniform_streets.UniformStreets",
          "translated-paths":"translated_paths.TranslatedPaths","room-reserve":"room_roads.RoomReserve",
          "weighted-room-reserve":"room_roads.WeightedRoomReserve","room-evidence":"room_roads.RoomEvidence",
          "no-probe-streets":"no_probe.NoProbeStreets","last-direction":"no_probe.LastDirection",
          "known-landing":"known_landing.KnownLanding","boundary-landing":"known_landing.BoundaryLanding",
          "sampled-sortie":"sampled_sortie.SampledSortie","local-corridors":"translated_paths.LocalCorridors",
          "mission-order-reserve":"mission_order.MissionOrderReserve","first-landing":"known_landing.FirstLanding",
          "shortcut-reserve":"shortcut_reserve.ShortcutReserve","after-landing":"known_landing.AfterLanding",
          "outward-after-landing":"known_landing.OutwardAfterLanding",
          "goal-window":"first_window.GoalWindow","margin-window":"first_window.MarginWindow",
          "smallest-detour":"proven_approach.SmallestDetour","gauge-detour":"proven_approach.GaugeDetour",
          "leg-survival":"leg_survival.LegSurvival","room-mission":"mission_order.RoomMission",
          "wall-window":"first_window.WallWindow","matching-survival":"leg_survival.MatchingSurvival",
          "covered-approach":"covered_approach.CoveredApproach","exit-flow":"exit_flow.ExitFlow",
          "boundary-flow":"exit_flow.BoundaryFlow","corridor-flow":"exit_flow.CorridorFlow",
          "exit-current":"exit_flow.ExitCurrent","observed-exit":"covered_approach.ObservedExit",
          "mission-exit-flow":"exit_flow.MissionExitFlow","path-particles":"path_particles.PathParticles",
          "mission-corridor-flow":"exit_flow.MissionCorridorFlow","sampled-margin":"sampled_margin.SampledMargin",
          "surveyed-room-mission":"mission_order.SurveyedRoomMission","simple-exit-paths":"simple_exit_paths.SimpleExitPaths",
          "nearest-station":"nearest_station.NearestStation","mission-simple-paths":"simple_exit_paths.MissionSimplePaths",
          "service-exit-paths":"service_exit_paths.ServiceExitPaths","service-contact":"service_contact.ServiceContact",
          "gateway-paths":"gateway_paths.GatewayPaths","approach-heading":"approach_heading.ApproachHeading",
          "feasible-windows":"feasible_windows.FeasibleWindows","joint-edge-paths":"edge_visibility.JointEdgePaths",
          "station-simple-paths":"nearest_station.StationSimplePaths","useful-service-paths":"service_exit_paths.UsefulServicePaths",
          "compatible-loops":"compatible_loops.CompatibleLoops","outpost-exits":"service_exit_paths.OutpostExits",
          "dominant-loops":"compatible_loops.DominantLoops","weighted-loops":"compatible_loops.WeightedLoops",
          "station-useful-paths":"service_exit_paths.StationUsefulPaths","specific-paths":"specific_paths.SpecificPaths",
          "conditional-exits":"conditional_exits.ConditionalExits","observed-loops":"compatible_loops.ObservedLoops",
          "protected-turn":"spare_turn.ProtectedTurn","spare-turn":"spare_turn.SpareTurn",
          "second-window":"second_window.SecondWindow","open-second-window":"second_window.OpenSecondWindow",
          "axis-loops":"axis_loops.AxisLoops","perimeter-exits":"perimeter_paths.PerimeterExits",
          "perimeter-approach":"perimeter_paths.PerimeterApproach","measured-detours":"measured_detours.MeasuredDetours",
          "bypass-station":"bypass_station.BypassStation","unanimous-bypass":"bypass_station.UnanimousBypass",
          "ordered-trunk":"ordered_trunk.OrderedTrunk","finish-trunk":"ordered_trunk.FinishTrunk",
          "return-paths":"specific_paths.ReturnPaths","last-return-paths":"specific_paths.LastReturnPaths",
          "overlapping-windows":"overlapping_windows.OverlappingWindows","return-windows":"second_window.ReturnWindows",
          "probe-pairs":"probe_pairs.ProbePairs","probe-then-direct":"probe_then_direct.ProbeThenDirect",
          "reference-roads":"reference_roads.ReferenceRoads",
          "backup-allowance":"branch_allowance.BackupAllowance","branch-allowance":"branch_allowance.BranchAllowance",
          "room-lengths":"room_lengths.RoomLengths","proved-finish":"proved_windows.ProvedFinish",
          "proved-windows":"proved_windows.ProvedWindows","route-shield":"route_shield.RouteShield",
          "unseen-return":"unseen_approach.UnseenReturn","unseen-approach":"unseen_approach.UnseenApproach",
          "standby-turn":"standby_turn.StandbyTurn","ranked-standby":"standby_turn.RankedStandby",
          "necessary-return":"necessary_visit.NecessaryReturn","necessary-approach":"necessary_visit.NecessaryApproach",
          "modeled-split":"modeled_trunk.ModeledSplit","modeled-trunk":"modeled_trunk.ModeledTrunk",
          "reference-shells":"reference_shells.ReferenceShells","consistent-shells":"reference_shells.ConsistentShells",
          "sensed-paths":"sensed_paths.SensedPaths","onward-services":"onward_services.OnwardServices",
          "standby-exploration":"standby_exploration.StandbyExploration","visible-standby":"standby_exploration.VisibleStandby",
          "connected-excursions":"connected_excursions.ConnectedExcursions",
          "conditional-standby":"conditional_standby.ConditionalStandby","conditional-probe":"conditional_standby.ConditionalProbe",
          "serviced-windows":"serviced_windows.ServicedWindows","three-windows":"serviced_windows.ThreeWindows",
          "window-patterns":"window_patterns.WindowPatterns","passage-patterns":"window_patterns.PassagePatterns",
          "anchored-directions":"anchored_directions.AnchoredDirections","mapped-directions":"anchored_directions.MappedDirections",
          "mission-shells":"mission_shells.MissionShells","standby-shells":"mission_shells.StandbyShells",
          "linked-services":"linked_services.LinkedServices","seen-linked-services":"linked_services.SeenLinkedServices",
          "exchanged-roads":"exchanged_roads.ExchangedRoads","preserved-exchanges":"exchanged_roads.PreservedExchanges",
          "terrain-trials":"terrain_trials.TerrainTrials","terrain-rooms":"terrain_trials.TerrainRooms",
          "mandatory-proof":"mandatory_supply.MandatoryProof","confirmed-supply":"mandatory_supply.ConfirmedSupply",
          "street-exits":"street_exits.StreetExits","bounded-street-exits":"street_exits.BoundedStreetExits",
          "corner-approaches":"corner_approaches.CornerApproaches","multiple-corners":"corner_approaches.MultipleCornerApproaches",
          "resource-deadline":"mandatory_supply.ResourceDeadline","directed-deadline":"mandatory_supply.DirectedDeadline",
          "gauge-exits":"gauge_exits.GaugeExits","fitted-gauge-exits":"gauge_exits.FittedGaugeExits",
          "side-step":"side_step.SideStep","paired-side-step":"side_step.PairedSideStep","two-side-step":"side_step.TwoSideStep",
          "available-exits":"available_exits.AvailableExits","flexible-exits":"available_exits.FlexibleExits",
          "exact-approaches":"precision_approaches.ExactApproaches","repeated-approaches":"precision_approaches.RepeatedApproaches"}
SETS = {"I": ROOT/"results/ARC-dot/source/ARC-stress-100/maps.jsonl",
        "II": ROOT/"results/E-0007/fresh.jsonl", "III": ROOT/"results/E-0009/fresh.jsonl",
        "val": ROOT/"maze/datasets/val.jsonl"}
MOVES = tuple(Move)


def cases():
    for split, path in SETS.items():
        if split == "val":
            for c in load_split("val"):
                yield split, c.name, str(c.stage), split, c.level
        else:
            for line in path.read_bytes().splitlines():
                r = json.loads(line)
                level = Level.parse("\n".join([r["name"], f'{r["fuel"]} {r["health"]}', *r["grid"]]))
                yield split, r["name"], r["tier"], r["family"], level


def clear_episode_caches():
    # Геометрия из предыдущего эпизода больше не нужна. Справочники,
    # зависящие только от размеров или фиксированного образца, остаются.
    shared={"geometry","lattice","edges","landscape","reference_roads"}
    for module_name,module in tuple(sys.modules.items()):
        if module_name.startswith("maze.") and module is not None:
            for key,fn in tuple(vars(module).items()):
                if key not in shared and getattr(fn,"__module__",None)==module_name and hasattr(fn,"cache_clear"):
                    fn.cache_clear()


def controller(agent):
    factory = AGENTS[agent]
    if isinstance(factory, str):
        module, name_in_module = factory.split(".")
        factory = getattr(importlib.import_module("maze."+module),name_in_module)
    return factory()


def run_group(agents, case):
    """У каждого робота своя среда и память. Вычисление ходов идёт в одном потоке."""
    split, name, tier, family, level = case
    clear_episode_caches()
    episodes = {agent: (Environment(level), controller(agent), bytearray(), Counter())
                for agent in agents}
    while episodes:
        for agent, (env, bot, actions, modes) in tuple(episodes.items()):
            move = bot.choose(env.view)
            modes[bot.activity] += 1
            actions.append(48 + MOVES.index(move))
            env.step(move)
            if env.status != Status.RUNNING:
                yield agent, dict(split=split, map=name, tier=tier, family=family,
                                  status=env.status.name, steps=env.steps, refuels=env.refuels,
                                  repairs=env.repairs, actions=actions.decode("ascii"), modes=modes)
                del episodes[agent]


def run_case(job):
    agent, case = job
    return next(run_group([agent],case))[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("agent", choices=AGENTS)
    p.add_argument("output", type=Path)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--resume", action="store_true", help="дописать отсутствующие карты собственной серии")
    args = p.parse_args()
    if args.workers < 1:
        p.error("Нужен хотя бы один исполнитель")
    selected = list(cases())
    assert Counter(c[0] for c in selected) == {"I": 100, "II": 100, "III": 100, "val": 36}
    assert len(set(c[1] for c in selected)) == len(selected)
    hashes = {k: sha256(v.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for k,v in SETS.items()}
    print(args.agent, hashes, flush=True)
    if args.resume:
        if args.output.stem != args.agent:
            p.error("Для возобновления имя файла должно совпадать с агентом")
        finished = [json.loads(line) for line in args.output.read_bytes().splitlines()]
        done = {(r["split"],r["map"],r["tier"],r["family"]) for r in finished}
        assert len(done) == len(finished) and done <= {c[:4] for c in selected}
        assert all(r["steps"] == len(r["actions"]) and r["status"] in Status.__members__ for r in finished)
        selected = [c for c in selected if c[:4] not in done]
    def measure():
        jobs=((args.agent,case) for case in selected)
        if args.workers == 1:
            yield from map(run_case,jobs)
            return
        with ProcessPoolExecutor(args.workers) as pool:
            futures=[pool.submit(run_case,job) for job in jobs]
            for future in as_completed(futures):
                yield future.result()
    with args.output.open("a" if args.resume else "x", encoding="utf8", newline="\n") as output:
        for i, row in enumerate(measure(), 1):
            output.write(json.dumps(row, ensure_ascii=False)+"\n")
            output.flush()
            if i % 50 == 0:
                print(args.agent, i, flush=True)


if __name__ == "__main__":
    main()
