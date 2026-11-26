import unittest
from dataclasses import replace
from importlib.util import find_spec

from maze.environment import Environment
from maze.interface import Move
from maze.levels import Level


@unittest.skipUnless(find_spec("scipy") and find_spec("numpy"), "Требуются библиотеки эксперимента E-0010")
class NavigationGuards(unittest.TestCase):
    def test_side_step_can_use_one_spare_move_only_if_the_diagonal_is_open(self):
        from maze.side_step import lateral_turn
        grid = ("#####","#...#","#...#","#...#","#####")
        self.assertEqual(lateral_turn(grid,(2,2),Move.NORTH,Move.EAST),Move.SE)
        blocked = grid[:1]+("#..##",)+grid[2:]
        self.assertIsNone(lateral_turn(blocked,(2,2),Move.NORTH,Move.EAST))
        self.assertIsNone(lateral_turn(grid,(2,2),Move.WEST,Move.EAST))

    def test_shortest_turn_requires_the_opposite_corner_to_be_closed(self):
        from maze.path_particles import implied_walls
        # Путь 0 → 1 → 4 укоротится до 0 → 4, если откроется клетка 3.
        visited = (1<<0)|(1<<1)
        opened = visited|(1<<4)
        self.assertEqual(implied_walls(4,1,visited,opened,3,3),1<<3)
        self.assertIsNone(implied_walls(4,1,visited,opened|(1<<3),3,3))

    def test_road_exchange_keeps_endpoints_connected(self):
        from maze.exchanged_roads import cycle_path
        neighbors = [[(1,0),(2,3)],[(0,0),(2,1)],[(1,1),(3,2),(0,3)],[(2,2)]]
        original = [True,True,True,False]
        self.assertEqual(cycle_path(0,2,neighbors,original),[1,0])
        exchanged = [False,True,True,True]
        self.assertEqual(cycle_path(0,1,neighbors,exchanged),[1,3])

    def test_analogue_needs_both_stations_in_the_same_placement(self):
        import numpy as np
        from maze.reference_shells import connected_stops
        fields = np.abs(np.arange(8)[:,None] - np.arange(8))
        # У второго положения нет ни одной совместимой мастерской.
        result = connected_stops(fields,np.array([0,0,1,2]),np.array([1,6,3,5]),
                                 np.array([0,0,2]),np.array([3,7,1]),1)
        self.assertEqual(result.tolist(),[False,True,False,False])

    def test_rotating_entire_window_preserves_predicted_closure(self):
        import numpy as np
        from maze.window_patterns import predicted_closure
        grid = ("#######","#..?.?#","##.####","#..F..#","#######")
        mirror = tuple(row[::-1] for row in grid)
        left = predicted_closure(grid,(2,1),(5,3))
        right = predicted_closure(mirror,(4,1),(1,3))
        self.assertTrue(np.isclose(left,right))

    def test_second_window_can_require_a_known_refuel(self):
        from maze.serviced_windows import backup_stocks
        memory = ("#######","#..F..#","#######")
        point = (5,1)
        options = backup_stocks(memory,(1,1),2,12,(point,))
        self.assertIn((98,8),options[point])
        self.assertNotIn(point,backup_stocks(memory,(1,1),1,12,(point,)))

    def test_shared_road_reduces_station_to_goal_length(self):
        from maze.connected_excursions import connection, connected_trips
        self.assertEqual(connection([7,4,3,2],[6,5,4,3,2],0),3)
        self.assertEqual(connection([7,4,3,2],[6,5,1],7),12)
        memory = ("#########", "#..??F?B#", "#########")
        trips = connected_trips(memory,(7,1),((2,1),),(1,1),(4,None))
        self.assertEqual(len(trips),32)
        self.assertTrue(all(t == ((2,1),5,(((2,1),3,"F",2),)) for t in trips))
        self.assertFalse(connected_trips(memory,(7,1),((2,1),),(1,1),(5,None)))

    def test_fuel_does_not_restore_health_along_an_inferred_route(self):
        from maze.onward_services import trip_fits
        self.assertTrue(trip_fits(7,[(3,"F")],(3,7),0,(3,None)))
        self.assertFalse(trip_fits(7,[(3,"F")],(3,6),0,(3,None)))
        self.assertFalse(trip_fits(7,[(3,"F")],(3,7),0,(4,None)))

    def test_false_shortcut_can_rule_out_a_move_but_unknown_budget_keeps_it(self):
        from maze.route_shield import still_possible, viable_in_layout
        base = ("###########", "#B.F......#", "##?######.#", "##..A.....#", "###########")
        level = Level.parse("обход\n13 60\n" + "\n".join(r.replace("?", "#") for r in base))
        memory = tuple(r.replace("A", ".") for r in base)
        view = replace(Environment(level).view, memory=memory)
        self.assertEqual(view.fuel_distance, 13)
        args = view, level.goal, (view.fuel_distance,None), None, (None,None)
        mask = sum(1<<i for i,c in enumerate("".join(memory)) if c != "#")
        self.assertTrue(viable_in_layout(view, level.goal, mask, Move.WEST, (None,None)))
        self.assertTrue(still_possible(*args, Move.WEST, limit=1))
        self.assertFalse(still_possible(*args, Move.WEST))
        self.assertTrue(still_possible(*args, Move.EAST))

    def test_dead_end_allowance_stops_at_a_service_and_an_unseen_exit(self):
        from maze.branch_allowance import branch_lengths
        base = ("#######", "###.###", "###.###", "#..B..#", "#######")
        # Три рукава длины 2 у развилки. Один закончился заправкой.
        grid = base[:3] + ("#F....#",base[4])
        self.assertEqual(dict(branch_lengths(grid, (0,0))), {0:1,4:2})
        grid = grid[:1] + ("###?###",) + grid[2:]
        self.assertEqual(dict(branch_lengths(grid, (0,0))), {0:2,4:1})

    def test_two_sensor_probe_distinguishes_remaining_choices(self):
        import numpy as np
        from maze.probe_pairs import two_attempts
        # Строка задаёт направление к первой станции, столбец — ко второй.
        pairs = np.array([[.2,0.,0.], [.4,0.,0.], [0.,.4,0.]])
        score, gain = two_attempts(pairs, (0,1,2), 0)
        self.assertAlmostEqual(score, 1.)
        self.assertAlmostEqual(gain, .4)

    def test_observed_open_edge_disproves_enclosing_wall(self):
        from maze.perimeter_paths import interior_graph, wall_probability
        from scipy.sparse.csgraph import dijkstra
        memory = ("?#???", ".....", "?????")
        self.assertEqual(wall_probability(memory, (2, 1)), 0.)
        memory = ("?#???", "?...?", "?????")
        self.assertGreater(wall_probability(memory, (2, 1)), .5)
        self.assertEqual(wall_probability(memory, (2, 0)), 0.)
        distances = dijkstra(interior_graph(memory, (2, 1)), indices=7, unweighted=True)
        self.assertEqual(distances[8], 1)
        self.assertEqual(distances[2], float("inf"))

    def test_window_blockage_depends_on_visible_diagonal_corners(self):
        from maze.second_window import closed_probability
        memory = ("#####", "##?##", "##.##", "#####")
        self.assertEqual(closed_probability(memory, (2, 2), (2, 1)), 0.)
        self.assertGreater(closed_probability(memory, (2, 2), (1, 1)), 0.)
        diagonal_only = ("#####", "#?###", "##.##", "#####")
        self.assertEqual(closed_probability(diagonal_only, (2, 2), (1, 1)), 1.)

    def test_conditional_exit_length_on_a_corridor_does_not_depend_on_hit_frequency(self):
        from maze.conditional_exits import conditional_lengths, hit_law
        memory = (".......",)
        points = ((0, 0), (6, 0))
        _, probabilities, source, _ = hit_law(memory,(1, 0),points)
        self.assertAlmostEqual(probabilities[source,0], 5/6)
        self.assertAlmostEqual(probabilities[source,1], 1/6)
        self.assertEqual(conditional_lengths(memory,(1,0),points,0), (1,)*32)
        self.assertEqual(conditional_lengths(memory,(1,0),points,1), (5,)*32)

    def test_compatible_density_interval_matches_exhaustive_physical_distances(self):
        from maze.compatible_loops import compatible_interval
        from maze.grid import wave
        from maze.route_witness import geometry
        width = 5
        walkway = (0, 1, 2, 3, 4, 5, 9, 10, 14)
        mask = sum(1 << p for p in walkway)
        masks = [mask]
        for p in (6, 8, 7, 11, 13, 12):
            masks.append(masks[-1] | (1 << p))
        actual = []
        for opened in masks:
            grid = tuple("".join("." if opened & (1 << (y*width+x)) else "#"
                                 for x in range(width)) for y in range(3))
            lengths = {q: d for q, d, _ in wave(grid, [(0, 2)])}
            actual.append((lengths[(4, 2)], lengths[(4, 1)]))
        for d1, d2 in set(actual) | {(0, 0)}:
            constraints = [(1 << 14, [(1 << 10, d1)]), (1 << 9, [(1 << 10, d2)])]
            start, end = compatible_interval(masks, geometry(width, 3)[0], constraints)
            self.assertEqual(list(range(start, end)), [i for i, lengths in enumerate(actual) if lengths == (d1, d2)])

    def test_joint_edge_requires_each_distinct_corner_once(self):
        import numpy as np
        from maze.edge_visibility import joint_chances
        from maze.road_hypotheses import lattice
        edges = lattice(2, 2)
        weights = joint_chances(np.full(4, .5), edges)
        diagonal = (edges[2] != edges[0]) & (edges[3] != edges[0])
        np.testing.assert_allclose(weights[diagonal], .5 ** 4)
        np.testing.assert_allclose(weights[~diagonal], .5 ** 2)

    def test_first_hits_match_physical_wave_stopped_at_each_window(self):
        from maze.feasible_windows import first_hits
        from maze.grid import wave
        from maze.route_witness import geometry
        for grid in (("....", ".#..", "..#.", "...."), ("..#.", "...#", "#...", "...#")):
            width = len(grid[0])
            encode = lambda p: p[1] * width + p[0]
            opened = sum(1 << i for i, t in enumerate("".join(grid)) if t == ".")
            stops = frozenset(((1, 0), (2, 1), (2, 3)))
            target = sum(1 << encode(p) for p in stops)
            for origin in ((0, 0), (0, 3)):
                expected = {encode(p): d for p, d, _ in wave(grid, [origin], stop_at=stops) if p in stops}
                self.assertEqual(first_hits(opened, encode(origin), target, geometry(width, len(grid))[0], 100), expected)

    def test_contact_uses_exact_remaining_length_after_the_known_approach(self):
        from maze.service_contact import first_contact_votes
        memory = ("#########", "#R?...?F#", "#########")
        # Из центра до правого края известной дороги один ход, ещё два до F.
        args = memory, (4, 1), ((7, 1),)
        self.assertGreater(first_contact_votes(*args, 3, None, None)[Move.EAST], 0)
        self.assertFalse(any(first_contact_votes(*args, 2, None, None).values()))
        # Если предыдущая клетка правее, дорожная гипотеза даёт от неё ровно два хода.
        self.assertFalse(any(first_contact_votes(*args, 3, 4, (5, 1)).values()))

    def test_remote_service_requires_both_a_real_reading_and_remaining_resource(self):
        from maze.service_exit_paths import interval_mass
        curve = (0., 0., .25, .5, .75)
        self.assertEqual(interval_mass(curve, 2, 2), .25)
        self.assertEqual(interval_mass(curve, 3, 2), 0.)
        self.assertEqual(interval_mass(curve, 0, 4), .75)
        self.assertEqual(interval_mass(curve, 3, 4), .5)

    def test_smallest_detour_requires_a_unique_compatible_direction(self):
        from maze.proven_approach import smallest_detour
        lengths = {Move.WEST: 8, Move.EAST: 5, Move.NORTH: 11}
        self.assertEqual(smallest_detour(lengths, set(), {Move.WEST, Move.EAST}), Move.WEST)
        self.assertIsNone(smallest_detour(lengths, {Move.EAST}, set(lengths)))
        self.assertIsNone(smallest_detour({Move.WEST: 8, Move.EAST: 8}, set(), set()))
        # Второй датчик показывает путь вдвое длиннее нижней оценки.
        # При показании 16 первый ход с оценкой 8 объясняет этот обход.
        self.assertEqual(smallest_detour(lengths, set(), set(lengths), (16, 10, 5)), Move.WEST)

    def test_first_window_keeps_the_fuel_detour_and_stops_at_first_observation(self):
        from maze.first_window import window_arrivals
        level = Level.parse("Выезд с возможной дозаправкой\n4 20\nF.A.....B")
        view = replace(Environment(level).view, memory=("F.....?.B",))
        arrivals = {(a.point,a.length,a.fuel,a.health,a.first) for a in window_arrivals(view)}
        self.assertEqual(arrivals, {((5,0),3,1,17,Move.EAST), ((5,0),7,95,13,Move.WEST)})

    def test_onward_support_includes_direct_finish_from_first_station(self):
        from maze.shortcut_reserve import onward_support
        level = Level.parse("После ремонта прямо к финишу\n6 4\nB..R..A..F")
        view = replace(Environment(level).view, memory=level.grid)
        # До R и оттуда до B по три хода. Путь между F и R длиннее
        # оставшегося второго запаса после любого из первых заездов.
        self.assertEqual(onward_support(view,(None,None)),1)
        self.assertEqual(onward_support(replace(view,fuel=5),(None,None)),0)

    def test_first_stop_can_be_the_farther_workshop(self):
        from maze.known_landing import FirstLanding, first_legs, known_first_stops, supported_stop
        level = Level.parse("Две мастерские\n6 4\nB..R..A.R......F")
        view = replace(Environment(level).view, memory=level.grid)
        sites = [((3,0),"R"), ((8,0),"R"), ((15,0),"F")]
        # Подъезд к F проходит через ближнюю мастерскую. Для финиша после
        # ремонта на дальнем пути хватает ровно трёх оставшихся единиц топлива.
        legs = first_legs(view.memory,view.position,level.goal)
        self.assertEqual({(p,d) for p,d,_ in legs}, {((3,0),3), ((8,0),2)})
        choices = known_first_stops(view,sites,level.goal)
        best = supported_stop(view,level.goal,choices)
        self.assertEqual((best.point,best.move), ((3,0),Move.WEST))
        self.assertEqual(FirstLanding().ongoing_stop(view,level.goal,sites[:2]),best)

    def test_translated_routes_obey_the_actual_map_boundary(self):
        import numpy as np
        from maze.grid import wave
        from maze.translated_paths import landscape, routes_to
        roads,depth,jump,boxes=landscape(4,1)
        width=roads.shape[1]
        checked=set()
        crops=[(0,0,9,9),(1,1,5,5),(3,2,4,5)]
        crops.extend((0,y,9,1) for y in range(1,8))
        for x,y,w,h in crops:
            grid=tuple("".join("." if c else "#" for c in r) for r in roads[y:y+h,x:x+w])
            points=[(i,j) for j,row in enumerate(grid) for i,v in enumerate(row) if v=="."]
            for start in points:
                paths={p:(d,m) for p,d,m in wave(grid,[start])}
                sources=np.full(len(points),(y+start[1])*width+x+start[0])
                targets=np.array([(y+b)*width+x+a for a,b in points])
                crop=tuple(np.full(len(points),v) for v in (x,y,x+w,y+h))
                lengths,steps=routes_to(sources,targets,depth,jump,boxes,crop)
                for i,p in enumerate(points):
                    d,m=paths.get(p,(10**6,None))
                    self.assertEqual(lengths[i],d)
                    checked.add(d==10**6)
                    if m:
                        self.assertEqual(steps[i],m.value[0]+width*m.value[1])
        self.assertEqual(checked,{False,True})

    def test_required_stock_for_a_relay_and_last_unit_at_goal(self):
        from maze.conditional_mission import required_after_stop
        from maze.stretch_sortie import tolerates
        sites = [("R",0),("F",55),("B",105)]
        fields = [[abs(i-q) for i in range(106)] for _,q in sites]
        import numpy as np
        need = required_after_stop(sites, np.array(fields))
        self.assertEqual(need[1], 50)
        self.assertEqual(need[0], float("inf"))
        self.assertTrue(tolerates((3,5), (3,6), ((0,2),(2,0)), ("F","B"), 1.))
        self.assertFalse(tolerates((3,4), (3,6), ((0,2),(2,0)), ("F","B"), 1.))

    def test_relay_station_rejoins_a_renewable_pair(self):
        from maze.renewable_routes import renewable_routes
        from maze.excursion_automaton import shelters
        memory = ("#" * 95, "#R" + "." * 39 + "F" + "." * 51 + "R#", "#" * 95)
        stops = (((1, 1), "R"), ((41, 1), "F"), ((93, 1), "R"))
        self.assertNotIn((93, 1), shelters(memory, stops))
        self.assertEqual(renewable_routes(memory, stops)[(93, 1)], ("R", 52))

    def test_flow_and_conditional_readings_on_a_known_corridor(self):
        from maze.passage_flow import flows
        from maze.implied_turn import support
        memory = ("#######", "#F...R#", "#######")
        sites = (((1, 1),), ((5, 1),))
        origin, target, goal = (3, 1), (2, 1), (4, 1)
        flow = flows(memory, origin, goal, sites[0], 2)
        self.assertGreater(flow[Move.WEST], 0.)
        self.assertEqual(flow.get(Move.EAST, 0.), 0.)
        self.assertEqual(support(memory, origin, target, goal, (2, 2), (1, 3), sites), 1.)
        self.assertEqual(support(memory, origin, target, goal, (2, 2), (1, 1), sites), 0.)

    def test_sensor_determines_hidden_gate_with_two_different_trip_lengths(self):
        from maze.deduced_roads import deductions
        memory=("#######","#...?F#","#.###.#","#R....#","#######")
        sites=(((5,1),),((1,3),))
        for length,terrain in ((4,"."),(8,"#")):
            inferred=deductions(memory,(1,1),(3,3),(length,2),None,(None,None),sites)
            self.assertEqual(inferred[1][4],terrain)
            self.assertEqual(memory[1][4],"?")

    def test_rotated_rectangular_memory_keeps_each_known_cell(self):
        from maze.symmetric_roads import oriented_grids
        memory=("abc","def")
        frames=oriented_grids(memory)
        self.assertEqual(len({grid for grid,*_ in frames}),8)
        for grid,(ox,oy),(ex,ey),(sx,sy) in frames:
            for y,row in enumerate(memory):
                for x,symbol in enumerate(row):
                    self.assertEqual(grid[oy+x*ey+y*sy][ox+x*ex+y*sx],symbol)

    def test_projected_trip_reaches_a_station_on_its_last_unit(self):
        from math import inf
        from maze.last_mission import trip_length
        sites=[("F",1),("R",3),("B",5)]
        fields=[[abs(i-p) for i in range(6)] for _,p in sites]
        self.assertEqual(trip_length(0,(1,3),sites,fields,{}),5)
        self.assertEqual(trip_length(0,(1,2),sites,fields,{}),inf)
        self.assertEqual(trip_length(0,(1,3),sites,fields,{"F":2}),inf)

    def test_last_unit_preserves_direct_finish(self):
        from maze.last_unit import surviving_moves
        level=Level.parse("проверка\n2 20\n#######\n#F.A.B#\n###.###\n#..R..#\n#######")
        view=replace(Environment(level).view,memory=level.grid)
        self.assertEqual(view.fuel,view.fuel_distance)
        self.assertEqual(surviving_moves(view,(None,None),Move.EAST),Move.EAST)
        self.assertEqual(surviving_moves(view,(None,None),Move.SOUTH),Move.WEST)

    def test_one_probe_spends_both_resources(self):
        from maze.fork_budget import chance
        self.assertEqual(chance(1,1,0,0),0.25)
        self.assertEqual(chance(1,1,0,2),0.5)
        self.assertEqual(chance(1,1,2,4),1.)

    def test_visible_fork_reduces_chance_of_reaching_service_without_a_probe(self):
        from maze.leg_survival import continuation_chance
        # Дороги 0-1-2-3 до станции и ложный выход 1-4.
        roads = [{1}, {0, 2, 4}, {1, 3}, {2}, {1}]
        field = [3, 2, 1, 0, 3]
        unknown = [0] * len(roads)
        self.assertEqual(continuation_chance(roads.__getitem__, field, unknown)(0, 1), .5)
        self.assertEqual(continuation_chance(roads.__getitem__, field, field)(0, 1), 1.)

    def test_first_new_wall_can_close_a_gate_or_leave_a_detour(self):
        from maze.first_window import length_after_wall
        sealed = ("#########", "#..??..B#", "#########", "#########")
        open_side = (sealed[0], sealed[1], "#.......#", sealed[3])
        self.assertEqual(length_after_wall(sealed, (7, 1), (2, 1)), float("inf"))
        self.assertEqual(length_after_wall(open_side, (7, 1), (2, 1)), 6)

    def test_service_bound_holds_for_every_completion_of_hidden_cells(self):
        from itertools import product
        from maze.covered_approach import arrival_bounds
        from maze.grid import destination
        from maze.proven_approach import service_field
        memory = ("#########", "#..??..F#", "#..###..#", "#..##...#", "#.......#", "#..??...#", "#########")
        start, sites = (1, 2), ((7, 1),)
        known = service_field(memory, sites)
        improved = False
        for fill in product(".#", repeat=4):
            values = iter(fill)
            grid = tuple("".join(next(values) if c == "?" else c for c in row) for row in memory)
            true = service_field(grid, sites)
            bounds = arrival_bounds(memory, start, true[start], None, None, sites)
            for move, worst in bounds.items():
                q = destination(memory, start, move)
                self.assertGreaterEqual(worst, 1 + true[q])
                improved |= worst < 1 + known[q]
        self.assertTrue(improved)

    def test_reverse_paths_stop_at_the_first_window(self):
        from maze.exit_flow import corridor_probabilities, eventual_exits, exit_probabilities
        memory = ("#########", "#...B...#", "#########")
        p = exit_probabilities(memory, (4, 1), ((1, 1), (2, 1), (6, 1)))
        self.assertEqual(p[(1, 1)][100], 0.)
        self.assertEqual(p[(2, 1)][1], 0.)
        self.assertAlmostEqual(p[(2, 1)][2], .25)
        self.assertAlmostEqual(p[(2, 1)][100], p[(6, 1)][100])
        straight = corridor_probabilities(memory, (4, 1), ((1, 1), (2, 1), (6, 1)))
        self.assertEqual(straight[(1, 1)][100], 0.)
        self.assertAlmostEqual(straight[(2, 1)][2], .5)
        self.assertAlmostEqual(straight[(6, 1)][100], .5)
        eventual = eventual_exits(memory, (4, 1), ((1, 1), (2, 1), (6, 1)))
        self.assertEqual(eventual[(2, 1)][1], 0.)
        self.assertAlmostEqual(eventual[(2, 1)][2], .5)
        self.assertAlmostEqual(eventual[(6, 1)][2], .5)
        from maze.simple_exit_paths import simple_exit_probabilities
        simple = simple_exit_probabilities(memory, (4, 1), ((1, 1), (2, 1), (6, 1)))
        self.assertAlmostEqual(simple[(2, 1)][2], .5)
        self.assertAlmostEqual(simple[(6, 1)][2], .5)

    def test_observing_a_diagonal_gate_requires_its_unknown_corner(self):
        from maze.covered_approach import visible_gates
        memory = ("#####", "#..?#", "#..?#", "#?..#", "#####")
        exits = [((2, 2), (3, 1), 1, (Move.NORTH,)), ((2, 3), (1, 3), 2, (Move.WEST,))]
        self.assertEqual(visible_gates(memory, exits, {(3, 1)}), 0.)
        self.assertEqual(visible_gates(memory, exits, {(3, 1), (3, 2)}), 1.)

    def test_resampled_roads_preserve_mass_and_match_a_measured_corridor(self):
        from random import Random
        from maze.path_particles import path_fractions, thin
        paths = {i: i + 1 for i in range(20)}
        kept = thin(paths, 4, Random(42))
        self.assertLessEqual(len(kept), 4)
        self.assertAlmostEqual(sum(kept.values()), sum(paths.values()))
        level = Level.parse("прямая\n2 10\n########\n#F.A.RB#\n########")
        view = replace(Environment(level).view, memory=level.grid)
        fractions = path_fractions(view, (2, 2), (None, None), None, (((1, 1),), ((5, 1),)),
                                   (Move.WEST, Move.EAST), 0)
        self.assertEqual(fractions, {Move.WEST: 1.})

    def test_visible_cell_prediction_distinguishes_dividers_from_a_periodic_grid(self):
        from maze.room_roads import room_predicts_memory
        rooms = tuple("".join("#" if (y in (2,7,12) and x not in (3,9)
                                      or x in (4,11) and y not in (5,9)) else "." for x in range(16))
                      for y in range(16))
        periodic = tuple("".join("#" if x%3 == 2 and y%3 == 2 else "." for x in range(16)) for y in range(16))
        for grid, expected in ((rooms, True), (periodic, False)):
            self.assertEqual(room_predicts_memory(grid), expected)
            self.assertEqual(room_predicts_memory(tuple("".join(row) for row in zip(*grid[::-1]))), expected)

    def test_hypothetical_corner_obeys_physical_movement(self):
        from maze.route_witness import agrees,geometry
        shifts,_=geometry(3,3)
        floor=(1<<0)|(1<<4)|(1<<1)
        self.assertTrue(agrees(floor,shifts,0,(1<<4,1<<1),(2,1),{0,1}))
        self.assertFalse(agrees(floor,shifts,0,(1<<4,1<<1),(1,1),{0,1}))
        self.assertTrue(agrees(floor|(1<<3),shifts,0,(1<<4,1<<1),(1,1),{0,1}))
