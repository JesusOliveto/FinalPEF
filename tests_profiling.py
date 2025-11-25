"""Tests y utilidades de profiling para el motor de ruteo.

Este módulo extrae todo el código de testing y profiling que antes vivía
en `logica.py`, para mantener el motor más limpio.
"""

from __future__ import annotations

import sys
import unittest
from typing import Tuple

from logica import (
	Algorithm,
	Graph,
	HistoricalTrafficModel,
	AStarRouter,
	DijkstraRouter,
	BFSRouter,
	PairwiseDistanceService,
	RouteCache,
	HeldKarpExact,
	HeuristicRoute,
	RouteSplicer,
	RouteRequest,
	RouteMode,
	run_profiled,
	profile_call,
)


# ==========================================================
# Helpers de construcción de servicio
# ==========================================================

def build_service(driver_max_kmh: float = 40.0) -> Tuple[Graph, HistoricalTrafficModel, "RoutingService"]:
	"""Helper para construir el servicio de ruteo listo para usar.

	Se deja aquí para que los tests y los perfiles no ensucien `logica.py`.
	"""

	from logica import RoutingService  # import diferido para evitar ciclos en tiempo de import

	g = Graph.build_jesus_maria_hardcoded()
	t = HistoricalTrafficModel(driver_max_kmh=driver_max_kmh)
	pair = PairwiseDistanceService(AStarRouter(driver_max_kmh), DijkstraRouter(), RouteCache())
	svc = RoutingService(g, t, pair, HeldKarpExact(), HeuristicRoute(), RouteSplicer())
	return g, t, svc


# ==========================================================
# Suite de tests (extraída de logica.py)
# ==========================================================


class TestJM(unittest.TestCase):
	def test_graph_non_empty(self):
		g = Graph.build_jesus_maria_hardcoded()
		self.assertGreater(len(list(g.iter_nodes())), 0)
		self.assertGreater(len(list(g.iter_edges())), 0)

	def test_routing_multi(self):
		g = Graph.build_jesus_maria_hardcoded()
		t = HistoricalTrafficModel()
		pair = PairwiseDistanceService(AStarRouter(), DijkstraRouter(), RouteCache())
		from logica import RoutingService  # import local para evitar ciclos

		svc = RoutingService(g, t, pair, HeldKarpExact(), HeuristicRoute(), RouteSplicer())
		nodes = list(g.iter_nodes())
		req = RouteRequest(
			origin=nodes[0].id,
			destinations=[nodes[5].id, nodes[20].id, nodes[-1].id],
			hour=8,
			mode=RouteMode.VISIT_ALL_OPEN,
			algorithm=Algorithm.ASTAR,
		)
		res = svc.route(req)
		self.assertGreater(len(res.legs), 0)


class TestEngine(unittest.TestCase):
	def setUp(self):
		self.graph = Graph.build_jesus_maria_hardcoded()
		self.traffic = HistoricalTrafficModel()
		self.nodes = list(self.graph.iter_nodes())

	def test_dijkstra_astar_same_cost(self):
		"""Comprueba que Dijkstra y A* coinciden en costo cuando hay ruta.

		Busca un destino alcanzable desde el origen en lugar de asumir que
		`nodes[-1]` siempre está conectado.
		"""
		src = self.nodes[0].id
		router_d = DijkstraRouter()
		router_a = AStarRouter()
		# buscamos el primer destino que tenga path con ambos algoritmos
		for candidate in self.nodes[1:]:
			dst = candidate.id
			leg_d, _ = router_d.route(self.graph, src, dst, hour=8, traffic=self.traffic)
			leg_a, _ = router_a.route(self.graph, src, dst, hour=8, traffic=self.traffic)
			if leg_d.path and leg_a.path:
				self.assertAlmostEqual(leg_d.seconds, leg_a.seconds, places=6)
				return
		self.fail("No se encontró ningún par (src, dst) alcanzable para comparar Dijkstra vs A*.")

	def test_bfs_route_valid(self):
		src, dst = self.nodes[0].id, self.nodes[len(self.nodes) // 2].id
		b_leg, _ = BFSRouter().route(
			self.graph, src, dst, hour=8, traffic=self.traffic
		)
		self.assertTrue(b_leg.path)
		self.assertTrue(b_leg.seconds > 0)

	def test_pairwise_matrix_shapes(self):
		wps = [self.nodes[i].id for i in [0, 3, 5, 8]]
		pair = PairwiseDistanceService(
			AStarRouter(), DijkstraRouter(), RouteCache(), max_workers=2
		)
		m, pm = pair.compute_matrix(
			self.graph,
			wps,
			hour=8,
			algorithm=Algorithm.DIJKSTRA,
			traffic=self.traffic,
		)
		n = len(wps)
		self.assertEqual(len(m), n)
		self.assertEqual(len(m[0]), n)
		for i in range(n):
			self.assertEqual(m[i][i], 0.0)
			for j in range(n):
				self.assertIn((i, j), pm)

	def test_circuit_adds_return_leg(self):
		pair = PairwiseDistanceService(AStarRouter(), DijkstraRouter(), RouteCache())
		from logica import RoutingService  # import local para evitar ciclos

		svc = RoutingService(
			self.graph, self.traffic, pair, HeldKarpExact(), HeuristicRoute(), RouteSplicer()
		)
		wps = [self.nodes[i].id for i in [0, 3, 5, 8]]
		req = RouteRequest(
			origin=wps[0],
			destinations=wps[1:],
			hour=8,
			mode=RouteMode.VISIT_ALL_CIRCUIT,
			algorithm=Algorithm.ASTAR,
		)
		res = svc.route(req)
		self.assertGreaterEqual(len(res.legs), 2)
		self.assertEqual(res.visit_order[0], res.visit_order[-1])
		# coherente con lo que ve el front: el último destino vuelve al origen
		self.assertIn(res.visit_order[0], req.destinations + [req.origin])

	def test_run_profiled_and_profile_call(self):
		def _tiny():
			sum(i * i for i in range(1000))

		report = run_profiled(_tiny)
		self.assertIsInstance(report, str)
		self.assertGreater(len(report), 0)

		(result, rep2) = profile_call(lambda x: x + 1, 41)
		self.assertEqual(result, 42)
		self.assertIsInstance(rep2, str)
		self.assertGreater(len(rep2), 0)

	def test_dijkstra_memoization(self):
		router = DijkstraRouter()
		src, dst, hour = self.nodes[0].id, self.nodes[-1].id, 8
		out1 = router.route(self.graph, src, dst, hour=hour, traffic=self.traffic)
		memo_key = (id(self.graph), id(self.traffic), src, dst, hour)
		self.assertIn(memo_key, router._memo)
		out2 = router.route(self.graph, src, dst, hour=hour, traffic=self.traffic)
		self.assertIs(out1, out2)


# ==========================================================
# CLI para ejecutar tests y profiling desde este módulo
# ==========================================================


def cmd_smoke(hour: int = 8, driver_max_kmh: float = 40.0) -> None:
	"""Smoke test end-to-end similar al front.

	- Construye el servicio como el front.
	- Busca el primer destino alcanzable desde un origen fijo.
	- Imprime tiempos y distancia en un formato legible.
	"""
	from logica import RoutingService  # import local

	g, _t, svc = build_service(driver_max_kmh)
	nodes = list(g.iter_nodes())
	if len(nodes) < 2:
		print("Smoke: grafo sin nodos suficientes")
		return

	src = nodes[0].id
	leg = None
	for candidate in nodes[1:]:
		dst = candidate.id
		leg = svc.route_single(src, dst, hour=hour)
		if leg.path and leg.seconds < float("inf"):
			break
	else:
		print("Smoke: no se encontró ningún destino alcanzable desde el origen")
		return

	print(
		f"Smoke OK: {src}→{dst} · "
		f"{leg.seconds/60:.1f} min · {leg.distance_m/1000:.2f} km"
	)


def cmd_tests(verbosity: int = 1) -> None:
	"""Ejecuta la suite de tests definida en este módulo.

	Args:
		verbosity: Nivel de verbosidad de `unittest` (1=normal, 2=detallado).

	Returns:
		None. Imprime el resultado de los tests en la salida estándar.
	"""
	argv = [sys.argv[0]]
	unittest.main(module=__name__, argv=argv, exit=False, verbosity=verbosity)


def cmd_profile(algo: str, mode: str, hour: int, n: int, driver_max_kmh: float) -> None:
	"""Ejecuta un perfilado de una ruta multi-destino similar al front.

	Args:
		algo: Algoritmo base a usar ("astar", "dijkstra" o "bfs").
		mode: Modo de ruta (valores de `RouteMode`, p.ej. "visit_all_open").
		hour: Hora del día (0..23) para el modelo de tráfico.
		n: Cantidad de waypoints a incluir (se recorta al tamaño del grafo).
		driver_max_kmh: Velocidad máxima del conductor en km/h.

	Returns:
		None. Imprime métricas de la ruta y el resumen de pstats en stdout.
	"""
	from logica import RoutingService  # import local

	g, t, svc = build_service(driver_max_kmh)
	nodes = list(g.iter_nodes())
	n = max(2, min(n, len(nodes)))
	waypoints = [nodes[i].id for i in range(n)]
	req = RouteRequest(
		origin=waypoints[0],
		destinations=waypoints[1:],
		hour=hour,
		mode=RouteMode(mode),
		algorithm=Algorithm(algo),
	)
	(res, report) = profile_call(svc.route, req)
	print("=== Profiling resultado ===")
	print(
		f"Total: {res.total_seconds:.1f}s · {res.total_distance_m/1000:.2f} km · algoritmo: {res.algorithm_summary}"
	)
	print("\n=== pstats (top 20 por cumtime) ===")
	print(report)


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Tests y profiling del motor de ruteo")
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument("--smoke", action="store_true", help="Ejecuta el smoke test (un tramo)")
	group.add_argument("--tests", action="store_true", help="Ejecuta los tests unitarios de este módulo")
	group.add_argument("--profile", action="store_true", help="Perfilado de una ruta multi-destino")

	parser.add_argument("--hour", type=int, default=8, help="Hora del día (0..23)")
	parser.add_argument("--driver-max-kmh", type=float, default=40.0, help="Velocidad máxima del conductor (km/h)")
	parser.add_argument("--algo", choices=[a.value for a in Algorithm], default=Algorithm.ASTAR.value, help="Algoritmo base (para --profile)")
	parser.add_argument("--mode", choices=[m.value for m in RouteMode], default=RouteMode.VISIT_ALL_OPEN.value, help="Modo de ruta (para --profile)")
	parser.add_argument("--n", type=int, default=4, help="Cantidad de waypoints (para --profile)")
	parser.add_argument("--verbosity", type=int, default=1, help="Verbosity de tests (para --tests)")

	args = parser.parse_args()
	if args.smoke:
		cmd_smoke(hour=args.hour, driver_max_kmh=args.driver_max_kmh)
	elif args.tests:
		cmd_tests(verbosity=args.verbosity)
	elif args.profile:
		cmd_profile(args.algo, args.mode, args.hour, args.n, args.driver_max_kmh)

