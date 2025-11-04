# -*- coding: utf-8 -*-
"""
logica.py — Motor de ruteo para “Pueblito: Rutas Inteligentes”
Foco: creación de rutas en una red hardcodeada que imita Jesús María (Córdoba).

Componentes:
- Graph.build_jesus_maria_hardcoded(): 3 parches de grilla + diagonal primaria (RN-9 aprox.)
- Algoritmos: Dijkstra y A* (heurística admisible). BFS opcional para depuración.
- Tráfico histórico por clase vial (PRIMARY / COLLECTOR / RESIDENTIAL).
- Cache LRU de tramos; matriz par-a-par concurrente con ThreadPoolExecutor.
- Solvers multi-stop: Held-Karp (DP) y heurístico (NN + 2-opt simplificado).
- Generador iter_path_edges para recorrer aristas de un camino.

NOTA: Se eliminaron explícitamente:
- SSSPMemo (no se usaba).
- La flag Edge.one_way (la direccionalidad surge de la existencia de la arista).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Iterable, Iterator, Optional, Any
from collections import defaultdict, OrderedDict
from math import radians, sin, cos, asin, sqrt
from concurrent.futures import ThreadPoolExecutor, as_completed
import heapq
import random

# ==========================================================
# Tipos y enums
# ==========================================================
class Algorithm(str, Enum):
    ASTAR = "astar"
    DIJKSTRA = "dijkstra"
    BFS = "bfs"  # útil para grafos no ponderados o depuración


class RouteMode(str, Enum):
    VISIT_ALL_OPEN = "visit_all_open"
    VISIT_ALL_CIRCUIT = "visit_all_circuit"
    FIXED_ORDER = "fixed_order"


class RoadClass(str, Enum):
    RESIDENTIAL = "residential"
    COLLECTOR = "collector"
    PRIMARY = "primary"


NodeId = int
Seconds = float
Meters = float
KmPerHour = float

# ==========================================================
# Grafo
# ==========================================================
@dataclass(frozen=True)
class Node:
    id: NodeId
    lat: float
    lon: float


@dataclass
class Edge:
    to: NodeId
    distance_m: Meters
    road_class: RoadClass
    freeflow_kmh: KmPerHour


class Graph:
    def __init__(self) -> None:
        self.nodes: Dict[NodeId, Node] = {}
        self.adj: Dict[NodeId, List[Edge]] = defaultdict(list)

    # ---- core
    def add_node(self, node_id: NodeId, lat: float, lon: float) -> None:
        self.nodes[node_id] = Node(node_id, lat, lon)

    def add_edge(
        self,
        u: NodeId,
        v: NodeId,
        distance_m: Meters,
        road_class: RoadClass = RoadClass.RESIDENTIAL,
        freeflow_kmh: KmPerHour = 40.0,
    ) -> None:
        self.adj[u].append(Edge(v, distance_m, road_class, freeflow_kmh))

    def add_bidirectional(
        self,
        u: NodeId,
        v: NodeId,
        distance_m: Meters,
        road_class: RoadClass,
        freeflow_kmh: KmPerHour,
    ) -> None:
        self.add_edge(u, v, distance_m, road_class, freeflow_kmh)
        self.add_edge(v, u, distance_m, road_class, freeflow_kmh)

    def neighbors(self, u: NodeId) -> Iterable[Edge]:
        return self.adj.get(u, [])

    def get_node(self, node_id: NodeId) -> Node:
        return self.nodes[node_id]

    def iter_nodes(self) -> Iterator[Node]:
        yield from self.nodes.values()

    def iter_edges(self) -> Iterator[Tuple[NodeId, Edge]]:
        for u, lst in self.adj.items():
            for e in lst:
                yield u, e

    # ---- utilidades
    @staticmethod
    def haversine_m(a: Node, b: Node) -> float:
        # distancia geodésica ~m
        R = 6371000.0
        lat1, lon1, lat2, lon2 = map(radians, [a.lat, a.lon, b.lat, b.lon])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        h = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        return 2 * R * asin(sqrt(h))

    # ---- Jesús María hardcodeado (simplificado y estable)
    @staticmethod
    def build_jesus_maria_hardcoded() -> "Graph":
        """Construye una red vial estilizada que imita Jesús María.

        Características:
        - 3 parches de grilla (centro rotado ~-23°, sudeste cardinal, noroeste chico rotado)
        - Diagonal primaria SW→NE (RN-9 aprox.) con enlaces
        - Políticas de mano única / doble mano por clase de vía (solo a través de aristas dirigidas)

        Returns:
            Grafo dirigido con nodos (lon/lat) y aristas con clases viales/velocidades.
        """
        g = Graph()

        LAT0 = -30.980556
        LON0 = -64.091944
        Kx = 111000.0 * cos(radians(LAT0))  # m/° lon
        Ky = 111000.0                        # m/° lat

        def to_lonlat(x_m: float, y_m: float) -> Tuple[float, float]:
            return (LON0 + x_m / Kx, LAT0 + y_m / Ky)

        next_id = 1
        def add_node_xy(x: float, y: float) -> int:
            nonlocal next_id
            lon, lat = to_lonlat(x, y)
            g.add_node(next_id, lat, lon)
            next_id += 1
            return next_id - 1

        def add_grid(x0: float, y0: float, nx: int, ny: int, step: float, *, rot_deg: float = 0.0,
                     road_primary: bool = False, bidir: bool = True) -> List[List[int]]:
            """Crea una grilla rectangular opcionalmente rotada. Retorna matriz de ids."""
            from math import radians as rads, cos as c, sin as s
            th = rads(rot_deg)
            out: List[List[int]] = []
            for r in range(ny):
                row: List[int] = []
                for cidx in range(nx):
                    x = x0 + cidx * step
                    y = y0 + r * step
                    xr = x * c(th) - y * s(th)
                    yr = x * s(th) + y * c(th)
                    row.append(add_node_xy(xr, yr))
                out.append(row)
            # Conectar calles
            v_primary = 70.0 if road_primary else 50.0
            rc = RoadClass.PRIMARY if road_primary else RoadClass.COLLECTOR
            for r in range(ny):
                for cidx in range(nx):
                    u = out[r][cidx]
                    if cidx + 1 < nx:
                        v = out[r][cidx+1]
                        d = Graph.haversine_m(g.get_node(u), g.get_node(v))
                        if bidir:
                            g.add_bidirectional(u, v, d, rc, v_primary)
                        else:
                            g.add_edge(u, v, d, rc, v_primary)
                    if r + 1 < ny:
                        v = out[r+1][cidx]
                        d = Graph.haversine_m(g.get_node(u), g.get_node(v))
                        if bidir:
                            g.add_bidirectional(u, v, d, rc, v_primary)
                        else:
                            g.add_edge(u, v, d, rc, v_primary)
            return out

        # Centro rotado
        grid_c = add_grid(-800, -400, nx=7, ny=6, step=120, rot_deg=-23, road_primary=False, bidir=True)
        # Sudeste cardinal (más residencial)
        grid_se = add_grid(200, -200, nx=6, ny=6, step=110, rot_deg=0, road_primary=False, bidir=True)
        # Noroeste chico rotado
        grid_nw = add_grid(-900, 250, nx=5, ny=4, step=110, rot_deg=-20, road_primary=False, bidir=True)

        # Diagonal primaria (RN-9 aprox.)
        diag: List[int] = []
        for k in range(10):
            diag.append(add_node_xy(-1000 + 180*k, -600 + 110*k))
        for a, b in zip(diag, diag[1:]):
            d = Graph.haversine_m(g.get_node(a), g.get_node(b))
            g.add_bidirectional(a, b, d, RoadClass.PRIMARY, 80.0)

        # Enlaces de la diagonal al centro
        for idx in range(0, len(grid_c[0]), 2):
            target = grid_c[2][idx]
            near = min(diag, key=lambda nid: Graph.haversine_m(g.get_node(nid), g.get_node(target)))
            d = Graph.haversine_m(g.get_node(near), g.get_node(target))
            g.add_bidirectional(near, target, d, RoadClass.COLLECTOR, 50.0)

        # Unas residenciales al azar
        all_nodes = list(g.nodes.keys())
        random.seed(7)
        for _ in range(40):
            u, v = random.sample(all_nodes, 2)
            d = Graph.haversine_m(g.get_node(u), g.get_node(v))
            g.add_bidirectional(u, v, d, RoadClass.RESIDENTIAL, 35.0)

        return g


# ==========================================================
# Modelo de tráfico
# ==========================================================
class TrafficModel:
    def travel_seconds(self, u: NodeId, e: Edge, *, hour: int) -> Seconds:
        raise NotImplementedError


class HistoricalTrafficModel(TrafficModel):
    """Modelo simple: factor por clase vial y hora (0-23)."""
    def __init__(self, driver_max_kmh: float = 120.0) -> None:
        self.driver_max_kmh = float(driver_max_kmh)
        # factores >=1.0 (más alto = más lento)
        base = {RoadClass.PRIMARY: 1.0, RoadClass.COLLECTOR: 1.0, RoadClass.RESIDENTIAL: 1.0}
        self._by_hour: Dict[int, Dict[RoadClass, float]] = {h: dict(base) for h in range(24)}
        # picos
        for h in range(7, 10):
            self._by_hour[h][RoadClass.PRIMARY] = 1.3
            self._by_hour[h][RoadClass.COLLECTOR] = 1.4
        for h in range(17, 20):
            self._by_hour[h][RoadClass.PRIMARY] = 1.35
            self._by_hour[h][RoadClass.COLLECTOR] = 1.5

    def factor(self, hour: int, rc: RoadClass) -> float:
        return self._by_hour.get(hour % 24, {}).get(rc, 1.0)

    def factors_by_hour(self) -> Dict[int, Dict[RoadClass, float]]:
        return self._by_hour

    def travel_seconds(self, u: NodeId, e: Edge, *, hour: int) -> Seconds:
        # limitar por max_kmh del conductor
        vmax = min(self.driver_max_kmh, e.freeflow_kmh)
        v_mps = vmax / 3.6
        # aplicar congestión (>=1 => más lento)
        factor = self.factor(hour, e.road_class)
        v_eff = max(0.1, v_mps / factor)
        return e.distance_m / v_eff


# ==========================================================
# Algoritmos de enrutamiento
# ==========================================================
class BaseRouter:
    def route(self, graph: Graph, src: NodeId, dst: NodeId, *, hour: int, traffic: TrafficModel) -> Tuple[Seconds, List[NodeId]]:
        raise NotImplementedError


class DijkstraRouter(BaseRouter):
    def route(self, graph: Graph, src: NodeId, dst: NodeId, *, hour: int, traffic: TrafficModel) -> Tuple[Seconds, List[NodeId]]:
        dist: Dict[NodeId, float] = {src: 0.0}
        parent: Dict[NodeId, Optional[NodeId]] = {src: None}
        pq: List[Tuple[float, NodeId]] = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == dst:
                break
            if d > dist.get(u, float("inf")):
                continue
            for e in graph.neighbors(u):
                w = e.to
                td = traffic.travel_seconds(u, e, hour=hour)
                nd = d + td
                if nd < dist.get(w, float("inf")):
                    dist[w] = nd
                    parent[w] = u
                    heapq.heappush(pq, (nd, w))
        if dst not in dist:
            return float("inf"), []
        # reconstruir
        path = []
        cur: Optional[NodeId] = dst
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur)
        path.reverse()
        return dist[dst], path


class AStarRouter(BaseRouter):
    def __init__(self, vmax_kmh: float = 120.0) -> None:
        self.vmax_mps = max(0.1, vmax_kmh) / 3.6

    def route(self, graph: Graph, src: NodeId, dst: NodeId, *, hour: int, traffic: TrafficModel) -> Tuple[Seconds, List[NodeId]]:
        h_cache: Dict[NodeId, float] = {}
        def h(n: NodeId) -> float:
            if n not in h_cache:
                a = graph.get_node(n)
                b = graph.get_node(dst)
                h_cache[n] = Graph.haversine_m(a, b) / self.vmax_mps
            return h_cache[n]

        g_score: Dict[NodeId, float] = {src: 0.0}
        parent: Dict[NodeId, Optional[NodeId]] = {src: None}
        pq: List[Tuple[float, NodeId]] = [(h(src), src)]
        while pq:
            f_u, u = heapq.heappop(pq)
            if u == dst:
                break
            for e in graph.neighbors(u):
                w = e.to
                td = traffic.travel_seconds(u, e, hour=hour)
                tentative = g_score[u] + td
                if tentative < g_score.get(w, float("inf")):
                    g_score[w] = tentative
                    parent[w] = u
                    heapq.heappush(pq, (tentative + h(w), w))
        if dst not in g_score:
            return float("inf"), []
        path = []
        cur: Optional[NodeId] = dst
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur)
        path.reverse()
        return g_score[dst], path


class BFSRouter(BaseRouter):
    """BFS con peso uniforme (de utilidad para grafos no ponderados)."""
    def route(self, graph: Graph, src: NodeId, dst: NodeId, *, hour: int, traffic: TrafficModel) -> Tuple[Seconds, List[NodeId]]:
        from collections import deque
        q = deque([src])
        parent: Dict[NodeId, Optional[NodeId]] = {src: None}
        while q:
            u = q.popleft()
            if u == dst:
                break
            for e in graph.neighbors(u):
                w = e.to
                if w not in parent:
                    parent[w] = u
                    q.append(w)
        if dst not in parent:
            return float("inf"), []
        # reconstruir y calcular tiempo real
        path = []
        cur: Optional[NodeId] = dst
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur)
        path.reverse()
        # sumar tiempos reales por tráfico
        secs = 0.0
        for u, v in iter_path_edges(path):
            # buscar arista (u->v)
            e = next(e for x, e in graph.iter_edges() if x == u and e.to == v)
            secs += traffic.travel_seconds(u, e, hour=hour)
        return secs, path


# ==========================================================
# Herramientas
# ==========================================================
def iter_path_edges(path: List[NodeId]) -> Iterator[Tuple[NodeId, NodeId]]:
    for a, b in zip(path, path[1:]):
        yield a, b


class RouteCache:
    """LRU simple para rutas entre pares (src, dst, hour, algorithm)."""
    def __init__(self, capacity: int = 2048) -> None:
        self.capacity = capacity
        self._store: "OrderedDict[Tuple[Any, ...], Tuple[Seconds, List[NodeId]]]" = OrderedDict()

    def get(self, key: Tuple[Any, ...]) -> Optional[Tuple[Seconds, List[NodeId]]]:
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def set(self, key: Tuple[Any, ...], value: Tuple[Seconds, List[NodeId]]) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)


class PairwiseDistanceService:
    def __init__(self, router_astar: AStarRouter, router_dijkstra: DijkstraRouter, route_cache: RouteCache, *, max_workers: int = 4) -> None:
        self.router_astar = router_astar
        self.router_dijkstra = router_dijkstra
        self.route_cache = route_cache
        self.max_workers = max_workers

    def compute_matrix(
        self,
        graph: Graph,
        waypoints: List[NodeId],
        *,
        hour: int,
        algorithm: Algorithm,
        traffic: TrafficModel,
    ) -> Tuple[List[List[Seconds]], Dict[Tuple[int, int], List[NodeId]]]:
        n = len(waypoints)
        time_matrix: List[List[Seconds]] = [[float("inf")] * n for _ in range(n)]
        path_map: Dict[Tuple[int, int], List[NodeId]] = {}

        def compute_pair(i: int, j: int) -> Tuple[int, int, Seconds, List[NodeId]]:
            if i == j:
                return i, j, 0.0, [waypoints[i]]
            src, dst = waypoints[i], waypoints[j]
            key = (src, dst, hour, algorithm.value)
            cached = self.route_cache.get(key)
            if cached is not None:
                secs, path = cached
                return i, j, secs, path
            if algorithm == Algorithm.ASTAR:
                secs, path = self.router_astar.route(graph, src, dst, hour=hour, traffic=traffic)
            elif algorithm == Algorithm.DIJKSTRA:
                secs, path = self.router_dijkstra.route(graph, src, dst, hour=hour, traffic=traffic)
            else:
                secs, path = BFSRouter().route(graph, src, dst, hour=hour, traffic=traffic)
            self.route_cache.set(key, (secs, path))
            return i, j, secs, path

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = []
            for i in range(n):
                for j in range(n):
                    futures.append(ex.submit(compute_pair, i, j))
            for fut in as_completed(futures):
                i, j, secs, path = fut.result()
                time_matrix[i][j] = secs
                path_map[(i, j)] = path
        return time_matrix, path_map


# ==========================================================
# Solvers multi-stop
# ==========================================================
class MultiStopSolver:
    def solve(self, waypoints: List[NodeId], time_matrix: List[List[Seconds]], *, mode: RouteMode) -> List[int]:
        raise NotImplementedError


class HeldKarpExact(MultiStopSolver):
    """TSP exacto con DP (Held-Karp). n <= 12 recomendado."""
    def solve(self, waypoints: List[NodeId], time_matrix: List[List[Seconds]], *, mode: RouteMode) -> List[int]:
        n = len(waypoints)
        if n <= 1:
            return list(range(n))

        # DP[mask][i] = (coste, prev_index)
        DP: Dict[Tuple[int, int], Tuple[float, Optional[int]]] = {}
        for i in range(1, n):
            DP[(1 << i, i)] = (time_matrix[0][i], 0)

        for mask in range(1, 1 << n):
            if not (mask & 1):  # el origen siempre fuera del mask
                for j in range(1, n):
                    if not (mask & (1 << j)):
                        continue
                    prev_mask = mask ^ (1 << j)
                    if prev_mask == 0:
                        continue
                    best = None
                    for i in range(1, n):
                        if not (prev_mask & (1 << i)):
                            continue
                        prev_cost, _ = DP.get((prev_mask, i), (float("inf"), None))
                        cand = prev_cost + time_matrix[i][j]
                        if best is None or cand < best[0]:
                            best = (cand, i)
                    if best is not None:
                        DP[(mask, j)] = best

        # reconstruir mejor final
        full_mask = (1 << n) - 1
        best_end = min(((DP.get((full_mask ^ 1, j), (float("inf"), None))[0] + (time_matrix[j][0] if mode == RouteMode.VISIT_ALL_CIRCUIT else 0.0), j) for j in range(1, n)), key=lambda x: x[0])
        order = [best_end[1]]
        mask = full_mask ^ 1
        cur = best_end[1]
        while cur != 0:
            cost, prev = DP[(mask, cur)]
            order.append(prev if prev is not None else 0)
            mask ^= (1 << cur)
            cur = prev if prev is not None else 0
        order.append(0)
        order.reverse()
        if mode == RouteMode.VISIT_ALL_OPEN:
            return order
        elif mode == RouteMode.VISIT_ALL_CIRCUIT:
            return order + [0]
        else:  # FIXED_ORDER (no reordena)
            return list(range(n))


class HeuristicRoute(MultiStopSolver):
    """Nearest Neighbor + 2-opt simple."""
    def __init__(self, restarts: int = 2) -> None:
        self.restarts = max(1, int(restarts))

    def solve(self, waypoints: List[NodeId], time_matrix: List[List[Seconds]], *, mode: RouteMode) -> List[int]:
        n = len(waypoints)
        if n <= 1:
            return list(range(n))

        def nn(start: int) -> List[int]:
            unvis = set(range(n))
            order = [start]
            unvis.remove(start)
            cur = start
            while unvis:
                nxt = min(unvis, key=lambda j: time_matrix[cur][j])
                order.append(nxt)
                unvis.remove(nxt)
                cur = nxt
            return order

        best = None
        starts = [0] + random.sample(range(1, n), min(self.restarts-1, n-1))
        for s in starts:
            cand = nn(s)
            # 2-opt muy simple
            improved = True
            while improved:
                improved = False
                for i in range(1, n-2):
                    for j in range(i+1, n-1):
                        delta = (time_matrix[cand[i-1]][cand[j]] + time_matrix[cand[i]][cand[j+1]]) - (time_matrix[cand[i-1]][cand[i]] + time_matrix[cand[j]][cand[j+1]])
                        if delta < -1e-6:
                            cand[i:j+1] = reversed(cand[i:j+1])
                            improved = True
            if best is None or self._route_cost(cand, time_matrix, mode) < self._route_cost(best, time_matrix, mode):
                best = cand

        if mode == RouteMode.VISIT_ALL_CIRCUIT:
            return best + [best[0]]
        return best

    @staticmethod
    def _route_cost(ordr: List[int], M: List[List[Seconds]], mode: RouteMode) -> float:
        cost = sum(M[a][b] for a, b in zip(ordr, ordr[1:]))
        if mode == RouteMode.VISIT_ALL_CIRCUIT:
            cost += M[ordr[-1]][ordr[0]]
        return cost


# ==========================================================
# Splicer y DTOs
# ==========================================================
@dataclass
class RouteLeg:
    src: NodeId
    dst: NodeId
    seconds: Seconds
    distance_m: Meters
    path: List[NodeId]


@dataclass
class ResultSummary:
    total_seconds: Seconds
    total_distance_m: Meters
    algorithm_summary: str


class RouteSplicer:
    def splice(self, order: List[int], waypoints: List[NodeId], path_map: Dict[Tuple[int, int], List[NodeId]], *, graph: Graph, hour: int, traffic: TrafficModel) -> Tuple[List[RouteLeg], ResultSummary]:
        legs: List[RouteLeg] = []
        total_s = 0.0
        total_m = 0.0
        for i, j in zip(order, order[1:]):
            path = path_map[(i, j)]
            src, dst = waypoints[i], waypoints[j]
            secs = 0.0
            dist_m = 0.0
            for u, v in iter_path_edges(path):
                e = next(e for x, e in graph.iter_edges() if x == u and e.to == v)
                secs += traffic.travel_seconds(u, e, hour=hour)
                dist_m += e.distance_m
            legs.append(RouteLeg(src, dst, secs, dist_m, path))
            total_s += secs
            total_m += dist_m
        return legs, ResultSummary(total_s, total_m, "par-a-par + splicing")


# ==========================================================
# Servicio principal
# ==========================================================
@dataclass
class RouteRequest:
    origin: NodeId
    destinations: List[NodeId]
    hour: int
    algorithm: Algorithm
    mode: RouteMode
    use_exact: bool = False  # True => Held-Karp; False => heurístico


class RoutingService:
    def __init__(self, graph: Graph, traffic: TrafficModel, pairwise_service: PairwiseDistanceService, *, solver_exact: MultiStopSolver, solver_heur: MultiStopSolver, splicer: RouteSplicer) -> None:
        self.graph = graph
        self.traffic = traffic
        self.pairwise = pairwise_service
        self.solver_exact = solver_exact
        self.solver_heur = solver_heur
        self.splicer = splicer

    def plan_route(self, req: RouteRequest) -> Tuple[List[RouteLeg], ResultSummary]:
        waypoints = [req.origin] + list(req.destinations)
        M, path_map = self.pairwise.compute_matrix(
            self.graph, waypoints, hour=req.hour, algorithm=req.algorithm, traffic=self.traffic
        )
        if req.mode == RouteMode.FIXED_ORDER:
            order = list(range(len(waypoints)))
        else:
            solver = self.solver_exact if req.use_exact else self.solver_heur
            order = solver.solve(waypoints, M, mode=req.mode)
        return self.splicer.splice(order, waypoints, path_map, graph=self.graph, hour=req.hour, traffic=self.traffic)
