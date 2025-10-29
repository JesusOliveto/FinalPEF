# -*- coding: utf-8 -*-
"""
logica.py — Motor de ruteo para “Pueblito: Rutas Inteligentes”
Versión sin OSM, con una ciudad hardcodeada que emula la traza de Jesús María.

Incluye:
- Graph.build_jesus_maria_hardcoded(): dos grillas rotadas + arteria primaria diagonal.
- Algoritmos: Dijkstra y A* (heurística admisible).
- Matriz par-a-par con batching/concurrencia y TSP (Held-Karp / NN+2opt).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Iterable, Iterator, Optional, Any, Callable, Mapping
from collections import defaultdict, OrderedDict
from math import radians, sin, cos, asin, sqrt
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import time
import cProfile, pstats, io
import unittest


# ==========================================================
# Tipos y enums
# ==========================================================
class Algorithm(str, Enum):
    ASTAR = "astar"
    DIJKSTRA = "dijkstra"
    BFS = "bfs"


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
    one_way: bool = False


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
        one_way: bool = False,
    ) -> None:
        self.adj[u].append(Edge(v, distance_m, road_class, freeflow_kmh, one_way))

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

    # ---- ciudad sintética (se mantiene por compatibilidad)
    @staticmethod
    def build_small_town(
        *,
        seed: int = 42,
        blocks_x: int = 10,
        blocks_y: int = 10,
        spacing_m: float = 120.0,
        base_lat: float = -30.9861543498,
        base_lon: float = -64.0895783905,
        two_way_ratio: float = 0.7,
        primary_ratio: float = 0.1,
        collector_ratio: float = 0.3,
    ) -> "Graph":
        rnd = random.Random(seed)
        g = Graph()
        dlat = spacing_m / 111000.0
        dlon = dlat / cos(radians(base_lat))

        def nid(r: int, c: int) -> int:
            return r * blocks_x + c

        for r in range(blocks_y):
            for c in range(blocks_x):
                g.add_node(nid(r, c), base_lat + r * dlat, base_lon + c * dlon)

        speed_by_class = {
            RoadClass.RESIDENTIAL: 35.0,
            RoadClass.COLLECTOR: 45.0,
            RoadClass.PRIMARY: 60.0,
        }

        def sample_road_class() -> RoadClass:
            x = rnd.random()
            if x < primary_ratio:
                return RoadClass.PRIMARY
            if x < primary_ratio + collector_ratio:
                return RoadClass.COLLECTOR
            return RoadClass.RESIDENTIAL

        for r in range(blocks_y):
            for c in range(blocks_x):
                u = nid(r, c)
                if c + 1 < blocks_x:
                    v = nid(r, c + 1)
                    n1, n2 = g.get_node(u), g.get_node(v)
                    dist_m = haversine_km(n1.lat, n1.lon, n2.lat, n2.lon) * 1000.0
                    rc = sample_road_class()
                    sp = speed_by_class[rc]
                    if rnd.random() < two_way_ratio:
                        g.add_edge(u, v, dist_m, rc, sp)
                        g.add_edge(v, u, dist_m, rc, sp)
                    else:
                        if r % 2 == 0:
                            g.add_edge(u, v, dist_m, rc, sp, True)
                        else:
                            g.add_edge(v, u, dist_m, rc, sp, True)
                if r + 1 < blocks_y:
                    v = nid(r + 1, c)
                    n1, n2 = g.get_node(u), g.get_node(v)
                    dist_m = haversine_km(n1.lat, n1.lon, n2.lat, n2.lon) * 1000.0
                    rc = sample_road_class()
                    sp = speed_by_class[rc]
                    if rnd.random() < two_way_ratio:
                        g.add_edge(u, v, dist_m, rc, sp)
                        g.add_edge(v, u, dist_m, rc, sp)
                    else:
                        if c % 2 == 0:
                            g.add_edge(u, v, dist_m, rc, sp, True)
                        else:
                            g.add_edge(v, u, dist_m, rc, sp, True)
        return g

    # ---- NUEVO: Jesús María hardcodeado (sin OSM)
    @staticmethod
    def build_jesus_maria_hardcoded() -> "Graph":
        """
        Genera una red vial que emula la traza de Jesús María:
        - Grilla principal (centro) rotada ~-22°.
        - Grilla sudeste (Malvinas Argentinas) sin rotación.
        - Arteria primaria diagonal que une ambas (RN-9/Av. Malvinas).
        """
        g = Graph()

        # Centro geográfico de referencia (aprox. Plaza San Martín)
        LAT0 = -30.9859
        LON0 = -64.0947

        def meters_to_deg(lat_ref: float, dx_m: float, dy_m: float) -> Tuple[float, float]:
            dlat = dy_m / 111000.0
            dlon = dx_m / (111000.0 * cos(radians(lat_ref)))
            return dlon, dlat  # (Δlon, Δlat)

        def rot(lon: float, lat: float, deg: float, cx: float, cy: float) -> Tuple[float, float]:
            if deg == 0:
                return lon, lat
            th = radians(deg)
            dx = lon - cx
            dy = lat - cy
            rx = dx * cos(th) - dy * sin(th)
            ry = dx * sin(th) + dy * cos(th)
            return cx + rx, cy + ry

        # Helper: añade grilla rectangular y devuelve matriz de ids
        def add_grid(cx: float, cy: float, nx: int, ny: int, step_m: float, rotation_deg: float, start_id: int,
                     primary_step: int = 4, collector_step: int = 2) -> Tuple[List[List[int]], int]:
            ids = [[-1] * nx for _ in range(ny)]
            # Crear nodos
            for r in range(ny):
                for c in range(nx):
                    dx = (c - (nx - 1) / 2.0) * step_m
                    dy = (r - (ny - 1) / 2.0) * step_m
                    dlon, dlat = meters_to_deg(cy, dx, dy)
                    lon, lat = LON0 + dlon, LAT0 + dlat
                    lon, lat = rot(lon, lat, rotation_deg, LON0, LAT0)
                    nid = start_id
                    start_id += 1
                    g.add_node(nid, lat, lon)
                    ids[r][c] = nid
            # Velocidades
            v_res, v_col, v_pri = 35.0, 45.0, 65.0

            # Función de clase por “avenidas” cada N calles
            def rc_for(r: int, c: int, horizontal: bool) -> RoadClass:
                idx = c if horizontal else r
                if idx % primary_step == 0:
                    return RoadClass.PRIMARY
                if idx % collector_step == 0:
                    return RoadClass.COLLECTOR
                return RoadClass.RESIDENTIAL

            # Aristas (dos manos)
            for r in range(ny):
                for c in range(nx):
                    u = ids[r][c]
                    if c + 1 < nx:
                        v = ids[r][c + 1]
                        n1, n2 = g.get_node(u), g.get_node(v)
                        dist = haversine_km(n1.lat, n1.lon, n2.lat, n2.lon) * 1000.0
                        rc = rc_for(r, c, True)
                        sp = v_pri if rc == RoadClass.PRIMARY else v_col if rc == RoadClass.COLLECTOR else v_res
                        g.add_edge(u, v, dist, rc, sp)
                        g.add_edge(v, u, dist, rc, sp)
                    if r + 1 < ny:
                        v = ids[r + 1][c]
                        n1, n2 = g.get_node(u), g.get_node(v)
                        dist = haversine_km(n1.lat, n1.lon, n2.lat, n2.lon) * 1000.0
                        rc = rc_for(r, c, False)
                        sp = v_pri if rc == RoadClass.PRIMARY else v_col if rc == RoadClass.COLLECTOR else v_res
                        g.add_edge(u, v, dist, rc, sp)
                        g.add_edge(v, u, dist, rc, sp)
            return ids, start_id

        # --- Grilla 1: centro histórico (rotada aprox. -22°) ---
        next_id = 0
        ids_centro, next_id = add_grid(
            cx=LON0, cy=LAT0, nx=16, ny=16, step_m=110.0, rotation_deg=-22.0, start_id=next_id
        )

        # --- Grilla 2: sudeste (sin rotación), desplazada ~1.2 km SE ---
        # Corrimiento: 1.2 km E y 1.0 km S aprox.
        dlon_e, dlat_s = meters_to_deg(LAT0, 1200.0, -1000.0)
        LON_SE = LON0 + dlon_e
        LAT_SE = LAT0 + dlat_s
        ids_se, next_id = add_grid(
            cx=LON_SE, cy=LAT_SE, nx=14, ny=12, step_m=115.0, rotation_deg=0.0, start_id=next_id
        )

        # --- Arteria primaria diagonal (RN-9 aproximada) que conecta ambos sectores ---
        def add_polyline(points_ll: List[Tuple[float, float]], v_kmh: float = 70.0):
            nonlocal next_id
            prev = None
            for lon, lat in points_ll:
                nid = next_id
                next_id += 1
                g.add_node(nid, lat, lon)
                if prev is not None:
                    n1, n2 = g.get_node(prev), g.get_node(nid)
                    dist = haversine_km(n1.lat, n1.lon, n2.lat, n2.lon) * 1000.0
                    g.add_edge(prev, nid, dist, RoadClass.PRIMARY, v_kmh)
                    g.add_edge(nid, prev, dist, RoadClass.PRIMARY, v_kmh)
                prev = nid
            return prev  # último id

        # Trazado aproximado: noroeste -> centro -> sudeste
        poly = [
            (LON0 - 0.018, LAT0 + 0.010),
            (LON0 - 0.007, LAT0 + 0.003),
            (LON0, LAT0),
            (LON0 + 0.006, LAT0 - 0.004),
            (LON_SE + 0.003, LAT_SE - 0.003),
            (LON_SE + 0.010, LAT_SE - 0.010),
        ]
        add_polyline(poly)

        return g


# ==========================================================
# Costos / Tráfico
# ==========================================================
class TrafficModel:
    def travel_time_seconds(self, edge: Edge, *, hour: int) -> Seconds:
        raise NotImplementedError


class HistoricalTrafficModel(TrafficModel):
    def __init__(self, factor_by_hour_and_class: Optional[Dict[int, Dict[RoadClass, float]]] = None) -> None:
        self.factor = factor_by_hour_and_class or self._default()

    @staticmethod
    def _default() -> Dict[int, Dict[RoadClass, float]]:
        base: Dict[int, Dict[RoadClass, float]] = {}
        for h in range(24):
            if 7 <= h <= 9 or 17 <= h <= 19:
                base[h] = {RoadClass.RESIDENTIAL: 1.2, RoadClass.COLLECTOR: 1.35, RoadClass.PRIMARY: 1.5}
            else:
                base[h] = {RoadClass.RESIDENTIAL: 1.05, RoadClass.COLLECTOR: 1.1, RoadClass.PRIMARY: 1.15}
        return base

    def travel_time_seconds(self, edge: Edge, *, hour: int) -> Seconds:
        factor = self.factor.get(hour, {}).get(edge.road_class, 1.1)
        hours = edge.distance_m / 1000.0 / max(edge.freeflow_kmh, 1e-6)
        return hours * 3600.0 * factor


# ==========================================================
# Heurísticas
# ==========================================================
class Heuristic:
    def estimate(self, graph: Graph, node_id: NodeId, goal_id: NodeId, *, hour: int) -> Seconds:
        raise NotImplementedError


class GeoLowerBoundHeuristic(Heuristic):
    def __init__(self, vmax_kmh: KmPerHour = 65.0) -> None:
        self.vmax_kmh = vmax_kmh

    def estimate(self, graph: Graph, node_id: NodeId, goal_id: NodeId, *, hour: int) -> Seconds:
        n, g = graph.get_node(node_id), graph.get_node(goal_id)
        dist_km = haversine_km(n.lat, n.lon, g.lat, g.lon)
        return (dist_km / max(self.vmax_kmh, 1e-6)) * 3600.0


class LearnedHistoricalHeuristic(Heuristic):
    def __init__(self, vmax95_by_hour: Optional[Dict[int, KmPerHour]] = None) -> None:
        self.v95 = vmax95_by_hour or {h: 65.0 for h in range(24)}

    def estimate(self, graph: Graph, node_id: NodeId, goal_id: NodeId, *, hour: int) -> Seconds:
        n, g = graph.get_node(node_id), graph.get_node(goal_id)
        dist_km = haversine_km(n.lat, n.lon, g.lat, g.lon)
        vmax = max(self.v95.get(hour, 60.0), 1e-6)
        return (dist_km / vmax) * 3600.0


class HybridConservativeHeuristic(Heuristic):
    def __init__(self, learned: LearnedHistoricalHeuristic, geo: GeoLowerBoundHeuristic) -> None:
        self.learned = learned
        self.geo = geo

    def estimate(self, graph: Graph, node_id: NodeId, goal_id: NodeId, *, hour: int) -> Seconds:
        return min(
            self.learned.estimate(graph, node_id, goal_id, hour=hour),
            self.geo.estimate(graph, node_id, goal_id, hour=hour),
        )


# ==========================================================
# Requests / Results / util
# ==========================================================
@dataclass
class RouteRequest:
    origin: NodeId
    destinations: List[NodeId]
    hour: int
    mode: RouteMode
    algorithm: Algorithm = Algorithm.ASTAR


@dataclass
class RouteLeg:
    src: NodeId
    dst: NodeId
    path: List[NodeId]
    seconds: Seconds
    distance_m: Meters
    expanded_nodes: int


@dataclass
class RouteResult:
    visit_order: List[NodeId]
    legs: List[RouteLeg]
    total_seconds: Seconds
    total_distance_m: Meters
    algorithm_summary: str
    cache_hits: int = 0
    cache_misses: int = 0


def iter_path_edges(path: List[NodeId]) -> Iterable[Tuple[NodeId, NodeId]]:
    for i in range(len(path) - 1):
        yield path[i], path[i + 1]


# ==========================================================
# Routers
# ==========================================================
class SearchStats:
    def __init__(self) -> None:
        self.expanded_nodes = 0
        self.queue_pushes = 0
        self.queue_pops = 0


def _reconstruct(parent: Dict[NodeId, Optional[NodeId]], src: NodeId, dst: NodeId) -> List[NodeId]:
    cur = dst
    chain = [cur]
    while cur is not None and cur != src:
        cur = parent.get(cur)
        if cur is None:
            return []
        chain.append(cur)
    chain.reverse()
    return chain


class DijkstraRouter:
    def route(self, graph: Graph, src: NodeId, dst: NodeId, *, hour: int, traffic: TrafficModel, heuristic: Optional[Heuristic] = None) -> Tuple[RouteLeg, SearchStats]:
        import heapq
        dist: Dict[NodeId, float] = defaultdict(lambda: float("inf"))
        parent: Dict[NodeId, Optional[NodeId]] = {src: None}
        stats = SearchStats()
        dist[src] = 0.0
        pq: List[Tuple[float, NodeId]] = [(0.0, src)]
        stats.queue_pushes += 1
        while pq:
            d, u = heapq.heappop(pq)
            stats.queue_pops += 1
            if d > dist[u]:
                continue
            stats.expanded_nodes += 1
            if u == dst:
                break
            for e in graph.neighbors(u):
                alt = d + traffic.travel_time_seconds(e, hour=hour)
                if alt < dist[e.to]:
                    dist[e.to] = alt
                    parent[e.to] = u
                    heapq.heappush(pq, (alt, e.to))
                    stats.queue_pushes += 1
        path = _reconstruct(parent, src, dst)
        seconds = dist[dst]
        distance_m = _path_distance_m(graph, path)
        return RouteLeg(src, dst, path, seconds, distance_m, stats.expanded_nodes), stats


class AStarRouter:
    def __init__(self, default_heuristic: Optional[Heuristic] = None) -> None:
        self.default_heuristic = default_heuristic

    def route(self, graph: Graph, src: NodeId, dst: NodeId, *, hour: int, traffic: TrafficModel, heuristic: Optional[Heuristic] = None) -> Tuple[RouteLeg, SearchStats]:
        import heapq
        h = heuristic or self.default_heuristic or GeoLowerBoundHeuristic()
        g_score: Dict[NodeId, float] = defaultdict(lambda: float("inf"))
        f_score: Dict[NodeId, float] = defaultdict(lambda: float("inf"))
        parent: Dict[NodeId, Optional[NodeId]] = {src: None}
        stats = SearchStats()

        def est(nid: NodeId) -> float:
            return h.estimate(graph, nid, dst, hour=hour)

        g_score[src] = 0.0
        f_score[src] = est(src)
        pq: List[Tuple[float, NodeId]] = [(f_score[src], src)]
        stats.queue_pushes += 1

        while pq:
            _, u = heapq.heappop(pq)
            stats.queue_pops += 1
            stats.expanded_nodes += 1
            if u == dst:
                break
            for e in graph.neighbors(u):
                tentative = g_score[u] + traffic.travel_time_seconds(e, hour=hour)
                if tentative < g_score[e.to]:
                    parent[e.to] = u
                    g_score[e.to] = tentative
                    f_score[e.to] = tentative + est(e.to)
                    heapq.heappush(pq, (f_score[e.to], e.to))
                    stats.queue_pushes += 1

        path = _reconstruct(parent, src, dst)
        seconds = g_score[dst]
        distance_m = _path_distance_m(graph, path)
        return RouteLeg(src, dst, path, seconds, distance_m, stats.expanded_nodes), stats


# ==========================================================
# Caches y pairwise
# ==========================================================
class LRUCache:
    def __init__(self, capacity: int = 1024) -> None:
        self.capacity = capacity
        self._store: OrderedDict[Tuple[Any, ...], Any] = OrderedDict()

    def get(self, key: Tuple[Any, ...]) -> Optional[Any]:
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def set(self, key: Tuple[Any, ...], value: Any) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)


class SSSPMemo:
    def __init__(self) -> None:
        self._store: Dict[Tuple[Any, ...], Tuple[Dict[NodeId, Seconds], Dict[NodeId, Optional[NodeId]]]] = {}

    def get(self, key: Tuple[Any, ...]) -> Optional[Tuple[Mapping[NodeId, Seconds], Mapping[NodeId, Optional[NodeId]]]]:
        return self._store.get(key)

    def set(self, key: Tuple[Any, ...], dist: Mapping[NodeId, Seconds], parent: Mapping[NodeId, Optional[NodeId]]) -> None:
        self._store[key] = (dict(dist), dict(parent))


class RouteCache:
    def __init__(self, capacity: int = 2048) -> None:
        self._lru = LRUCache(capacity)

    def get(self, key: Tuple[Any, ...]) -> Optional[RouteLeg]:
        return self._lru.get(key)

    def set(self, key: Tuple[Any, ...], leg: RouteLeg) -> None:
        self._lru.set(key, leg)


class PairwiseMatrixCache:
    def __init__(self) -> None:
        self._store: Dict[Tuple[Any, ...], Tuple[List[List[Seconds]], Dict[Tuple[int, int], List[NodeId]]]] = {}

    def get(self, signature: Tuple[Any, ...]) -> Optional[Tuple[List[List[Seconds]], Dict[Tuple[int, int], List[NodeId]]]]:
        return self._store.get(signature)

    def set(self, signature: Tuple[Any, ...], time_matrix: List[List[Seconds]], path_map: Dict[Tuple[int, int], List[NodeId]]) -> None:
        self._store[signature] = (time_matrix, path_map)


class PairwiseDistanceService:
    def __init__(self, router_astar: AStarRouter, router_dijkstra: DijkstraRouter, route_cache: RouteCache, ssp_memo: SSSPMemo, *, max_workers: int = 4) -> None:
        self.router_astar = router_astar
        self.router_dijkstra = router_dijkstra
        self.route_cache = route_cache
        self.ssp_memo = ssp_memo
        self.max_workers = max_workers

    def compute_matrix(
        self,
        graph: Graph,
        waypoints: List[NodeId],
        *,
        hour: int,
        algorithm: Algorithm,
        traffic: TrafficModel,
        heuristic: Heuristic,
    ) -> Tuple[List[List[Seconds]], Dict[Tuple[int, int], List[NodeId]]]:
        n = len(waypoints)
        time_matrix: List[List[Seconds]] = [[float("inf")] * n for _ in range(n)]
        path_map: Dict[Tuple[int, int], List[NodeId]] = {}

        def solve_from(i: int) -> Tuple[int, Dict[NodeId, Seconds], Dict[NodeId, Optional[NodeId]]]:
            src = waypoints[i]
            key = ("SSSP", src, hour, "dijkstra")
            cached = self.ssp_memo.get(key)
            if cached:
                dm, pm = cached
                return i, dict(dm), dict(pm)

            import heapq
            dist: Dict[NodeId, float] = defaultdict(lambda: float("inf"))
            parent: Dict[NodeId, Optional[NodeId]] = {src: None}
            dist[src] = 0.0
            pq: List[Tuple[float, NodeId]] = [(0.0, src)]
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u]:
                    continue
                for e in graph.neighbors(u):
                    alt = d + traffic.travel_time_seconds(e, hour=hour)
                    if alt < dist[e.to]:
                        dist[e.to] = alt
                        parent[e.to] = u
                        heapq.heappush(pq, (alt, e.to))

            self.ssp_memo.set(key, dist, parent)
            return i, dist, parent

        results: Dict[int, Tuple[Dict[NodeId, Seconds], Dict[NodeId, Optional[NodeId]]]] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = [ex.submit(solve_from, i) for i in range(n)]
            for fut in as_completed(futs):
                i, dist, parent = fut.result()
                results[i] = (dist, parent)

        for i in range(n):
            dist, parent = results[i]
            for j in range(n):
                if i == j:
                    time_matrix[i][j] = 0.0
                    path_map[(i, j)] = [waypoints[i]]
                    continue
                dst = waypoints[j]
                time_matrix[i][j] = float(dist.get(dst, float("inf")))
                path_map[(i, j)] = _reconstruct(parent, waypoints[i], dst)
        return time_matrix, path_map


# ==========================================================
# Solvers y splicer
# ==========================================================
class MultiStopSolver:
    def solve(self, waypoints: List[NodeId], time_matrix: List[List[Seconds]], *, mode: RouteMode) -> List[int]:
        raise NotImplementedError


class HeldKarpExact(MultiStopSolver):
    def solve(self, waypoints: List[NodeId], time_matrix: List[List[Seconds]], *, mode: RouteMode) -> List[int]:
        n = len(waypoints)
        if n <= 1:
            return list(range(n))
        from math import inf
        ALL = 1 << n
        dp = [[inf] * n for _ in range(ALL)]
        prv = [[-1] * n for _ in range(ALL)]
        origin = 0
        dp[1 << origin][origin] = 0.0
        for mask in range(ALL):
            if not (mask & (1 << origin)): continue
            for j in range(n):
                if not (mask & (1 << j)): continue
                cj = dp[mask][j]
                if cj == inf: continue
                for k in range(n):
                    if mask & (1 << k): continue
                    nm = mask | (1 << k)
                    cand = cj + time_matrix[j][k]
                    if cand < dp[nm][k]:
                        dp[nm][k] = cand; prv[nm][k] = j

        full = ALL - 1
        if mode == RouteMode.VISIT_ALL_CIRCUIT:
            best, last = inf, -1
            for j in range(n):
                cand = dp[full][j] + time_matrix[j][0]
                if cand < best:
                    best, last = cand, j
        else:
            last = min(range(n), key=lambda j: dp[full][j])

        order = []
        mask = full
        while last != -1:
            order.append(last)
            p = prv[mask][last]
            if p == -1: break
            mask &= ~(1 << last)
            last = p
        order.append(0)
        order.reverse()
        return order


class HeuristicRoute(MultiStopSolver):
    def __init__(self, restarts: int = 4) -> None:
        self.restarts = max(1, int(restarts))

    def solve(self, waypoints: List[NodeId], time_matrix: List[List[Seconds]], *, mode: RouteMode) -> List[int]:
        n = len(waypoints)
        if n <= 1: return list(range(n))
        circuit = mode == RouteMode.VISIT_ALL_CIRCUIT

        def route_cost(order: List[int]) -> float:
            c = 0.0
            for a, b in zip(order, order[1:]): c += time_matrix[a][b]
            if circuit: c += time_matrix[order[-1]][order[0]]
            return c

        def two_opt(order: List[int]) -> List[int]:
            best = order[:]; bestc = route_cost(best); improved = True
            while improved:
                improved = False
                for i in range(1, n - 2):
                    for k in range(i + 1, n - (0 if circuit else 1)):
                        cand = best[:i] + list(reversed(best[i:k])) + best[k:]
                        cc = route_cost(cand)
                        if cc + 1e-9 < bestc:
                            best, bestc, improved = cand, cc, True
            return best

        origin = 0
        seeds = [origin] + [i for i in range(1, n)]
        seeds = seeds[: self.restarts]
        best_order, best_cost = list(range(n)), float("inf")
        for s in seeds:
            un = set(range(n)); un.remove(s)
            order = [s]; cur = s
            while un:
                nxt = min(un, key=lambda j: time_matrix[cur][j])
                order.append(nxt); un.remove(nxt); cur = nxt
            if origin in order:
                idx = order.index(origin); order = order[idx:] + order[:idx]
            order = two_opt(order)
            c = route_cost(order)
            if c < best_cost: best_cost, best_order = c, order
        return best_order


class RouteSplicer:
    def splice(self, waypoints: List[NodeId], visit_order_idx: List[int], path_map: Dict[Tuple[int, int], List[NodeId]], graph: Graph, time_matrix: List[List[Seconds]]) -> List[RouteLeg]:
        legs: List[RouteLeg] = []
        for a, b in zip(visit_order_idx, visit_order_idx[1:]):
            src = waypoints[a]; dst = waypoints[b]
            path = path_map[(a, b)]
            seconds = time_matrix[a][b]
            dist_m = _path_distance_m(graph, path)
            legs.append(RouteLeg(src, dst, path, seconds, dist_m, 0))
        return legs


# ==========================================================
# Servicio de ruteo
# ==========================================================
class RoutingService:
    def __init__(self, graph: Graph, traffic: TrafficModel, heuristic: Heuristic, pairwise_service: PairwiseDistanceService, solver_exact: MultiStopSolver, solver_heur: MultiStopSolver, splicer: RouteSplicer) -> None:
        self.graph = graph
        self.traffic = traffic
        self.heuristic = heuristic
        self.pairwise = pairwise_service
        self.solver_exact = solver_exact
        self.solver_heur = solver_heur
        self.splicer = splicer
        self.router_dijkstra = DijkstraRouter()
        self.router_astar = AStarRouter(default_heuristic=heuristic)

    def route_single(self, src: NodeId, dst: NodeId, *, hour: int, algorithm: Algorithm = Algorithm.ASTAR) -> RouteLeg:
        if algorithm == Algorithm.DIJKSTRA or algorithm == Algorithm.BFS:
            leg, _ = self.router_dijkstra.route(self.graph, src, dst, hour=hour, traffic=self.traffic)
        else:
            leg, _ = self.router_astar.route(self.graph, src, dst, hour=hour, traffic=self.traffic, heuristic=self.heuristic)
        return leg

    def route(self, request: RouteRequest) -> RouteResult:
        waypoints = [request.origin] + list(request.destinations)
        time_matrix, path_map = self.pairwise.compute_matrix(
            self.graph, waypoints, hour=request.hour, algorithm=request.algorithm, traffic=self.traffic, heuristic=self.heuristic
        )
        n = len(waypoints)
        if request.mode == RouteMode.FIXED_ORDER:
            order_idx = list(range(n)); alg = f"{request.algorithm.value}"
        else:
            if n <= 13:
                order_idx = self.solver_exact.solve(waypoints, time_matrix, mode=request.mode); alg = f"{request.algorithm.value} + Held-Karp"
            else:
                order_idx = self.solver_heur.solve(waypoints, time_matrix, mode=request.mode); alg = f"{request.algorithm.value} + NN/2opt"
        legs = self.splicer.splice(waypoints, order_idx, path_map, self.graph, time_matrix)
        total_s = sum(l.seconds for l in legs)
        total_m = sum(l.distance_m for l in legs)
        visit_order = [waypoints[i] for i in order_idx]
        return RouteResult(visit_order, legs, total_s, total_m, alg)


# ==========================================================
# Helpers geo
# ==========================================================
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * (2 * asin(sqrt(a)))


def _path_distance_m(graph: Graph, path: List[NodeId]) -> float:
    d = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        for e in graph.neighbors(u):
            if e.to == v:
                d += e.distance_m
                break
    return d


# ==========================================================
# Tests mínimos
# ==========================================================
class TestJM(unittest.TestCase):
    def test_graph_non_empty(self):
        g = Graph.build_jesus_maria_hardcoded()
        self.assertGreater(len(list(g.iter_nodes())), 0)
        self.assertGreater(len(list(g.iter_edges())), 0)


if __name__ == "__main__":
    # Smoke test
    g = Graph.build_jesus_maria_hardcoded()
    traffic = HistoricalTrafficModel()
    heuristic = HybridConservativeHeuristic(LearnedHistoricalHeuristic(), GeoLowerBoundHeuristic())
    route_cache = RouteCache()
    memo = SSSPMemo()
    pairwise = PairwiseDistanceService(AStarRouter(heuristic), DijkstraRouter(), route_cache, memo)
    service = RoutingService(g, traffic, heuristic, pairwise, HeldKarpExact(), HeuristicRoute(), RouteSplicer())
    # tomar dos nodos cualesquiera
    nodes = list(g.iter_nodes())
    src, dst = nodes[0].id, nodes[-1].id
    leg = service.route_single(src, dst, hour=8)
    print(f"Smoke: {src}→{dst} en {leg.seconds:.1f}s, {leg.distance_m/1000:.2f} km")
