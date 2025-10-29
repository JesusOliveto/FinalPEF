"""
logica.py — Motor de ruteo multi-destino para ciudad tipo "pueblito" (PEF 2025)

- Grafo dirigido con calles de uno y dos sentidos.
- Costos dependientes de la hora (tráfico histórico por clase de vía).
- Ruteo par-a-par (Dijkstra / A*) con heurística admisible (geométrica + aprendida).
- Multi-destino (TSP/waypoints): Held-Karp (exacto) y heurístico (NN + 2-opt).
- Batching para matriz par-a-par, memoización SSSP, caches LRU, concurrencia y profiling.
- Generadores para iterar caminos sin materializar estructuras enormes.
- Incluye tests unitarios (unittest) ejecutables con `python logica.py -m test` o `pytest`.

Nota: Este módulo está pensado para ser consumido por un front (p. ej., Streamlit) en un archivo aparte.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import (
    Dict,
    List,
    Tuple,
    Iterable,
    Iterator,
    Optional,
    Any,
    Callable,
    Mapping,
)
from math import radians, sin, cos, asin, sqrt
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import cProfile
import pstats
import io
import random
import time
import unittest

# ==========================================================
# 1) Enumeraciones y alias de tipos
# ==========================================================

class Algorithm(str, Enum):
    """Algoritmos disponibles para ruteo par-a-par."""

    ASTAR = "astar"
    DIJKSTRA = "dijkstra"
    BFS = "bfs"  # educativo; en este motor se resuelve como Dijkstra costo=tiempo


class RouteMode(str, Enum):
    """Modos de ruteo multi-destino."""

    VISIT_ALL_OPEN = "visit_all_open"  # TSP abierto (no vuelve al origen)
    VISIT_ALL_CIRCUIT = "visit_all_circuit"  # TSP circuito (vuelve al origen)
    FIXED_ORDER = "fixed_order"  # seguir el orden dado (waypoints)


class RoadClass(str, Enum):
    """Clases de vía para modelar velocidades y congestión."""

    RESIDENTIAL = "residential"
    COLLECTOR = "collector"
    PRIMARY = "primary"


NodeId = int
Seconds = float
Meters = float
KmPerHour = float


# ==========================================================
# 2) Dominio: Nodos, Aristas, Grafo
# ==========================================================


@dataclass(frozen=True)
class Node:
    """Nodo del grafo.

    Args:
        id: identificador único.
        lat: latitud (grados decimales).
        lon: longitud (grados decimales).
    """

    id: NodeId
    lat: float
    lon: float


@dataclass
class Edge:
    """Arista dirigida u -> v con atributos de transporte.

    Args:
        to: id del nodo destino.
        distance_m: distancia en metros.
        road_class: clase de vía (afecta velocidades/horarios).
        freeflow_kmh: velocidad libre de congestión (km/h).
        one_way: True si el segmento es de un solo sentido (esta arista lo respeta).
    """

    to: NodeId
    distance_m: Meters
    road_class: RoadClass
    freeflow_kmh: KmPerHour
    one_way: bool = False


class Graph:
    """Grafo dirigido con listas de adyacencia.

    Responsabilidades:
        - Gestionar nodos/aristas.
        - Entregar vecinos eficientemente.
        - Proveer constructores de ciudad simulada ("pueblito").
    """

    def __init__(self) -> None:
        self.nodes: Dict[NodeId, Node] = {}
        self.adj: Dict[NodeId, List[Edge]] = defaultdict(list)

    # ---- Alta y acceso ----
    def add_node(self, node_id: NodeId, lat: float, lon: float) -> None:
        """Agrega un nodo al grafo."""
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
        """Agrega una arista dirigida u -> v con atributos de transporte."""
        self.adj[u].append(
            Edge(to=v, distance_m=distance_m, road_class=road_class, freeflow_kmh=freeflow_kmh, one_way=one_way)
        )

    def neighbors(self, u: NodeId) -> Iterable[Edge]:
        """Itera aristas salientes desde `u` (generador/iterable)."""
        return self.adj.get(u, [])

    def get_node(self, node_id: NodeId) -> Node:
        """Obtiene un nodo por id."""
        return self.nodes[node_id]

    def iter_nodes(self) -> Iterator[Node]:
        """Generador de todos los nodos."""
        for n in self.nodes.values():
            yield n

    def iter_edges(self) -> Iterator[Tuple[NodeId, Edge]]:
        """Generador de todas las aristas (u, Edge(u->v))."""
        for u, lst in self.adj.items():
            for e in lst:
                yield u, e

    # ---- Constructores de ciudad ----
    @staticmethod
    def build_small_town(
        *,
        seed: int = 42,
        blocks_x: int = 10,
        blocks_y: int = 10,
        spacing_m: float = 120.0,
        base_lat: float = -30.986154349785995,
        base_lon: float = -64.08957839045158,
        two_way_ratio: float = 0.7,
        primary_ratio: float = 0.1,
        collector_ratio: float = 0.3,
    ) -> "Graph":
        """Construye un "pueblito" reproducible con plaza central, grilla y mezcla de sentidos.

        Reglas:
          - Nodos en grilla (blocks_y x blocks_x).
          - Aristas horizontales y verticales.
          - Algunas calles de un solo sentido (patrón alternado y aleatoriedad controlada).
          - Clases viales con distintas velocidades libres.
        """
        rnd = random.Random(seed)
        g = Graph()

        # Conversión simple grados ↔ metros (aprox.)
        dlat = (spacing_m / 1000.0) / 111.0
        dlon = dlat / cos(radians(base_lat))

        def nid(r: int, c: int) -> int:
            return r * blocks_x + c

        # Crear nodos
        for r in range(blocks_y):
            for c in range(blocks_x):
                g.add_node(nid(r, c), base_lat + r * dlat, base_lon + c * dlon)

        # Velocidades por clase
        speed_by_class = {RoadClass.RESIDENTIAL: 35.0, RoadClass.COLLECTOR: 45.0, RoadClass.PRIMARY: 60.0}

        def sample_road_class() -> RoadClass:
            x = rnd.random()
            if x < primary_ratio:
                return RoadClass.PRIMARY
            if x < primary_ratio + collector_ratio:
                return RoadClass.COLLECTOR
            return RoadClass.RESIDENTIAL

        # Crear aristas (con mezcla de sentidos)
        for r in range(blocks_y):
            for c in range(blocks_x):
                u = nid(r, c)
                # derecha
                if c + 1 < blocks_x:
                    v = nid(r, c + 1)
                    n1, n2 = g.get_node(u), g.get_node(v)
                    dist_m = haversine_km(n1.lat, n1.lon, n2.lat, n2.lon) * 1000.0
                    rc = sample_road_class()
                    sp = speed_by_class[rc]
                    if rnd.random() < two_way_ratio:
                        g.add_edge(u, v, dist_m, rc, sp, one_way=False)
                        g.add_edge(v, u, dist_m, rc, sp, one_way=False)
                    else:
                        # patrón alternado por fila
                        if r % 2 == 0:
                            g.add_edge(u, v, dist_m, rc, sp, one_way=True)
                        else:
                            g.add_edge(v, u, dist_m, rc, sp, one_way=True)
                # abajo
                if r + 1 < blocks_y:
                    v = nid(r + 1, c)
                    n1, n2 = g.get_node(u), g.get_node(v)
                    dist_m = haversine_km(n1.lat, n1.lon, n2.lat, n2.lon) * 1000.0
                    rc = sample_road_class()
                    sp = speed_by_class[rc]
                    if rnd.random() < two_way_ratio:
                        g.add_edge(u, v, dist_m, rc, sp, one_way=False)
                        g.add_edge(v, u, dist_m, rc, sp, one_way=False)
                    else:
                        # patrón alternado por columna
                        if c % 2 == 0:
                            g.add_edge(u, v, dist_m, rc, sp, one_way=True)
                        else:
                            g.add_edge(v, u, dist_m, rc, sp, one_way=True)
        return g

    @staticmethod
    def from_csv(nodes_csv: str, edges_csv: str) -> "Graph":
        """Carga un grafo desde CSVs (nodos y aristas) exportados de una ciudad real.

        Formatos esperados:
            nodes.csv: id,lat,lon
            edges.csv: u,v,distance_m,road_class,freeflow_kmh,one_way
        """
        import csv

        g = Graph()
        with open(nodes_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                g.add_node(int(row["id"]), float(row["lat"]), float(row["lon"]))
        with open(edges_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                g.add_edge(
                    int(row["u"]),
                    int(row["v"]),
                    float(row["distance_m"]),
                    RoadClass(row["road_class"]),
                    float(row["freeflow_kmh"]),
                    row.get("one_way", "false").strip().lower() in {"1", "true", "t", "yes", "y"},
                )
        return g


# ==========================================================
# 3) Costos y Tráfico (Strategy)
# ==========================================================


class TrafficModel:
    """Interfaz de cálculo de costo temporal por arista según el contexto (hora)."""

    def travel_time_seconds(self, edge: Edge, *, hour: int) -> Seconds:  # pragma: no cover - interfaz
        raise NotImplementedError


class HistoricalTrafficModel(TrafficModel):
    """Uso de factor de congestión por (hora × clase de vía).

    `factor >= 1.0` (1.0 == free-flow). Se aplica sobre el tiempo base (distancia / freeflow_kmh).
    """

    def __init__(self, factor_by_hour_and_class: Optional[Dict[int, Dict[RoadClass, float]]] = None) -> None:
        self.factor = factor_by_hour_and_class or self._default_factors()

    @staticmethod
    def _default_factors() -> Dict[int, Dict[RoadClass, float]]:
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
# 4) Heurísticas (Strategy) – admisibles por diseño
# ==========================================================


class Heuristic:
    """Interfaz de heurística para A* (devuelve cota inferior de tiempo restante)."""

    def estimate(self, graph: Graph, node_id: NodeId, goal_id: NodeId, *, hour: int) -> Seconds:  # pragma: no cover
        raise NotImplementedError


class GeoLowerBoundHeuristic(Heuristic):
    """Cota inferior geométrica: distancia en línea recta / v_max.

    Mantiene admisibilidad porque asume velocidad alta (límite inferior de tiempo).
    """

    def __init__(self, vmax_kmh: KmPerHour = 65.0) -> None:
        self.vmax_kmh = vmax_kmh

    def estimate(self, graph: Graph, node_id: NodeId, goal_id: NodeId, *, hour: int) -> Seconds:
        n = graph.get_node(node_id)
        g = graph.get_node(goal_id)
        dist_km = haversine_km(n.lat, n.lon, g.lat, g.lon)
        vmax = max(self.vmax_kmh, 1e-6)
        return (dist_km / vmax) * 3600.0


class LearnedHistoricalHeuristic(Heuristic):
    """Heurística aprendida: usa una velocidad alta plausible por hora (p.ej., p95).

    Debe ser conservadora para no sobreestimar; se recomienda combinar con la geométrica.
    """

    def __init__(self, vmax95_by_hour: Optional[Dict[int, KmPerHour]] = None) -> None:
        self.vmax95_by_hour = vmax95_by_hour or {h: 65.0 for h in range(24)}

    def estimate(self, graph: Graph, node_id: NodeId, goal_id: NodeId, *, hour: int) -> Seconds:
        n = graph.get_node(node_id)
        g = graph.get_node(goal_id)
        dist_km = haversine_km(n.lat, n.lon, g.lat, g.lon)
        vmax = max(self.vmax95_by_hour.get(hour, 60.0), 1e-6)
        return (dist_km / vmax) * 3600.0


class HybridConservativeHeuristic(Heuristic):
    """Heurística híbrida: `min(heurística aprendida, geométrica)` → admisible."""

    def __init__(self, learned: LearnedHistoricalHeuristic, geo: GeoLowerBoundHeuristic) -> None:
        self.learned = learned
        self.geo = geo

    def estimate(self, graph: Graph, node_id: NodeId, goal_id: NodeId, *, hour: int) -> Seconds:
        return min(
            self.learned.estimate(graph, node_id, goal_id, hour=hour),
            self.geo.estimate(graph, node_id, goal_id, hour=hour),
        )


# ==========================================================
# 5) Requests/Results y utilidades
# ==========================================================


@dataclass
class RouteRequest:
    """Solicitud de ruteo multi-destino.

    Args:
        origin: nodo de inicio.
        destinations: lista de nodos a visitar.
        hour: franja horaria (0–23) para costos/heurística.
        mode: modo de ruteo (TSP abierto/circuito o fixed order).
        algorithm: algoritmo base para tramos par-a-par.
    """

    origin: NodeId
    destinations: List[NodeId]
    hour: int
    mode: RouteMode
    algorithm: Algorithm = Algorithm.ASTAR


@dataclass
class RouteLeg:
    """Un tramo par-a-par dentro de una ruta completa."""

    src: NodeId
    dst: NodeId
    path: List[NodeId]
    seconds: Seconds
    distance_m: Meters
    expanded_nodes: int


@dataclass
class RouteResult:
    """Resultado final de ruteo (multi-destino o single)."""

    visit_order: List[NodeId]  # orden de visita (incluye origin y destinos en orden)
    legs: List[RouteLeg]  # descomposición por tramos (src→dst)
    total_seconds: Seconds
    total_distance_m: Meters
    algorithm_summary: str  # p.ej., "A* + Held-Karp" o "A* + NN+2opt"
    cache_hits: int = 0
    cache_misses: int = 0


def iter_path_edges(path: List[NodeId]) -> Iterator[Tuple[NodeId, NodeId]]:
    """Generador que emite aristas consecutivas (u, v) de un camino dado por nodos."""
    for i in range(len(path) - 1):
        yield path[i], path[i + 1]


# ==========================================================
# 6) Routers par-a-par (Strategy)
# ==========================================================


class SearchStats:
    """Métricas del algoritmo de búsqueda (para profiling y UI)."""

    def __init__(self) -> None:
        self.expanded_nodes: int = 0
        self.queue_pushes: int = 0
        self.queue_pops: int = 0


class Router:
    """Interfaz de router par-a-par."""

    def route(
        self,
        graph: Graph,
        src: NodeId,
        dst: NodeId,
        *,
        hour: int,
        traffic: TrafficModel,
        heuristic: Optional[Heuristic] = None,
    ) -> Tuple[RouteLeg, SearchStats]:  # pragma: no cover
        raise NotImplementedError


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


class DijkstraRouter(Router):
    """Implementación Dijkstra (óptimo para costos no negativos)."""

    def route(
        self,
        graph: Graph,
        src: NodeId,
        dst: NodeId,
        *,
        hour: int,
        traffic: TrafficModel,
        heuristic: Optional[Heuristic] = None,
    ) -> Tuple[RouteLeg, SearchStats]:
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


class AStarRouter(Router):
    """Implementación A* (usa heurística admisible para acelerar)."""

    def __init__(self, default_heuristic: Optional[Heuristic] = None) -> None:
        self.default_heuristic = default_heuristic

    def route(
        self,
        graph: Graph,
        src: NodeId,
        dst: NodeId,
        *,
        hour: int,
        traffic: TrafficModel,
        heuristic: Optional[Heuristic] = None,
    ) -> Tuple[RouteLeg, SearchStats]:
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
# 7) Caches y memoización
# ==========================================================


class LRUCache:
    """Cache LRU genérica parametrizable por capacidad.

    Se usa para cachear rutas, subrutas y valores de heurística.
    """

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

    def clear(self) -> None:
        self._store.clear()


class SSSPMemo:
    """Memoización de SSSP (árbol de distancias y padres) por clave."""

    def __init__(self) -> None:
        self._store: Dict[Tuple[Any, ...], Tuple[Dict[NodeId, Seconds], Dict[NodeId, Optional[NodeId]]]] = {}

    def get(self, key: Tuple[Any, ...]) -> Optional[Tuple[Mapping[NodeId, Seconds], Mapping[NodeId, Optional[NodeId]]]]:
        return self._store.get(key)

    def set(self, key: Tuple[Any, ...], dist: Mapping[NodeId, Seconds], parent: Mapping[NodeId, Optional[NodeId]]) -> None:
        self._store[key] = (dict(dist), dict(parent))

    def clear(self) -> None:
        self._store.clear()


class RouteCache:
    """Cache específica de rutas completas (src, dst, hour, algo, heur_ver, graph_ver)."""

    def __init__(self, capacity: int = 2048) -> None:
        self._lru = LRUCache(capacity)

    def get(self, key: Tuple[Any, ...]) -> Optional[RouteLeg]:
        return self._lru.get(key)

    def set(self, key: Tuple[Any, ...], leg: RouteLeg) -> None:
        self._lru.set(key, leg)


class PairwiseMatrixCache:
    """Cache para matrices par-a-par (tiempos + paths) sobre un conjunto de waypoints."""

    def __init__(self) -> None:
        self._store: Dict[Tuple[Any, ...], Tuple[List[List[Seconds]], Dict[Tuple[int, int], List[NodeId]]]] = {}

    def get(self, signature: Tuple[Any, ...]) -> Optional[Tuple[List[List[Seconds]], Dict[Tuple[int, int], List[NodeId]]]]:
        return self._store.get(signature)

    def set(
        self,
        signature: Tuple[Any, ...],
        time_matrix: List[List[Seconds]],
        path_map: Dict[Tuple[int, int], List[NodeId]],
    ) -> None:
        self._store[signature] = (time_matrix, path_map)


# ==========================================================
# 8) Servicios par-a-par y multi-destino
# ==========================================================


def _path_distance_m(graph: Graph, path: List[NodeId]) -> float:
    dist = 0.0
    for u, v in iter_path_edges(path):
        for e in graph.neighbors(u):
            if e.to == v:
                dist += e.distance_m
                break
    return dist


class PairwiseDistanceService:
    """Calcula, con batching y concurrencia, todas las rutas par-a-par entre waypoints.

    - Agrupa por (src, hour, algorithm) para reutilizar SSSP donde convenga.
    - Paraleliza por grupos y llena time_matrix y path_map.
    - Respeta caches (RouteCache, SSSPMemo, PairwiseMatrixCache).

    Nota: Para completar la matriz eficientemente, aunque el usuario haya seleccionado
    ASTAR, este servicio ejecuta **Dijkstra single-source** por cada `src` del conjunto,
    ya que A* multi-objetivo no aporta ventajas en la construcción de matrices completas.
    """

    def __init__(
        self,
        router_astar: Router,
        router_dijkstra: Router,
        route_cache: RouteCache,
        ssp_memo: SSSPMemo,
        *,
        max_workers: int = 4,
    ) -> None:
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

        # Trabajo por fuente: un SSSP (Dijkstra) por cada waypoint fuente.
        def solve_from(i: int) -> Tuple[int, Dict[NodeId, Seconds], Dict[NodeId, Optional[NodeId]]]:
            src = waypoints[i]
            key = ("SSSP", src, hour, "dijkstra")
            cached = self.ssp_memo.get(key)
            if cached:
                dist_map, parent_map = cached
                return i, dict(dist_map), dict(parent_map)

            # Dijkstra SSSP desde src
            dist: Dict[NodeId, float] = defaultdict(lambda: float("inf"))
            parent: Dict[NodeId, Optional[NodeId]] = {src: None}
            import heapq

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

        # Paralelizar por fuentes
        results: Dict[int, Tuple[Dict[NodeId, Seconds], Dict[NodeId, Optional[NodeId]]]] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = [ex.submit(solve_from, i) for i in range(n)]
            for fut in as_completed(futs):
                i, dist, parent = fut.result()
                results[i] = (dist, parent)

        # Completar matriz y paths
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


class MultiStopSolver:
    """Interfaz del solver multi-destino (TSP/Waypoints)."""

    def solve(self, waypoints: List[NodeId], time_matrix: List[List[Seconds]], *, mode: RouteMode) -> List[int]:  # pragma: no cover
        raise NotImplementedError


class HeldKarpExact(MultiStopSolver):
    """DP exacto para TSP (apto hasta ~12–14 destinos).

    Implementa variante con origen fijo en índice 0. Soporta circuito y abierto.
    """

    def solve(self, waypoints: List[NodeId], time_matrix: List[List[Seconds]], *, mode: RouteMode) -> List[int]:
        n = len(waypoints)
        if n <= 1:
            return list(range(n))
        # Map (subset_mask, last_idx) -> (cost, prev_idx)
        from math import inf

        ALL = 1 << n
        dp_cost = [[inf] * n for _ in range(ALL)]
        dp_prev = [[-1] * n for _ in range(ALL)]
        origin = 0
        dp_cost[1 << origin][origin] = 0.0

        for mask in range(ALL):
            if not (mask & (1 << origin)):
                continue
            for j in range(n):
                if not (mask & (1 << j)):
                    continue
                cost_j = dp_cost[mask][j]
                if cost_j == inf:
                    continue
                for k in range(n):
                    if mask & (1 << k):
                        continue
                    new_mask = mask | (1 << k)
                    cand = cost_j + time_matrix[j][k]
                    if cand < dp_cost[new_mask][k]:
                        dp_cost[new_mask][k] = cand
                        dp_prev[new_mask][k] = j

        # Reconstrucción
        if mode == RouteMode.VISIT_ALL_CIRCUIT:
            best_cost = inf
            last = -1
            full = ALL - 1
            for j in range(n):
                cand = dp_cost[full][j] + time_matrix[j][origin]
                if cand < best_cost:
                    best_cost = cand
                    last = j
            order = []
            mask = full
            while last != -1:
                order.append(last)
                prev = dp_prev[mask][last]
                if prev == -1:
                    break
                mask &= ~(1 << last)
                last = prev
            order.append(origin)
            order.reverse()
            return order
        else:  # abierto: no vuelve al origen
            full = ALL - 1
            last = min(range(n), key=lambda j: dp_cost[full][j])
            order = []
            mask = full
            while last != -1:
                order.append(last)
                prev = dp_prev[mask][last]
                if prev == -1:
                    break
                mask &= ~(1 << last)
                last = prev
            order.append(origin)
            order.reverse()
            return order


class HeuristicRoute(MultiStopSolver):
    """Heurístico para TSP/Waypoints: Nearest Neighbor + 2-opt."""

    def __init__(self, restarts: int = 4, use_3opt: bool = False) -> None:
        self.restarts = max(1, int(restarts))
        self.use_3opt = bool(use_3opt)

    def solve(self, waypoints: List[NodeId], time_matrix: List[List[Seconds]], *, mode: RouteMode) -> List[int]:
        n = len(waypoints)
        if n <= 1:
            return list(range(n))

        def route_cost(order: List[int], circuit: bool) -> float:
            c = 0.0
            for a, b in zip(order, order[1:]):
                c += time_matrix[a][b]
            if circuit:
                c += time_matrix[order[-1]][order[0]]
            return c

        def two_opt(order: List[int], circuit: bool) -> List[int]:
            improved = True
            best = order[:]
            best_cost = route_cost(best, circuit)
            while improved:
                improved = False
                for i in range(1, n - 2):
                    for k in range(i + 1, n - (0 if circuit else 1)):
                        new_order = best[:i] + list(reversed(best[i:k])) + best[k:]
                        new_cost = route_cost(new_order, circuit)
                        if new_cost + 1e-9 < best_cost:
                            best, best_cost = new_order, new_cost
                            improved = True
            return best

        circuit = mode == RouteMode.VISIT_ALL_CIRCUIT
        origin = 0
        best_order = list(range(n))
        best_cost = float("inf")
        seeds = [origin] + [i for i in range(1, n)]
        seeds = seeds[: self.restarts]
        for seed in seeds:
            unvis = set(range(n))
            unvis.remove(seed)
            order = [seed]
            cur = seed
            while unvis:
                nxt = min(unvis, key=lambda j: time_matrix[cur][j])
                order.append(nxt)
                unvis.remove(nxt)
                cur = nxt
            # asegurar origen en la primera posición
            if origin in order:
                idx = order.index(origin)
                order = order[idx:] + order[:idx]
            order = two_opt(order, circuit)
            cost = route_cost(order, circuit)
            if cost < best_cost:
                best_cost, best_order = cost, order

        return best_order


class RouteSplicer:
    """Reconstruye la ruta final concatenando los tramos par-a-par del orden elegido."""

    def splice(
        self,
        waypoints: List[NodeId],
        visit_order_idx: List[int],
        path_map: Dict[Tuple[int, int], List[NodeId]],
        graph: Graph,
        time_matrix: List[List[Seconds]],
    ) -> List[RouteLeg]:
        legs: List[RouteLeg] = []
        for a_idx, b_idx in zip(visit_order_idx, visit_order_idx[1:]):
            src = waypoints[a_idx]
            dst = waypoints[b_idx]
            path = path_map[(a_idx, b_idx)]
            seconds = time_matrix[a_idx][b_idx]
            distance_m = _path_distance_m(graph, path)
            legs.append(RouteLeg(src, dst, path, seconds, distance_m, expanded_nodes=0))
        return legs


# ==========================================================
# 9) Concurrencia y profiling
# ==========================================================


@dataclass
class ProfileReport:
    """Reporte de profiling (resumen textual + métricas clave)."""

    text: str
    total_seconds: float
    call_count: int


class Profiler:
    """Wrapper sobre cProfile/tiempos por etapa para mostrar en la UI."""

    def run(self, func: Callable[..., Any], *args, **kwargs) -> ProfileReport:
        pr = cProfile.Profile()
        t0 = time.perf_counter()
        pr.enable()
        func(*args, **kwargs)
        pr.disable()
        total = time.perf_counter() - t0
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats(25)
        return ProfileReport(text=s.getvalue(), total_seconds=total, call_count=int(ps.total_calls))


class ConcurrencyExecutor:
    """Ejecutor simple basado en ThreadPool (suficiente para Streamlit Cloud)."""

    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max_workers

    def map(self, fn: Callable[..., Any], tasks: List[Tuple[tuple, dict]]) -> List[Any]:
        results: List[Any] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = [ex.submit(fn, *args, **kwargs) for args, kwargs in tasks]
            for f in as_completed(futs):
                results.append(f.result())
        return results


# ==========================================================
# 10) Servicio orquestador de ruteo
# ==========================================================


class RoutingService:
    """Orquesta el pipeline de ruteo single y multi-destino.

    Flujo multi-destino:
        1) (time_matrix, path_map) con PairwiseDistanceService (batching + cache/memoización).
        2) Elegir solver (Held-Karp o Heurístico) según tamaño y modo.
        3) Splice de tramos para la ruta final.
        4) Armar RouteResult con métricas y metadata.
    """

    def __init__(
        self,
        graph: Graph,
        traffic: TrafficModel,
        heuristic: Heuristic,
        pairwise_service: PairwiseDistanceService,
        solver_exact: MultiStopSolver,
        solver_heur: MultiStopSolver,
        splicer: RouteSplicer,
    ) -> None:
        self.graph = graph
        self.traffic = traffic
        self.heuristic = heuristic
        self.pairwise = pairwise_service
        self.solver_exact = solver_exact
        self.solver_heur = solver_heur
        self.splicer = splicer

        # Routers para single
        self.router_dijkstra = DijkstraRouter()
        self.router_astar = AStarRouter(default_heuristic=heuristic)

    # ---- Single (par-a-par) ----
    def route_single(
        self,
        src: NodeId,
        dst: NodeId,
        *,
        hour: int,
        algorithm: Algorithm = Algorithm.ASTAR,
    ) -> RouteLeg:
        """Devuelve un tramo óptimo entre src y dst (atajo para UI y tests)."""
        if algorithm == Algorithm.DIJKSTRA or algorithm == Algorithm.BFS:
            leg, _ = self.router_dijkstra.route(self.graph, src, dst, hour=hour, traffic=self.traffic)
        else:
            leg, _ = self.router_astar.route(self.graph, src, dst, hour=hour, traffic=self.traffic, heuristic=self.heuristic)
        return leg

    # ---- Multi-destino ----
    def route(self, request: RouteRequest) -> RouteResult:
        waypoints = [request.origin] + list(request.destinations)
        time_matrix, path_map = self.pairwise.compute_matrix(
            self.graph,
            waypoints,
            hour=request.hour,
            algorithm=request.algorithm,
            traffic=self.traffic,
            heuristic=self.heuristic,
        )
        n = len(waypoints)
        # elegir solver
        if request.mode == RouteMode.FIXED_ORDER:
            visit_order_idx = list(range(n))
        else:
            if n <= 13:  # exacto hasta ~12-14
                visit_order_idx = self.solver_exact.solve(waypoints, time_matrix, mode=request.mode)
                alg_sum = f"{request.algorithm.value} + Held-Karp"
            else:
                visit_order_idx = self.solver_heur.solve(waypoints, time_matrix, mode=request.mode)
                alg_sum = f"{request.algorithm.value} + NN/2opt"
        # si es FIXED_ORDER y n>1, usar heurístico para pequeños ajustes? dejamos orden fijo.
            alg_sum = locals().get("alg_sum", f"{request.algorithm.value}")

        legs = self.splicer.splice(waypoints, visit_order_idx, path_map, self.graph, time_matrix)
        total_seconds = sum(l.seconds for l in legs)
        total_distance_m = sum(l.distance_m for l in legs)
        visit_order = [waypoints[i] for i in visit_order_idx]
        return RouteResult(visit_order, legs, total_seconds, total_distance_m, algorithm_summary=alg_sum)


# ==========================================================
# 11) Helpers matemáticos/geo
# ==========================================================


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia Haversine aproximada en km (para heurísticas y generación de grafo)."""
    r = 6371.0  # km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return r * c


# ==========================================================
# 12) Tests unitarios
# ==========================================================


class _BaseFixture:
    @staticmethod
    def make_small_graph() -> Graph:
        return Graph.build_small_town(seed=7, blocks_x=8, blocks_y=8, spacing_m=100.0)

    @staticmethod
    def make_engine_components(graph: Graph):
        traffic = HistoricalTrafficModel()
        heuristic = HybridConservativeHeuristic(LearnedHistoricalHeuristic(), GeoLowerBoundHeuristic())
        route_cache = RouteCache()
        ssp_memo = SSSPMemo()
        psvc = PairwiseDistanceService(AStarRouter(heuristic), DijkstraRouter(), route_cache, ssp_memo, max_workers=4)
        return traffic, heuristic, psvc


class TestGeo(unittest.TestCase):
    def test_haversine_symmetry(self):
        a = haversine_km(-31.4, -64.2, -31.41, -64.19)
        b = haversine_km(-31.41, -64.19, -31.4, -64.2)
        self.assertAlmostEqual(a, b, places=9)


class TestGraph(unittest.TestCase):
    def test_build_small_town(self):
        g = _BaseFixture.make_small_graph()
        nodes = sum(1 for _ in g.iter_nodes())
        edges = sum(1 for _ in g.iter_edges())
        self.assertGreater(nodes, 0)
        self.assertGreater(edges, 0)
        # Debe haber alguna arista de un solo sentido
        self.assertTrue(any(e.one_way for _, e in g.iter_edges()))


class TestRouters(unittest.TestCase):
    def test_astar_equals_dijkstra(self):
        g = _BaseFixture.make_small_graph()
        traffic, heuristic, _ = _BaseFixture.make_engine_components(g)
        # Tomar dos nodos razonablemente apartados
        src, dst, hour = 0, 62, 8
        leg_d, _ = DijkstraRouter().route(g, src, dst, hour=hour, traffic=traffic)
        leg_a, _ = AStarRouter(heuristic).route(g, src, dst, hour=hour, traffic=traffic, heuristic=heuristic)
        # Mismo costo óptimo (tiempo); A* puede expandir menos nodos
        self.assertAlmostEqual(leg_d.seconds, leg_a.seconds, places=6)
        self.assertGreaterEqual(leg_d.distance_m, 0.0)
        self.assertGreaterEqual(leg_a.distance_m, 0.0)


class TestPairwise(unittest.TestCase):
    def test_pairwise_matrix(self):
        g = _BaseFixture.make_small_graph()
        traffic, heuristic, psvc = _BaseFixture.make_engine_components(g)
        waypoints = [0, 7, 56, 63]
        tm, pm = psvc.compute_matrix(g, waypoints, hour=8, algorithm=Algorithm.ASTAR, traffic=traffic, heuristic=heuristic)
        self.assertEqual(len(tm), len(waypoints))
        self.assertTrue(all(len(row) == len(waypoints) for row in tm))
        # caminos i->i son triviales
        for i in range(len(waypoints)):
            self.assertEqual(tm[i][i], 0.0)
            self.assertEqual(pm[(i, i)], [waypoints[i]])


class TestSolvers(unittest.TestCase):
    def test_held_karp_small(self):
        g = _BaseFixture.make_small_graph()
        traffic, heuristic, psvc = _BaseFixture.make_engine_components(g)
        waypoints = [0, 9, 56, 63, 7]
        tm, pm = psvc.compute_matrix(g, waypoints, hour=8, algorithm=Algorithm.DIJKSTRA, traffic=traffic, heuristic=heuristic)
        order_idx = HeldKarpExact().solve(waypoints, tm, mode=RouteMode.VISIT_ALL_OPEN)
        self.assertEqual(order_idx[0], 0)  # origen fijo
        self.assertEqual(len(order_idx), len(waypoints))

    def test_nn_2opt(self):
        g = _BaseFixture.make_small_graph()
        traffic, heuristic, psvc = _BaseFixture.make_engine_components(g)
        waypoints = [0, 5, 10, 18, 23, 40]
        tm, pm = psvc.compute_matrix(g, waypoints, hour=18, algorithm=Algorithm.ASTAR, traffic=traffic, heuristic=heuristic)
        order_idx = HeuristicRoute(restarts=3).solve(waypoints, tm, mode=RouteMode.VISIT_ALL_CIRCUIT)
        self.assertEqual(order_idx[0], 0)  # suele rotarse al origen
        self.assertEqual(len(order_idx), len(waypoints))


class TestRoutingService(unittest.TestCase):
    def test_route_end_to_end(self):
        g = _BaseFixture.make_small_graph()
        traffic = HistoricalTrafficModel()
        heuristic = HybridConservativeHeuristic(LearnedHistoricalHeuristic(), GeoLowerBoundHeuristic())
        route_cache = RouteCache()
        ssp_memo = SSSPMemo()
        psvc = PairwiseDistanceService(AStarRouter(heuristic), DijkstraRouter(), route_cache, ssp_memo, max_workers=4)
        service = RoutingService(
            g,
            traffic,
            heuristic,
            psvc,
            HeldKarpExact(),
            HeuristicRoute(restarts=2),
            RouteSplicer(),
        )
        req = RouteRequest(origin=0, destinations=[7, 56, 63], hour=8, mode=RouteMode.VISIT_ALL_OPEN, algorithm=Algorithm.ASTAR)
        res = service.route(req)
        self.assertGreater(res.total_seconds, 0.0)
        self.assertEqual(res.visit_order[0], req.origin)
        self.assertEqual(len(res.legs), len(res.visit_order) - 1)


# ----------------------------------------------------------
# Soporte para ejecutar tests con `python logica.py -m test`
# ----------------------------------------------------------
if __name__ == "__main__":
    import sys

    if "-m" in sys.argv and "test" in sys.argv:
        unittest.main(argv=[sys.argv[0]])
    else:
        # Smoke test rápido
        g = Graph.build_small_town(seed=5, blocks_x=8, blocks_y=8)
        traffic = HistoricalTrafficModel()
        heuristic = HybridConservativeHeuristic(LearnedHistoricalHeuristic(), GeoLowerBoundHeuristic())
        route_cache = RouteCache()
        ssp_memo = SSSPMemo()
        psvc = PairwiseDistanceService(AStarRouter(heuristic), DijkstraRouter(), route_cache, ssp_memo)
        service = RoutingService(
            g,
            traffic,
            heuristic,
            psvc,
            HeldKarpExact(),
            HeuristicRoute(restarts=2),
            RouteSplicer(),
        )
        req = RouteRequest(origin=0, destinations=[7, 56, 63], hour=8, mode=RouteMode.VISIT_ALL_OPEN)
        out = service.route(req)
        print(
            f"Ruta con {len(out.legs)} tramos, ETA={out.total_seconds:.1f}s, Dist={out.total_distance_m/1000:.2f} km, Alg={out.algorithm_summary}"
        )
