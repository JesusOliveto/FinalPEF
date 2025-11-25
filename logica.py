# -*- coding: utf-8 -*-
"""
logica.py — Motor de ruteo para “Pueblito: Rutas Inteligentes”
Foco: creación de rutas en una red hardcodeada que imita Jesús María (Córdoba).
- Algoritmos: Dijkstra, A*, BFS (opcional).
- Tráfico histórico por clase vial (PRIMARY / COLLECTOR / RESIDENTIAL).
- Memoización SSSP + LRU de tramos; matriz par-a-par concurrente.
- Solvers multi-stop: Held-Karp (DP) y heurístico (NN + 2-opt).
- Generador iter_path_edges para recorrer aristas de un camino.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Iterable, Iterator, Optional, Any, Callable, Mapping
from collections import defaultdict, OrderedDict
from math import radians, sin, cos, asin, sqrt
from concurrent.futures import ThreadPoolExecutor, as_completed
import cProfile
import pstats
import io
import time
import tracemalloc


# ==========================================================
# Tipos y enums
# ==========================================================
class Algorithm(str, Enum):
    """Algoritmo base para el cálculo par-a-par.

    Valores:
    - ASTAR: A* con heurística geográfica de tiempo optimista.
    - DIJKSTRA: Búsqueda de costo uniforme sobre tiempos de viaje.
    - BFS: Búsqueda en anchura (camino no ponderado; tiempos se suman a posteriori).
    """
    ASTAR = "astar"
    DIJKSTRA = "dijkstra"
    BFS = "bfs"  # útil para grafos no ponderados o depuración


class RouteMode(str, Enum):
    """Modo global de la ruta multi-destino.

    - VISIT_ALL_OPEN: ruta abierta (no vuelve al origen).
    - VISIT_ALL_CIRCUIT: circuito (vuelve al origen y se muestra explícitamente).
    - FIXED_ORDER: respeta el orden provisto (no re-optimiza).
    """
    VISIT_ALL_OPEN = "visit_all_open"
    VISIT_ALL_CIRCUIT = "visit_all_circuit"
    FIXED_ORDER = "fixed_order"


class RoadClass(str, Enum):
    """Clasificación vial de una arista (calle/carretera)."""
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
    """Nodo del grafo (un punto geográfico).

    Atributos:
    - id: identificador entero único.
    - lat, lon: coordenadas geográficas en grados.
    """
    id: NodeId
    lat: float
    lon: float


@dataclass
class Edge:
    """Arista dirigida entre nodos con atributos viales.

    Atributos:
    - to: id de nodo destino.
    - distance_m: longitud en metros.
    - road_class: clase vial (afecta congestión/velocidad).
    - freeflow_kmh: velocidad libre (km/h).
    - one_way: indica si es de un solo sentido.
    """
    to: NodeId
    distance_m: Meters
    road_class: RoadClass
    freeflow_kmh: KmPerHour
    one_way: bool = False


class Graph:
    """Grafo dirigido con nodos georreferenciados y aristas con atributos viales."""

    def __init__(self) -> None:
        """Crea un grafo vacío con listas de adyacencia."""
        self.nodes: Dict[NodeId, Node] = {}
        self.adj: Dict[NodeId, List[Edge]] = defaultdict(list)

    # ---- core
    def add_node(self, node_id: NodeId, lat: float, lon: float) -> None:
        """Agrega un nodo con id y coordenadas geográficas."""
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
        """Agrega una arista dirigida u→v con atributos viales."""
        self.adj[u].append(
            Edge(
                v,
                distance_m,
                road_class,
                freeflow_kmh,
                one_way))

    def add_bidirectional(
        self,
        u: NodeId,
        v: NodeId,
        distance_m: Meters,
        road_class: RoadClass,
        freeflow_kmh: KmPerHour,
    ) -> None:
        """Agrega aristas en ambos sentidos entre u y v con los mismos atributos."""
        self.add_edge(u, v, distance_m, road_class, freeflow_kmh)
        self.add_edge(v, u, distance_m, road_class, freeflow_kmh)

    def neighbors(self, u: NodeId) -> Iterable[Edge]:
        """Itera sobre aristas salientes de u (puede estar vacío)."""
        return self.adj.get(u, [])

    def get_node(self, node_id: NodeId) -> Node:
        """Obtiene el nodo por id. Lanza KeyError si no existe."""
        return self.nodes[node_id]

    def iter_nodes(self) -> Iterator[Node]:
        """Itera todos los nodos del grafo (sin orden particular)."""
        yield from self.nodes.values()

    def iter_edges(self) -> Iterator[Tuple[NodeId, Edge]]:
        """Itera todas las aristas (u, e) del grafo (sin orden particular)."""
        for u, lst in self.adj.items():
            for e in lst:
                yield u, e

    # ---- Jesús María hardcodeado (mantiene tu configuración manual)
    @staticmethod
    def build_jesus_maria_hardcoded() -> "Graph":
        """Construye una red vial estilizada que imita Jesús María.

        Características:
        - 3 parches de grilla (centro rotado ~-23°, sudeste cardinal, noroeste chico rotado)
        - Diagonal primaria SW→NE (RN-9 aprox.) con enlaces
        - Políticas de mano única / doble mano por clase de vía

        Returns:
            Grafo dirigido con nodos (lon/lat) y aristas con clases viales/velocidades.
        """
        g = Graph()

        LAT0 = -30.980556
        LON0 = -64.091944
        Kx = 111000.0 * cos(radians(LAT0))  # m/° lon
        Ky = 111000.0                        # m/° lat

        def to_lonlat(x_m: float, y_m: float) -> Tuple[float, float]:
            """Convierte coordenadas en metros relativos (x,y) a (lon, lat) grados."""
            return (LON0 + x_m / Kx, LAT0 + y_m / Ky)

        def rot_xy(dx: float, dy: float, deg: float) -> Tuple[float, float]:
            """Rota un vector (dx,dy) grados alrededor del origen (convención antihoraria)."""
            if deg == 0.0:
                return dx, dy
            th = radians(deg)
            return dx * cos(th) - dy * sin(th), dx * sin(th) + dy * cos(th)

        # Políticas de doble mano
        BIDIR = {
            RoadClass.PRIMARY: True,
            RoadClass.COLLECTOR: False,
            RoadClass.RESIDENTIAL: False,
        }
        BIDIR_RN9 = True
        BIDIR_CONNECTORS = False

        def add_grid_patch(
            *,
            center_xy_m: Tuple[float, float],
            nx: int, ny: int,
            step_m: float,
            rotation_deg: float,
            primary_mod: int,
            collector_mod: int,
            start_id: int
        ) -> Tuple[List[List[int]], List[int], int]:
            """Agrega un parche de grilla al grafo con rotación y clases viales.

            Devuelve la matriz de ids, la lista de ids de borde y el próximo id disponible.
            """
            cx, cy = center_xy_m
            ids = [[-1] * nx for _ in range(ny)]
            next_id = start_id

            # nodos
            for r in range(ny):
                for c in range(nx):
                    dx = (c - (nx - 1) / 2.0) * step_m
                    dy = (r - (ny - 1) / 2.0) * step_m
                    rx, ry = rot_xy(dx, dy, rotation_deg)
                    X, Y = cx + rx, cy + ry
                    lon, lat = to_lonlat(X, Y)
                    g.add_node(next_id, lat, lon)
                    ids[r][c] = next_id
                    next_id += 1

            v_res, v_col, v_pri = 35.0, 45.0, 65.0

            def class_h(row: int) -> RoadClass:
                """Clase vial para horizontales según la fila."""
                if row % primary_mod == 0:
                    return RoadClass.PRIMARY
                if row % collector_mod == 0:
                    return RoadClass.COLLECTOR
                return RoadClass.RESIDENTIAL

            def class_v(col: int) -> RoadClass:
                """Clase vial para verticales según la columna."""
                if col % primary_mod == 0:
                    return RoadClass.PRIMARY
                if col % collector_mod == 0:
                    return RoadClass.COLLECTOR
                return RoadClass.RESIDENTIAL

            def add_one_way(
                u: int,
                v: int,
                dist: float,
                rc: RoadClass,
                sp: float,
                *,
                is_horizontal: bool,
                r: int,
                    c: int):
                """Agrega una arista de un sentido con patrón alternado por fila/columna."""
                if is_horizontal:
                    if (r % 2) == 0:
                        g.add_edge(u, v, dist, rc, sp, one_way=True)
                    else:
                        g.add_edge(v, u, dist, rc, sp, one_way=True)
                else:
                    if (c % 2) == 0:
                        g.add_edge(u, v, dist, rc, sp, one_way=True)
                    else:
                        g.add_edge(v, u, dist, rc, sp, one_way=True)

            for r in range(ny):
                for c in range(nx):
                    u = ids[r][c]
                    if c + 1 < nx:
                        v = ids[r][c + 1]
                        node_a, node_b = g.get_node(u), g.get_node(v)
                        distance_m = haversine_km(
                            node_a.lat, node_a.lon, node_b.lat, node_b.lon) * 1000.0
                        road_class = class_h(r)
                        speed_kmh = v_pri if road_class == RoadClass.PRIMARY else (
                            v_col if road_class == RoadClass.COLLECTOR else v_res)
                        if BIDIR[road_class]:
                            g.add_bidirectional(
                                u, v, distance_m, road_class, speed_kmh)
                        else:
                            add_one_way(
                                u,
                                v,
                                distance_m,
                                road_class,
                                speed_kmh,
                                is_horizontal=True,
                                r=r,
                                c=c)
                    if r + 1 < ny:
                        v = ids[r + 1][c]
                        node_a, node_b = g.get_node(u), g.get_node(v)
                        distance_m = haversine_km(
                            node_a.lat, node_a.lon, node_b.lat, node_b.lon) * 1000.0
                        road_class = class_v(c)
                        speed_kmh = v_pri if road_class == RoadClass.PRIMARY else (
                            v_col if road_class == RoadClass.COLLECTOR else v_res)
                        if BIDIR[road_class]:
                            g.add_bidirectional(
                                u, v, distance_m, road_class, speed_kmh)
                        else:
                            add_one_way(
                                u,
                                v,
                                distance_m,
                                road_class,
                                speed_kmh,
                                is_horizontal=False,
                                r=r,
                                c=c)

            border: List[int] = []
            for r in range(ny):
                for c in range(nx):
                    if r in (0, ny - 1) or c in (0, nx - 1):
                        border.append(ids[r][c])
            return ids, border, next_id

        def connect_nearest(
                border_a: List[int],
                border_b: List[int],
                *,
                k_pairs: int,
                max_dist_m: float,
                road: RoadClass,
                v_kmh: float):
            """Conecta pares más cercanos entre dos bordes de parches respetando un máx. de distancia."""
            pairs: List[Tuple[float, int, int]] = []
            for u in border_a:
                n1 = g.get_node(u)
                for v in border_b:
                    n2 = g.get_node(v)
                    distance_m = haversine_km(
                        n1.lat, n1.lon, n2.lat, n2.lon) * 1000.0
                    if distance_m <= max_dist_m:
                        pairs.append((distance_m, u, v))
            pairs.sort(key=lambda x: x[0])
            used_a, used_b = set(), set()
            cnt = 0
            for d, u, v in pairs:
                if u in used_a or v in used_b:
                    continue
                if BIDIR_CONNECTORS:
                    g.add_bidirectional(u, v, d, road, v_kmh)
                else:
                    if (cnt % 2) == 0:
                        g.add_edge(u, v, d, road, v_kmh, one_way=True)
                    else:
                        g.add_edge(v, u, d, road, v_kmh, one_way=True)
                used_a.add(u)
                used_b.add(v)
                cnt += 1
                if cnt >= k_pairs:
                    break

        next_id = 0

        # Parches de grilla
        _, border_c, next_id = add_grid_patch(center_xy_m=(
            0.0, 0.0), nx=10, ny=13, step_m=115.0, rotation_deg=-23, primary_mod=4, collector_mod=2, start_id=next_id)
        _, border_se, next_id = add_grid_patch(center_xy_m=(
            850.0, -550.0), nx=5, ny=16, step_m=100.0, rotation_deg=0.0, primary_mod=4, collector_mod=2, start_id=next_id)
        _, border_nw, next_id = add_grid_patch(center_xy_m=(-900.0, 600.0), nx=8, ny=6,
                                               step_m=115.0, rotation_deg=-22, primary_mod=4, collector_mod=2, start_id=next_id)

        # Diagonal RN-9 SW→NE
        rn9_xy = [(-2000, -1300), (-1200, -800), (-400, -300),
                  (0, 0), (600, 350), (1200, 800), (1900, 1250)]
        v_pri = 70.0
        prev_id = None
        rn9_ids: List[int] = []
        for x, y in rn9_xy:
            lon, lat = to_lonlat(x, y)
            g.add_node(next_id, lat, lon)
            rn9_ids.append(next_id)
            if prev_id is not None:
                n1, n2 = g.get_node(prev_id), g.get_node(next_id)
                dist = haversine_km(n1.lat, n1.lon, n2.lat, n2.lon) * 1000.0
                if BIDIR_RN9:
                    g.add_bidirectional(
                        prev_id, next_id, dist, RoadClass.PRIMARY, v_pri)
                else:
                    g.add_edge(
                        prev_id,
                        next_id,
                        dist,
                        RoadClass.PRIMARY,
                        v_pri,
                        one_way=True)
            prev_id = next_id
            next_id += 1

        # Enlaces RN-9 → parches
        def attach_poly_to_patch(
                poly_ids: List[int],
                border: List[int],
                *,
                every: int,
                max_dist_m: float):
            """Conecta un polígono (RN-9) con un borde de parche cada N nodos si la distancia lo permite."""
            for i, pid in enumerate(poly_ids):
                if i % every != 0:
                    continue
                pn = g.get_node(pid)
                best, best_d = None, 1e18
                for b in border:
                    bn = g.get_node(b)
                    d = haversine_km(pn.lat, pn.lon, bn.lat, bn.lon) * 1000.0
                    if d < best_d:
                        best_d, best = d, b
                if best is not None and best_d <= max_dist_m:
                    if BIDIR_CONNECTORS:
                        g.add_bidirectional(
                            pid, best, best_d, RoadClass.COLLECTOR, 50.0)
                    else:
                        if (i % 2) == 0:
                            g.add_edge(
                                pid, best, best_d, RoadClass.COLLECTOR, 50.0, one_way=True)
                        else:
                            g.add_edge(
                                best, pid, best_d, RoadClass.COLLECTOR, 50.0, one_way=True)

        attach_poly_to_patch(rn9_ids, border_c, every=1, max_dist_m=220.0)
        attach_poly_to_patch(rn9_ids, border_se, every=1, max_dist_m=220.0)
        attach_poly_to_patch(rn9_ids, border_nw, every=2, max_dist_m=250.0)

        # Conectores entre parches
        connect_nearest(
            border_c,
            border_se,
            k_pairs=10,
            max_dist_m=240.0,
            road=RoadClass.COLLECTOR,
            v_kmh=45.0)
        connect_nearest(
            border_c,
            border_nw,
            k_pairs=6,
            max_dist_m=260.0,
            road=RoadClass.COLLECTOR,
            v_kmh=45.0)

        return g


# ==========================================================
# Costos / Tráfico
# ==========================================================
class TrafficModel:
    def travel_time_seconds(self, edge: Edge, *, hour: int) -> Seconds:
        """Devuelve el tiempo de viaje estimado (en segundos) para recorrer una arista a una hora dada.

        Debe ser implementado por subclases que modelen la congestión o el costo temporal.
        """
        raise NotImplementedError


class HistoricalTrafficModel(TrafficModel):
    def __init__(
        self,
        factor_by_hour_and_class: Optional[Dict[int, Dict[RoadClass, float]]] = None,
        *,
        driver_max_kmh: float = 40.0,
    ) -> None:
        """Modelo de tráfico histórico con límite de velocidad del vehículo.

        Args:
            factor_by_hour_and_class: Mapa {hora -> {clase_vial -> factor_congestión}}.
            driver_max_kmh: Velocidad máxima realista del conductor (km/h). Por defecto 40.
        """
        self.factor = factor_by_hour_and_class or self._default()
        self.driver_max_kmh = max(1.0, float(driver_max_kmh))

    @staticmethod
    def _default() -> Dict[int, Dict[RoadClass, float]]:
        base: Dict[int, Dict[RoadClass, float]] = {}
        for h in range(24):
            if 7 <= h <= 9 or 17 <= h <= 19:
                base[h] = {
                    RoadClass.RESIDENTIAL: 1.2,
                    RoadClass.COLLECTOR: 1.35,
                    RoadClass.PRIMARY: 1.5}
            else:
                base[h] = {
                    RoadClass.RESIDENTIAL: 1.05,
                    RoadClass.COLLECTOR: 1.1,
                    RoadClass.PRIMARY: 1.15}
        return base

    def travel_time_seconds(self, edge: Edge, *, hour: int) -> Seconds:
        """Calcula el tiempo de viaje ajustado por congestión y límite del conductor.

        Se toma la velocidad efectiva como el mínimo entre la velocidad libre de la arista
        y la velocidad máxima del conductor (p.ej., 40 km/h), y luego se multiplica por
        el factor de congestión correspondiente a la hora y clase vial.
        """
        factor = self.factor.get(hour, {}).get(edge.road_class, 1.1)
        effective_kmh = max(min(edge.freeflow_kmh, self.driver_max_kmh), 1e-6)
        hours = edge.distance_m / 1000.0 / effective_kmh
        return hours * 3600.0 * factor


# ==========================================================
# Requests / Results / util
# ==========================================================
@dataclass
class RouteRequest:
    """Solicitud de ruteo de origen a múltiples destinos.

    Atributos:
    - origin: id de nodo inicio.
    - destinations: ids de nodos a visitar en el recorrido.
    - hour: hora del día (0..23) para el modelo de tráfico.
    - mode: modo de ruta (abierta, circuito, fijo).
    - algorithm: algoritmo base para calcular tramos.
    """
    origin: NodeId
    destinations: List[NodeId]
    hour: int
    mode: RouteMode
    algorithm: Algorithm = Algorithm.ASTAR


@dataclass
class RouteLeg:
    """Tramo elemental de una ruta (de src a dst por un path).

    - path: secuencia de nodos desde src a dst (incluye extremos).
    - seconds: tiempo total estimado (segundos).
    - distance_m: distancia acumulada (metros).
    - expanded_nodes: nodos expandidos por el algoritmo de búsqueda (telemetría).
    """
    src: NodeId
    dst: NodeId
    path: List[NodeId]
    seconds: Seconds
    distance_m: Meters
    expanded_nodes: int


@dataclass
class RouteResult:
    """Resultado completo de una ruta multi-destino."""
    visit_order: List[NodeId]
    legs: List[RouteLeg]
    total_seconds: Seconds
    total_distance_m: Meters
    algorithm_summary: str
    cache_hits: int = 0
    cache_misses: int = 0


def iter_path_edges(path: List[NodeId]) -> Iterable[Tuple[NodeId, NodeId]]:
    """Itera pares consecutivos (u, v) a lo largo del path.

    Args:
        path: lista de ids de nodos que define un camino.
    Yields:
        Tuplas (u, v) para cada segmento del camino.
    """
    for i in range(len(path) - 1):
        yield path[i], path[i + 1]


# ==========================================================
# Routers
# ==========================================================
class SearchStats:
    """Estadísticas simples de una búsqueda (telemetría)."""

    def __init__(self) -> None:
        self.expanded_nodes = 0
        self.queue_pushes = 0
        self.queue_pops = 0


def _reconstruct(parent: Dict[NodeId,
                              Optional[NodeId]],
                 src: NodeId,
                 dst: NodeId) -> List[NodeId]:
    """Reconstruye el camino src→dst usando punteros padre; devuelve [] si no hay ruta."""
    cur = dst
    chain = [cur]
    while cur is not None and cur != src:
        cur = parent.get(cur)
        if cur is None:
            return []
        chain.append(cur)
    chain.reverse()
    return chain


# ====== helper para batching con generator (yield) ======
def batcher(lista, batch_size: int):
    """
    Generador que produce sublistas de tamaño batch_size.
    Cumple el requisito de usar yield y el patrón pedido.
    """
    for i in range(0, len(lista), batch_size):
        yield lista[i:i + batch_size]


# ====== Dijkstra con batching + threads + memo ======
class DijkstraRouter:
    """Dijkstra con batching de vecinos, concurrencia y memoización.

    - Batching: procesa vecinos en lotes para amortizar overhead.
    - Concurrencia: paraleliza cálculo de costos de aristas dentro de cada lote.
    - Memoización: cachea resultados (RouteLeg, SearchStats) por clave de parámetros.
    """

    def __init__(self, *, batch_size: int = 64, max_workers: int = 4) -> None:
        """
        batch_size: cuántos edges procesar por lote (batch).
        max_workers: grado de paralelismo para calcular costos de aristas.
        """
        self.batch_size = max(1, int(batch_size))
        self.max_workers = max(1, int(max_workers))
        # Memoización por (grafo, modelo_tráfico, src, dst, hora)
        # Guarda el resultado completo (RouteLeg, SearchStats)
        self._memo: Dict[Tuple[int, int, NodeId, NodeId, int],
                         Tuple[RouteLeg, "SearchStats"]] = {}

    def route(
        self,
        graph: Graph,
        src: NodeId,
        dst: NodeId,
        *,
        hour: int,
        traffic: TrafficModel,
    ) -> Tuple[RouteLeg, "SearchStats"]:
        """
        Dijkstra clásico, pero:
        - Calcula relajaciones en batches (batcher + yield).
        - Paraleliza el cálculo de tiempos de viaje por arista con ThreadPoolExecutor.
        - Aplica memoización por (id(graph), id(traffic), src, dst, hour).
        """
        # ---------- memoización ----------
        memo_key = (id(graph), id(traffic), src, dst, hour)
        cached = self._memo.get(memo_key)
        if cached is not None:
            return cached  # (RouteLeg, SearchStats)

        import heapq
        from collections import defaultdict
        from concurrent.futures import ThreadPoolExecutor

        dist: Dict[NodeId, float] = defaultdict(lambda: float("inf"))
        parent: Dict[NodeId, Optional[NodeId]] = {src: None}
        stats = SearchStats()

        dist[src] = 0.0
        pq: List[Tuple[float, NodeId]] = [(0.0, src)]
        stats.queue_pushes += 1

        # Un único pool para toda la búsqueda (evita overhead por batch)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            while pq:
                d, u = heapq.heappop(pq)
                stats.queue_pops += 1
                if d > dist[u]:
                    continue
                stats.expanded_nodes += 1
                if u == dst:
                    break

                # Tomamos los vecinos y los procesamos en LOTES (batching)
                vecinos = list(graph.neighbors(u))
                for lote in batcher(vecinos, self.batch_size):
                    # Función de relajación (pura) que corre en paralelo
                    def _relax_edge(e: Edge) -> Tuple[NodeId, float]:
                        # Calcula el costo alternativo sin tocar estado
                        # compartido
                        alt = d + traffic.travel_time_seconds(e, hour=hour)
                        return e.to, alt

                    # Ejecutamos el cálculo de todos los alt del batch en
                    # threads
                    resultados = list(pool.map(_relax_edge, lote))

                    # Aplicamos los resultados secuencialmente (evita race
                    # conditions)
                    for v, alt in resultados:
                        if alt < dist[v]:
                            dist[v] = alt
                            parent[v] = u
                            heapq.heappush(pq, (alt, v))
                            stats.queue_pushes += 1

        path = _reconstruct(parent, src, dst)
        seconds = dist[dst]
        distance_m = _path_distance_m(graph, path)
        leg = RouteLeg(
            src,
            dst,
            path,
            seconds,
            distance_m,
            stats.expanded_nodes)

        # Guardamos en memo y devolvemos
        out = (leg, stats)
        self._memo[memo_key] = out
        return out


class AStarRouter:
    def __init__(self, vmax_kmh_for_h: KmPerHour = 50.0) -> None:
        """A* con heurística inline: tiempo geo optimista usando vmax_kmh_for_h."""
        self.vmax_h = max(1e-6, float(vmax_kmh_for_h))

    def route(self, graph: Graph, src: NodeId, dst: NodeId, *, hour: int,
              traffic: TrafficModel) -> Tuple[RouteLeg, SearchStats]:
        """Calcula un tramo con A* usando costo temporal y heurística de tiempo geográfico.

        La heurística estima el tiempo directo (gran círculo) con velocidad `vmax_h`.
        """
        import heapq
        g_score: Dict[NodeId, float] = defaultdict(lambda: float("inf"))
        f_score: Dict[NodeId, float] = defaultdict(lambda: float("inf"))
        parent: Dict[NodeId, Optional[NodeId]] = {src: None}
        stats = SearchStats()

        def est(nid: NodeId) -> float:
            n, g2 = graph.get_node(nid), graph.get_node(dst)
            dist_km = haversine_km(n.lat, n.lon, g2.lat, g2.lon)
            return (dist_km / self.vmax_h) * 3600.0

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
                tentative = g_score[u] + \
                    traffic.travel_time_seconds(e, hour=hour)
                if tentative < g_score[e.to]:
                    parent[e.to] = u
                    g_score[e.to] = tentative
                    f_score[e.to] = tentative + est(e.to)
                    heapq.heappush(pq, (f_score[e.to], e.to))
                    stats.queue_pushes += 1

        path = _reconstruct(parent, src, dst)
        seconds = g_score[dst]
        distance_m = _path_distance_m(graph, path)
        return RouteLeg(src, dst, path, seconds, distance_m,
                        stats.expanded_nodes), stats


class BFSRouter:
    """BFS para grafos no ponderados (o ponderar todo como 1). Útil como baseline/depuración."""

    def route(self, graph: Graph, src: NodeId, dst: NodeId, *, hour: int,
              traffic: TrafficModel) -> Tuple[RouteLeg, SearchStats]:
        """Calcula un camino por BFS y luego suma tiempos reales sobre ese camino."""
        from collections import deque
        q = deque([src])
        parent: Dict[NodeId, Optional[NodeId]] = {src: None}
        stats = SearchStats()
        stats.queue_pushes += 1
        while q:
            u = q.popleft()
            stats.queue_pops += 1
            stats.expanded_nodes += 1
            if u == dst:
                break
            for e in graph.neighbors(u):
                if e.to not in parent:
                    parent[e.to] = u
                    q.append(e.to)
                    stats.queue_pushes += 1
        path = _reconstruct(parent, src, dst)
        # tiempo estimado: suma tiempos reales para no romper métricas
        seconds = sum(
            traffic.travel_time_seconds(
                next(
                    e for e in graph.neighbors(u) if e.to == v),
                hour=hour) for u,
            v in iter_path_edges(path)) if path else float("inf")
        dist_m = _path_distance_m(graph, path)
        return RouteLeg(src, dst, path, seconds, dist_m,
                        stats.expanded_nodes), stats


# ==========================================================
# Caches y par-a-par (batching)
# ==========================================================
class LRUCache:
    """Cache LRU simple basada en OrderedDict.

    No es thread-safe por sí misma; su uso concurrente debe coordinarse externamente.
    """

    def __init__(self, capacity: int = 1024) -> None:
        self.capacity = capacity
        self._store: OrderedDict[Tuple[Any, ...], Any] = OrderedDict()

    def get(self, key: Tuple[Any, ...]) -> Optional[Any]:
        """Obtiene un valor moviendo la clave al final (más reciente)."""
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def set(self, key: Tuple[Any, ...], value: Any) -> None:
        """Inserta/actualiza un valor y recorta si excede la capacidad."""
        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)


class RouteCache:
    """Capa de caché para tramos (RouteLeg) con política LRU."""

    def __init__(self, capacity: int = 2048) -> None:
        self._lru = LRUCache(capacity)

    def get(self, key: Tuple[Any, ...]) -> Optional[RouteLeg]:
        """Devuelve RouteLeg cacheado si existe para la clave dada."""
        return self._lru.get(key)

    def set(self, key: Tuple[Any, ...], leg: RouteLeg) -> None:
        """Guarda un RouteLeg asociado a la clave dada."""
        self._lru.set(key, leg)


class PairwiseDistanceService:
    """Calcula matriz de tiempos y paths par-a-par entre waypoints.

    - Selecciona router según `algorithm` (A*, Dijkstra o BFS).
    - Aplica concurrencia por pares (optimizable a SSSP por origen).
    - Usa `RouteCache` para memoizar tramos (src, dst, hora, algoritmo).
    """

    def __init__(
            self,
            router_astar: AStarRouter,
            router_dijkstra: DijkstraRouter,
            route_cache: RouteCache,
            *,
            max_workers: int = 4) -> None:
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
        """Construye la matriz de tiempos y el mapa de paths para los `waypoints`.

        Args:
            graph: grafo vial.
            waypoints: lista de nodos a considerar (incluye origen en índice 0).
            hour: hora del día (afecta costos del tráfico).
            algorithm: algoritmo base para cada tramo.
            traffic: modelo de tráfico.
        Returns:
            time_matrix: matriz NxN de segundos entre cada par (i→j).
            path_map: dict {(i, j) -> path de nodos}.
        """
        n = len(waypoints)
        time_matrix: List[List[Seconds]] = [
            [float("inf")] * n for _ in range(n)]
        path_map: Dict[Tuple[int, int], List[NodeId]] = {}

        def route_pair(i: int,
                       j: int) -> Tuple[int,
                                        int,
                                        Seconds,
                                        List[NodeId],
                                        RouteLeg]:
            if i == j:
                return i, j, 0.0, [
                    waypoints[i]], RouteLeg(
                    waypoints[i], waypoints[j], [
                        waypoints[i]], 0.0, 0.0, 0)
            src, dst = waypoints[i], waypoints[j]
            cache_key = ("LEG", src, dst, hour, algorithm.value)
            cached = self.route_cache.get(cache_key)
            if cached:
                return i, j, cached.seconds, cached.path, cached
            # seleccionar router
            if algorithm == Algorithm.ASTAR:
                leg, _ = self.router_astar.route(
                    graph, src, dst, hour=hour, traffic=traffic)
            elif algorithm == Algorithm.DIJKSTRA:
                leg, _ = self.router_dijkstra.route(
                    graph, src, dst, hour=hour, traffic=traffic)
            else:
                # BFS directo
                bfs = BFSRouter()
                leg, _ = bfs.route(graph, src, dst, hour=hour, traffic=traffic)
            self.route_cache.set(cache_key, leg)
            return i, j, leg.seconds, leg.path, leg

        # Ejecutar en paralelo todos los pares
        tasks = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            for i in range(n):
                for j in range(n):
                    tasks.append(ex.submit(route_pair, i, j))
            for fut in as_completed(tasks):
                i, j, seconds, path, _leg = fut.result()
                time_matrix[i][j] = seconds
                path_map[(i, j)] = path
        return time_matrix, path_map


# ==========================================================
# Solvers y splicer
# ==========================================================
class MultiStopSolver:
    """Interfaz de solver de orden de visita (TSP/VRP simple)."""

    def solve(self,
              waypoints: List[NodeId],
              time_matrix: List[List[Seconds]],
              *,
              mode: RouteMode) -> List[int]:
        raise NotImplementedError


class HeldKarpExact(MultiStopSolver):
    """Solver exacto Held-Karp (DP) para TSP con origen fijo.

    - Complejidad O(n^2 2^n), adecuado para n<=13 aprox.
    - Soporta modo circuito agregando coste de regreso al origen en el cierre.
    """

    def solve(self,
              waypoints: List[NodeId],
              time_matrix: List[List[Seconds]],
              *,
              mode: RouteMode) -> List[int]:
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
            if not (mask & (1 << origin)):
                continue
            for j in range(n):
                if not (mask & (1 << j)):
                    continue
                cj = dp[mask][j]
                if cj == inf:
                    continue
                for k in range(n):
                    if mask & (1 << k):
                        continue
                    nm = mask | (1 << k)
                    cand = cj + time_matrix[j][k]
                    if cand < dp[nm][k]:
                        dp[nm][k] = cand
                        prv[nm][k] = j
        full = ALL - 1
        if mode == RouteMode.VISIT_ALL_CIRCUIT:
            best, last = float("inf"), -1
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
            if p == -1:
                break
            mask &= ~(1 << last)
            last = p
        order.append(0)
        order.reverse()
        return order


class HeuristicRoute(MultiStopSolver):
    """Heurística NN + 2-opt con múltiples reinicios para TSP sencillo."""

    def __init__(self, restarts: int = 4) -> None:
        self.restarts = max(1, int(restarts))

    def solve(self,
              waypoints: List[NodeId],
              time_matrix: List[List[Seconds]],
              *,
              mode: RouteMode) -> List[int]:
        n = len(waypoints)
        if n <= 1:
            return list(range(n))
        circuit = mode == RouteMode.VISIT_ALL_CIRCUIT

        def route_cost(order: List[int]) -> float:
            c = 0.0
            for a, b in zip(order, order[1:]):
                c += time_matrix[a][b]
            if circuit:
                c += time_matrix[order[-1]][order[0]]
            return c

        def two_opt(order: List[int]) -> List[int]:
            best = order[:]
            bestc = route_cost(best)
            improved = True
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
            un = set(range(n))
            un.remove(s)
            order = [s]
            cur = s
            while un:
                nxt = min(un, key=lambda j: time_matrix[cur][j])
                order.append(nxt)
                un.remove(nxt)
                cur = nxt
            if origin in order:
                idx = order.index(origin)
                order = order[idx:] + order[:idx]
            order = two_opt(order)
            c = route_cost(order)
            if c < best_cost:
                best_cost, best_order = c, order
        return best_order


class RouteSplicer:
    """Ensamblador de tramos individuales en una lista de RouteLegs según un orden dado."""

    def splice(self,
               waypoints: List[NodeId],
               visit_order_idx: List[int],
               path_map: Dict[Tuple[int,
                                    int],
                              List[NodeId]],
               graph: Graph,
               time_matrix: List[List[Seconds]]) -> List[RouteLeg]:
        """Concatena legs según `visit_order_idx` usando `path_map` y `time_matrix`."""
        legs: List[RouteLeg] = []
        for a, b in zip(visit_order_idx, visit_order_idx[1:]):
            src = waypoints[a]
            dst = waypoints[b]
            path = path_map[(a, b)]
            seconds = time_matrix[a][b]
            dist_m = _path_distance_m(graph, path)
            legs.append(RouteLeg(src, dst, path, seconds, dist_m, 0))
        return legs


# ==========================================================
# Servicio de ruteo
# ==========================================================
class RoutingService:
    """Servicio de alto nivel: calcula rutas multi-destino y arma resultados completos."""

    def __init__(
            self,
            graph: Graph,
            traffic: TrafficModel,
            pairwise_service: PairwiseDistanceService,
            solver_exact: MultiStopSolver,
            solver_heur: MultiStopSolver,
            splicer: RouteSplicer) -> None:
        """Inyecta dependencias: grafo, tráfico, servicio par-a-par, solvers y splicer."""
        self.graph = graph
        self.traffic = traffic
        self.pairwise = pairwise_service
        self.solver_exact = solver_exact
        self.solver_heur = solver_heur
        self.splicer = splicer
        self.router_dijkstra = DijkstraRouter()
        self.router_astar = AStarRouter()
        self.router_bfs = BFSRouter()

    def route_single(
            self,
            src: NodeId,
            dst: NodeId,
            *,
            hour: int,
            algorithm: Algorithm = Algorithm.ASTAR) -> RouteLeg:
        """Calcula un único tramo src→dst con el algoritmo indicado."""
        if algorithm == Algorithm.DIJKSTRA:
            leg, _ = self.router_dijkstra.route(
                self.graph, src, dst, hour=hour, traffic=self.traffic)
        elif algorithm == Algorithm.BFS:
            leg, _ = self.router_bfs.route(
                self.graph, src, dst, hour=hour, traffic=self.traffic)
        else:
            leg, _ = self.router_astar.route(
                self.graph, src, dst, hour=hour, traffic=self.traffic)
        return leg

    def route(self, request: RouteRequest) -> RouteResult:
        """Calcula una ruta multi-destino completa según el modo y algoritmo seleccionados."""
        waypoints = [request.origin] + list(request.destinations)
        time_matrix, path_map = self.pairwise.compute_matrix(
            self.graph, waypoints, hour=request.hour, algorithm=request.algorithm, traffic=self.traffic)
        n = len(waypoints)
        if request.mode == RouteMode.FIXED_ORDER:
            order_idx = list(range(n))
            alg = f"{request.algorithm.value}"
        else:
            if n <= 13:
                order_idx = self.solver_exact.solve(
                    waypoints, time_matrix, mode=request.mode)
                alg = f"{request.algorithm.value} + Held-Karp"
            else:
                order_idx = self.solver_heur.solve(
                    waypoints, time_matrix, mode=request.mode)
                alg = f"{request.algorithm.value} + NN/2opt"
        # En modo circuito, añadimos el regreso al origen para que sea visible
        # en el mapa y en métricas
        if request.mode == RouteMode.VISIT_ALL_CIRCUIT and order_idx and order_idx[-1] != order_idx[0]:
            order_idx = order_idx + [order_idx[0]]
        legs = self.splicer.splice(
            waypoints,
            order_idx,
            path_map,
            self.graph,
            time_matrix)
        total_s = sum(l.seconds for l in legs)
        total_m = sum(l.distance_m for l in legs)
        visit_order = [waypoints[i] for i in order_idx]
        return RouteResult(visit_order, legs, total_s, total_m, alg)


# =================== Integración Gemini (orden de visita vía IA) ========
def route_with_gemini(
    service: "RoutingService",
    request: "RouteRequest",
    *,
    api_key: str,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.1,
) -> "RouteResult":
    """
    Usa Gemini para decidir el orden de visita (TSP) a partir de la matriz de tiempos calculada
    por el motor local (A*, Dijkstra o BFS). Luego ensambla los tramos con tus rutas óptimas.

    - Mantiene tus caminos/tiempos por tramo (no inventa trayectos).
    - Devuelve un RouteResult idéntico al flujo normal.
    """
    import json
    import re

    # 1) Calculamos matriz de tiempos y los paths óptimos par-a-par con tu
    # servicio
    waypoints = [request.origin] + list(request.destinations)
    time_matrix, path_map = service.pairwise.compute_matrix(
        service.graph,
        waypoints,
        hour=request.hour,
        algorithm=request.algorithm,
        traffic=service.traffic,
    )
    n = len(waypoints)

    # Mapeo índice->node_id solo a efectos de trazabilidad en la explicación
    idx_to_node = {i: int(waypoints[i]) for i in range(n)}

    # 2) Armamos instrucciones estrictas para obtener SOLO JSON con el orden
    #    (Gemini decide orden minimizando suma de tiempos de la matriz)
    mode = request.mode.value
    circuit = (request.mode == RouteMode.VISIT_ALL_CIRCUIT)
    constraints = (f"- Usar índices 0..{n -
                                        1}\n" f"- Debe comenzar en 0 (origen)\n" +
                   ("- Debe terminar nuevamente en 0 (circuito)\n" if circuit else "- No repetir 0 al final (camino abierto)\n") +
                   "- Incluir cada índice exactamente una vez (excepto repetir 0 si es circuito)\n")

    sys_instr = (
        "You are a routing optimizer. "
        "Given a non-negative time matrix (seconds) for traveling between waypoint indices, "
        "return ONLY a strict JSON object with the visiting order that minimizes total time "
        "under the constraints. Do not include code fences or extra keys.")

    prompt = (
        "TASK: Choose the visiting order that minimizes total travel time.\n"
        f"MODE: {mode}\n"
        f"ALGORITHM_FOR_LEGS: {request.algorithm.value} (precomputed)\n"
        "CONSTRAINTS:\n"
        f"{constraints}\n"
        "INPUT:\n"
        f" - waypoints_index_to_node_id = {json.dumps(idx_to_node, ensure_ascii=False)}\n"
        f" - time_matrix_seconds = {json.dumps(time_matrix)}\n\n"
        "Return ONLY this JSON schema (no markdown, no explanations outside JSON):\n"
        "{\n"
        '  "order_idx": [int, ...],\n'
        '  "reasoning": "one brief sentence"\n'
        "}"
    )

    # 3) Llamamos a Gemini (import lazy para no romper entornos sin la lib)
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model,
            config=types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=sys_instr,
            ),
            contents=prompt,
        )

        raw = getattr(
            resp,
            "text",
            None) or getattr(
            resp,
            "output_text",
            None) or ""
        # Limpieza por si viniera envuelto en ```json ... ```
        raw = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw.strip())
        data = json.loads(raw)
        order_idx = list(map(int, data.get("order_idx", [])))
    except Exception as e:
        # Fallback: usa tu solver local si Gemini falla
        # (mantiene robustness y no rompe el flujo)
        if n <= 13:
            order_idx = service.solver_exact.solve(
                waypoints, time_matrix, mode=request.mode)
            alg_sum = f"{request.algorithm.value} + Held-Karp (fallback)"
        else:
            order_idx = service.solver_heur.solve(
                waypoints, time_matrix, mode=request.mode)
            alg_sum = f"{request.algorithm.value} + NN/2opt (fallback)"
    else:
        # 4) Validaciones suaves de contrato
        def valid_order(idx):
            return (
                isinstance(idx, list)
                and all(isinstance(x, int) and 0 <= x < n for x in idx)
                and (idx[0] == 0)
                and (len(set(idx[1:])) == (n - 1))  # todos los demás únicos
                and ((idx[-1] == 0) if circuit else (idx[-1] != 0))
                and (len(idx) == (n + (1 if circuit else 0)))
            )

        if not valid_order(order_idx):
            # Si Gemini devolvió algo raro, aplicamos fallback local
            if n <= 13:
                order_idx = service.solver_exact.solve(
                    waypoints, time_matrix, mode=request.mode)
                alg_sum = f"{
                    request.algorithm.value} + Held-Karp (fallback-bad-json)"
            else:
                order_idx = service.solver_heur.solve(
                    waypoints, time_matrix, mode=request.mode)
                alg_sum = f"{
                    request.algorithm.value} + NN/2opt (fallback-bad-json)"
        else:
            alg_sum = f"gemini/{model} + {request.algorithm.value} (tramos)"

    # 5) Ensamblamos con tus tramos óptimos ya calculados
    legs = service.splicer.splice(
        waypoints,
        order_idx,
        path_map,
        service.graph,
        time_matrix)
    total_s = sum(l.seconds for l in legs)
    total_m = sum(l.distance_m for l in legs)
    visit_order = [waypoints[i] for i in order_idx]

    return RouteResult(
        visit_order=visit_order,
        legs=legs,
        total_seconds=total_s,
        total_distance_m=total_m,
        algorithm_summary=alg_sum,
    )


# ==========================================================
# Helpers geo / profiling
# ==========================================================
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia de gran círculo en kilómetros entre dos puntos (lat, lon)."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * \
        cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * (2 * asin(sqrt(a)))


def _path_distance_m(graph: Graph, path: List[NodeId]) -> float:
    """Suma las distancias de aristas a lo largo de un path de nodos."""
    d = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        for e in graph.neighbors(u):
            if e.to == v:
                d += e.distance_m
                break
    return d


def run_profiled(fn: Callable, *args, **kwargs) -> str:
    """Perfila una llamada y devuelve un resumen de CPU, tiempo y memoria.

    Este helper ejecuta ``fn(*args, **kwargs)`` y devuelve una cadena de texto
    con un resumen combinado de:

    - Tiempo wall-clock total (segundos) medido con ``time.perf_counter``.
    - Estadísticas de CPU a través de ``cProfile`` (top 20 por ``cumtime``).
    - Uso de memoria (pico) medido con ``tracemalloc``.
    """
    pr = cProfile.Profile()
    tracemalloc.start()
    t0 = time.perf_counter()
    pr.enable()
    fn(*args, **kwargs)
    pr.disable()
    elapsed = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    s = io.StringIO()
    s.write(f"Wall time: {elapsed:.4f} s\n")
    s.write(f"Peak memory: {peak / (1024 * 1024):.3f} MiB\n")
    s.write("\nTop 20 functions by cumulative CPU time (cProfile):\n")
    pstats.Stats(pr, stream=s).sort_stats("cumtime").print_stats(20)
    return s.getvalue()


def profile_call(fn: Callable, *args, **kwargs) -> Tuple[Any, str]:
    """Ejecuta una función y devuelve (resultado, reporte de CPU/tiempo/memoria).

    Args:
        fn: Función a ejecutar.
        *args, **kwargs: Argumentos posicionales y con nombre para la función.

    Returns:
        tuple[result, str]:
            - ``result``: resultado de ``fn(*args, **kwargs)``.
            - ``str``: reporte textual con tiempo wall-clock, pico de memoria
              y top de funciones según ``cProfile`` (ordenadas por ``cumtime``).
    """
    pr = cProfile.Profile()
    tracemalloc.start()
    t0 = time.perf_counter()
    pr.enable()
    result = fn(*args, **kwargs)
    pr.disable()
    elapsed = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    s = io.StringIO()
    s.write(f"Wall time: {elapsed:.4f} s\n")
    s.write(f"Peak memory: {peak / (1024 * 1024):.3f} MiB\n")
    s.write("\nTop 20 functions by cumulative CPU time (cProfile):\n")
    pstats.Stats(pr, stream=s).sort_stats("cumtime").print_stats(20)
    return result, s.getvalue()


class ProfileTimer:
    """Context manager para medir tiempo con perf_counter.

    Uso:
        with ProfileTimer() as t:
            ...
        print(t.elapsed)
    """

    def __enter__(self):
        self._t0 = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self._t0
        return False



