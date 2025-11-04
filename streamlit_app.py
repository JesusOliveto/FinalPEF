# -*- coding: utf-8 -*-
"""
streamlit_app.py — UI para “Pueblito: Rutas Inteligentes”
Foco: calcular mejores rutas en Jesús María (sin OSM).

Cambios solicitados:
- ❌ Eliminado SSSPMemo.
- ❌ Eliminada la flag Edge.one_way (la direccionalidad se deduce por existencia de aristas).
- ✅ El botón “Limpiar destinos” limpia también la selección del multiselect (usa session_state).
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

import streamlit as st
import pydeck as pdk

from logica import (
    Algorithm, Graph, HistoricalTrafficModel,
    AStarRouter, DijkstraRouter, PairwiseDistanceService,
    RouteCache, HeldKarpExact, HeuristicRoute, RouteSplicer,
    RouteLeg, RouteMode, RouteRequest, RoutingService
)

# =================== UI config ===================
st.set_page_config(page_title="Pueblito · Rutas Inteligentes", page_icon="🏘️", layout="wide")

COLOR_ROAD_PRIMARY   = [45, 85, 255]
COLOR_ROAD_COLLECTOR = [92, 112, 177]
COLOR_ROAD_RES       = [150, 160, 180]
COLOR_ROUTE          = [12, 180, 105]
LAT_JM = -30.9859
LON_JM = -64.0947


# =================== Helpers de visualización ===================
def build_road_layer(graph: Graph, *, hour: int, traffic: HistoricalTrafficModel, color_by: str = "congestion") -> pdk.Layer:
    """Construye una PathLayer para las calles del grafo.
    color_by: "class" usa color fijo por clase; "congestion" añade rojo según factor horario.
    """
    traffic_factors = traffic.factors_by_hour()  # {hour: {RoadClass: factor}}
    road_data: List[Dict] = []
    for u, e in graph.iter_edges():
        nu = graph.get_node(u)
        nv = graph.get_node(e.to)
        # ancho por clase
        if e.road_class is e.road_class.PRIMARY:
            base_color = COLOR_ROAD_PRIMARY
            width = 6
        elif e.road_class is e.road_class.COLLECTOR:
            base_color = COLOR_ROAD_COLLECTOR
            width = 4
        else:
            base_color = COLOR_ROAD_RES
            width = 2.5

        if color_by == "congestion":
            factor = traffic_factors.get(hour, {}).get(e.road_class, 1.0)
            red = min(255, int(100 + (factor - 1.0) * 240))
            rgba = [red, base_color[1], base_color[2], 160]
        else:
            rgba = [*base_color, 160]

        road_data.append({
            "path": [[nu.lon, nu.lat], [nv.lon, nv.lat]],
            "width": width,
            "color": rgba,
        })

    return pdk.Layer(
        "PathLayer", data=road_data, get_path="path", get_width="width", get_color="color",
        width_min_pixels=2, pickable=False
    )


def build_route_layer(graph: Graph, route_legs: List[RouteLeg]) -> pdk.Layer:
    """Capa para la ruta óptima resultante."""
    if not route_legs:
        return pdk.Layer("PathLayer", data=[])
    segments = []
    for leg in route_legs:
        path = []
        for nid in leg.path:
            node = graph.get_node(nid)
            path.append([node.lon, node.lat])
        segments.append({"path": path, "width": 8, "color": [*COLOR_ROUTE, 220]})
    return pdk.Layer("PathLayer", data=segments, get_path="path", get_width="width", get_color="color",
                     width_min_pixels=2, pickable=False)


# =================== Servicios base ===================
def make_services(*, driver_max_kmh: float = 80.0) -> Tuple[Graph, HistoricalTrafficModel, RoutingService]:
    """Construye (graph, traffic, service) lista para la UI."""
    graph = Graph.build_jesus_maria_hardcoded()
    traffic = HistoricalTrafficModel(driver_max_kmh=driver_max_kmh)
    pairwise = PairwiseDistanceService(AStarRouter(driver_max_kmh), DijkstraRouter(), RouteCache(), max_workers=4)
    service = RoutingService(
        graph=graph, traffic=traffic, pairwise_service=pairwise,
        solver_exact=HeldKarpExact(), solver_heur=HeuristicRoute(restarts=4), splicer=RouteSplicer(),
    )
    return graph, traffic, service


def make_pois(graph: Graph) -> Dict[str, int]:
    """Crea un pequeño set de POIs reproducible."""
    nodes = list(graph.iter_nodes())
    random.seed(42)
    sampled = random.sample(nodes, k=min(15, len(nodes)))
    labels = [
        "Plaza Central", "Escuela #1", "Mercado", "Centro Cívico", "Club Social",
        "Comisaría", "Capilla", "Biblioteca", "Terminal", "Hospital",
        "Parque Norte", "Anfiteatro", "Estadio", "Museo", "Estación"
    ]
    pois: Dict[str, int] = {labels[0]: sampled[0].id}
    for idx, node in enumerate(sampled[1:len(labels)], start=1):
        pois[labels[idx]] = node.id
    return pois


# =================== Encabezado ===================
st.title("🏘️ Pueblito: Rutas Inteligentes")
st.caption("A* / Dijkstra / BFS · batching par-a-par y TSP (Held-Karp / NN + 2-opt)")

graph, traffic, service = make_services(driver_max_kmh=80.0)
POIS = make_pois(graph)

# =================== Sidebar ===================
st.sidebar.title("⚙️ Configuración")
algorithm = st.sidebar.selectbox("Algoritmo base (tramos)", [Algorithm.ASTAR.value, Algorithm.DIJKSTRA.value, Algorithm.BFS.value], index=0)
mode = st.sidebar.selectbox(
    "Modo multi-stop", [RouteMode.VISIT_ALL_OPEN.value, RouteMode.VISIT_ALL_CIRCUIT.value, RouteMode.FIXED_ORDER.value], index=0
)
hour = st.sidebar.slider("Hora del día", min_value=0, max_value=23, value=8)
color_by = st.sidebar.selectbox("Color de calles", ["congestion", "class"], index=0)

st.sidebar.markdown("---")
origin_label = st.sidebar.selectbox("Origen (POI)", list(POIS.keys()), index=0)
origin_id = POIS[origin_label]
choices = [k for k in POIS.keys() if POIS[k] != origin_id]

# ✅ MULTISELECT con key para poder limpiarlo desde el botón
selected_labels = st.sidebar.multiselect("Destinos (POIs)", choices, default=choices[:3], key="POIS_MULTI")
destinations = [POIS[l] for l in st.session_state.get("POIS_MULTI", [])]

st.sidebar.markdown("---")
calc = st.sidebar.button("🧭 Calcular mejor ruta", use_container_width=True)
clear = st.sidebar.button("🧹 Limpiar destinos", use_container_width=True)
if clear:
    # ✅ limpiar la selección visual y el estado interno
    st.session_state["POIS_MULTI"] = []

# =================== Cámara y capas ===================
nodes = list(graph.iter_nodes())
lat_c = sum(n.lat for n in nodes) / len(nodes)
lon_c = sum(n.lon for n in nodes) / len(nodes)
view_state = pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=14.3, pitch=0)

roads_layer = build_road_layer(graph, hour=hour, traffic=traffic, color_by=color_by)

route_leg_list: List[RouteLeg] = []
result_summary = None

if calc and destinations:
    req = RouteRequest(
        origin=int(origin_id),
        destinations=destinations,
        hour=int(hour),
        algorithm=Algorithm(algorithm),
        mode=RouteMode(mode),
        use_exact=False,  # cambiar a True para Held-Karp exacto
    )
    route_leg_list, result_summary = service.plan_route(req)

# =================== Mapa ===================
layers = [roads_layer]
if route_leg_list:
    layers.append(build_route_layer(graph, route_leg_list))

st.pydeck_chart(pdk.Deck(
    map_style=None,
    initial_view_state=view_state,
    layers=layers,
))

# =================== Métricas ===================
st.subheader("📊 Métricas")
if result_summary:
    st.success(
        f"Tiempo total estimado: **{result_summary.total_seconds:.1f}s** · "
        f"Distancia: **{result_summary.total_distance_m/1000:.2f} km** · "
        f"Algoritmo: {result_summary.algorithm_summary}"
    )
    with st.expander("Detalle de tramos"):
        for i, leg in enumerate(route_leg_list, 1):
            st.write(f"{i}. #{leg.src} → #{leg.dst} · {leg.seconds:.1f}s · {leg.distance_m/1000:.3f} km")
else:
    st.info("Calculá una ruta para ver métricas.")
