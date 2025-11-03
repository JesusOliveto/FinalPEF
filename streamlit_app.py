# -*- coding: utf-8 -*-
"""
streamlit_app.py — UI para “Pueblito: Rutas Inteligentes”
Foco: calcular mejores rutas en Jesús María (sin OSM).
"""
from __future__ import annotations

import json
import random
from typing import Dict, List, Tuple

import streamlit as st
import pydeck as pdk

from logica import (
    Algorithm, Graph, HistoricalTrafficModel,
    GeoLowerBoundHeuristic, LearnedHistoricalHeuristic, HybridConservativeHeuristic,
    AStarRouter, DijkstraRouter, PairwiseDistanceService,
    RouteCache, SSSPMemo, HeldKarpExact, HeuristicRoute, RouteSplicer,
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

# =================== Motor (cache) ===================
@st.cache_resource(show_spinner=False)
def load_services(driver_max_kmh: float = 40.0):
    """Inicializa y cachea el motor de ruteo y sus servicios.

    Args:
        driver_max_kmh: Velocidad máxima del conductor (km/h) para limitar tiempos.

    Returns:
        Tupla con (graph, traffic, heuristic, service) lista para usar en la UI.
    """
    graph = Graph.build_jesus_maria_hardcoded()
    traffic = HistoricalTrafficModel(driver_max_kmh=driver_max_kmh)
    heuristic = HybridConservativeHeuristic(LearnedHistoricalHeuristic(), GeoLowerBoundHeuristic())
    pairwise = PairwiseDistanceService(AStarRouter(heuristic), DijkstraRouter(), RouteCache(), SSSPMemo(), max_workers=4)
    service = RoutingService(
        graph=graph, traffic=traffic, heuristic=heuristic, pairwise_service=pairwise,
        solver_exact=HeldKarpExact(), solver_heur=HeuristicRoute(restarts=4), splicer=RouteSplicer(),
    )
    return graph, traffic, heuristic, service

# =================== Sidebar ===================
st.sidebar.title("⚙️ Configuración")
algorithm = st.sidebar.selectbox("Algoritmo base (tramos)", [Algorithm.ASTAR.value, Algorithm.DIJKSTRA.value, Algorithm.BFS.value], index=0)
mode = st.sidebar.selectbox("Modo de ruta", [RouteMode.VISIT_ALL_OPEN.value, RouteMode.VISIT_ALL_CIRCUIT.value, RouteMode.FIXED_ORDER.value], index=0)
hour = st.sidebar.slider("Hora del día", 0, 23, 8)
color_by = st.sidebar.radio("Color de calles", ["class", "traffic"], index=0, horizontal=True)

with st.sidebar.expander("Tráfico y vehículo", expanded=False):
    driver_speed = st.slider("Velocidad del conductor (km/h)", min_value=20, max_value=80, value=40, step=5)

# Inicialización del motor con la velocidad elegida
graph, traffic, heuristic, service = load_services(driver_speed)

with st.sidebar.expander("Heurística (v95 por hora)"):
    st.caption("Subí un JSON {hora:int → vmax95_kmh:float} para A*.")
    vmax_file = st.file_uploader("v95.json", type=["json"], accept_multiple_files=False)
    if vmax_file:
        vmax = json.load(vmax_file)
        if isinstance(heuristic, HybridConservativeHeuristic):
            heuristic.learned.v95 = {int(k): float(v) for k, v in vmax.items()}
            st.success("Heurística actualizada.")

# =================== Helpers de visualización ===================
def build_road_layer(graph: Graph, *, hour: int, color_by: str = "class"):
    """Construye la capa de calles (PathLayer) para pydeck.

    Args:
        graph: Grafo de la ciudad.
        hour: Hora del día usada para colorear por tráfico si corresponde.
        color_by: "class" para colorear por clase vial, "traffic" para congestión.

    Returns:
        Capa de pydeck con todos los segmentos viales.
    """
    road_data = []
    traffic_factor_by_hour = getattr(traffic, "factor", {})
    for from_node_id, edge in graph.iter_edges():
        node_from = graph.get_node(from_node_id)
        node_to = graph.get_node(edge.to)
        path_width_px = 6 if edge.road_class.value == "primary" else (4 if edge.road_class.value == "collector" else 2.5)
        base_color = (
            COLOR_ROAD_PRIMARY if edge.road_class.value == "primary" else (
                COLOR_ROAD_COLLECTOR if edge.road_class.value == "collector" else COLOR_ROAD_RES
            )
        )
        if color_by == "traffic":
            congestion_factor = traffic_factor_by_hour.get(hour, {}).get(edge.road_class, 1.0)
            red_channel = min(255, int(100 + (congestion_factor - 1.0) * 240))
            color_rgba = [red_channel, base_color[1], base_color[2], 160]
        else:
            color_rgba = [*base_color, 160]
        road_data.append({
            "path": [[node_from.lon, node_from.lat], [node_to.lon, node_to.lat]],
            "width": path_width_px,
            "color": color_rgba,
        })
    return pdk.Layer(
        "PathLayer", data=road_data, get_path="path", get_width="width", get_color="color", width_min_pixels=2, pickable=False
    )

def build_route_layers(route_legs: List[RouteLeg], graph: Graph):
    """Construye las capas de ruta (PathLayer) y marcadores (ScatterplotLayer).

    Args:
        route_legs: Tramos de la ruta calculada por el motor.
        graph: Grafo para traducir ids a coordenadas lon/lat.

    Returns:
        Tupla (layer_path, layer_points). Si no hay ruta, devuelve (None, None).
    """
    if not route_legs:
        return None, None
    path_coords: List[Dict[str, List[List[float]]]] = []
    marker_points: List[Dict[str, float]] = []
    origin_node = graph.get_node(route_legs[0].src)
    marker_points.append({"lon": origin_node.lon, "lat": origin_node.lat, "kind": "Origen"})
    for leg in route_legs:
        leg_coords: List[List[float]] = []
        for node_id in leg.path:
            current_node = graph.get_node(node_id)
            leg_coords.append([current_node.lon, current_node.lat])
        path_coords.append({"path": leg_coords})
        destination_node = graph.get_node(leg.dst)
        marker_points.append({"lon": destination_node.lon, "lat": destination_node.lat, "kind": "Destino"})
    layer_path = pdk.Layer(
        "PathLayer", data=path_coords, get_path="path", get_width=7, get_color=COLOR_ROUTE, width_min_pixels=3
    )
    layer_points = pdk.Layer(
        "ScatterplotLayer", data=marker_points, get_position="[lon, lat]", get_radius=30, get_fill_color=[20, 200, 120], pickable=True
    )
    return layer_path, layer_points

# =================== POIs ===================
@st.cache_data(show_spinner=False)
def make_pois(_graph: Graph) -> Dict[str, int]:
    """Genera un conjunto pequeño de POIs de ejemplo mapeados a ids de nodos.

    Args:
        _graph: Grafo desde el cual se seleccionarán nodos representativos.

    Returns:
        Diccionario {nombre_POI -> id_nodo}.
    """
    rnd = random.Random(99)
    nodes = list(_graph.iter_nodes())
    center_node = nodes[len(nodes)//2]
    sampled_nodes = rnd.sample(nodes, k=min(40, len(nodes)))
    labels = [
        "Plaza Central","Escuela #1","Mercado","Centro Cívico","Club Social",
        "Comisaría","Capilla","Biblioteca","Terminal","Hospital"
    ]
    pois: Dict[str, int] = {labels[0]: center_node.id}
    for idx, node in enumerate(sampled_nodes[:len(labels)-1], start=1):
        pois[labels[idx]] = node.id
    return pois

# =================== Encabezado ===================
st.title("🏘️ Pueblito: Rutas Inteligentes")
st.caption("A* / Dijkstra / BFS · batching par-a-par y TSP (Held-Karp / NN + 2-opt)")

POIS = make_pois(graph)

st.sidebar.markdown("---")
origin_label = st.sidebar.selectbox("Origen (POI)", list(POIS.keys()), index=0)
origin_id = POIS[origin_label]
choices = [k for k in POIS.keys() if POIS[k] != origin_id]
selected_labels = st.sidebar.multiselect("Destinos (POIs)", choices, default=choices[:3])
default_ids = [POIS[l] for l in selected_labels]

with st.sidebar.expander("O ingresar IDs manuales"):
    src_id = st.number_input("Origen (id)", min_value=0, value=int(origin_id), step=1)
    dst_ids_str = st.text_input("Destinos (ids separados por coma)", value=",".join(str(i) for i in default_ids))
    try:
        manual_ids = [int(x.strip()) for x in dst_ids_str.split(",") if x.strip()]
    except Exception:
        manual_ids = default_ids
use_manual = st.sidebar.checkbox("Usar IDs manuales", value=False)
origin = int(src_id) if use_manual else int(origin_id)
destinations = manual_ids if use_manual else [POIS[l] for l in selected_labels]

st.sidebar.markdown("---")
calc = st.sidebar.button("🧭 Calcular mejor ruta", use_container_width=True)
clear = st.sidebar.button("🧹 Limpiar destinos", use_container_width=True)
if clear:
    destinations = []

# =================== Cámara y capas ===================
with st.sidebar.expander("Mapa / Cámara", expanded=False):
    center_mode = st.radio("Centrar mapa en:", ("Centro de ciudad", "Origen seleccionado"), index=0)
    pitch = st.slider("Inclinación (pitch)", 0, 60, 0)
    zoom = st.slider("Zoom", 8.0, 18.0, 14.3)

nodes = list(graph.iter_nodes())
if center_mode == "Origen seleccionado":
    try:
        n0 = graph.get_node(origin)
        lat_c, lon_c = n0.lat, n0.lon
    except Exception:
        lat_c, lon_c = LAT_JM, LON_JM
else:
    lat_c = sum(n.lat for n in nodes)/len(nodes)
    lon_c = sum(n.lon for n in nodes)/len(nodes)

roads_layer = build_road_layer(graph, hour=hour, color_by=color_by)

route_leg_list: List[RouteLeg] = []
result_summary = None
if calc and destinations:
    req = RouteRequest(origin=int(origin), destinations=[int(x) for x in destinations], hour=int(hour), mode=RouteMode(mode), algorithm=Algorithm(algorithm))
    res = service.route(req)
    route_leg_list = res.legs
    result_summary = res

route_layer, points_layer = build_route_layers(route_leg_list, graph)

layers = [roads_layer]
if route_layer: layers.append(route_layer)
if points_layer: layers.append(points_layer)

st.pydeck_chart(
    pdk.Deck(
        initial_view_state=pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=zoom, pitch=pitch),
        layers=layers, map_style=None,
    ),
    use_container_width=True,
)

# =================== Panel info ===================
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("📍 Selección")
    st.write(f"**Origen:** #{origin}")
    st.write("**Destinos:** " + (", ".join(f"#{d}" for d in destinations) if destinations else "—"))

with col2:
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
