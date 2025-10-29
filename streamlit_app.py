# -*- coding: utf-8 -*-
"""
streamlit_app.py — UI para “Pueblito: Rutas Inteligentes”
Versión sin OSM. Opción “Jesús María (Hardcoded)”.
"""
from __future__ import annotations

import math
import random
import json
from typing import Dict, List, Tuple, Optional

import streamlit as st
import pydeck as pdk

from logica import (
    Algorithm, Graph, HistoricalTrafficModel,
    GeoLowerBoundHeuristic, LearnedHistoricalHeuristic, HybridConservativeHeuristic,
    AStarRouter, DijkstraRouter, PairwiseDistanceService,
    RouteCache, SSSPMemo, HeldKarpExact, HeuristicRoute, RouteSplicer,
    RouteLeg, RouteMode, RouteRequest
)

# =================== UI config ===================
st.set_page_config(page_title="Pueblito · Rutas Inteligentes", page_icon="🏘️", layout="wide")

COLOR_BG = [14, 18, 24]
COLOR_ROAD_PRIMARY = [45, 85, 255]
COLOR_ROAD_COLLECTOR = [92, 112, 177]
COLOR_ROAD_RES = [150, 160, 180]
COLOR_ROUTE = [12, 180, 105]
COLOR_PLAZA = [90, 200, 100, 110]
COLOR_PARK = [64, 170, 80, 70]
COLOR_WATER = [120, 170, 255, 80]
COLOR_HOUSES = [220, 140, 90]
LAT_JM = -30.9859
LON_JM = -64.0947

# =================== Motor (cache) ===================
@st.cache_resource(show_spinner=False)
def load_services(source: str):
    if source == "jm_hc":
        graph = Graph.build_jesus_maria_hardcoded()
    else:
        graph = Graph.build_small_town()

    traffic = HistoricalTrafficModel()
    heuristic = HybridConservativeHeuristic(LearnedHistoricalHeuristic(), GeoLowerBoundHeuristic())
    route_cache = RouteCache()
    memo = SSSPMemo()
    pairwise = PairwiseDistanceService(AStarRouter(heuristic), DijkstraRouter(), route_cache, memo, max_workers=4)
    service = RoutingService(
        graph=graph,
        traffic=traffic,
        heuristic=heuristic,
        pairwise_service=pairwise,
        solver_exact=HeldKarpExact(),
        solver_heur=HeuristicRoute(restarts=4),
        splicer=RouteSplicer(),
    )
    return graph, traffic, heuristic, service

class RoutingService:  # ligera envoltura para type-hints de Streamlit
    def __init__(self, graph, traffic, heuristic, pairwise_service, solver_exact, solver_heur, splicer) -> None:
        self.graph = graph
        self.traffic = traffic
        self.heuristic = heuristic
        self.pairwise = pairwise_service
        self.solver_exact = solver_exact
        self.solver_heur = solver_heur
        self.splicer = splicer
        self.router_dijkstra = DijkstraRouter()
        self.router_astar = AStarRouter(default_heuristic=heuristic)

    def route(self, req: RouteRequest):
        waypoints = [req.origin] + list(req.destinations)
        tm, pm = self.pairwise.compute_matrix(self.graph, waypoints, hour=req.hour, algorithm=req.algorithm, traffic=self.traffic, heuristic=self.heuristic)
        n = len(waypoints)
        if req.mode == RouteMode.FIXED_ORDER:
            order = list(range(n)); alg = f"{req.algorithm.value}"
        else:
            if n <= 13:
                order = self.solver_exact.solve(waypoints, tm, mode=req.mode); alg = f"{req.algorithm.value} + Held-Karp"
            else:
                order = self.solver_heur.solve(waypoints, tm, mode=req.mode); alg = f"{req.algorithm.value} + NN/2opt"
        legs = RouteSplicer().splice(waypoints, order, pm, self.graph, tm)
        return type("Obj", (), dict(legs=legs, total_seconds=sum(l.seconds for l in legs), total_distance_m=sum(l.distance_m for l in legs), algorithm_summary=alg, visit_order=[waypoints[i] for i in order]))

# =================== Sidebar ===================
st.sidebar.title("⚙️ Configuración")
src_choice = st.sidebar.radio("Fuente de calles", ["Jesús María (Hardcoded)", "Sintético (grilla)"], index=0)
algorithm = st.sidebar.selectbox("Algoritmo base (tramos)", [Algorithm.ASTAR.value, Algorithm.DIJKSTRA.value], index=0)
mode = st.sidebar.selectbox("Modo de ruta", [RouteMode.VISIT_ALL_OPEN.value, RouteMode.VISIT_ALL_CIRCUIT.value, RouteMode.FIXED_ORDER.value], index=0)
hour = st.sidebar.slider("Hora del día", 0, 23, 8)
color_by = st.sidebar.radio("Color de calles", ["class", "traffic"], index=0, horizontal=True)

with st.sidebar.expander("Heurística (v95 por hora)"):
    st.caption("Podés subir un JSON {hora:int → vmax95_kmh:float}.")
    vmax_file = st.file_uploader("v95.json", type=["json"], accept_multiple_files=False)

# =================== Cargar motor ===================
source_param = "jm_hc" if src_choice.startswith("Jesús") else "synthetic"
graph, traffic, heuristic, service = load_services(source_param)

if vmax_file:
    vmax = json.load(vmax_file)
    if isinstance(heuristic, HybridConservativeHeuristic):
        heuristic.learned.v95 = {int(k): float(v) for k, v in vmax.items()}
        st.success("Heurística actualizada.")

# =================== Helpers de visualización ===================
def _rotate_point(lon: float, lat: float, angle_deg: float, center_lon: float, center_lat: float) -> Tuple[float, float]:
    if angle_deg == 0: return lon, lat
    theta = math.radians(angle_deg)
    dx = lon - center_lon; dy = lat - center_lat
    rx = dx * math.cos(theta) - dy * math.sin(theta)
    ry = dx * math.sin(theta) + dy * math.cos(theta)
    return center_lon + rx, center_lat + ry

def edges_geo_layers(graph: Graph, *, hour: int, color_by: str = "class", rotation_deg: float = 0.0, center_lon: float = 0.0, center_lat: float = 0.0):
    road_data, arrow_data = [], []
    factor = traffic.factor if hasattr(traffic, "factor") else {}
    for u, e in graph.iter_edges():
        n1, n2 = graph.get_node(u), graph.get_node(e.to)
        width = 6 if e.road_class.value == "primary" else (4 if e.road_class.value == "collector" else 2.5)
        base_color = COLOR_ROAD_PRIMARY if e.road_class.value == "primary" else (COLOR_ROAD_COLLECTOR if e.road_class.value == "collector" else COLOR_ROAD_RES)
        if color_by == "traffic":
            f = factor.get(hour, {}).get(e.road_class, 1.0)
            red = min(255, int(100 + (f - 1.0) * 240)); col = [red, base_color[1], base_color[2], 160]
        else:
            col = [*base_color, 160]
        lon1, lat1 = _rotate_point(n1.lon, n1.lat, rotation_deg, center_lon, center_lat)
        lon2, lat2 = _rotate_point(n2.lon, n2.lat, rotation_deg, center_lon, center_lat)
        road_data.append({"path": [[lon1, lat1], [lon2, lat2]], "width": width, "color": col})
        if e.one_way:
            mid_lon, mid_lat = (n1.lon + n2.lon) / 2, (n1.lat + n2.lat) / 2
            mid_lon, mid_lat = _rotate_point(mid_lon, mid_lat, rotation_deg, center_lon, center_lat)
            arrow_data.append({"position": [mid_lon, mid_lat], "text": "→", "angle": 0, "size": 16})
    roads = pdk.Layer("PathLayer", data=road_data, get_path="path", get_width="width", get_color="color", width_min_pixels=2)
    arrows = pdk.Layer("TextLayer", data=arrow_data, get_position="position", get_text="text", get_size="size")
    return roads, arrows

def route_layers(route_legs: List[RouteLeg], graph: Graph, *, rotation_deg: float = 0.0, center_lon: float = 0.0, center_lat: float = 0.0):
    if not route_legs: return None, None
    path_coords, markers = [], []
    n0 = graph.get_node(route_legs[0].src)
    lon0, lat0 = _rotate_point(n0.lon, n0.lat, rotation_deg, center_lon, center_lat)
    markers.append({"lon": lon0, "lat": lat0, "kind": "Origen"})
    for leg in route_legs:
        coords = []
        for nid in leg.path:
            nn = graph.get_node(nid)
            lon_r, lat_r = _rotate_point(nn.lon, nn.lat, rotation_deg, center_lon, center_lat)
            coords.append([lon_r, lat_r])
        path_coords.append({"path": coords})
        dnode = graph.get_node(leg.dst)
        dlon, dlat = _rotate_point(dnode.lon, dnode.lat, rotation_deg, center_lon, center_lat)
        markers.append({"lon": dlon, "lat": dlat, "kind": "Destino"})
    lay_path = pdk.Layer("PathLayer", data=path_coords, get_path="path", get_width=7, get_color=COLOR_ROUTE, width_min_pixels=3)
    lay_points = pdk.Layer("ScatterplotLayer", data=markers, get_position="[lon, lat]", get_radius=30, get_fill_color=[20, 200, 120], pickable=True)
    return lay_path, lay_points

# =================== POIs ===================
@st.cache_data(show_spinner=False)
def make_pois(_graph: Graph, *, version_key: str) -> Dict[str, int]:
    rnd = random.Random(99)
    nodes = list(_graph.iter_nodes())
    center = nodes[len(nodes)//2]
    picks = rnd.sample(nodes, k=min(40, len(nodes)))
    labels = ["Plaza Central","Escuela #1","Mercado","Centro Cívico","Club Social","Comisaría","Capilla","Biblioteca","Terminal","Hospital"]
    pois: Dict[str, int] = {labels[0]: center.id}
    for i, n in enumerate(picks[:len(labels)-1], start=1):
        pois[labels[i]] = n.id
    return pois

# =================== Encabezado ===================
st.title("🏘️ Pueblito: Rutas Inteligentes")
st.caption("A* / Dijkstra + heurística admisible, batching par-a-par y TSP (Held-Karp / NN + 2-opt)")

nodes_list = list(graph.iter_nodes())
edges_count = sum(1 for _ in graph.iter_edges())
_lat_sig = sum(n.lat for n in nodes_list)/len(nodes_list)
_lon_sig = sum(n.lon for n in nodes_list)/len(nodes_list)
_sig = f"{source_param}-{len(nodes_list)}-{edges_count}-{round(_lat_sig,6)}-{round(_lon_sig,6)}"
POIS = make_pois(graph, version_key=_sig)

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
    bearing = st.slider("Orientación (bearing)", -180, 180, 0)
    pitch = st.slider("Inclinación (pitch)", 0, 60, 0)
    zoom = st.slider("Zoom", 8.0, 18.0, 14.2)

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

roads_layer, arrows_layer = edges_geo_layers(graph, hour=hour, color_by=color_by, rotation_deg=bearing, center_lon=lon_c, center_lat=lat_c)

route_leg_list: List[RouteLeg] = []
result_summary = None
if calc and destinations:
    req = RouteRequest(origin=int(origin), destinations=[int(x) for x in destinations], hour=int(hour), mode=RouteMode(mode), algorithm=Algorithm(algorithm))
    res = service.route(req)
    route_leg_list = res.legs
    result_summary = res

route_layer, points_layer = route_layers(route_leg_list, graph, rotation_deg=bearing, center_lon=lon_c, center_lat=lat_c)

layers = [roads_layer, arrows_layer]
if route_layer: layers.append(route_layer)
if points_layer: layers.append(points_layer)

st.pydeck_chart(
    pdk.Deck(
        initial_view_state=pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=zoom, pitch=pitch, bearing=bearing),
        layers=layers, map_style=None, parameters={"clearColor": COLOR_BG}, tooltip={"text": "{kind}"}
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
