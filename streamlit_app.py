"""
Streamlit front-end para el "Pueblito" (PEF 2025)

- Visual super gráfica con pydeck: calles, flechas de sentido, plaza/zonas verdes y casas.
- Selección de origen y múltiples destinos (POIs o IDs) y cálculo de la mejor ruta.
- Soporta TSP abierto/circuito y orden fijo. A* / Dijkstra según preferencia.
- Carga opcional de heurística aprendida (v95 por hora) mediante JSON.

Requiere: streamlit, pydeck y el módulo local `logica.py` en el mismo proyecto.

Ejecutar:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from typing import Dict, List, Tuple

import pydeck as pdk
import streamlit as st

from logica import (
    Algorithm,
    AStarRouter,
    DijkstraRouter,
    GeoLowerBoundHeuristic,
    Graph,
    HeuristicRoute,
    HeldKarpExact,
    HistoricalTrafficModel,
    HybridConservativeHeuristic,
    LearnedHistoricalHeuristic,
    PairwiseDistanceService,
    RouteCache,
    RouteLeg,
    RouteMode,
    RouteRequest,
    RouteSplicer,
    RoutingService,
    SSSPMemo,
    haversine_km,
)

# ==========================================================
# Config y estado
# ==========================================================

st.set_page_config(page_title="Pueblito · Rutas Inteligentes", page_icon="🏘️", layout="wide")

# Paleta y helpers visuales
COLOR_BG = [244, 246, 249]
COLOR_ROAD_PRIMARY = [45, 85, 255]
COLOR_ROAD_COLLECTOR = [92, 112, 177]
COLOR_ROAD_RES = [150, 160, 180]
COLOR_ROUTE = [12, 180, 105]
COLOR_PLAZA = [90, 200, 100, 110]
COLOR_PARK = [64, 170, 80, 70]
COLOR_WATER = [120, 170, 255, 80]
COLOR_HOUSES = [220, 140, 90]


# ==========================================================
# Carga de engine (cacheado)
# ==========================================================

@st.cache_resource(show_spinner=False)
def load_services(
    *,
    seed: int = 7,
    blocks_x: int = 14,
    blocks_y: int = 14,
    spacing_m: float = 110.0,
    two_way_ratio: float = 0.72,
    primary_ratio: float = 0.12,
    collector_ratio: float = 0.28,
):
    """Construye el grafo "pueblito" y prepara todos los servicios del motor."""
    graph = Graph.build_small_town(
        seed=seed,
        blocks_x=blocks_x,
        blocks_y=blocks_y,
        spacing_m=spacing_m,
        two_way_ratio=two_way_ratio,
        primary_ratio=primary_ratio,
        collector_ratio=collector_ratio,
    )
    traffic = HistoricalTrafficModel()
    heuristic = HybridConservativeHeuristic(LearnedHistoricalHeuristic(), GeoLowerBoundHeuristic())
    route_cache = RouteCache()
    sssp_memo = SSSPMemo()

    pairwise = PairwiseDistanceService(
        router_astar=AStarRouter(heuristic),
        router_dijkstra=DijkstraRouter(),
        route_cache=route_cache,
        ssp_memo=sssp_memo,
        max_workers=4,
    )

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


graph, traffic, heuristic, service = load_services()


# ==========================================================
# Utilidades de visualización
# ==========================================================

def _unique_sorted(values: List[float]) -> List[float]:
    s = sorted(set(values))
    return s


def _rotate_point(lon: float, lat: float, angle_deg: float, center_lon: float, center_lat: float) -> Tuple[float, float]:
    """Rota un punto (lon, lat) alrededor de (center_lon, center_lat) por angle_deg grados.

    Nota: para la escala de la ciudad pequeña tratamos lon/lat como coordenadas planas
    (bueno para visualización local). No se altera la topología del grafo, sólo
    cómo se dibuja en el mapa.
    """
    if angle_deg == 0:
        return lon, lat
    theta = math.radians(angle_deg)
    dx = lon - center_lon
    dy = lat - center_lat
    rx = dx * math.cos(theta) - dy * math.sin(theta)
    ry = dx * math.sin(theta) + dy * math.cos(theta)
    return center_lon + rx, center_lat + ry


def guess_steps(graph: Graph) -> Tuple[float, float]:
    """Estima la separación (lat, lon) media de la grilla del pueblito.
    Sirve para dibujar plaza y zonas.
    """
    lats = [n.lat for n in graph.iter_nodes()]
    lons = [n.lon for n in graph.iter_nodes()]
    lats_u = _unique_sorted(lats)
    lons_u = _unique_sorted(lons)
    dlat = min(b - a for a, b in zip(lats_u, lats_u[1:])) if len(lats_u) > 1 else 0.001
    dlon = min(b - a for a, b in zip(lons_u, lons_u[1:])) if len(lons_u) > 1 else 0.001
    return dlat, dlon


def edges_geo_layers(graph: Graph, *, hour: int, color_by: str = "class", rotation_deg: float = 0.0, center_lon: float = 0.0, center_lat: float = 0.0) -> Tuple[pdk.Layer, pdk.Layer]:
    """Construye capa de calles y flechas de sentido (TextLayer)."""
    road_data = []
    arrow_data = []
    factor = traffic.factor if hasattr(traffic, "factor") else {}

    for u, e in graph.iter_edges():
        n1 = graph.get_node(u)
        n2 = graph.get_node(e.to)
        width = 6 if e.road_class.value == "primary" else (4 if e.road_class.value == "collector" else 2.5)
        base_color = (
            COLOR_ROAD_PRIMARY if e.road_class.value == "primary" else COLOR_ROAD_COLLECTOR if e.road_class.value == "collector" else COLOR_ROAD_RES
        )
        if color_by == "traffic":
            f = factor.get(hour, {}).get(e.road_class, 1.0)
            # más congestión → más rojizo
            red = min(255, int(100 + (f - 1.0) * 240))
            col = [red, base_color[1], base_color[2], int(160)]
        else:
            col = [*base_color, 120]

        lon1, lat1 = _rotate_point(n1.lon, n1.lat, rotation_deg, center_lon, center_lat)
        lon2, lat2 = _rotate_point(n2.lon, n2.lat, rotation_deg, center_lon, center_lat)
        road_data.append({
            "path": [[lon1, lat1], [lon2, lat2]],
            "width": width,
            "color": col,
        })

        if e.one_way:
            mid_lon = (n1.lon + n2.lon) / 2
            mid_lat = (n1.lat + n2.lat) / 2
            mid_lon_r, mid_lat_r = _rotate_point(mid_lon, mid_lat, rotation_deg, center_lon, center_lat)
            angle = math.degrees(math.atan2(n2.lat - n1.lat, n2.lon - n1.lon)) + rotation_deg
            arrow_data.append({
                "position": [mid_lon_r, mid_lat_r],
                "text": "→",
                "angle": angle,
                "size": 16,
            })

    roads = pdk.Layer(
        "PathLayer",
        data=road_data,
        get_path="path",
        get_width="width",
        get_color="color",
        width_min_pixels=2,
        pickable=False,
    )
    arrows = pdk.Layer(
        "TextLayer",
        data=arrow_data,
        get_position="position",
        get_text="text",
        get_angle="angle",
        get_size="size",
        get_text_anchor="middle",
        get_alignment_baseline="center",
        pickable=False,
    )
    return roads, arrows


def town_layers(graph: Graph, *, rotation_deg: float = 0.0, center_lon: float = 0.0, center_lat: float = 0.0) -> List[pdk.Layer]:
    """Parques, plaza central, agüita y casitas para darle look de pueblito."""
    dlat, dlon = guess_steps(graph)
    nodes = list(graph.iter_nodes())
    lat_c = sum(n.lat for n in nodes) / len(nodes)
    lon_c = sum(n.lon for n in nodes) / len(nodes)

    # crear y rotar polígonos
    def rot(lon, lat):
        return _rotate_point(lon, lat, rotation_deg, center_lon, center_lat)

    plaza = {
        "polygon": [
            list(rot(lon_c - 1.5 * dlon, lat_c - 1.5 * dlat)),
            list(rot(lon_c + 1.5 * dlon, lat_c - 1.5 * dlat)),
            list(rot(lon_c + 1.5 * dlon, lat_c + 1.5 * dlat)),
            list(rot(lon_c - 1.5 * dlon, lat_c + 1.5 * dlat)),
        ]
    }
    park_sw = {"polygon": [
        list(rot(lon_c - 3.8 * dlon, lat_c - 3.8 * dlat)),
        list(rot(lon_c - 2.4 * dlon, lat_c - 3.8 * dlat)),
        list(rot(lon_c - 2.4 * dlon, lat_c - 2.6 * dlat)),
        list(rot(lon_c - 3.8 * dlon, lat_c - 2.6 * dlat)),
    ]}
    laguna = {"polygon": [
        list(rot(lon_c + 3.0 * dlon, lat_c + 2.2 * dlat)),
        list(rot(lon_c + 4.4 * dlon, lat_c + 2.0 * dlat)),
        list(rot(lon_c + 4.8 * dlon, lat_c + 3.0 * dlat)),
        list(rot(lon_c + 3.5 * dlon, lat_c + 3.2 * dlat)),
    ]}

    # Casitas: muestreamos algunos nodos alejados de la plaza
    rnd = random.Random(17)
    house_pts = []
    for n in rnd.sample(nodes, k=min(120, len(nodes))):
        if haversine_km(n.lat, n.lon, lat_c, lon_c) < 0.25:  # cerca de la plaza: menos casas
            continue
        lon_r, lat_r = _rotate_point(n.lon, n.lat, rotation_deg, center_lon, center_lat)
        house_pts.append({"lon": lon_r, "lat": lat_r})

    lay_plaza = pdk.Layer(
        "PolygonLayer",
        data=[plaza],
        get_polygon="polygon",
        get_fill_color=COLOR_PLAZA,
        stroked=True,
        get_line_color=[60, 140, 80],
        line_width_min_pixels=1,
    )

    lay_park = pdk.Layer(
        "PolygonLayer",
        data=[park_sw],
        get_polygon="polygon",
        get_fill_color=COLOR_PARK,
        stroked=False,
    )

    lay_laguna = pdk.Layer(
        "PolygonLayer",
        data=[laguna],
        get_polygon="polygon",
        get_fill_color=COLOR_WATER,
        stroked=False,
    )

    lay_houses = pdk.Layer(
        "ScatterplotLayer",
        data=house_pts,
        get_position="[lon, lat]",
        get_radius=6,
        get_fill_color=COLOR_HOUSES,
        opacity=0.8,
    )

    return [lay_laguna, lay_park, lay_plaza, lay_houses]


def route_layers(route_legs: List[RouteLeg], graph: Graph, *, rotation_deg: float = 0.0, center_lon: float = 0.0, center_lat: float = 0.0) -> Tuple[pdk.Layer, pdk.Layer]:
    """Capa de la ruta final y marcadores (origen/destinos)."""
    if not route_legs:
        return None, None  # type: ignore

    path_coords = []
    markers = []

    # Origen
    n0 = graph.get_node(route_legs[0].src)
    lon0, lat0 = _rotate_point(n0.lon, n0.lat, rotation_deg, center_lon, center_lat)
    markers.append({"lon": lon0, "lat": lat0, "kind": "Origen"})

    for leg in route_legs:
        coords = []
        for n in leg.path:
            nn = graph.get_node(n)
            lon_r, lat_r = _rotate_point(nn.lon, nn.lat, rotation_deg, center_lon, center_lat)
            coords.append([lon_r, lat_r])
        path_coords.append({"path": coords})
        dnode = graph.get_node(leg.dst)
        dlon_r, dlat_r = _rotate_point(dnode.lon, dnode.lat, rotation_deg, center_lon, center_lat)
        markers.append({"lon": dlon_r, "lat": dlat_r, "kind": "Destino"})

    lay_path = pdk.Layer(
        "PathLayer",
        data=path_coords,
        get_path="path",
        get_width=7,
        get_color=COLOR_ROUTE,
        width_min_pixels=3,
    )
    lay_points = pdk.Layer(
        "ScatterplotLayer",
        data=markers,
        get_position="[lon, lat]",
        get_radius=30,
        get_fill_color=[20, 200, 120],
        pickable=True,
    )
    return lay_path, lay_points


# ==========================================================
# Sidebar - controles
# ==========================================================

st.sidebar.title("⚙️ Configuración")
with st.sidebar.expander("Generar pueblito", expanded=False):
    seed = st.number_input("Semilla", value=7, step=1)
    bx = st.slider("Manzanas (X)", 8, 22, 14)
    by = st.slider("Manzanas (Y)", 8, 22, 14)
    spacing = st.slider("Distancia entre calles (m)", 70, 180, 110)
    two_way = st.slider("% Doble mano", 40, 95, 72)
    primary_ratio = st.slider("% Avenidas (primary)", 5, 25, 12)
    collector_ratio = st.slider("% Colectoras", 20, 40, 28)
    regen = st.button("🔁 Regenerar ciudad", use_container_width=True)

if regen:
    # Reinicia recursos cacheados con nueva ciudad
    load_services.clear()
    graph, traffic, heuristic, service = load_services(
        seed=int(seed),
        blocks_x=int(bx),
        blocks_y=int(by),
        spacing_m=float(spacing),
        two_way_ratio=float(two_way) / 100.0,
        primary_ratio=float(primary_ratio) / 100.0,
        collector_ratio=float(collector_ratio) / 100.0,
    )

algorithm = st.sidebar.selectbox("Algoritmo base (tramos)", [Algorithm.ASTAR.value, Algorithm.DIJKSTRA.value], index=0)
mode = st.sidebar.selectbox(
    "Modo de ruta",
    [RouteMode.VISIT_ALL_OPEN.value, RouteMode.VISIT_ALL_CIRCUIT.value, RouteMode.FIXED_ORDER.value],
    index=0,
)
hour = st.sidebar.slider("Hora del día", 0, 23, 8)
color_by = st.sidebar.radio("Color de calles", ["class", "traffic"], index=0, horizontal=True)

with st.sidebar.expander("Heurística (v95 por hora)"):
    st.caption("Sube un JSON {hora:int → vmax95_kmh:float} para guiar A* y mantener admisibilidad.")
    vmax_file = st.file_uploader("v95.json", type=["json"], accept_multiple_files=False)
    if vmax_file:
        vmax = json.load(vmax_file)
        if isinstance(heuristic, HybridConservativeHeuristic):
            heuristic.learned.vmax95_by_hour = {int(k): float(v) for k, v in vmax.items()}
            st.success("Heurística actualizada.")

# ==========================================================
# POIs (origen y destinos)
# ==========================================================

@st.cache_data(show_spinner=False)
def make_pois(_graph: Graph, *, version_key: str) -> Dict[str, int]:
    """Crea un set chico de POIs lindos: Plaza, Escuela, Mercado, etc. -> node_id."""
    rnd = random.Random(99)
    nodes = list(graph.iter_nodes())
    # Tomamos esquinas representativas
    corners = sorted(nodes, key=lambda n: (n.lat, n.lon))
    center = nodes[len(nodes) // 2]
    picks = rnd.sample(nodes, k=min(30, len(nodes)))
    labels = [
        "Plaza Central",
        "Escuela #1",
        "Mercado",
        "Centro Cívico",
        "Club Social",
        "Comisaría",
        "Capilla",
        "Biblioteca",
        "Terminal",
        "Hospital",
    ]
    pois: Dict[str, int] = {}
    pois[labels[0]] = center.id
    for i, n in enumerate(picks[: len(labels) - 1], start=1):
        pois[labels[i]] = n.id
    return pois


_nodes_for_sig = list(graph.iter_nodes())
_edges_for_sig = sum(1 for _ in graph.iter_edges())
_lat_sig = sum(n.lat for n in _nodes_for_sig)/len(_nodes_for_sig)
_lon_sig = sum(n.lon for n in _nodes_for_sig)/len(_nodes_for_sig)
_sig = f"{len(_nodes_for_sig)}-{_edges_for_sig}-{round(_lat_sig,6)}-{round(_lon_sig,6)}"
POIS = make_pois(graph, version_key=_sig)

st.sidebar.markdown("---")
origin_label = st.sidebar.selectbox("Origen (POI)", list(POIS.keys()), index=0)
origin_id = POIS[origin_label]

# Multi-select de destinos
choices = [k for k in POIS.keys() if POIS[k] != origin_id]
selected_labels = st.sidebar.multiselect("Destinos (POIs)", choices, default=choices[:3])

default_ids = [POIS[l] for l in selected_labels]

with st.sidebar.expander("O ingresar IDs de nodos manualmente"):
    src_id = st.number_input("Origen (id)", min_value=0, value=int(origin_id), step=1)
    dst_ids_str = st.text_input("Destinos (ids separados por coma)", value=",".join(str(i) for i in default_ids))
    try:
        manual_ids = [int(x.strip()) for x in dst_ids_str.split(",") if x.strip()]
    except Exception:
        manual_ids = default_ids

use_manual = st.sidebar.checkbox("Usar IDs manuales en lugar de POIs", value=False)

if use_manual:
    origin = int(src_id)
    destinations = manual_ids
else:
    origin = POIS[origin_label]
    destinations = [POIS[l] for l in selected_labels]

st.sidebar.markdown("---")
calc = st.sidebar.button("🧭 Calcular mejor ruta", use_container_width=True)
clear = st.sidebar.button("🧹 Limpiar destinos", use_container_width=True)

if clear:
    selected_labels = []
    destinations = []

# ==========================================================
# Vista principal: mapa, ruta y métricas
# ==========================================================

st.title("🏘️ Pueblito: Rutas Inteligentes")
st.caption("A* / Dijkstra + heurística admisible, batching par-a-par y TSP (Held-Karp / NN + 2-opt)")

# Las capas se construyen más abajo para poder pasar rotation (bearing) y centro
roads_layer = arrows_layer = None
extras = []
route_leg_list: List[RouteLeg] = []
result_summary = None

if calc and destinations:
    req = RouteRequest(
        origin=int(origin),
        destinations=[int(x) for x in destinations],
        hour=int(hour),
        mode=RouteMode(mode),
        algorithm=Algorithm(algorithm),
    )
    res = service.route(req)
    route_leg_list = res.legs
    result_summary = res

# (Las capas se construyen más abajo, después de leer sliders de cámara)

# Cámara: centrada en el centro del grafo (o en origen / coordenadas manuales)
# Para inicializar el mapa en coordenadas específicas pedidas por el usuario
# usamos estas constantes por defecto. Si preferís el centro de la ciudad,
# activá "Centro de ciudad" en el sidebar.
nodes = list(graph.iter_nodes())
# coordenadas solicitadas
lat_c_default = -30.986154349785995
lon_c_default = -64.08957839045158

with st.sidebar.expander("Mapa / Cámara", expanded=False):
    center_mode = st.radio(
        "Centrar mapa en:", ("Centro de ciudad", "Origen seleccionado", "Coordenadas manuales"), index=2
    )
    bearing = st.slider("Orientación (bearing, grados)", -180, 180, 0)
    pitch = st.slider("Inclinación (pitch)", 0, 60, 0)
    zoom = st.slider("Zoom inicial", 8.0, 18.0, 14.2)
    if center_mode == "Coordenadas manuales":
        manual_lat = st.number_input("Latitud manual", value=float(lat_c_default))
        manual_lon = st.number_input("Longitud manual", value=float(lon_c_default))

# Resolver centro según modo seleccionado
if center_mode == "Centro de ciudad":
    lat_c, lon_c = lat_c_default, lon_c_default
elif center_mode == "Origen seleccionado":
    try:
        n0 = graph.get_node(origin)
        lat_c, lon_c = n0.lat, n0.lon
    except Exception:
        # si algo falla (p.ej. origin no definido), fallback al centro por defecto
        lat_c, lon_c = lat_c_default, lon_c_default
else:  # Coordenadas manuales
    lat_c, lon_c = manual_lat, manual_lon

# Construir capas con rotación de la generación de la ciudad (bearing)
roads_layer, arrows_layer = edges_geo_layers(graph, hour=hour, color_by=color_by, rotation_deg=bearing, center_lon=lon_c, center_lat=lat_c)
extras = town_layers(graph, rotation_deg=bearing, center_lon=lon_c, center_lat=lat_c)
route_layer, points_layer = route_layers(route_leg_list, graph, rotation_deg=bearing, center_lon=lon_c, center_lat=lat_c)

layers = [roads_layer, arrows_layer, *(extras or [])]
if route_layer:
    layers.append(route_layer)
if points_layer:
    layers.append(points_layer)

st.pydeck_chart(
    pdk.Deck(
        initial_view_state=pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=zoom, pitch=pitch, bearing=bearing),
        layers=layers,
        map_style=None,
        parameters={"clearColor": COLOR_BG},
        tooltip={"text": "{kind}"},
    ),
    use_container_width=True,
)

# Panel de resultados
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("📍 Selección")
    st.write(f"**Origen:** #{origin}")
    if destinations:
        st.write("**Destinos:** ", ", ".join(f"#{d}" for d in destinations))
    else:
        st.info("Agregá algunos destinos en la barra lateral para calcular la ruta.")

with col2:
    st.subheader("📊 Métricas")
    if result_summary:
        st.success(
            f"Tiempo total estimado: **{result_summary.total_seconds:.1f}s** · Distancia: **{result_summary.total_distance_m/1000:.2f} km**\n\n"
            f"Algoritmo: {result_summary.algorithm_summary}"
        )
        with st.expander("Detalle de tramos"):
            for i, leg in enumerate(route_leg_list, 1):
                st.write(
                    f"{i}. #{leg.src} → #{leg.dst} · {leg.seconds:.1f}s · {leg.distance_m/1000:.3f} km"
                )
    else:
        st.info("Calculá una ruta para ver métricas.")

# Ayuda / info
with st.expander("ℹ️ Cómo usar"):
    st.markdown(
        """
        - Elegí el **origen** y una lista de **destinos** desde los POIs o ingresa IDs manuales.
        - Seleccioná el **modo** (visitar todos / circuito / orden fijo) y la **hora**.
        - Podés cambiar **colores** de calles por clase o por **congestión** estimada.
        - En *Generar pueblito* podés regenerar la ciudad con diferentes semillas y densidades.
        - Para A* más rápido, cargá un JSON con **v95 por hora** en la sección de Heurística.
        """
    )
