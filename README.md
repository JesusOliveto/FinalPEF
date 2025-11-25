# Final PEF 2025 – Sistema de Recomendación de Rutas en una Ciudad Inteligente

Este proyecto implementa un **motor de ruteo** y una **aplicación web interactiva** para
recomendar rutas dentro de una ciudad ficticia que imita el trazado de **Jesús María (Córdoba)**.

El foco está en:

- Modelar una red vial realista (calles residenciales, colectoras y primarias).
- Incluir un modelo de tráfico histórico y un límite de velocidad realista del conductor.
- Ofrecer distintos algoritmos de ruteo (A*, Dijkstra, BFS) y modos de ruta multi‑destino.
- Proveer una interfaz en Streamlit para que el usuario explore rutas y métricas.
- Incluir herramientas de **testing** y **profiling** para evaluar el rendimiento del motor.

---

## 1. Propósito del sistema

El sistema busca responder a la pregunta:

> “Dado un origen y varios puntos de interés dentro de la ciudad, ¿cuál es la mejor ruta
>  para visitarlos, bajo diferentes modos (camino abierto o circuito), horarios y niveles
>  de tráfico?”

Casos de uso típicos:

- Planificación de recorridos para servicios municipales (recolección, inspecciones, etc.).
- Rutas para logística liviana dentro de la ciudad.
- Análisis de impacto del tráfico en diferentes horarios.

El sistema permite **experimentar** con:

- Algoritmo base de ruteo (A*, Dijkstra o BFS).
- Modo de visita:
	- `visit_all_open`: recorre todos los destinos y termina en el último.
	- `visit_all_circuit`: recorre todos los destinos y vuelve al origen (circuito).
	- `fixed_order`: respeta el orden en que se entregan los destinos.
- Horario del día (afecta congestión) y velocidad máxima del conductor.

---

## 2. Arquitectura general

El proyecto se organiza en tres componentes principales:

1. `logica.py` – Motor de ruteo
2. `streamlit_app.py` – Interfaz web (front en Streamlit)
3. `tests_profiling.py` – Tests unitarios y herramientas de profiling

### 2.1 Motor de ruteo (`logica.py`)

**Modelo de grafo**

- `Graph`, `Node`, `Edge` representan la red vial.
- La red de Jesús María se construye en `Graph.build_jesus_maria_hardcoded()`, que:
	- Genera parches de grilla rotados (centro, sudeste, noroeste).
	- Modela la RN‑9 como una diagonal primaria con conectores.
	- Asigna clases viales (`PRIMARY`, `COLLECTOR`, `RESIDENTIAL`) y velocidades.
	- Define qué segmentos son doble mano y cuáles son mano única.

**Modelo de tráfico**

- `HistoricalTrafficModel` hereda de `TrafficModel` y define un método
	`travel_time_seconds(edge, hour=...)` que:
	- Aplica un **factor de congestión** según hora del día y tipo de calle.
	- Limita la velocidad efectiva por un parámetro `driver_max_kmh` (ej. 40 km/h),
		para simular el comportamiento real de un conductor.

**Routers (algoritmos de ruteo)**

- `DijkstraRouter`:
	- Implementación de Dijkstra sobre tiempos de viaje.
	- Soporta **batching** de vecinos (uso de un generador `batcher`).
	- Usa `ThreadPoolExecutor` para paralelizar el cálculo de tiempos por arista.
	- Aplica **memoización** de resultados por clave `(graph, traffic, src, dst, hour)`.

- `AStarRouter`:
	- Implementa A* con **heurística temporal** basada en distancia geodésica (haversine)
		y una velocidad máxima optimista `vmax_h`.
	- Usa el mismo modelo de tráfico para el costo real de las aristas.

- `BFSRouter`:
	- Realiza una búsqueda en anchura (útil como baseline/depuración).
	- Encuentra un camino no ponderado y luego suma tiempos reales sobre ese camino.

**Servicio par‑a‑par y caché**

- `PairwiseDistanceService`:
	- Calcula una **matriz de tiempos** y un mapa de paths para una lista de waypoints.
	- Selecciona `AStarRouter`, `DijkstraRouter` o `BFSRouter` según `Algorithm`.
	- Usa `RouteCache` (LRU) para cachear tramos individuales (src, dst, hora, algoritmo).
	- Ejecuta los pares `(i, j)` en paralelo con `ThreadPoolExecutor`.

**Solvers multi‑stop**

- `HeldKarpExact`: implementación del algoritmo Held‑Karp (DP) para TSP con origen fijo.
	- Complejidad `O(n^2 2^n)`; se usa para problemas pequeños (n ≤ ~13).
	- Soporta modo circuito agregando el costo de regreso al origen.

- `HeuristicRoute`: heurística **Nearest Neighbor + 2‑opt**, con varios reinicios.
	- Se usa para casos con mayor cantidad de destinos.

**Servicio de alto nivel**

- `RoutingService` combina todas las piezas:
	- Llama a `PairwiseDistanceService.compute_matrix` para obtener tiempos y paths.
	- Selecciona solver exacto o heurístico según el tamaño del problema.
	- Ensambla la lista de `RouteLeg` usando `RouteSplicer`.
	- Devuelve un `RouteResult` con:
		- Orden de visita (`visit_order`).
		- Lista de tramos (`legs`).
		- Tiempo total (`total_seconds`) y distancia total (`total_distance_m`).
		- Resumen del algoritmo utilizado.

**Integración con IA (Gemini)**

- La función `route_with_gemini` permite delegar la decisión del orden de visita a
	un modelo de lenguaje (Gemini), mientras que los caminos y tiempos por tramo
	siguen siendo calculados por el motor local.
- Está implementada de forma opcional y robusta: ante errores o JSON inválido,
	aplica un **fallback local** (Held‑Karp o heurística).

### 2.2 Interfaz web (`streamlit_app.py`)

La app de Streamlit ofrece una UI interactiva para configurar y visualizar rutas:

- **Configuración lateral (sidebar)**:
	- Selección de algoritmo base: A*, Dijkstra, BFS.
	- Modo de ruta: abierto, circuito o en orden fijo.
	- Opción para usar IA (Gemini) en el ordenamiento de visitas.
	- Color de calles por clase o por nivel de tráfico estimado.
	- Controles de “Hora del día” y “Velocidad del conductor (km/h)”.

- **POIs (puntos de interés)**:
	- `make_pois` genera un conjunto pequeño de POIs (Base Central, Plaza, Escuela, etc.)
		mapeados a nodos del grafo.
	- El usuario elige un **origen** y uno o varios **destinos** por nombre.

- **Visualización con PyDeck**:
	- `build_road_layer` dibuja la red vial (PathLayer) y colorea las calles según
		clase o congestión.
	- `build_route_layers` dibuja la ruta resultante y los marcadores de origen/destinos.

- **Panel de métricas**:
	- Muestra tiempo total estimado en minutos, distancia total en km y algoritmo usado.
	- Detalla cada tramo (`RouteLeg`): origen/destino (por nombre), segundos y km.

### 2.3 Tests y profiling (`tests_profiling.py`)

Este módulo concentra:

- **Tests unitarios** sobre:
	- Construcción del grafo y no vacuidad.
	- Consistencia de costos entre Dijkstra y A* cuando existe ruta.
	- Validez básica de rutas BFS.
	- Formas y contenido de la matriz par‑a‑par.
	- Comportamiento de circuito (regreso al origen).
	- Herramientas de profiling (`run_profiled`, `profile_call`).
	- Memoización en `DijkstraRouter`.

- **Utilidades CLI** para:
	- `cmd_smoke`: smoke test end‑to‑end (un tramo representativo).
	- `cmd_tests`: ejecutar toda la suite de tests.
	- `cmd_profile`: perfilar una ruta multi‑destino y ver el top de funciones
		más costosas según `pstats`.

---

## 3. Tecnologías y técnicas utilizadas

- **Lenguaje**: Python 3.
- **Front**: Streamlit + PyDeck.
- **Estructuras de datos**: grafos dirigidos, listas de adyacencia, LRU cache.
- **Algoritmos de ruteo**: Dijkstra, A*, BFS.
- **Optimización**:
	- Batching de vecinos y uso de generators (`yield`).
	- Concurrencia con `ThreadPoolExecutor` para par‑a‑par.
	- Memoización de Dijkstra y caché LRU de tramos.
- **Optimización de rutas multi‑stop**: Held‑Karp (exacto) y heurística NN + 2‑opt.
- **Profiling**: `cProfile`, `pstats`, utilidades `run_profiled`, `profile_call`,
	y `ProfileTimer` (context manager para tiempos wall‑clock).
- **Testing**: `unittest`, con casos que reflejan tanto el motor interno como
	comportamientos cercanos al front.

---

## 4. Guía de uso desde la terminal

Para los siguientes comandos se asume que estás en el directorio raíz del proyecto:

```powershell
cd C:\Users\Gsu\Documents\FinalPEF
```

Y que estás utilizando el entorno virtual configurado en `.venv`.

### 4.1 Ejecutar la aplicación Streamlit

Para levantar el front web:

```powershell
C:/Users/Gsu/Documents/FinalPEF/.venv/Scripts/streamlit.exe run streamlit_app.py
```

Esto abrirá (o actualizará) la app en tu navegador en `http://localhost:8501`.

### 4.2 Ejecutar los tests unitarios

Para correr toda la suite de tests definida en `tests_profiling.py`:

```powershell
C:/Users/Gsu/Documents/FinalPEF/.venv/Scripts/python.exe tests_profiling.py --tests
```

Opcionalmente, puedes aumentar la verbosidad:

```powershell
C:/Users/Gsu/Documents/FinalPEF/.venv/Scripts/python.exe tests_profiling.py --tests --verbosity 2
```

### 4.3 Ejecutar el smoke test

El smoke test valida rápidamente que el motor puede calcular al menos un tramo
alcanzable y muestra una métrica sencilla en consola:

```powershell
C:/Users/Gsu/Documents/FinalPEF/.venv/Scripts/python.exe tests_profiling.py --smoke
```

Salida típica:

```text
Smoke OK: 0→1 · 0.3 min · 0.12 km
```

### 4.4 Ejecutar profiling de una ruta multi‑destino

Para perfilar una ruta con varios puntos (similar al uso del front):

```powershell
C:/Users/Gsu/Documents/FinalPEF/.venv/Scripts/python.exe tests_profiling.py \
	--profile \
	--algo astar \
	--mode visit_all_circuit \
	--n 6 \
	--hour 8 \
	--driver-max-kmh 40
```

Parámetros clave:

- `--algo`: `astar`, `dijkstra` o `bfs`.
- `--mode`: `visit_all_open`, `visit_all_circuit` o `fixed_order`.
- `--n`: cantidad de waypoints tomados de la red (mínimo 2).
- `--hour`: hora del día (0–23) para el modelo de tráfico.
- `--driver-max-kmh`: velocidad máxima del conductor.

La salida incluye un resumen de la ruta (tiempo y distancia total) y el top de
funciones más costosas según `pstats` (cumtime).

---

## 5. Resumen

Este proyecto integra:

- Un **motor de ruteo configurable** con varios algoritmos y un modelo de tráfico
	realista.
- Una **UI interactiva** que permite explorar rutas, algoritmos y modos de visita
	sobre una ciudad ficticia pero coherente.
- Un set de **tests y herramientas de profiling** que facilitan validar la
	corrección y medir el rendimiento.

Este README funciona como informe técnico del sistema, cubriendo propósito,
arquitectura, tecnologías y forma de ejecución, y sirve como base para
extensiones futuras (por ejemplo, incorporar datos reales de OSM o más heurísticas
de optimización de rutas).

