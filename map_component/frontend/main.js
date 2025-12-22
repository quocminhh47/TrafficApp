import L from "leaflet";
import "leaflet/dist/leaflet.css";

const SET_COMPONENT_VALUE = "streamlit:setComponentValue";
const RENDER = "streamlit:render";
const COMPONENT_READY = "streamlit:componentReady";
const SET_FRAME_HEIGHT = "streamlit:setFrameHeight";

function sendMessage(type, data) {
  window.parent.postMessage(
    {
      isStreamlitMessage: true,
      type,
      ...data,
    },
    "*"
  );
}

function init() {
  sendMessage(COMPONENT_READY, { apiVersion: 1 });
}

function sendValue(val) {
  sendMessage(SET_COMPONENT_VALUE, { value: val });
}

function setFrameHeight(height) {
  sendMessage(SET_FRAME_HEIGHT, { height });
}

// ====================== LEAFLET ======================
let map = null;
let linesGroup = null;
let markersGroup = null;
let markersById = {};
let linesById = {};
let globalBounds = null;
let resetAdded = false;
let legendAdded = false;
let firstRender = true;
let routesPaneCreated = false;
let routeRenderer = null;

function ensureMap() {
  if (!map) {
    map = L.map("map").setView([0, 0], 3);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "© OpenStreetMap contributors",
    }).addTo(map);

    // Pane riêng cho tuyến đường để không che nhãn nền
    map.createPane("routes");
    map.getPane("routes").style.zIndex = 400;
    routesPaneCreated = true;
    routeRenderer = L.canvas({ padding: 0.5 });

    linesGroup = L.featureGroup().addTo(map);
    markersGroup = L.featureGroup().addTo(map);
  }
}

function getCongestionInfo(p) {
  const value = Number(p);
  if (!Number.isFinite(value)) {
    return { level: "unknown", color: "#6c757d", label: "Unknown", value: null };
  }

  if (value <= 0.3) {
    return { level: "low", color: "#2ecc71", label: "Low", value };
  }
  if (value <= 0.7) {
    return { level: "medium", color: "#f1c40f", label: "Medium", value };
  }
  return { level: "high", color: "#e74c3c", label: "High", value };
}

function getIcon(isSelected) {
  const size = isSelected ? 40 : 28;
  const color = isSelected ? "#ff3333" : "#3388ff";

  return L.divIcon({
    html: `
      <svg width="${size}" height="${size}" viewBox="0 0 24 24"
           fill="${color}" stroke="white" stroke-width="1.5"
           xmlns="http://www.w3.org/2000/svg"
           style="filter: drop-shadow(1px 2px 2px rgba(0,0,0,0.3));">
        <path d="M12 2C8.13 2 5 5.13 5 9
                 c0 5.25 7 13 7 13s7-7.75 7-13
                 c0-3.87-3.13-7-7-7z"/>
        <circle cx="12" cy="9" r="2.5" fill="white"/>
      </svg>
    `,
    className: "",
    iconSize: [size, size],
    iconAnchor: [size / 2, size],
  });
}

function computeBounds(list) {
  let minLat = 90,
    maxLat = -90,
    minLon = 180,
    maxLon = -180;

  list.forEach((r) => {
    const lat = Number(r.lat);
    const lon = Number(r.lon);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;

    minLat = Math.min(minLat, lat);
    maxLat = Math.max(maxLat, lat);
    minLon = Math.min(minLon, lon);
    maxLon = Math.max(maxLon, lon);
  });

  if (minLat > maxLat) return null;

  return L.latLngBounds(
    L.latLng(minLat, minLon),
    L.latLng(maxLat, maxLon)
  );
}

function addResetButton() {
  if (resetAdded || !map) return;

  const ResetControl = L.Control.extend({
    onAdd() {
      const btn = L.DomUtil.create("button", "reset-btn");
      btn.innerHTML = "Reset View";
      btn.title = "Show all supported routes";

      L.DomEvent.on(btn, "click", () => {
        if (globalBounds) {
          map.fitBounds(globalBounds, { padding: [80, 80] });
        }
      });

      return btn;
    },
  });

  new ResetControl({ position: "topleft" }).addTo(map);
  resetAdded = true;
}

function addLegend() {
  if (legendAdded || !map) return;

  const legend = L.control({ position: "bottomleft" });

  legend.onAdd = function () {
    const div = L.DomUtil.create("div", "traffic-legend");
    div.innerHTML = `
      <div style="background: rgba(255,255,255,0.92); padding:8px 10px; border-radius:8px; font-size:12px; box-shadow: 0 1px 3px rgba(0,0,0,0.25);">
        <div style="font-weight:600; margin-bottom:4px;">Tuyến đường</div>
        <div style="margin-bottom:4px; display:flex; align-items:center; gap:6px;">
          <span style="display:inline-block;width:20px;height:4px;background:#2ecc71;border-radius:2px;"></span>
          <span>Low (p ≤ 0.3)</span>
        </div>
        <div style="margin-bottom:4px; display:flex; align-items:center; gap:6px;">
          <span style="display:inline-block;width:20px;height:4px;background:#f1c40f;border-radius:2px;"></span>
          <span>Medium (0.3 < p ≤ 0.7)</span>
        </div>
        <div style="margin-bottom:8px; display:flex; align-items:center; gap:6px;">
          <span style="display:inline-block;width:20px;height:4px;background:#e74c3c;border-radius:2px;"></span>
          <span>High (p > 0.7)</span>
        </div>
        <div style="font-weight:600; margin-bottom:4px;">Marker</div>
        <div style="display:flex; align-items:center; gap:6px; margin-bottom:2px;">
          <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#ff3333;border:1px solid #fff;"></span>
          <span>Đang chọn</span>
        </div>
        <div style="display:flex; align-items:center; gap:6px;">
          <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#3388ff;border:1px solid #fff;"></span>
          <span>Khác</span>
        </div>
      </div>
    `;
    return div;
  };

  legend.addTo(map);
  legendAdded = true;
}

function getLineStyle(color, isSelected) {
  return {
    color,
    weight: isSelected ? 7 : 4,
    opacity: 0.85,
    lineCap: "round",
    lineJoin: "round",
    pane: routesPaneCreated ? "routes" : undefined,
    renderer: routeRenderer || undefined,
  };
}

function coordsToLatLngs(coords) {
  if (!Array.isArray(coords)) return [];
  return coords
    .map((pair) => {
      if (!Array.isArray(pair) || pair.length < 2) return null;
      const [lon, lat] = pair;
      if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
      return [lat, lon];
    })
    .filter(Boolean);
}

function parseGeojson(data) {
  if (!data) return null;
  let obj = data;
  if (typeof data === "string") {
    try {
      obj = JSON.parse(data);
    } catch (err) {
      console.error("Invalid GeoJSON string", err);
      return null;
    }
  }

  if (!obj || typeof obj !== "object") return null;
  if (obj.type === "FeatureCollection" && Array.isArray(obj.features)) {
    return obj.features;
  }
  if (obj.type === "Feature") {
    return [obj];
  }
  if (Array.isArray(obj.features)) {
    return obj.features;
  }
  return null;
}

function applySelection(routeId) {
  Object.entries(markersById).forEach(([rid, marker]) => {
    marker.setIcon(getIcon(rid === routeId));
  });

  Object.entries(linesById).forEach(([rid, info]) => {
    const isSelected = rid === routeId;
    info.layer.setStyle(getLineStyle(info.color, isSelected));
    if (isSelected && info.layer.bringToFront) {
      info.layer.bringToFront();
    }
  });
}

function updateLines(routeLinesGeojson, routeCongestion, selectedRouteId) {
  ensureMap();

  if (!linesGroup) return;
  linesGroup.clearLayers();
  linesById = {};

  const features = parseGeojson(routeLinesGeojson);
  if (!features || features.length === 0) return;

  features.forEach((feat) => {
    const geometry = feat.geometry || {};
    const props = feat.properties || {};
    const routeId = props.route_id || props.routeId;
    if (!routeId || !geometry.coordinates) return;

    const congestionInfo = getCongestionInfo(
      routeCongestion ? routeCongestion[routeId] : null
    );
    const isSelected = routeId === selectedRouteId;

    const featureLayer = L.geoJSON(feat, {
      coordsToLatLng: ([lon, lat]) => L.latLng(lat, lon),
      style: () => getLineStyle(congestionInfo.color, isSelected),
      pane: routesPaneCreated ? "routes" : undefined,
      renderer: routeRenderer || undefined,
      onEachFeature: (_, layer) => {
        const name = props.name || routeId;
        const percent = congestionInfo.value != null
          ? `${(congestionInfo.value * 100).toFixed(0)}%`
          : "N/A";
        const tooltipContent = `
          <div><b>${name}</b></div>
          <div><b>Route:</b> ${routeId}</div>
          <div><b>Level:</b> ${congestionInfo.label}</div>
          <div><b>p_cong:</b> ${percent}</div>
        `;
        layer.bindTooltip(tooltipContent, { sticky: true, className: "custom-tooltip" });

        layer.on("click", () => {
          sendValue(routeId);
          applySelection(routeId);
          const bounds = layer.getBounds();
          if (bounds && bounds.isValid && bounds.isValid()) {
            map.fitBounds(bounds, { padding: [60, 60] });
          }
          if (layer.bringToFront) {
            layer.bringToFront();
          }
        });
      },
    });

    featureLayer.addTo(linesGroup);
    linesById[routeId] = { layer: featureLayer, color: congestionInfo.color };
  });
}


function updateMarkers(routesData, selectedRouteId, allRoutes) {
  ensureMap();

  // Luôn vẽ marker cho TẤT CẢ các route (nhiều city)
  const list = (allRoutes && allRoutes.length > 0)
    ? allRoutes
    : (routesData || []);

  markersGroup.clearLayers();
  markersById = {};

  // Global bounds tính trên toàn bộ marker
  globalBounds = computeBounds(list);
  if (linesGroup && linesGroup.getLayers().length > 0) {
    const lineBounds = linesGroup.getBounds();
    globalBounds = globalBounds ? globalBounds.extend(lineBounds) : lineBounds;
  }

  list.forEach((r) => {
    const lat = Number(r.lat);
    const lon = Number(r.lon);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;

    const routeId = r.route_id;
    const selected = routeId === selectedRouteId;

    const marker = L.marker([lat, lon], { icon: getIcon(selected) });

    const name = r.name || routeId;
    const tooltipContent = `
      <div><b>${name}</b></div>
      <div><b>Route:</b> ${routeId}</div>
      <div><b>City:</b> ${r.city || ""}</div>
      <div><b>Zone:</b> ${r.zone || ""}</div>
    `;

    marker.bindTooltip(tooltipContent, {
      direction: "top",
      className: "custom-tooltip",
      offset: [0, -30],
    });

    marker.on("click", () => {
      // gửi route_id về Python
      sendValue(routeId);

      applySelection(routeId);

      map.setView([lat, lon], 15);
    });

    marker.addTo(markersGroup);
    markersById[routeId] = marker;
  });

  applySelection(selectedRouteId);

  // Điều khiển view
  if (selectedRouteId && markersById[selectedRouteId]) {
    // Có route đang chọn → zoom vào luôn
    const latlng = markersById[selectedRouteId].getLatLng();
    map.setView(latlng, 15);
    firstRender = false;
  } else if (firstRender) {
    // Lần đầu chưa có route → overview tất cả
    if (globalBounds) {
      map.fitBounds(globalBounds, { padding: [80, 80] });
    }
    firstRender = false;
  } else if (globalBounds) {
    // Fallback: overview
    map.fitBounds(globalBounds, { padding: [80, 80] });
  }


  addResetButton();
  addLegend();
}

function createMarker(lat, lon, isSelected, routeId, name) {
  const radius = isSelected ? 10 : 7;

  const color = isSelected ? "#ff4b4b" : "#2c7be5";
  const fillColor = isSelected ? "#ff4b4b" : "#2c7be5";

  const marker = L.circleMarker([lat, lon], {
    radius,
    color: "#ffffff",       // viền trắng
    weight: isSelected ? 3 : 1,
    fillColor,
    fillOpacity: isSelected ? 0.95 : 0.8,
  });

  marker.bindTooltip(name, {direction: "top", offset: [0, -8]});
  marker.on("click", () => {
    sendValue(routeId);
  });

  return marker;
}

// ====================== STREAMLIT HOOKS ======================
function handleRender(args) {
  const routesData = args.data || [];
  const selectedRouteId = args.selected_route_id || null;
  const allRoutes = args.all_routes || [];
  const routeLinesGeojson = args.route_lines_geojson || null;
  const routeCongestion = args.route_congestion || null;

  updateLines(routeLinesGeojson, routeCongestion, selectedRouteId);
  updateMarkers(routesData, selectedRouteId, allRoutes);

  const h =
    document.getElementById("map")?.getBoundingClientRect()?.height || 500;
  setFrameHeight(h);
}

window.addEventListener("message", (e) => {
  if (e.data?.type === RENDER) {
    handleRender(e.data.args);
  }
});

window.addEventListener("load", init);
