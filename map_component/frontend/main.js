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
let markersGroup = null;
let markersById = {};
let globalBounds = null;
let resetAdded = false;
let firstRender = true;

function ensureMap() {
  if (!map) {
    map = L.map("map").setView([0, 0], 3);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "© OpenStreetMap contributors",
    }).addTo(map);

    markersGroup = L.featureGroup().addTo(map);
  }
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

      // highlight marker được chọn
      Object.entries(markersById).forEach(([rid, m]) => {
        const sel = rid === routeId;
        m.setIcon(getIcon(sel));
      });

      map.setView([lat, lon], 15);
    });

    marker.addTo(markersGroup);
    markersById[routeId] = marker;
  });

  // Điều khiển view
  if (firstRender) {
    // Lần đầu: luôn overview tất cả route
    if (globalBounds) {
      map.fitBounds(globalBounds, { padding: [80, 80] });
    }
    firstRender = false;
  } else {
    if (selectedRouteId && markersById[selectedRouteId]) {
      // Có route đang chọn → zoom vào
      const latlng = markersById[selectedRouteId].getLatLng();
      map.setView(latlng, 15);
    } else if (globalBounds) {
      // Fallback: overview tất cả
      map.fitBounds(globalBounds, { padding: [80, 80] });
    }
  }

  addResetButton();
}

// ====================== STREAMLIT HOOKS ======================
function handleRender(args) {
  const routesData = args.data || [];
  const selectedRouteId = args.selected_route_id || null;
  const allRoutes = args.all_routes || [];

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
