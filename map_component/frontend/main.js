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
let legendControl = null;
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

function getIcon(isSelected, baseColor = "#3388ff", selectedColor = "#ff3333") {
  const size = isSelected ? 40 : 28;
  const color = isSelected ? selectedColor : baseColor;

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

function boundsFromArray(boundsArr) {
  if (!Array.isArray(boundsArr) || boundsArr.length !== 2) return null;
  const [sw, ne] = boundsArr;
  if (
    !Array.isArray(sw) ||
    !Array.isArray(ne) ||
    sw.length !== 2 ||
    ne.length !== 2
  ) {
    return null;
  }

  const [minLat, minLon] = sw.map(Number);
  const [maxLat, maxLon] = ne.map(Number);

  if (
    !Number.isFinite(minLat) ||
    !Number.isFinite(minLon) ||
    !Number.isFinite(maxLat) ||
    !Number.isFinite(maxLon)
  ) {
    return null;
  }

  return L.latLngBounds(L.latLng(minLat, minLon), L.latLng(maxLat, maxLon));
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

function updateMarkers(
  routesData,
  selectedRouteId,
  allRoutes,
  focusBoundsArr,
  districtCenter
) {
  ensureMap();

  const list = routesData && routesData.length > 0
    ? routesData
    : (allRoutes || []);

  const hasRiskLevels = list.some((r) => (r.level || "").toString().length > 0);

  markersGroup.clearLayers();
  markersById = {};

  // Global bounds tính trên toàn bộ marker (để Reset View hiển thị lại tất cả)
  const allBoundsList = allRoutes && allRoutes.length > 0 ? allRoutes : list;
  globalBounds = computeBounds(allBoundsList);

  const currentBounds = computeBounds(list);
  const focusBounds = boundsFromArray(focusBoundsArr) || currentBounds || globalBounds;
  const hasDistrictCenter =
    districtCenter &&
    Number.isFinite(Number(districtCenter.lat)) &&
    Number.isFinite(Number(districtCenter.lon));

  const riskBaseColors = {
    high: "#e74c3c", // đỏ
    medium: "#f1c40f", // vàng
    low: "#2ecc71", // xanh lá
  };

  const riskSelectedColors = {
    high: "#c0392b",
    medium: "#d4ac0d",
    low: "#1e8449",
  };

  const defaultBaseColor = hasDistrictCenter ? "#3388ff" : "#3388ff";
  const defaultSelectedColor = hasDistrictCenter ? "#1b8a5a" : "#ff3333";

  list.forEach((r) => {
    const lat = Number(r.lat);
    const lon = Number(r.lon);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;

    const routeId = r.route_id;
    const level = (r.level || "").toString().toLowerCase();

    const baseColor = riskBaseColors[level] || defaultBaseColor;
    const selectedColor = riskSelectedColors[level] || defaultSelectedColor;
    const selected = routeId === selectedRouteId;

    const marker = L.marker([lat, lon], {
      icon: getIcon(selected, baseColor, selectedColor),
    });

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
        m.setIcon(getIcon(sel, baseColor, selectedColor));
      });

      map.setView([lat, lon], 14);
    });

    marker.addTo(markersGroup);
    markersById[routeId] = marker;
  });

  if (hasDistrictCenter) {
    const lat = Number(districtCenter.lat);
    const lon = Number(districtCenter.lon);
    const marker = L.marker([lat, lon], {
      icon: getIcon(true, "#3498db", "#3498db"),
    });

    marker.bindTooltip(
      `<div><b>${districtCenter.district || "Trung tâm quận"}</b></div>`,
      { direction: "top", offset: [0, -30] }
    );

    marker.addTo(markersGroup);
  }

  // Điều khiển view
  if (selectedRouteId && markersById[selectedRouteId]) {
    const latlng = markersById[selectedRouteId].getLatLng();
    map.setView(latlng, 14);
    firstRender = false;
  } else if (focusBounds) {
    map.fitBounds(focusBounds, { padding: [80, 80], maxZoom: 14 });
    firstRender = false;
  } else if (firstRender) {
    if (globalBounds) {
      map.fitBounds(globalBounds, { padding: [80, 80] });
    }
    firstRender = false;
  } else if (globalBounds) {
    map.fitBounds(globalBounds, { padding: [80, 80] });
  }

  addResetButton();
  if (legendControl) {
    legendControl.remove();
  }

  legendControl = L.control({
    position: "bottomleft",
    hasDistrictCenter,
    hasRiskLevels,
  });
  legendControl.onAdd = function () {
    const div = L.DomUtil.create("div", "traffic-legend");
    const commonWrapper =
      "background: rgba(255,255,255,0.9); padding:6px 10px; border-radius:8px; font-size:12px; box-shadow: 0 1px 3px rgba(0,0,0,0.25);";

    if (hasDistrictCenter) {
      div.innerHTML = `
        <div style="${commonWrapper}">
          <div style="margin-bottom:2px;">
            <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#3498db;margin-right:4px;border:1px solid #fff;"></span>
            Trung tâm quận
          </div>
          ${
            hasRiskLevels
              ? `
                <div style="margin-top:2px;">
                  <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#e74c3c;margin-right:4px;border:1px solid #fff;"></span>
                  Tuyến rủi ro cao
                </div>
                <div style="margin-top:2px;">
                  <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#f1c40f;margin-right:4px;border:1px solid #fff;"></span>
                  Tuyến rủi ro trung bình
                </div>
                <div style="margin-top:2px;">
                  <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#2ecc71;margin-right:4px;border:1px solid #fff;"></span>
                  Tuyến rủi ro thấp
                </div>
                <div style="margin-top:2px;">
                  <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#3388ff;margin-right:4px;border:1px solid #fff;"></span>
                  Tuyến chưa có nhãn rủi ro
                </div>`
              : `
                <div style="margin-top:2px;">
                  <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#3388ff;margin-right:4px;border:1px solid #fff;"></span>
                  Tuyến trong quận
                </div>`
          }
        </div>`;
      return div;
    }

    div.innerHTML = `
      <div style="${commonWrapper}">
        <div style="margin-bottom:2px;">
          <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#ff3333;margin-right:4px;border:1px solid #fff;"></span>
          Tuyến đang chọn
        </div>
        <div>
          <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#3388ff;margin-right:4px;border:1px solid #fff;"></span>
          Tuyến khác
        </div>
      </div>`;
    return div;
  };

  legendControl.addTo(map);
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
    Streamlit.setComponentValue(routeId);
  });

  return marker;
}

// ====================== STREAMLIT HOOKS ======================
function handleRender(args) {
  const routesData = args.data || [];
  const selectedRouteId = args.selected_route_id || null;
  const allRoutes = args.all_routes || [];
  const focusBoundsArr = args.focus_bounds || null;
  const districtCenter = args.district_center || null;

  updateMarkers(routesData, selectedRouteId, allRoutes, focusBoundsArr, districtCenter);

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
