const deckContainer = document.getElementById("deck-container");

const data = window.parent.streamlit.getData();

mapboxgl.accessToken =
  "YOUR_MAPBOX_TOKEN"; // bạn thêm token mapbox của bạn

const INITIAL_VIEW_STATE = {
  longitude: data[0].lon,
  latitude: data[0].lat,
  zoom: 11,
  pitch: 0,
  bearing: 0,
};

const scatterLayer = new deck.ScatterplotLayer({
  id: "scatter-layer",
  data: data,
  getPosition: d => [d.lon, d.lat],
  getRadius: 300,
  getFillColor: d => [d.is_selected ? 255 : 80, 0, d.is_selected ? 0 : 80, 200],
  pickable: true,
  onClick: info => {
    if (info.object) {
      const msg = info.object.route_id;
      window.parent.streamlit.setComponentValue(msg);
    }
  },
});

const deckgl = new deck.DeckGL({
  container: "deck-container",
  mapStyle: "mapbox://styles/mapbox/light-v10",
  initialViewState: INITIAL_VIEW_STATE,
  layers: [scatterLayer],
});

window.parent.streamlit.ready();
