import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet'; // Import Leaflet for custom icons
import 'leaflet/dist/leaflet.css';
import './App.css';

// Define custom icons outside the component to avoid re-creation
const icons = {
  flood: L.icon({
    iconUrl: 'https://cdn-icons-png.flaticon.com/512/5809/5809490.png', // Flood icon
    iconSize: [32, 32],
    iconAnchor: [16, 32],
    popupAnchor: [0, -32]
  }),
  water: L.icon({
    iconUrl: 'https://cdn-icons-png.flaticon.com/512/5809/5809490.png', // Same as flood
    iconSize: [32, 32],
    iconAnchor: [16, 32],
    popupAnchor: [0, -32]
  }),
  fire: L.icon({
    iconUrl: 'https://cdn-icons-png.flaticon.com/512/1453/1453025.png', // Fire icon
    iconSize: [32, 32],
    iconAnchor: [16, 32],
    popupAnchor: [0, -32]
  }),
  food: L.icon({
    iconUrl: 'https://cdn-icons-png.flaticon.com/512/1037/1037762.png', // Food icon
    iconSize: [32, 32],
    iconAnchor: [16, 32],
    popupAnchor: [0, -32]
  }),
  trapped: L.icon({
    iconUrl: 'https://cdn-icons-png.flaticon.com/512/3208/3208247.png', // Trapped/rescue icon
    iconSize: [32, 32],
    iconAnchor: [16, 32],
    popupAnchor: [0, -32]
  }),
  rescue: L.icon({
    iconUrl: 'https://cdn-icons-png.flaticon.com/512/3208/3208247.png', // Same as trapped
    iconSize: [32, 32],
    iconAnchor: [16, 32],
    popupAnchor: [0, -32]
  }),
  help: L.icon({
    iconUrl: 'https://cdn-icons-png.flaticon.com/512/3208/3208247.png', // Help icon
    iconSize: [32, 32],
    iconAnchor: [16, 32],
    popupAnchor: [0, -32]
  }),
  urgent: L.icon({
    iconUrl: 'https://cdn-icons-png.flaticon.com/512/7596/7596805.png', // Same as help
    iconSize: [32, 32],
    iconAnchor: [16, 32],
    popupAnchor: [0, -32]
  }),
  'need help': L.icon({
    iconUrl: 'https://cdn-icons-png.flaticon.com/512/3208/3208247.png', // Same as help
    iconSize: [32, 32],
    iconAnchor: [16, 32],
    popupAnchor: [0, -32]
  }),
  'send help': L.icon({
    iconUrl: 'https://cdn-icons-png.flaticon.com/512/3208/3208247.png', // Same as help
    iconSize: [32, 32],
    iconAnchor: [16, 32],
    popupAnchor: [0, -32]
  }),
  None: L.icon({
    iconUrl: 'https://unpkg.com/leaflet@1.9.3/dist/images/marker-icon.png', // Default Leaflet marker
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34]
  })
};

// Fix default marker icon issue in React Leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.3/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.3/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.3/dist/images/marker-shadow.png'
});

function MapController({ center }) {
  const map = useMap();
  useEffect(() => {
    if (center) {
      map.flyTo(center, 10); // Zoom to level 10
    }
  }, [center, map]);
  return null;
}

function App() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [sortConfig, setSortConfig] = useState({ key: 'urgency', direction: 'descending' });
  const [mapCenter, setMapCenter] = useState(null);

  const fetchData = () => {
    setLoading(true);
    axios.get('http://localhost:5000/analyze?limit=1000')
      .then(response => {
        setData(response.data);
        setLoading(false);
      })
      .catch(error => {
        console.error('Error fetching data:', error);
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchData();
  }, []);

  const sortData = (key) => {
    let direction = 'ascending';
    if (sortConfig.key === key && sortConfig.direction === 'ascending') {
      direction = 'descending';
    }
    setSortConfig({ key, direction });
    const sortedData = [...data].sort((a, b) => {
      if (a[key] < b[key]) return direction === 'ascending' ? -1 : 1;
      if (a[key] > b[key]) return direction === 'ascending' ? 1 : -1;
      return 0;
    });
    setData(sortedData);
  };

  const handleRowClick = (coords) => {
    setMapCenter(coords);
  };

  return (
    <div className="App">
      <h1>Disaster Response Simulator</h1>
      <button onClick={fetchData} disabled={loading}>
        {loading ? 'Loading...' : 'Refresh Data'}
      </button>
      {loading && <p>Loading data, please wait...</p>}
      {!loading && data.length === 0 && <p>No data available.</p>}
      {!loading && data.length > 0 && (
        <div className="content">
          <div className="table-section">
            <h2>Priority Needs ({data.length} Entries)</h2>
            <table>
              <thead>
                <tr>
                  <th onClick={() => sortData('post')}>Post</th>
                  <th onClick={() => sortData('need')}>Need</th>
                  <th onClick={() => sortData('urgency')}>Urgency</th>
                  <th onClick={() => sortData('sentiment')}>Sentiment</th>
                  <th onClick={() => sortData('location')}>Location</th>
                </tr>
              </thead>
              <tbody>
                {data.map((item, index) => (
                  <tr
                    key={index}
                    className={item.urgency > 7 ? 'high-urgency' : ''}
                    onClick={() => handleRowClick(item.coordinates)}
                    style={{ cursor: 'pointer' }}
                  >
                    <td>{item.post}</td>
                    <td>{item.need !== 'None' ? item.need : '-'}</td>
                    <td>{item.urgency}</td>
                    <td>{item.sentiment}</td>
                    <td>{item.location !== 'Unknown' ? item.location : '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="map-section">
            <h2>Disaster Map (Top 50 by Urgency)</h2>
            <MapContainer center={[40, -100]} zoom={4} style={{ height: '400px', width: '100%' }}>
              <TileLayer
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                attribution='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              />
              <MapController center={mapCenter} />
              {data
                .filter(item => item.coordinates)
                .slice(0, 50)
                .map((item, index) => (
                  <Marker
                    key={index}
                    position={item.coordinates}
                    icon={icons[item.need] || icons.None} // Use need-specific icon or default
                  >
                    <Popup>
                      <b>{item.need !== 'None' ? item.need : 'No Need'}</b> (Urgency: {item.urgency})<br />
                      {item.post}<br />
                      Location: {item.location}
                    </Popup>
                  </Marker>
                ))}
            </MapContainer>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;