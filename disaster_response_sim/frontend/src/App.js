import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import './App.css';

function App() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [sortConfig, setSortConfig] = useState({ key: 'urgency', direction: 'descending' });

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
    fetchData(); // Initial fetch
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

  const locationCoords = {
    "Manitou": [38.8597, -104.9172],
    "La Ronge": [55.1001, -105.2842],
    "California": [36.7783, -119.4179],
    "Unknown": null
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
            <h2>Priority Needs</h2>
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
                  <tr key={index} className={item.urgency > 7 ? 'high-urgency' : ''}>
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
            <h2>Disaster Map</h2>
            <MapContainer center={[40, -100]} zoom={4} style={{ height: '400px', width: '100%' }}>
              <TileLayer
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                attribution='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              />
              {data
                .filter(item => locationCoords[item.location])
                .map((item, index) => (
                  <Marker key={index} position={locationCoords[item.location]}>
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