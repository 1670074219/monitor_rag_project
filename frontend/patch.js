const fs = require('fs');
let code = fs.readFileSync('src/components/ThreeDMap.vue', 'utf8');

// Add io import
code = code.replace("import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'", "import { io } from 'socket.io-client';\nimport { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'");

// Find the end before defineExpose
const insertPos = code.lastIndexOf('defineExpose({');

const snippet = `
// ==================== Real-time Tracking ====================
const realtimeSocket = ref(null);
const activeTracks = new Map();
const realtimeTimeoutMs = 3000;
const MAX_TRAIL_POINTS = 60;
let cleanupInterval = null;

function colorByTrackId(trackId) {
  const hue = (trackId * 67) % 360;
  return \`hsl(\${hue}, 85%, 52%)\`;
}

const connectTracking = () => {
  if (realtimeSocket.value) return;
  console.log("Connecting real-time tracking websocket...");
  realtimeSocket.value = io('http://' + window.location.hostname + ':5000/ws/tracking');

  realtimeSocket.value.on('tracking_point', (payload) => {
    const trackId = Number(payload.track_id);
    
    // Map coords
    const normalizedX = payload.x / floorplanImageWidth.value;
    const normalizedY = payload.y / floorplanImageHeight.value;
    const sceneX = (normalizedX - 0.5) * floorplanPlaneWidth.value;
    const sceneZ = (normalizedY - 0.5) * floorplanPlaneHeight.value; 

    const existing = activeTracks.get(trackId);
    if (!existing) {
      // Create marker geometries
      const color = colorByTrackId(trackId);
      
      // We can use a Sphere for the head marker
      const geometry = new THREE.SphereGeometry(0.15, 16, 16);
      const material = new THREE.MeshBasicMaterial({ color: new THREE.Color(color) });
      const marker = new THREE.Mesh(geometry, material);
      marker.position.set(sceneX, 0.1, sceneZ);
      scene.value.add(marker);

      // We'll store history of scene coords to rebuild line
      activeTracks.set(trackId, {
        marker,
        line: null,   // Will be created on 2nd point
        history: [{x: sceneX, y: 0.05, z: sceneZ}],
        color,
        lastSeenAt: Date.now()
      });
    } else {
      existing.marker.position.set(sceneX, 0.1, sceneZ);
      existing.history.push({x: sceneX, y: 0.05, z: sceneZ});
      if (existing.history.length > MAX_TRAIL_POINTS) {
        existing.history.shift();
      }
      existing.lastSeenAt = Date.now();
      
      // Update line
      if (existing.line) {
        scene.value.remove(existing.line);
        existing.line.geometry.dispose();
      }
      if (existing.history.length > 1) {
        // use Line2
        const positions = [];
        existing.history.forEach(coord => {
          positions.push(coord.x, coord.y, coord.z);
        });
        const geometry = new LineGeometry();
        geometry.setPositions(positions);
        const material = new LineMaterial({
          color: new THREE.Color(existing.color),
          linewidth: 3, 
          dashed: false,
          transparent: true,
          opacity: 0.7,
        });
        material.resolution.set(container.value.clientWidth, container.value.clientHeight);
        const line = new Line2(geometry, material);
        line.computeLineDistances();
        scene.value.add(line);
        existing.line = line;
      }
    }
  });

  cleanupInterval = setInterval(() => {
    const now = Date.now();
    activeTracks.forEach((track, trackId) => {
      if (now - track.lastSeenAt > realtimeTimeoutMs) {
        removeRealtimeTrack(trackId);
      }
    });
  }, 1000);
};

const removeRealtimeTrack = (trackId) => {
  const track = activeTracks.get(trackId);
  if (track) {
    if (track.marker) {
      scene.value.remove(track.marker);
      track.marker.geometry.dispose();
      track.marker.material.dispose();
    }
    if (track.line) {
      scene.value.remove(track.line);
      track.line.geometry.dispose();
      track.line.material.dispose();
    }
    activeTracks.delete(trackId);
  }
}

const disconnectTracking = () => {
  if (realtimeSocket.value) {
    realtimeSocket.value.disconnect();
    realtimeSocket.value = null;
  }
  if (cleanupInterval) {
    clearInterval(cleanupInterval);
    cleanupInterval = null;
  }
  activeTracks.forEach((_, trackId) => {
    removeRealtimeTrack(trackId);
  });
};

`

code = code.substring(0, insertPos) + snippet + code.substring(insertPos);
code = code.replace("defineExpose({", "defineExpose({\n  connectTracking,\n  disconnectTracking,");

fs.writeFileSync('src/components/ThreeDMap.vue', code);
