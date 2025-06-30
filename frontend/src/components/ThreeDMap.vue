<template>
  <div class="three-d-map" ref="container" @click="handleClick"></div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, shallowRef } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'

// 场景元素 - 使用 shallowRef 来避免深层代理
const container = ref(null)
const scene = shallowRef(null)
const camera = shallowRef(null)
const renderer = shallowRef(null)
const controls = shallowRef(null)
const raycaster = shallowRef(new THREE.Raycaster())
const mouse = shallowRef(new THREE.Vector2())

// 标记点管理
const markers = shallowRef([])
const currentPlane = shallowRef(null)
const isFloorplanLoaded = ref(false)

// 轨迹线条管理
const trajectoryLines = shallowRef([])
const currentTrajectoryEventId = ref(null)

// Store floorplan dimensions for coordinate conversion
const floorplanImageWidth = ref(0)
const floorplanImageHeight = ref(0)
const floorplanPlaneWidth = ref(0)
const floorplanPlaneHeight = ref(0)

// 初始化场景
const initScene = () => {
  if (!container.value) {
    console.error('Container not found')
    return
  }

  try {
    // 创建场景
    const newScene = new THREE.Scene()
    newScene.background = new THREE.Color(0x001529)
    scene.value = newScene

    // 创建相机
    const aspect = container.value.clientWidth / container.value.clientHeight
    const newCamera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000)
    newCamera.position.set(0, 5, 10)
    newCamera.lookAt(0, 0, 0)
    camera.value = newCamera

    // 创建渲染器
    const newRenderer = new THREE.WebGLRenderer({ 
      antialias: true,
      alpha: true
    })
    newRenderer.setSize(container.value.clientWidth, container.value.clientHeight)
    newRenderer.setPixelRatio(window.devicePixelRatio)
    container.value.appendChild(newRenderer.domElement)
    renderer.value = newRenderer

    // 添加控制器
    const newControls = new OrbitControls(camera.value, renderer.value.domElement)
    newControls.enableDamping = true
    newControls.dampingFactor = 0.05
    newControls.minDistance = 2
    newControls.maxDistance = 20
    controls.value = newControls

    // 添加光源
    const ambientLight = new THREE.AmbientLight(0x404040, 1)
    scene.value.add(ambientLight)

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8)
    directionalLight.position.set(1, 1, 1)
    scene.value.add(directionalLight)

    console.log('Scene initialized successfully')
    
    // 开始动画循环
    animate()
  } catch (error) {
    console.error('Error initializing scene:', error)
  }
}

// 创建标记点 (Now expects scene coordinates)
const createMarker = (scenePosition, data) => {
  const isAbnormal = data.is_abnormal || false;
  const markerColor = isAbnormal ? 0xff0000 : 0x00ffff; // Red for abnormal, Cyan for normal
  const emissiveIntensityFactor = isAbnormal ? 0.6 : 0.5; // Slightly stronger glow for abnormal?

  // 创建圆锥体作为主体
  const coneGeometry = new THREE.ConeGeometry(0.15, 0.5, 32)
  const coneMaterial = new THREE.MeshPhongMaterial({
    color: markerColor,
    emissive: markerColor,
    emissiveIntensity: 0.5 * emissiveIntensityFactor, // Adjust base intensity
    shininess: 100,
    transparent: true,
    opacity: 0.8
  })
  
  const marker = new THREE.Group()
  const cone = new THREE.Mesh(coneGeometry, coneMaterial)
  cone.position.set(0, 0.25, 0)
  cone.rotation.x = Math.PI
  marker.add(cone)
  
  // 添加底座
  const baseGeometry = new THREE.CylinderGeometry(0.2, 0.2, 0.05, 32)
  const baseMaterial = new THREE.MeshPhongMaterial({
    color: markerColor,
    emissive: markerColor,
    emissiveIntensity: 0.3 * emissiveIntensityFactor,
    transparent: true,
    opacity: 0.5
  })
  const base = new THREE.Mesh(baseGeometry, baseMaterial)
  base.position.set(0, 0, 0)
  marker.add(base)
  
  // 添加发光环
  const ringGeometry = new THREE.TorusGeometry(0.2, 0.025, 16, 32)
  const ringMaterial = new THREE.MeshPhongMaterial({
    color: markerColor,
    emissive: markerColor,
    emissiveIntensity: 0.8 * emissiveIntensityFactor,
    transparent: true,
    opacity: 0.6
  })
  const ring = new THREE.Mesh(ringGeometry, ringMaterial)
  ring.rotation.x = Math.PI / 2
  ring.position.set(0, 0.025, 0)
  marker.add(ring)
  
  // 添加上升的光柱
  const beamGeometry = new THREE.CylinderGeometry(0.025, 0.025, 1, 8)
  const beamMaterial = new THREE.MeshBasicMaterial({
    color: markerColor, // Use marker color for beam too
    transparent: true,
    opacity: 0.3
  })
  const beam = new THREE.Mesh(beamGeometry, beamMaterial)
  beam.position.set(0, 0.5, 0)
  marker.add(beam)
  
  // 设置标记点位置 (Use scenePosition, slight offset in Y)
  const markerYOffset = 0.1; // Raise marker slightly above the plane
  marker.position.set(scenePosition.x, markerYOffset, scenePosition.z)
  marker.userData = data
  
  // 添加动画 (Adjust animation based on type? Optional)
  const animate = () => {
    ring.rotation.y += 0.02;
    // Maybe make abnormal pulse faster or different?
    const pulseSpeed = isAbnormal ? 0.005 : 0.003;
    const baseOpacity = isAbnormal ? 0.4 : 0.3;
    const opacityVariation = isAbnormal ? 0.3 : 0.2;
    beamMaterial.opacity = baseOpacity + Math.sin(Date.now() * pulseSpeed) * opacityVariation;
    coneMaterial.emissiveIntensity = (0.5 + Math.sin(Date.now() * (pulseSpeed * 0.8))) * emissiveIntensityFactor;
  }
  
  // 将动画函数添加到场景的更新列表中
  const updateFunctions = scene.value.userData.updateFunctions || []
  updateFunctions.push(animate)
  scene.value.userData.updateFunctions = updateFunctions
  
  scene.value.add(marker)
  markers.value = [...markers.value, marker]
  
  return marker
}

// 添加标记点 (Now accepts pixel coordinates and converts them)
const addMarker = (pixelPosition, data) => {
  console.log('%cAttempting to add marker with pixel coords:', 'color: orange;', pixelPosition, data);

  if (!isFloorplanLoaded.value || !floorplanPlaneWidth.value || !floorplanImageWidth.value) {
    console.warn('Floorplan not loaded or dimensions not available, cannot add marker yet.');
    return;
  }

  if (typeof pixelPosition.pixel_x !== 'number' || typeof pixelPosition.pixel_y !== 'number') {
      console.error('Invalid pixelPosition object received:', pixelPosition);
      return;
  }

  try {
    // 1. Normalize Pixel Coordinates
    const normalizedX = pixelPosition.pixel_x / floorplanImageWidth.value;
    const normalizedY = pixelPosition.pixel_y / floorplanImageHeight.value;

    // 2. Map to Plane Coordinates (Origin top-left -> Scene center, Y NOT inverted)
    const sceneX = (normalizedX - 0.5) * floorplanPlaneWidth.value;
    const sceneZ = (normalizedY - 0.5) * floorplanPlaneHeight.value; // REMOVED Negative sign to stop Y inversion

    console.log(`Converted pixel (${pixelPosition.pixel_x}, ${pixelPosition.pixel_y}) to scene (${sceneX.toFixed(2)}, ${sceneZ.toFixed(2)})`);

    // 3. Call createMarker with calculated scene coordinates
    return createMarker({ x: sceneX, z: sceneZ }, data);

  } catch (error) {
      console.error('Error converting pixel coordinates to scene coordinates:', error, pixelPosition);
      return null;
  }
}

// 清除所有标记点
const clearMarkers = () => {
  markers.value.forEach(marker => {
    scene.value.remove(marker)
  })
  markers.value = []
}

// 创建轨迹线条
const createTrajectoryLine = (coordinates, color, trackId) => {
  if (!coordinates || coordinates.length < 2) {
    console.warn(`轨迹 ${trackId} 坐标点不足，无法创建线条`)
    return null
  }

  try {
    // 将坐标转换为Three.js Vector3数组
    const points = coordinates.map(coord => 
      new THREE.Vector3(coord.x, coord.y + 0.05, coord.z) // 稍微抬高一点，避免与地面重叠
    )

    // 创建线条几何体
    const geometry = new THREE.BufferGeometry().setFromPoints(points)
    
    // 创建线条材质
    const material = new THREE.LineBasicMaterial({
      color: color,
      linewidth: 3,
      transparent: true,
      opacity: 0.8
    })

    // 创建线条对象
    const line = new THREE.Line(geometry, material)
    line.userData = {
      trackId: trackId,
      type: 'trajectory',
      color: color,
      pointCount: coordinates.length
    }

    // 添加轨迹点标记（可选）
    const pointMaterial = new THREE.MeshBasicMaterial({
      color: color,
      transparent: true,
      opacity: 0.6
    })
    
    coordinates.forEach((coord, index) => {
      // 起点和终点使用不同大小的球体
      const isStart = index === 0
      const isEnd = index === coordinates.length - 1
      const radius = isStart || isEnd ? 0.08 : 0.04
      
      const pointGeometry = new THREE.SphereGeometry(radius, 8, 6)
      const point = new THREE.Mesh(pointGeometry, pointMaterial.clone())
      
      if (isStart) {
        point.material.color.setHex(0x00ff00) // 起点绿色
      } else if (isEnd) {
        point.material.color.setHex(0xff0000) // 终点红色
      }
      
      point.position.set(coord.x, coord.y + 0.06, coord.z)
      point.userData = {
        trackId: trackId,
        type: 'trajectory_point',
        index: index,
        isStart: isStart,
        isEnd: isEnd
      }
      
      line.add(point)
    })

    console.log(`创建轨迹线条 - Track ID: ${trackId}, 颜色: ${color}, 点数: ${coordinates.length}`)
    return line

  } catch (error) {
    console.error(`创建轨迹线条失败 - Track ID: ${trackId}:`, error)
    return null
  }
}

// 显示事件轨迹
const showEventTrajectory = async (eventId) => {
  if (!isFloorplanLoaded.value) {
    console.warn('平面图未加载，无法显示轨迹')
    return false
  }

  try {
    console.log(`开始加载事件 ${eventId} 的轨迹数据`)
    
    // 清除当前轨迹
    clearTrajectories()
    
    // 获取轨迹数据
    const response = await fetch(`/api/trajectory/${eventId}/scene_coords`)
    
    if (!response.ok) {
      const errorData = await response.json()
      console.error('获取轨迹数据失败:', errorData.error)
      return false
    }

    const trajectoryData = await response.json()
    console.log('轨迹数据获取成功:', trajectoryData)

    if (!trajectoryData.trajectories || trajectoryData.trajectories.length === 0) {
      console.warn('该事件没有轨迹数据')
      return false
    }

    // 创建每条轨迹的线条
    const createdLines = []
    for (const trajectory of trajectoryData.trajectories) {
      const line = createTrajectoryLine(
        trajectory.coordinates,
        trajectory.color,
        trajectory.track_id
      )
      
      if (line) {
        scene.value.add(line)
        createdLines.push(line)
      }
    }

    trajectoryLines.value = createdLines
    currentTrajectoryEventId.value = eventId

    console.log(`成功显示 ${createdLines.length} 条轨迹线条`)
    return true

  } catch (error) {
    console.error('显示轨迹失败:', error)
    return false
  }
}

// 清除所有轨迹线条
const clearTrajectories = () => {
  trajectoryLines.value.forEach(line => {
    if (scene.value) {
      scene.value.remove(line)
    }
    
    // 清理几何体和材质
    if (line.geometry) {
      line.geometry.dispose()
    }
    if (line.material) {
      line.material.dispose()
    }
    
    // 清理子对象（轨迹点）
    line.children.forEach(child => {
      if (child.geometry) child.geometry.dispose()
      if (child.material) child.material.dispose()
    })
  })
  
  trajectoryLines.value = []
  currentTrajectoryEventId.value = null
  console.log('已清除所有轨迹线条')
}

// 切换轨迹显示
const toggleTrajectory = async (eventId) => {
  console.log(`toggleTrajectory 被调用 - eventId: ${eventId}`)
  console.log(`当前显示的轨迹事件ID: ${currentTrajectoryEventId.value}`)
  console.log(`平面图是否已加载: ${isFloorplanLoaded.value}`)
  console.log(`3D场景是否存在: ${!!scene.value}`)
  
  if (currentTrajectoryEventId.value === eventId) {
    // 如果当前显示的就是这个事件的轨迹，则清除
    console.log('清除当前轨迹')
    clearTrajectories()
    return false
  } else {
    // 显示新的轨迹
    console.log('显示新轨迹')
    return await showEventTrajectory(eventId)
  }
}

// 加载平面图
const loadFloorPlan = (imagePath, onLoadCallback) => {
  isFloorplanLoaded.value = false
  // Reset dimensions
  floorplanImageWidth.value = 0;
  floorplanImageHeight.value = 0;
  floorplanPlaneWidth.value = 0;
  floorplanPlaneHeight.value = 0;

  console.log('Loading floor plan:', imagePath)
  const textureLoader = new THREE.TextureLoader()
  textureLoader.load(
    imagePath,
    (texture) => {
      console.log('Texture loaded successfully');
      // Store image dimensions
      floorplanImageWidth.value = texture.image.width;
      floorplanImageHeight.value = texture.image.height;

      // 计算图片宽高比
      const imageAspect = floorplanImageWidth.value / floorplanImageHeight.value;
      // 增大平面宽度
      const planeWidth = 25;
      const planeHeight = planeWidth / imageAspect;
      // Store plane dimensions
      floorplanPlaneWidth.value = planeWidth;
      floorplanPlaneHeight.value = planeHeight;
      console.log(`Image size: ${floorplanImageWidth.value}x${floorplanImageHeight.value}`);
      console.log(`Setting plane size: width=${planeWidth}, height=${planeHeight}`);

      // 创建平面
      const geometry = new THREE.PlaneGeometry(planeWidth, planeHeight)
      const material = new THREE.MeshPhongMaterial({
        map: texture,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.9
      })

      // 移除旧的平面
      if (currentPlane.value) {
        scene.value.remove(currentPlane.value)
      }

      // 创建新平面
      const plane = new THREE.Mesh(geometry, material)
      plane.rotation.x = -Math.PI / 2
      scene.value.add(plane)
      currentPlane.value = plane

      // 调整相机位置
      const maxDimension = Math.max(planeWidth, planeHeight)
      camera.value.position.set(0, maxDimension * 0.8, maxDimension * 1.2)
      camera.value.lookAt(0, 0, 0)
      controls.value.update()
      
      // Set the flag when the plane is ready
      isFloorplanLoaded.value = true
      console.log('Floorplan loaded flag set to true. Dimensions stored.');

      // Call the callback after plane is ready
      if (onLoadCallback) {
          console.log('Executing onLoadCallback after floor plan loaded.')
          onLoadCallback()
      }
    },
    undefined,
    (error) => {
      console.error('Error loading floor plan:', error)
    }
  )
}

// 动画循环
let animationFrameId = null

const animate = () => {
  if (!renderer.value || !scene.value || !camera.value) {
    if (animationFrameId) {
      cancelAnimationFrame(animationFrameId)
    }
    return
  }
  
  animationFrameId = requestAnimationFrame(animate)
  
  if (controls.value) {
    controls.value.update()
  }
  
  // 更新所有标记点的动画
  if (scene.value.userData.updateFunctions) {
    scene.value.userData.updateFunctions.forEach(fn => fn())
  }
  
  try {
    renderer.value.render(scene.value, camera.value)
  } catch (error) {
    console.error('Render error:', error)
    if (animationFrameId) {
      cancelAnimationFrame(animationFrameId)
    }
  }
}

// 清理函数
const cleanup = () => {
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId)
  }
  
  // 清除轨迹线条
  clearTrajectories()
  
  if (scene.value) {
    // 清除更新函数
    scene.value.userData.updateFunctions = []
    
    scene.value.traverse((object) => {
      if (object.geometry) {
        object.geometry.dispose()
      }
      if (object.material) {
        if (Array.isArray(object.material)) {
          object.material.forEach(material => material.dispose())
        } else {
          object.material.dispose()
        }
      }
    })
  }
  
  if (renderer.value) {
    renderer.value.dispose()
  }
  
  markers.value = []
  trajectoryLines.value = []
  currentPlane.value = null
  currentTrajectoryEventId.value = null
  scene.value = null
  camera.value = null
  controls.value = null
  renderer.value = null
}

// 窗口大小改变时调整渲染器大小
const handleResize = () => {
  if (!container.value || !camera.value || !renderer.value) return
  
  camera.value.aspect = container.value.clientWidth / container.value.clientHeight
  camera.value.updateProjectionMatrix()
  renderer.value.setSize(container.value.clientWidth, container.value.clientHeight)
}

// 处理点击事件
const handleClick = (event) => {
  if (!container.value || !camera.value || !scene.value) return

  // 计算鼠标在归一化设备坐标中的位置
  const rect = container.value.getBoundingClientRect()
  mouse.value.x = ((event.clientX - rect.left) / container.value.clientWidth) * 2 - 1
  mouse.value.y = -((event.clientY - rect.top) / container.value.clientHeight) * 2 + 1

  // 更新射线
  raycaster.value.setFromCamera(mouse.value, camera.value)

  // 获取所有可点击的对象
  const clickableObjects = markers.value.map(marker => {
    // 收集标记点组中的所有网格
    const meshes = []
    marker.traverse((child) => {
      if (child.isMesh) {
        child.userData = marker.userData // 确保子网格也有相同的用户数据
        meshes.push(child)
      }
    })
    return meshes
  }).flat()

  // 检测与标记点的交点
  const intersects = raycaster.value.intersectObjects(clickableObjects)
  
  if (intersects.length > 0) {
    const clickedObject = intersects[0].object
    console.log('Clicked marker data:', clickedObject.userData)
    emit('markerClick', clickedObject.userData)
  }
}

// 生命周期钩子
onMounted(() => {
  console.log('ThreeDMap mounted')
  initScene()
  window.addEventListener('resize', handleResize)
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', handleResize)
  cleanup()
})

// 定义事件
const emit = defineEmits(['markerClick'])

// 暴露方法
defineExpose({
  loadFloorPlan,
  addMarker,
  clearMarkers,
  initScene,
  isFloorplanLoaded,
  showEventTrajectory,
  clearTrajectories,
  toggleTrajectory,
  currentTrajectoryEventId
})
</script>

<style scoped>
.three-d-map {
  width: 100%;
  height: 100%;
  background-color: #001529;
  position: relative;
}
</style> 