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
  currentPlane.value = null
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
  isFloorplanLoaded
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