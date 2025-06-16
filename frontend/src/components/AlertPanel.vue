<template>
  <div class="alert-panel">
    <div class="alert-header">
      <h3>🚨 实时预警</h3>
      <div class="alert-controls">
        <button @click="toggleAutoRefresh" :class="['btn', autoRefresh ? 'btn-active' : 'btn-inactive']">
          {{ autoRefresh ? '自动刷新' : '手动刷新' }}
        </button>
        <button @click="refreshAlerts" class="btn btn-refresh">刷新</button>
        <button @click="clearAlerts" class="btn btn-danger">清空</button>
      </div>
    </div>

    <!-- 预警统计 -->
    <div class="alert-stats">
      <div class="stat-item">
        <span class="stat-number">{{ stats.total_alerts }}</span>
        <span class="stat-label">总预警</span>
      </div>
      <div class="stat-item">
        <span class="stat-number">{{ stats.last_hour }}</span>
        <span class="stat-label">过去1小时</span>
      </div>
      <div class="stat-item high">
        <span class="stat-number">{{ stats.by_severity.high || 0 }}</span>
        <span class="stat-label">高危预警</span>
      </div>
    </div>

    <!-- 预警过滤器 -->
    <div class="alert-filters">
      <select v-model="filterSeverity" @change="applyFilters">
        <option value="">所有级别</option>
        <option value="high">高危</option>
        <option value="medium">中等</option>
        <option value="low">低危</option>
      </select>
      
      <select v-model="filterType" @change="applyFilters">
        <option value="">所有类型</option>
        <option value="fall">跌倒</option>
        <option value="fight">打架</option>
        <option value="loitering">异常停留</option>
        <option value="abnormal_pose">异常姿态</option>
      </select>
      
      <select v-model="filterCamera" @change="applyFilters">
        <option value="">所有摄像头</option>
        <option v-for="camera in cameras" :key="camera" :value="camera">{{ camera }}</option>
      </select>
    </div>

    <!-- 预警列表 -->
    <div class="alert-list" ref="alertList">
      <div 
        v-for="alert in filteredAlerts" 
        :key="alert.alert_id"
        :class="['alert-item', `severity-${alert.severity}`]"
        @click="showAlertDetail(alert)"
      >
        <div class="alert-icon">
          {{ getAlertIcon(alert.type) }}
        </div>
        
        <div class="alert-content">
          <div class="alert-title">
            {{ getAlertTitle(alert.type) }}
          </div>
          <div class="alert-description">
            {{ alert.description }}
          </div>
          <div class="alert-meta">
            <span class="alert-time">{{ alert.alert_time }}</span>
            <span class="alert-camera">{{ alert.camera_id }}</span>
            <span :class="['alert-severity', `severity-${alert.severity}`]">
              {{ getSeverityText(alert.severity) }}
            </span>
          </div>
        </div>
        
        <div class="alert-actions">
          <button @click.stop="viewAlertVideo(alert)" class="btn-action" title="查看视频">
            📹
          </button>
          <button @click.stop="dismissAlert(alert)" class="btn-action" title="忽略">
            ✕
          </button>
        </div>
      </div>
      
      <div v-if="filteredAlerts.length === 0" class="no-alerts">
        <div class="no-alerts-icon">✅</div>
        <div class="no-alerts-text">暂无预警信息</div>
      </div>
    </div>

    <!-- 预警详情弹窗 -->
    <div v-if="selectedAlert" class="alert-modal" @click="closeAlertDetail">
      <div class="alert-modal-content" @click.stop>
        <div class="modal-header">
          <h4>{{ getAlertTitle(selectedAlert.type) }} - 详情</h4>
          <button @click="closeAlertDetail" class="close-btn">✕</button>
        </div>
        
        <div class="modal-body">
          <div class="detail-section">
            <label>预警类型：</label>
            <span>{{ getAlertTitle(selectedAlert.type) }}</span>
          </div>
          
          <div class="detail-section">
            <label>严重程度：</label>
            <span :class="['severity-badge', `severity-${selectedAlert.severity}`]">
              {{ getSeverityText(selectedAlert.severity) }}
            </span>
          </div>
          
          <div class="detail-section">
            <label>摄像头：</label>
            <span>{{ selectedAlert.camera_id }}</span>
          </div>
          
          <div class="detail-section">
            <label>发生时间：</label>
            <span>{{ selectedAlert.alert_time }}</span>
          </div>
          
          <div class="detail-section">
            <label>详细描述：</label>
            <p>{{ selectedAlert.description }}</p>
          </div>
          
          <div class="detail-section">
            <label>置信度：</label>
            <span>{{ (selectedAlert.confidence * 100).toFixed(1) }}%</span>
          </div>
          
          <div v-if="selectedAlert.bbox" class="detail-section">
            <label>检测区域：</label>
            <span>{{ formatBbox(selectedAlert.bbox) }}</span>
          </div>
        </div>
        
        <div class="modal-footer">
          <button @click="viewAlertVideo(selectedAlert)" class="btn btn-primary">
            查看相关视频
          </button>
          <button @click="dismissAlert(selectedAlert)" class="btn btn-secondary">
            忽略此预警
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, reactive, onMounted, onUnmounted, computed } from 'vue'

export default {
  name: 'AlertPanel',
  emits: ['play-video', 'alert-dismissed'],
  setup(props, { emit }) {
    const alerts = ref([])
    const stats = reactive({
      total_alerts: 0,
      last_hour: 0,
      last_24_hours: 0,
      by_type: {},
      by_severity: { high: 0, medium: 0, low: 0 },
      by_camera: {}
    })
    
    const autoRefresh = ref(true)
    const filterSeverity = ref('')
    const filterType = ref('')
    const filterCamera = ref('')
    const selectedAlert = ref(null)
    const alertList = ref(null)
    
    let refreshInterval = null
    
    // 计算属性
    const cameras = computed(() => {
      return [...new Set(alerts.value.map(alert => alert.camera_id))]
    })
    
    const filteredAlerts = computed(() => {
      let filtered = alerts.value
      
      if (filterSeverity.value) {
        filtered = filtered.filter(alert => alert.severity === filterSeverity.value)
      }
      
      if (filterType.value) {
        filtered = filtered.filter(alert => alert.type === filterType.value)
      }
      
      if (filterCamera.value) {
        filtered = filtered.filter(alert => alert.camera_id === filterCamera.value)
      }
      
      return filtered
    })
    
    // 方法
    const fetchAlerts = async () => {
      try {
        const response = await fetch('/api/alerts?limit=100')
        const data = await response.json()
        
        if (data.status === 'success') {
          alerts.value = data.data.alerts
          
          // 滚动到顶部显示最新预警
          if (alertList.value && alerts.value.length > 0) {
            alertList.value.scrollTop = 0
          }
        }
      } catch (error) {
        console.error('获取预警信息失败:', error)
      }
    }
    
    const fetchStats = async () => {
      try {
        const response = await fetch('/api/alerts/stats')
        const data = await response.json()
        
        if (data.status === 'success') {
          Object.assign(stats, data.data)
        }
      } catch (error) {
        console.error('获取预警统计失败:', error)
      }
    }
    
    const refreshAlerts = async () => {
      await Promise.all([fetchAlerts(), fetchStats()])
    }
    
    const toggleAutoRefresh = () => {
      autoRefresh.value = !autoRefresh.value
      
      if (autoRefresh.value) {
        startAutoRefresh()
      } else {
        stopAutoRefresh()
      }
    }
    
    const startAutoRefresh = () => {
      if (refreshInterval) clearInterval(refreshInterval)
      
      refreshInterval = setInterval(() => {
        refreshAlerts()
      }, 5000) // 每5秒刷新一次
    }
    
    const stopAutoRefresh = () => {
      if (refreshInterval) {
        clearInterval(refreshInterval)
        refreshInterval = null
      }
    }
    
    const clearAlerts = async () => {
      if (confirm('确定要清空所有预警信息吗？')) {
        alerts.value = []
        Object.assign(stats, {
          total_alerts: 0,
          last_hour: 0,
          last_24_hours: 0,
          by_type: {},
          by_severity: { high: 0, medium: 0, low: 0 },
          by_camera: {}
        })
      }
    }
    
    const applyFilters = () => {
      // 过滤逻辑在computed中处理
    }
    
    const showAlertDetail = (alert) => {
      selectedAlert.value = alert
    }
    
    const closeAlertDetail = () => {
      selectedAlert.value = null
    }
    
    const viewAlertVideo = (alert) => {
      // 发送事件给父组件播放视频
      emit('play-video', {
        camera_id: alert.camera_id,
        timestamp: alert.timestamp,
        bbox: alert.bbox
      })
      closeAlertDetail()
    }
    
    const dismissAlert = (alert) => {
      const index = alerts.value.findIndex(a => a.alert_id === alert.alert_id)
      if (index !== -1) {
        alerts.value.splice(index, 1)
        emit('alert-dismissed', alert)
      }
      closeAlertDetail()
    }
    
    const getAlertIcon = (type) => {
      const icons = {
        fall: '🤕',
        fight: '👊',
        loitering: '⏰',
        abnormal_pose: '🚶',
        intrusion: '🚫'
      }
      return icons[type] || '⚠️'
    }
    
    const getAlertTitle = (type) => {
      const titles = {
        fall: '跌倒检测',
        fight: '打架检测',
        loitering: '异常停留',
        abnormal_pose: '异常姿态',
        intrusion: '入侵检测'
      }
      return titles[type] || '未知预警'
    }
    
    const getSeverityText = (severity) => {
      const texts = {
        high: '高危',
        medium: '中等', 
        low: '低危'
      }
      return texts[severity] || severity
    }
    
    const formatBbox = (bbox) => {
      if (!bbox || !Array.isArray(bbox)) return '未知'
      return `(${bbox[0]}, ${bbox[1]}) - (${bbox[2]}, ${bbox[3]})`
    }
    
    // 生命周期
    onMounted(() => {
      refreshAlerts()
      if (autoRefresh.value) {
        startAutoRefresh()
      }
    })
    
    onUnmounted(() => {
      stopAutoRefresh()
    })
    
    return {
      alerts,
      stats,
      autoRefresh,
      filterSeverity,
      filterType,
      filterCamera,
      selectedAlert,
      alertList,
      cameras,
      filteredAlerts,
      refreshAlerts,
      toggleAutoRefresh,
      clearAlerts,
      applyFilters,
      showAlertDetail,
      closeAlertDetail,
      viewAlertVideo,
      dismissAlert,
      getAlertIcon,
      getAlertTitle,
      getSeverityText,
      formatBbox
    }
  }
}
</script>

<style scoped>
.alert-panel {
  height: 100%;
  display: flex;
  flex-direction: column;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.alert-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px;
  border-bottom: 1px solid #eee;
}

.alert-header h3 {
  margin: 0;
  color: #333;
  font-size: 18px;
}

.alert-controls {
  display: flex;
  gap: 8px;
}

.btn {
  padding: 6px 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
  background: #fff;
  cursor: pointer;
  font-size: 12px;
  transition: all 0.2s;
}

.btn-active {
  background: #4CAF50;
  color: white;
  border-color: #4CAF50;
}

.btn-inactive {
  background: #f5f5f5;
  color: #666;
}

.btn-refresh {
  background: #2196F3;
  color: white;
  border-color: #2196F3;
}

.btn-danger {
  background: #f44336;
  color: white;
  border-color: #f44336;
}

.btn:hover {
  opacity: 0.8;
}

.alert-stats {
  display: flex;
  padding: 12px 16px;
  background: #f8f9fa;
  border-bottom: 1px solid #eee;
}

.stat-item {
  flex: 1;
  text-align: center;
}

.stat-item.high .stat-number {
  color: #f44336;
}

.stat-number {
  display: block;
  font-size: 20px;
  font-weight: bold;
  color: #333;
}

.stat-label {
  display: block;
  font-size: 12px;
  color: #666;
  margin-top: 4px;
}

.alert-filters {
  display: flex;
  gap: 8px;
  padding: 12px 16px;
  border-bottom: 1px solid #eee;
}

.alert-filters select {
  flex: 1;
  padding: 6px 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 12px;
}

.alert-list {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}

.alert-item {
  display: flex;
  align-items: center;
  padding: 12px;
  margin-bottom: 8px;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
  border-left: 4px solid;
}

.alert-item.severity-high {
  border-left-color: #f44336;
  background: #fff5f5;
}

.alert-item.severity-medium {
  border-left-color: #ff9800;
  background: #fff8f0;
}

.alert-item.severity-low {
  border-left-color: #4CAF50;
  background: #f8fff8;
}

.alert-item:hover {
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.alert-icon {
  font-size: 24px;
  margin-right: 12px;
}

.alert-content {
  flex: 1;
}

.alert-title {
  font-weight: bold;
  color: #333;
  margin-bottom: 4px;
}

.alert-description {
  color: #666;
  font-size: 14px;
  margin-bottom: 6px;
}

.alert-meta {
  display: flex;
  gap: 12px;
  font-size: 12px;
  color: #999;
}

.alert-severity.severity-high {
  color: #f44336;
  font-weight: bold;
}

.alert-severity.severity-medium {
  color: #ff9800;
  font-weight: bold;
}

.alert-severity.severity-low {
  color: #4CAF50;
  font-weight: bold;
}

.alert-actions {
  display: flex;
  gap: 4px;
}

.btn-action {
  width: 32px;
  height: 32px;
  border: none;
  border-radius: 4px;
  background: #f5f5f5;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  transition: all 0.2s;
}

.btn-action:hover {
  background: #e0e0e0;
}

.no-alerts {
  text-align: center;
  padding: 40px 20px;
  color: #999;
}

.no-alerts-icon {
  font-size: 48px;
  margin-bottom: 12px;
}

.no-alerts-text {
  font-size: 16px;
}

/* 弹窗样式 */
.alert-modal {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0,0,0,0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.alert-modal-content {
  background: white;
  border-radius: 8px;
  width: 500px;
  max-width: 90vw;
  max-height: 80vh;
  overflow-y: auto;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px;
  border-bottom: 1px solid #eee;
}

.modal-header h4 {
  margin: 0;
  color: #333;
}

.close-btn {
  background: none;
  border: none;
  font-size: 20px;
  cursor: pointer;
  color: #999;
}

.close-btn:hover {
  color: #333;
}

.modal-body {
  padding: 16px;
}

.detail-section {
  margin-bottom: 16px;
}

.detail-section label {
  display: inline-block;
  width: 80px;
  font-weight: bold;
  color: #333;
}

.severity-badge {
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: bold;
}

.severity-badge.severity-high {
  background: #ffebee;
  color: #f44336;
}

.severity-badge.severity-medium {
  background: #fff3e0;
  color: #ff9800;
}

.severity-badge.severity-low {
  background: #e8f5e8;
  color: #4CAF50;
}

.modal-footer {
  display: flex;
  gap: 8px;
  padding: 16px;
  border-top: 1px solid #eee;
}

.modal-footer .btn {
  flex: 1;
  padding: 10px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

.btn-primary {
  background: #2196F3;
  color: white;
}

.btn-secondary {
  background: #f5f5f5;
  color: #333;
}

.modal-footer .btn:hover {
  opacity: 0.8;
}
</style> 