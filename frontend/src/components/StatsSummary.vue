<template>
  <div class="stats-summary-panel">
    <h3>统计摘要</h3>
    <div class="stats-grid">
      <div class="stat-item">
        <span class="stat-label">选中日期总事件:</span>
        <span class="stat-value">{{ selectedDateTotalEvents }}</span>
      </div>
      <div class="stat-item">
        <span class="stat-label">当前可见事件:</span>
        <span class="stat-value">{{ visibleEventsCount }}</span>
      </div>
      <!-- Add more stats here as needed -->
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue';

// Props definition
const props = defineProps({
  selectedDateTotalEvents: {
    type: Number,
    required: true,
    default: 0
  },
  visibleEvents: {
    type: Array,
    required: true,
    default: () => []
  },
  cameraActivityData: {
    type: Array,
    required: true,
    default: () => []
  }
});

// Computed property for visible events count
const visibleEventsCount = computed(() => props.visibleEvents.length);
</script>

<style scoped>
.stats-summary-panel {
  padding: 15px;
  background-color: rgba(0, 41, 82, 0.5); /* Match other panels */
  border-radius: 4px;
  color: #eee;
  /* height: 100%; Removed to allow auto-sizing */
  display: flex;
  flex-direction: column;
  gap: 15px; /* Add gap between sections */
}

.stats-summary-panel h3 {
  margin-top: 0;
  margin-bottom: 15px;
  color: var(--primary-color, #00ffff);
  text-align: center;
  border-bottom: 1px solid rgba(0, 255, 255, 0.2);
  padding-bottom: 10px;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); /* Responsive grid */
  gap: 10px;
  margin-bottom: 15px;
}

.stat-item {
  background-color: rgba(0, 21, 41, 0.7);
  padding: 10px;
  border-radius: 3px;
  text-align: center;
}

.stat-label {
  display: block;
  font-size: 0.9em;
  color: #aaa;
  margin-bottom: 5px;
}

.stat-value {
  display: block;
  font-size: 1.4em;
  font-weight: bold;
  color: var(--primary-color, #00ffff);
}
</style> 