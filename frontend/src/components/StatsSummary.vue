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
    <div class="camera-activity-chart-container">
        <h4>摄像头活跃度 (选中日期):</h4>
        <div ref="activityPieChartRef" class="pie-chart"></div>
         <div v-if="!hasActivityData" class="no-data-chart">
             选中日期无摄像头活跃数据
         </div>
      </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue';
import * as echarts from 'echarts/core';
import {
    TooltipComponent,
    LegendComponent,
    TitleComponent // Import TitleComponent if using title in options
} from 'echarts/components';
import {
    PieChart
} from 'echarts/charts';
import {
    CanvasRenderer
} from 'echarts/renderers';

// Register necessary components
echarts.use(
    [TitleComponent, TooltipComponent, LegendComponent, PieChart, CanvasRenderer]
);

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

// Ref for the chart container
const activityPieChartRef = ref(null);
// ECharts instance
let chartInstance = null;

// Computed property to check if there is data
const hasActivityData = computed(() => props.cameraActivityData && props.cameraActivityData.length > 0);

// Function to initialize the chart
const initChart = () => {
  if (activityPieChartRef.value) {
    chartInstance = echarts.init(activityPieChartRef.value);
    updateChartOptions(); // Initial setup
  } else {
    console.error("Pie chart container ref not available for init.");
  }
};

// Function to update chart options
const updateChartOptions = () => {
  if (!chartInstance) return;

  const pieData = props.cameraActivityData.map(item => ({
      name: `CAM ${item.id}`, // Format name as CAM XX
      value: item.count
  }));

  const option = {
    tooltip: {
        trigger: 'item',
        formatter: '{a} <br/>{b} : {c} ({d}%)' // Tooltip format
    },
    legend: {
        orient: 'vertical',
        left: 'left',
        top: 'center',
        textStyle: {
            color: '#ccc'
        },
        // Ensure legend items don't overflow container if too many cams
        type: 'scroll' // Add scroll type for legend
    },
    series: [
        {
            name: '摄像头事件数', // Series name for tooltip
            type: 'pie',
            radius: ['40%', '70%'], // Make it a donut chart
            center: ['65%', '50%'], // Adjust center to make space for legend
            avoidLabelOverlap: false,
            itemStyle: {
                borderRadius: 5,
                borderColor: '#001529',
                borderWidth: 1
            },
            label: {
                show: false, // Hide labels on slices
                position: 'center'
            },
            emphasis: {
                label: {
                    show: true,
                    fontSize: '16',
                    fontWeight: 'bold',
                    formatter: '{b}\n{c}次' // Show name and count on hover
                }
            },
            labelLine: {
                show: false
            },
            data: pieData.length > 0 ? pieData : [] // Use formatted data
            // Optionally add colors
            // color: ['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de', '#3ba272', '#fc8452', '#9a60b4', '#ea7ccc']
        }
    ]
  };

  chartInstance.setOption(option);
};

// Watch for changes in camera activity data
watch(() => props.cameraActivityData, (newData) => {
    console.log("Camera activity data changed, updating pie chart.");
    updateChartOptions();
}, { deep: true });

// Mount and unmount hooks
onMounted(() => {
  initChart();
});

onUnmounted(() => {
  if (chartInstance) {
    chartInstance.dispose();
    chartInstance = null;
  }
});

</script>

<style scoped>
.stats-summary-panel {
  padding: 15px;
  background-color: rgba(0, 41, 82, 0.5); /* Match other panels */
  border-radius: 4px;
  color: #eee;
  height: 100%; /* Try to fill the panel */
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

.camera-activity-chart-container {
    flex-grow: 1; /* Allow this section to take remaining space */
    display: flex;
    flex-direction: column;
    overflow: hidden; /* Important for chart resizing */
    background-color: rgba(0, 21, 41, 0.7);
    padding: 10px;
    border-radius: 3px;
}

.camera-activity-chart-container h4 {
    margin-top: 0;
    margin-bottom: 10px;
    color: #ccc;
    font-size: 1em;
    text-align: center;
    flex-shrink: 0; /* Prevent title from shrinking */
}

.pie-chart {
    width: 100%;
    height: 100%;
    flex-grow: 1; /* Allow chart to fill available space */
    min-height: 150px; /* Ensure minimum height */
}

.no-data-chart {
    position: absolute; /* Position over the chart area */
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: #666;
    font-style: italic;
}
</style> 