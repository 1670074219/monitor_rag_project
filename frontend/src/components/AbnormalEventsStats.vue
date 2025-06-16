<template>
  <div class="abnormal-events-panel abnormal-chart-panel">
    <h4>周异常事件趋势</h4>
    <div ref="weeklyAbnormalChartRef" class="line-chart"></div>
    <div v-if="!hasData" class="no-data-chart">
      暂无异常事件数据
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue';
import * as echarts from 'echarts/core';
import {
    TooltipComponent,
    GridComponent,
    LegendComponent
} from 'echarts/components';
import {
    LineChart
} from 'echarts/charts';
import {
    UniversalTransition
} from 'echarts/features';
import {
    CanvasRenderer
} from 'echarts/renderers';

// Register necessary components
echarts.use([
    TooltipComponent, GridComponent, LegendComponent, LineChart, CanvasRenderer, UniversalTransition
]);

// Props: Expecting weekly data now
const props = defineProps({
  weeklyAbnormalData: {
    type: Array,
    required: true,
    default: () => []
  }
});

// Ref for the chart container
const weeklyAbnormalChartRef = ref(null);
// ECharts instance
let chartInstance = null;

// Computed property to check if data exists
const hasData = computed(() => props.weeklyAbnormalData && props.weeklyAbnormalData.length > 0);

// Function to initialize the chart
const initChart = () => {
  if (weeklyAbnormalChartRef.value) {
    chartInstance = echarts.init(weeklyAbnormalChartRef.value);
    updateChartOptions(); // Initial setup
    // Add resize listener
    // Consider using a ResizeObserver for better performance if available
    window.addEventListener('resize', handleResize);
  } else {
    console.error("Line chart container ref not available for init.");
  }
};

// Function to update chart options
const updateChartOptions = () => {
  if (!chartInstance || !props.weeklyAbnormalData) return;

  // Ensure data is sorted by date for the line chart
  const sortedData = [...props.weeklyAbnormalData].sort((a, b) => new Date(a.date) - new Date(b.date));

  const dates = sortedData.map(item => item.date.substring(5)); // Extract MM-DD for label
  const counts = sortedData.map(item => item.count);

  const option = {
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'cross',
                label: {
                    backgroundColor: '#6a7985'
                }
            }
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '3%',
            containLabel: true
        },
        xAxis: [
            {
                type: 'category',
                boundaryGap: false,
                data: dates,
                axisLabel: {
                     color: '#ccc'
                },
                 axisLine: {
                     lineStyle: { color: '#555' }
                }
            }
        ],
        yAxis: [
            {
                type: 'value',
                minInterval: 1,
                splitLine: { lineStyle: { color: '#2a2a2a' } },
                 axisLabel: {
                     color: '#ccc'
                },
                 axisLine: {
                     show: true,
                     lineStyle: { color: '#555' }
                }
            }
        ],
        series: [
            {
                name: '异常事件数',
                type: 'line',
                stack: 'Total',
                smooth: true,
                lineStyle: {
                    width: 2,
                     color: '#ff8080'
                },
                showSymbol: false,
                areaStyle: {
                    opacity: 0.2,
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        {
                            offset: 0,
                            color: 'rgba(255, 128, 128, 0.5)'
                        },
                        {
                            offset: 1,
                            color: 'rgba(255, 128, 128, 0)'
                        }
                    ])
                },
                emphasis: {
                    focus: 'series',
                     label: {
                        show: true,
                        position: 'top'
                    }
                },
                data: counts
            }
        ]
    };

  chartInstance.setOption(option);
};

// Resize handler
const handleResize = () => {
    if (chartInstance) {
        chartInstance.resize();
    }
};

// Watch for changes in data
watch(() => props.weeklyAbnormalData, (newData) => {
    console.log("Weekly abnormal data changed, updating line chart.");
    updateChartOptions();
}, { deep: true });

// Mount and unmount hooks
onMounted(() => {
  initChart();
});

onUnmounted(() => {
   window.removeEventListener('resize', handleResize);
  if (chartInstance) {
    chartInstance.dispose();
    chartInstance = null;
  }
});

</script>

<style scoped>
.abnormal-chart-panel {
  background-color: rgba(41, 0, 0, 0.5);
  border: 1px solid rgba(255, 50, 50, 0.3);
  border-radius: 4px;
  padding: 10px;
  color: #eee;
  display: flex;
  flex-direction: column;
  flex-grow: 1;
  overflow: hidden;
  position: relative;
}

.abnormal-chart-panel h4 {
  margin-top: 0;
  margin-bottom: 10px;
  color: #ffcccc;
  text-align: center;
  font-size: 1.1em;
  flex-shrink: 0;
}

.line-chart {
    width: 100%;
    height: 100%;
    flex-grow: 1;
    min-height: 150px;
}

.no-data-chart {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: #aaa;
    font-style: italic;
}
</style> 