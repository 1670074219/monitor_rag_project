<template>
  <div class="daily-event-type-chart-panel">
    <h5>事件类型分布 (选中日期)</h5>
    <div ref="pieChartRef" class="pie-chart"></div>
    <div v-if="!hasData" class="no-data-chart">
        该日期无事件数据
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue';
import * as echarts from 'echarts/core';
import {
    TooltipComponent,
    LegendComponent,
    TitleComponent
} from 'echarts/components';
import {
    PieChart
} from 'echarts/charts';
import {
    CanvasRenderer
} from 'echarts/renderers';

echarts.use(
    [TitleComponent, TooltipComponent, LegendComponent, PieChart, CanvasRenderer]
);

// Props: Expecting events for the selected date
const props = defineProps({
  eventsForDate: { 
    type: Array,
    required: true,
    default: () => []
  }
});

// Ref for the chart container
const pieChartRef = ref(null);
// ECharts instance
let chartInstance = null;

// Check if there is data
const hasData = computed(() => props.eventsForDate && props.eventsForDate.length > 0);

// Calculate normal vs abnormal counts
const eventTypeCounts = computed(() => {
    let normalCount = 0;
    let abnormalCount = 0;

    props.eventsForDate.forEach(event => {
        // Use the is_abnormal flag directly from the event data
        if (event.is_abnormal) {
            abnormalCount++;
        } else {
            normalCount++;
        }
    });

    return {
        normal: normalCount,
        abnormal: abnormalCount
    };
});

// Function to initialize the chart
const initChart = () => {
  if (pieChartRef.value) {
    chartInstance = echarts.init(pieChartRef.value);
    updateChartOptions(); 
  } else {
    console.error("Daily Event Type Pie chart container ref not available for init.");
  }
};

// Function to update chart options
const updateChartOptions = () => {
  if (!chartInstance) return;

  const counts = eventTypeCounts.value;
  const pieData = [
      { value: counts.normal, name: '正常事件' },
      { value: counts.abnormal, name: '异常事件' }
  ].filter(item => item.value > 0); // Only show types with count > 0

  const option = {
    tooltip: {
        trigger: 'item',
        formatter: '{a} <br/>{b} : {c} ({d}%)' 
    },
    legend: {
        orient: 'horizontal',
        top: '0%',
        left: 'center',
        textStyle: {
            color: '#ccc',
            fontSize: 10
        }
    },
    color: ['#3ba272', '#ff8080'], // Green for normal, Red for abnormal
    series: [
        {
            name: '事件类型', 
            type: 'pie',
            radius: '75%',
            center: ['50%', '45%'],
            avoidLabelOverlap: true,
            itemStyle: {
                borderRadius: 5,
                borderColor: '#001529',
                borderWidth: 1
            },
            label: {
                show: true,
                position: 'outer',
                formatter: '{b}: {d}%',
                color: '#ccc',
                fontSize: 11
            },
            emphasis: {
                label: {
                    show: true,
                },
                itemStyle: {
                    shadowBlur: 10,
                    shadowOffsetX: 0,
                    shadowColor: 'rgba(0, 0, 0, 0.5)'
                }
            },
            labelLine: {
                show: true,
                length: 8,
                length2: 10,
                smooth: true,
                lineStyle: {
                    color: '#555'
                }
            },
            data: pieData 
        }
    ]
  };

  chartInstance.setOption(option);
};

// Watch for changes in input data
watch(() => props.eventsForDate, (newData) => {
    console.log("Events for date changed, updating event type pie chart.");
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
.daily-event-type-chart-panel {
  padding: 10px;
  background-color: rgba(0, 41, 82, 0.5);
  border-radius: 4px;
  color: #eee;
  flex-grow: 1; /* Allow panel to take space */
  display: flex;
  flex-direction: column;
  overflow: hidden; /* Hide overflow */
  min-height: 200px; /* Ensure minimum height */
  position: relative;
}

.daily-event-type-chart-panel h5 {
  margin-top: 0;
  margin-bottom: 5px;
  color: var(--primary-color, #00ffff);
  text-align: center;
  font-size: 1em;
  flex-shrink: 0; /* Prevent shrinking */
}

.pie-chart {
    width: 100%;
    height: 100%;
    flex-grow: 1; 
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