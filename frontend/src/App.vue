<template>
  <div class="smart-city">
    <header class="header">
      <h1>智慧监控数据可视化展示</h1>
       <!-- Add Dropdown Toggle Button -->
       <button @click="toggleDropdownChat" class="chat-dropdown-toggle-button">
         <span v-if="!isChatDropdownVisible">对话</span>
         <span v-else>收起</span>
       </button>
    </header>
    <div class="content">
      <div class="left-panel">
        <div class="date-picker-container">
          <label for="event-date">选择日期:</label>
          <input 
            type="date" 
            id="event-date" 
            v-model="selectedDate" 
          >
        </div>
        

        
        <!-- Re-add Event Details Display -->
        <div v-if="selectedEvent" class="event-details">
          <h3>事件详情</h3>
          <div class="event-content">
            <pre>{{ selectedEvent.content || '(无内容)' }}</pre>
            <button
              v-if="selectedEvent.videoFile"
              @click="playVideo(selectedEvent.videoFile)"
              class="play-button"
            >
              播放视频
            </button>
          </div>
        </div>
        <div v-else-if="!eventsLoading && events.length === 0 && !initialLoadPending" class="no-logs-message">
          {{ selectedDate }} 无日志记录
        </div>
        <div v-else-if="eventsLoading" class="loading-message">
          正在加载 {{selectedDate}} 的日志...
        </div>
        <div v-else class="no-selection">
           点击地图标记或选择日期查看详情
        </div>
        
        <!-- Add the new Pie Chart Section -->
        <div class="left-panel-section left-panel-bottom-chart">
           <DailyEventTypeChart :events-for-date="eventsForSelectedDate" />
        </div>
      </div>
      <div class="center-area-wrapper">
        <div class="center-panel">
          <ThreeDMap ref="threeDMap" @markerClick="handleMarkerClick" :onLoadCallback="onFloorplanLoad" />
        </div>
        <div class="center-bottom-panel">
           <TimeLine
             :events="events" 
             v-model="timeOfDaySeconds" 
             :displayDate="selectedDate"
             @markerClick="handleMarkerClick"
           />
        </div>
      </div>
      <!-- Populate Right Panel -->
      <div class="right-panel">
         <!-- Top Section: Stats Summary -->
         <div class="right-panel-section right-panel-top">
            <StatsSummary 
              :selected-date-total-events="eventsForSelectedDate.length" 
              :visible-events="visibleEvents"
              :camera-activity-data="cameraActivityForSelectedDate"
            />
         </div>
         
         <!-- Bottom Section: Abnormal Events -->
         <div class="right-panel-section right-panel-bottom">
            <AbnormalEventsStats 
              :weekly-abnormal-data="weeklyAbnormalData" 
            />
         </div>
      </div>
    </div>

    <!-- Chat Dropdown Panel -->
    <div :class="['chat-dropdown-panel', { 'is-visible': isChatDropdownVisible }]">
       <RagChat 
         v-if="isChatDropdownVisible" 
         @play-video-from-chat="handlePlayVideoFromChat" 
       /> 
    </div>
    <!-- End Chat Dropdown Panel -->

    <div v-if="videoUrl" class="video-modal">
      <div class="video-container">
        <button @click="closeVideoPlayer" class="close-button">&times;</button>
        <video :src="videoUrl" controls autoplay></video>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed, watch, nextTick } from 'vue'
import * as echarts from 'echarts'
import ThreeDMap from './components/ThreeDMap.vue'
import TimeLine from './components/TimeLine.vue'
import RagChat from './components/RagChat.vue'
import StatsSummary from './components/StatsSummary.vue'
import AbnormalEventsStats from './components/AbnormalEventsStats.vue'
import DailyEventTypeChart from './components/DailyEventTypeChart.vue'

// Remove state for the previous collapsible sidebar
// const isChatVisible = ref(true); 
// const toggleChat = () => { ... };

// Add state for the new dropdown chat
const isChatDropdownVisible = ref(false);

const toggleDropdownChat = () => {
  isChatDropdownVisible.value = !isChatDropdownVisible.value;
};


// 使用数据中存在的日期，而不是今天
const formatDate = (date) => date.toISOString().split('T')[0]

// 设置为2025-06-09，因为API数据中有这个日期的事件
const selectedDate = ref('2025-06-09')
const timeOfDaySeconds = ref(0)

// Restore absoluteTimestampForFiltering and other removed refs
const absoluteTimestampForFiltering = computed(() => {
  try {
    const dateStr = selectedDate.value
    const seconds = timeOfDaySeconds.value
    if (!dateStr || typeof seconds !== 'number') {
      console.warn("Invalid date or time for timestamp calculation", dateStr, seconds)
      return Math.floor(Date.now() / 1000)
    }
    const [year, month, day] = dateStr.split('-').map(Number)
    const date = new Date(year, month - 1, day)
    date.setSeconds(date.getSeconds() + seconds)
    return Math.floor(date.getTime() / 1000)
  } catch (e) {
    console.error("Error calculating absolute timestamp:", e)
    return Math.floor(Date.now() / 1000)
  }
})

const selectedEvent = ref(null)
const events = ref([])
const videoUrl = ref(null)
const eventsLoading = ref(false)
// End of restored refs

const initialLoadPending = ref(true) // New ref to track if initial load via onFloorplanLoad is pending
const threeDMap = ref(null)
const weeklyAbnormalData = ref([])
// 直接使用区域随机布局

const timeWindow = 600; // 调整时间窗口到2分钟（120秒）

const visibleEvents = computed(() => {
  if (!events.value || events.value.length === 0) {
    return [];
  }
  
  // 首先按选择的日期过滤事件
  const selectedDateEvents = events.value.filter(event => {
    if (typeof event.timestamp !== 'number') {
      return false;
    }
    
    // 将事件时间戳转换为日期字符串格式 YYYY-MM-DD
    const eventDate = new Date(event.timestamp * 1000);
    const eventDateStr = formatDate(eventDate);
    
    // 与选定日期比较
    return eventDateStr === selectedDate.value;
  });
  
  // 如果没有选定日期的事件，返回空数组
  if (selectedDateEvents.length === 0) {
    console.log(`没有找到日期 ${selectedDate.value} 的事件`);
    return [];
  }
  
  console.log(`找到 ${selectedDateEvents.length} 个日期为 ${selectedDate.value} 的事件`);
  
  // 然后按时间窗口过滤
  const currentFilterTime = absoluteTimestampForFiltering.value;
  const filteredEvents = selectedDateEvents.filter(event => {
    const diff = Math.abs(event.timestamp - currentFilterTime);
    const isInWindow = diff <= timeWindow;
    if (!isInWindow) {
      console.log(`事件 ${event.id} 不在时间窗口内: 事件时间=${new Date(event.timestamp * 1000).toLocaleTimeString()}, 当前时间=${new Date(currentFilterTime * 1000).toLocaleTimeString()}, 差值=${diff}秒`);
    }
    return isInWindow;
  });
  
  console.log(`时间窗口过滤后显示 ${filteredEvents.length} 个事件`);
  return filteredEvents;
});

const abnormalVisibleEvents = computed(() => {
    return visibleEvents.value.filter(event => {
        // --- Placeholder Logic for identifying abnormal events --- 
        // Option 1: Check for a specific type/flag (if available)
        // return event.type === 'abnormal'; 
        
        // Option 2: Check content for keywords (adjust keywords as needed)
        if (event.content && typeof event.content === 'string') {
            return event.content.includes('异常') || 
                   event.content.includes('打架') || 
                   event.content.includes('闯入');
        }
        return false; 
        // -----------------------------------------------------------
    });
});

const updateVisibleMarkers = (currentVisibleEvents) => { // Accept the list as argument
  if (!threeDMap.value) {
    console.warn('updateVisibleMarkers called but ThreeDMap component not ready');
    return;
  }

  console.log(`%c正在更新地图标记: ${currentVisibleEvents.length} 个可见事件`, 'color: lightblue; font-weight: bold;');
  console.log('当前可见事件详情:', currentVisibleEvents);

  threeDMap.value.clearMarkers(); // Clear existing markers

  let successCount = 0;
  currentVisibleEvents.forEach((event, index) => {
    // Use pixel_x and pixel_y from the event position data
    if (event.position && typeof event.position.pixel_x === 'number' && typeof event.position.pixel_y === 'number') {
        console.log(`添加标记 ${index + 1}/${currentVisibleEvents.length}: 事件ID=${event.id}, 位置=(${event.position.pixel_x}, ${event.position.pixel_y})`);
        // Pass the pixel coordinates object directly to addMarker
        threeDMap.value.addMarker(event.position, {
          id: event.id,
          timestamp: event.timestamp,
          content: event.content,
          videoFile: event.videoFile,
          cam_id: event.cam_id,
          date_str: event.date_str,
          time_str: event.time_str,
          index: event.index,
          sourceFile: event.sourceFile,
          // Pass original pixel coords in metadata too, might be useful
          pixel_x: event.position.pixel_x,
          pixel_y: event.position.pixel_y,
          is_abnormal: event.is_abnormal || false // Pass the abnormal flag
        });
        successCount++;
    } else {
        console.warn(`跳过标记添加 - 位置信息无效:`, event);
    }
  });
  
  console.log(`%c成功添加 ${successCount} 个标记到地图`, 'color: green; font-weight: bold;');
};

// Watch the computed property directly to update markers when it changes
watch(visibleEvents, (newVisibleEvents) => {
     console.log("Computed visibleEvents changed, triggering marker update.");
     updateVisibleMarkers(newVisibleEvents);
}, { deep: true }); // deep might be needed if event objects themselves change, but usually not for filtering

const handleMarkerClick = (markerData) => {
  console.log('Marker/Event clicked:', markerData);
  selectedEvent.value = markerData; // 更新左侧显示的事件详情
  // 当点击标记时，同时更新时间线到该事件的时间
  // REMOVED: This part caused the time slider to jump on click
  /*
  if (markerData && typeof markerData.timestamp === 'number') {
    const clickedDate = new Date(markerData.timestamp * 1000);
    const midnight = new Date(clickedDate);
    midnight.setHours(0, 0, 0, 0);
    // 计算点击事件的时间距离当天零点的秒数，并更新时间线滑块
    timeOfDaySeconds.value = Math.floor((clickedDate.getTime() - midnight.getTime()) / 1000);
  }
  */
};

const playVideo = (videoFileName) => {
  if (!videoFileName) {
    console.warn('playVideo called with empty filename');
    return;
  }
  console.log('playVideo called with:', videoFileName);
  videoUrl.value = `/api/video/${videoFileName}`;
  console.log('Video URL set to:', videoUrl.value);
};

const closeVideoPlayer = () => {
  videoUrl.value = null;
};

const handlePlayVideoFromChat = (videoFileName) => {
  console.log('handlePlayVideoFromChat called with:', videoFileName);
  if (videoFileName) {
    playVideo(videoFileName);
  } else {
    console.warn('handlePlayVideoFromChat received empty filename');
  }
};

const barChart = ref(null)
const eventStatsChartInstance = ref(null)
const initBarChart = () => {
  if (!barChart.value) return
  const myChart = echarts.init(barChart.value)
  const option = {
    title: { text: '数据概览', textStyle: { color: '#fff' } },
    tooltip: {},
    xAxis: {
      data: ['类别A', '类别B', '类别C', '类别D', '类别E'],
      axisLine: { lineStyle: { color: '#00ffff' } },
      axisLabel: { color: '#fff' }
    },
    yAxis: {
      axisLine: { lineStyle: { color: '#00ffff' } },
      axisLabel: { color: '#fff' },
      splitLine: { lineStyle: { color: 'rgba(0, 255, 255, 0.1)' } }
    },
    series: [{
      name: '数量', type: 'bar',
      data: [50, 200, 360, 100, 150],
      itemStyle: { color: '#00ffff' }
    }],
    grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
    }
  }
  myChart.setOption(option)
}

const initEventStatsChart = () => {
  const chartDom = document.getElementById('eventStatsChart')
  if (!chartDom) {
      console.warn("Element with id 'eventStatsChart' not found for initialization.");
      return; // Exit if the element doesn't exist yet
  }
  eventStatsChartInstance.value = echarts.init(chartDom)
  const option = {
    title: {
      text: '事件类型分布',
      left: 'center',
      textStyle: { color: '#fff' }
    },
    tooltip: {
      trigger: 'item'
    },
    legend: {
      top: 'bottom',
      textStyle: { color: '#fff' }
    },
    series: [
      {
        name: '事件类型',
        type: 'pie',
        radius: ['40%', '70%'],
        avoidLabelOverlap: false,
        itemStyle: {
            borderRadius: 10,
            borderColor: '#001529',
            borderWidth: 2
        },
        label: {
            show: false,
            position: 'center'
        },
        emphasis: {
             label: {
                show: true,
                fontSize: '20',
                fontWeight: 'bold',
                color: '#fff'
            },
          itemStyle: {
            shadowBlur: 10,
            shadowOffsetX: 0,
            shadowColor: 'rgba(0, 0, 0, 0.5)'
          }
        },
         labelLine: { 
             show: false
         },
        data: [ // Default/placeholder data
          { value: 0, name: '无数据' } 
        ]
      }
    ]
  }
  eventStatsChartInstance.value.setOption(option)
}

const fetchEvents = async () => {
  eventsLoading.value = true;
  console.log(`正在获取事件数据...`);
  try {
    // 直接使用区域随机布局
    const apiEndpoint = '/api/events_3d';
    const layoutType = '区域内随机分布';
    
    const response = await fetch(apiEndpoint);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    console.log(`获取到 ${data.length} 个事件（${layoutType}）`);

    if (Array.isArray(data)) { 
      const processedEvents = data.map((event, index) => ({
        id: event.id || `event-${Date.now()}-${index}`, 
        timestamp: typeof event.timestamp === 'number' ? event.timestamp : (event.datetime ? Math.floor(new Date(event.datetime).getTime() / 1000) : null), // Ensure timestamp is a number or null
        position: event.position, 
        content: event.content || '(无内容)', 
        videoFile: event.videoFile || null,
        cam_id: event.cam_id || 'N/A', // Ensure cam_id exists
        is_abnormal: event.is_abnormal || false // Extract and store the abnormal flag
      })).filter(e => e.timestamp !== null); // Filter out events where timestamp couldn't be parsed
      
      events.value = processedEvents; // Assign processed events
      console.log(`处理后的事件数量: ${events.value.length}`);
      
      // Update chart
      updateEventStatsChart(null); 
      
       // Markers will update automatically via the visibleEvents watcher

       // Select the first event by default if available
        if (events.value.length > 0 && !selectedEvent.value) {
          if(events.value[0].timestamp) { 
              handleMarkerClick(events.value[0]); 
          }
        } else if (events.value.length === 0) {
          selectedEvent.value = null; 
        }
       
    } else {
      console.warn('收到了意外的数据结构（不是数组）: ', data);
      events.value = []; 
      selectedEvent.value = null; 
      updateEventStatsChart(null); 
    }

  } catch (error) {
    console.error('获取事件数据出错:', error);
    events.value = []; 
    selectedEvent.value = null; 
    updateEventStatsChart(null); 
  } finally {
    eventsLoading.value = false;
    initialLoadPending.value = false; // Mark that initial load attempt has completed
  }
};

const updateEventStatsChart = (stats) => {
   if (!eventStatsChartInstance.value) {
       console.warn("Attempted to update event stats chart, but instance is not ready.");
       // Optionally try to re-initialize if the DOM element exists now
       const chartDom = document.getElementById('eventStatsChart');
       if (chartDom) initEventStatsChart(); 
       else return; // Still not ready
       // If re-init was successful, try setting options again, otherwise return
       if (!eventStatsChartInstance.value) return;
   }
   if (!stats) {
       console.warn("No stats data provided to updateEventStatsChart.");
       // Set to empty state
       eventStatsChartInstance.value.setOption({
           series: [{ data: [{ name: '无数据', value: 0 }] }]
       });
       return;
   }
  
  // Format stats data for ECharts pie chart
  const pieData = Object.entries(stats).map(([name, value]) => ({ name, value }));

  eventStatsChartInstance.value.setOption({
    series: [
      {
        // Ensure the series 'name' matches if needed for tooltips/legends
        name: '事件类型', 
        data: pieData.length > 0 ? pieData : [{name: '无数据', value: 0}] // Show placeholder if no data
      }
    ]
  });
};

// Function to be called when the floorplan image is loaded (called by ThreeDMap)
// Function to be called when the floorplan image is loaded
const onFloorplanLoad = () => {
  console.log("Floorplan loaded, fetching initial events...");
  initialLoadPending.value = true; // Set before fetchEvents
  fetchEvents(); 
  fetchWeeklyAbnormalData(); // Fetch weekly data too
};

// Fetch events when the selected date changes
watch(selectedDate, (newDate, oldDate) => {
  if (newDate !== oldDate) {
    initialLoadPending.value = false; // User changed date, not initial load
    fetchEvents();
  }
}, { immediate: false }); // Don't run immediately, wait for floorplan load

// --- New Computed Properties for Stats Summary --- 
// Filter events based on the selected date in the date picker
const eventsForSelectedDate = computed(() => {
    if (!selectedDate.value || !events.value) return [];
    return events.value.filter(event => {
        if (!event.timestamp) return false;
        const eventDate = new Date(event.timestamp * 1000);
        return formatDate(eventDate) === selectedDate.value;
    });
});

// Calculate camera activity based on events for the selected date
const cameraActivityForSelectedDate = computed(() => {
    if (!eventsForSelectedDate.value || eventsForSelectedDate.value.length === 0) {
        return [];
    }
    const counts = eventsForSelectedDate.value.reduce((acc, event) => {
        if (event.cam_id) {
            acc[event.cam_id] = (acc[event.cam_id] || 0) + 1;
        }
        return acc;
    }, {});
    
    return Object.entries(counts).map(([id, count]) => ({ id, count }));
});
// --- End New Computed Properties --- 

// --- TODO: Add a new function to fetch weekly abnormal data --- 
const fetchWeeklyAbnormalData = async () => {
    console.log("Fetching weekly abnormal data (placeholder)...");
    // Replace this with your actual API call to the backend endpoint
    // that returns data like: [{ date: 'YYYY-MM-DD', count: X }, ...]
    // Example Placeholder Data (replace with fetch):
    const todayStr = formatDate(new Date());
    const yesterday = new Date(); yesterday.setDate(yesterday.getDate() - 1); 
    const dayBefore = new Date(); dayBefore.setDate(dayBefore.getDate() - 2);
    
    weeklyAbnormalData.value = [
        { date: formatDate(dayBefore), count: Math.floor(Math.random() * 5) },
        { date: formatDate(yesterday), count: Math.floor(Math.random() * 8) },
        { date: todayStr, count: abnormalVisibleEvents.value.length }, // Use current visible count for today as placeholder
        // ... add more placeholder days if needed for a full week
    ];
    console.log("Using placeholder weekly data:", weeklyAbnormalData.value);
    // try {
    //   const response = await fetch('/api/events/abnormal_summary?days=7'); // Example new endpoint
    //   if (!response.ok) throw new Error('Failed to fetch weekly data');
    //   const data = await response.json();
    //   weeklyAbnormalData.value = data; // Assuming backend returns the correct format
    // } catch (error) {
    //   console.error("Error fetching weekly abnormal data:", error);
    //   weeklyAbnormalData.value = []; // Clear on error
    // }
};
// --- End TODO --- 

// --- Lifecycle Hooks --- 
onMounted(async () => {
  console.log("App component mounted");
  initBarChart(); // Initialize the bar chart

  // Create and initialize the event stats pie chart
  await nextTick(); // Ensure DOM is ready for chart container
  const chartContainer = document.getElementById('eventStatsChartContainer');
  if (chartContainer && !document.getElementById('eventStatsChart')) { // Prevent creating multiple chart divs
    const newChartDom = document.createElement('div');
    newChartDom.id = 'eventStatsChart';
    newChartDom.style.width = '100%';
    newChartDom.style.height = '100%'; // Let container define height
    chartContainer.appendChild(newChartDom);
    initEventStatsChart(); // Initialize after creating the element
  } else if (!chartContainer) {
     console.error("Element with id 'eventStatsChartContainer' not found.");
  } else {
     console.log("Event stats chart element already exists.");
     // Ensure chart is initialized if element exists but instance is null
     if(!eventStatsChartInstance.value) initEventStatsChart();
  }
  // Ensure ThreeDMap component's ref is available and load the floor plan
  await nextTick(); // Ensure refs are populated after initial render
  if (threeDMap.value) {
    // Explicitly call the method to load the floor plan
    // Use the correct path to your floorplan image (relative to public folder)
    // Pass the onFloorplanLoad function as the callback
    console.log("Calling threeDMap.value.loadFloorPlan...");
    threeDMap.value.loadFloorPlan('/floorplan.png', onFloorplanLoad);
  } else {
    console.error("ThreeDMap ref not available on mount to load floor plan.");
  }
});
</script>

<style lang="scss" scoped>
// Define primary color if not defined globally
:root {
  --primary-color: #00ffff; 
}

.smart-city {
  width: 100vw;
  height: 100vh;
  background-color: #001529;
  color: #fff;
  box-sizing: border-box;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  position: relative; /* Needed for absolute positioning of dropdown */

  .header {
    text-align: center;
    padding: 10px 0;
    flex-shrink: 0;
    background-color: rgba(0, 21, 41, 0.8);
    border-bottom: 1px solid var(--primary-color, #00ffff); // Use CSS variable with fallback
    display: flex;
    justify-content: center;
    align-items: center;
    position: relative; // Relative positioning for the button
    z-index: 10; // Ensure header is above content panels

    h1 {
      color: var(--primary-color, #00ffff);
      font-size: 24px;
      margin: 0;
    }
  }

  .content {
    display: flex;
    flex-grow: 1;
    overflow: hidden; 
    gap: 15px; 
    padding: 15px;

    .left-panel, .right-panel { // Keep right-panel style definition
       // Apply box-sizing explicitly
       box-sizing: border-box;
       
       // Use flex-basis instead of width for better flexibility with gap
       flex-basis: 25%; 
       max-width: 350px; 
       min-width: 250px; 
       padding: 15px;
       display: flex;
       flex-direction: column;
       gap: 15px;
       background-color: rgba(0, 21, 41, 0.7);
       border-radius: 4px;
       border: 1px solid rgba(0, 255, 255, 0.1);
       overflow: hidden; // Keep hidden for panel itself
       flex-shrink: 0; 
    }
    
    .right-panel {
        // Specific styles if needed, e.g., if it holds other content
        // If it's truly empty now, consider removing it or hiding it
    }


    .center-area-wrapper {
       // Apply box-sizing explicitly
       box-sizing: border-box;
       flex-grow: 1; 
       display: flex;
       flex-direction: column; 
       overflow: hidden; 
       gap: 15px; 
    }

    .center-panel {
      flex-grow: 1; 
      position: relative; 
      background-color: rgba(0, 21, 41, 0.7); 
      border-radius: 4px;
      border: 1px solid rgba(0, 255, 255, 0.1);
      overflow: hidden; 
      min-height: 300px; 
      display: flex; // Ensure map component can stretch
      > :deep(div) { // Target direct child (ThreeDMap container)
         flex-grow: 1;
         display: flex; // Allow canvas inside to potentially resize
      }
    }

    .center-bottom-panel {
      flex-shrink: 0; 
      background-color: rgba(0, 21, 41, 0.7); 
      padding: 10px; // Reduced padding slightly
      border-radius: 4px;
      border: 1px solid rgba(0, 255, 255, 0.1);
      height: 180px; // Keep fixed height for timeline consistency
      display: flex; // Use flex for centering/stretching timeline
      flex-direction: column;
      overflow: hidden; // Hide overflow from timeline
    }
  }

  .date-picker-container {
    padding: 10px;
    background-color: rgba(0, 41, 82, 0.5);
    border-radius: 4px;
    margin-bottom: 15px;
    color: #ccc;
  }
  .date-picker-container label {
    margin-right: 10px;
  }
  .date-picker-container input[type="date"] {
    background-color: #001529;
    color: #fff;
    border: 1px solid var(--primary-color, #00ffff);
    padding: 5px;
    border-radius: 3px;
  }
  
  .left-panel { 
    // Reset left panel styles to original purpose
    width: 25%;
    max-width: 350px; 
    min-width: 250px; 
    padding: 15px;
    display: flex;
    flex-direction: column; 
    gap: 15px;
    background-color: rgba(0, 21, 41, 0.7);
    border-radius: 4px;
    border: 1px solid rgba(0, 255, 255, 0.1);
    overflow: hidden; // Keep hidden for panel itself
    flex-shrink: 0; 
    
    .date-picker-container {
        flex-shrink: 0; 
        // ... existing date picker styles ...
    }
    
    // Re-add event details styles
    .event-details {
        background-color: rgba(0, 41, 82, 0.5); 
        padding: 10px;
        border-radius: 4px;
        margin-bottom: 15px; 
        flex-shrink: 0; // Don't let it shrink too much
        max-height: 30%; // Reduce slightly more for chart
        overflow-y: auto;
    }
    .event-details h3 {
        margin-top: 0;
        color: var(--primary-color, #00ffff);
        margin-bottom: 10px; 
    }
    .event-content pre {
        white-space: pre-wrap; 
        word-wrap: break-word;
        max-height: 150px; 
        overflow-y: auto; 
        background-color: rgba(0,0,0,0.2);
        padding: 5px;
        border-radius: 3px;
        color: #eee; 
        font-family: monospace;
        font-size: 0.9em;
    }
    .play-button {
        background-color: var(--primary-color, #00ffff);
        color: #001529;
        border: none;
        padding: 5px 10px;
        border-radius: 4px;
        cursor: pointer;
        margin-top: 10px;
        transition: background-color 0.3s;
        &:hover { background-color: #fff; }
    }
    .no-selection {
        flex-grow: 0; // Don't let it grow too much initially
        color: #666;
        text-align: center;
        padding: 20px;
        font-style: italic;
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 50px; // Give it some base height
    }
    
    // Style for the new chart section
    .left-panel-section.left-panel-bottom-chart {
        flex-grow: 1; 
        min-height: 200px; // Increase min-height slightly
        overflow: hidden;
        display: flex;
        flex-direction: column;
        /* Add border/background if desired */
        /* border-top: 1px solid rgba(0, 255, 255, 0.1); */
        /* padding-top: 10px; */
    }
  }
  
  .right-panel {
       // Apply vertical flex layout to right panel
       width: 25%;
       max-width: 350px; 
       min-width: 250px; 
       padding: 15px;
       display: flex;
       flex-direction: column; // Stack top and bottom sections
       gap: 15px; // Gap between sections
       background-color: rgba(0, 21, 41, 0.7);
       border-radius: 4px;
       border: 1px solid rgba(0, 255, 255, 0.1);
       overflow: hidden; // Hide overflow for the panel
       flex-shrink: 0; 
       
       .right-panel-section { // Style for sections within right panel
           overflow: hidden; // Allow internal scrolling
           display: flex; // Needed for flex-grow on child component
           flex-direction: column;
       }
       
       .right-panel-top { 
           flex-shrink: 0; /* Prevent top from shrinking too much */
           /* Adjust height/max-height if needed */
           max-height: 45%; // Example max height
       }
       
       .right-panel-bottom {
           flex-grow: 1; /* Allow bottom to fill remaining space */
           min-height: 150px; /* Optional min height */
       }
  }

  .video-modal {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1100; // Ensure video is above dropdown

    .video-container {
      position: relative;
      background-color: #001529;
      padding: 20px;
      border-radius: 8px;
      border: 1px solid rgba(0, 255, 255, 0.2);
      max-width: 80vw;
      max-height: 80vh;

      video {
        display: block;
        max-width: 100%;
        max-height: calc(80vh - 60px);
      }

      .close-button {
        position: absolute;
        top: -10px;
        right: -10px;
        background-color: #ff4d4f;
        color: white;
        border: none;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        font-size: 20px;
        line-height: 30px;
        text-align: center;
        cursor: pointer;
        box-shadow: 0 2px 5px rgba(0,0,0,0.3);
        z-index: 1101; // Ensure close button is above video
      }
    }
  }
}

// Styles for the NEW Dropdown Chat
.chat-dropdown-toggle-button {
  position: absolute;
  top: 50%;
  right: 20px;
  transform: translateY(-50%);
  padding: 8px 15px;
  background-color: var(--primary-color, #00ffff);
  color: #001529;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-weight: bold;
  z-index: 11; // Ensure button is above content but below dropdown panel maybe
  transition: background-color 0.3s;

  &:hover {
    background-color: #fff;
  }
}

.chat-dropdown-panel {
  position: absolute;
  top: 60px; // Position below the header (adjust if header height changes)
  right: 15px; // Align with content padding
  width: 350px; // Match the width of the original right panel idea
  max-height: calc(100vh - 80px); // Limit height, leave some space at bottom
  background-color: rgba(0, 21, 41, 0.95); // More opaque background
  border: 1px solid var(--primary-color, #00ffff);
  border-top: none; // Optional: remove top border if it touches header
  border-radius: 0 0 4px 4px; // Round bottom corners
  box-shadow: 0 5px 15px rgba(0, 255, 255, 0.2);
  z-index: 1050; // Ensure it's above content but below modals
  overflow: hidden; // Prevent content overflow before transition
  display: flex; // Use flex for internal layout
  flex-direction: column;
  
  // Animation part
  transform: translateY(-110%); // Start completely hidden above
  transition: transform 0.4s ease-in-out, opacity 0.4s ease-in-out;
  opacity: 0; // Start invisible
  pointer-events: none; // Prevent interaction when hidden

  &.is-visible {
    transform: translateY(0); // Slide down into view
    opacity: 1; // Become visible
    pointer-events: auto; // Allow interaction when visible
  }

  // Ensure RagChat component inside fills the space and scrolls
  > :deep(.rag-chat-container) { // Assuming RagChat has a root element with this class
      flex-grow: 1;
      overflow-y: auto; // Allow internal scrolling
      padding: 15px; // Add padding inside the dropdown
      height: 100%; // Try to force height (might depend on RagChat structure)
  }
   // Fallback if RagChat's root is different or deeper
   > :deep(div:first-child) { 
       flex-grow: 1;
       display: flex;
       flex-direction: column;
       overflow-y: auto;
       padding: 15px; 
   }

}



</style> 