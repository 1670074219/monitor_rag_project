<template>
  <div class="timeline-container">
    <div class="time-display">{{ formattedTime }}</div>
    <input 
      type="range" 
      :min="0" 
      :max="86399" 
      :value="modelValue" 
      @input="updateTime" 
      class="timeline-slider"
    >
    <!-- Hour Scale (Step by 2) -->
    <div class="hour-scale">
      <!-- Loop through even hours only -->
      <span v-for="hour in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]" :key="hour" class="hour-mark" :style="getHourMarkStyle(hour)">
        {{ hour.toString().padStart(2, '0') }}:00
      </span>
      <!-- Optionally add the last mark for 24:00 (or end of day) -->
      <span class="hour-mark" :style="getHourMarkStyle(24)" style="transform: translateX(-100%);">24:00</span>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue';

const props = defineProps({
  // events: { type: Array, default: () => [] }, // No longer directly needed for scale
  modelValue: { type: Number, default: 0 }, // Represents time of day in seconds (0-86399)
  displayDate: { type: String, default: '' } // Optional: receive date for context
});

const emit = defineEmits(['update:modelValue', 'markerClick']);

// Format the time of day seconds into HH:MM:SS
const formattedTime = computed(() => {
    const totalSeconds = props.modelValue;
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
});

// Update time when slider moves
const updateTime = (event) => {
  emit('update:modelValue', parseInt(event.target.value, 10));
};

// Calculate style for hour marks
const getHourMarkStyle = (hour) => {
    const totalSecondsInDay = 86400;
    const hourInSeconds = hour * 3600;
    const percentage = (hourInSeconds / totalSecondsInDay) * 100;
    // Clamp percentage between 0 and 100 to avoid visual overflow
    const clampedPercentage = Math.max(0, Math.min(100, percentage)); 
    return { left: `${clampedPercentage}%` };
};

// --- Marker click handling on timeline (If needed) ---
// This part might need to be re-evaluated or removed if markers are solely on 3D map.
// If you still want clickable markers on the timeline itself:
// 1. You'd need the `events` prop back.
// 2. You'd need logic to calculate marker positions based on `timeOfDaySeconds`.
// 3. You'd need click handling logic similar to before, emitting marker data.
// For now, assuming markers are only on the 3D map, this part is simplified/removed.

</script>

<style scoped>
.timeline-container {
  padding: 15px 10px;
  background-color: rgba(0, 41, 82, 0.3);
  border-radius: 4px;
  position: relative; /* Needed for absolute positioning of scale */
}

.time-display {
  text-align: center;
  font-size: 1.1em;
  margin-bottom: 10px;
  color: var(--primary-color);
  font-family: 'Courier New', Courier, monospace;
}

.timeline-slider {
  width: 100%;
  cursor: pointer;
  height: 8px;
  background: linear-gradient(to right, var(--primary-color), #008f8f);
  border-radius: 4px;
  appearance: none; /* Override default look */
  -webkit-appearance: none;
  outline: none;
}

/* Style the thumb */
.timeline-slider::-webkit-slider-thumb {
  -webkit-appearance: none; /* Override default look */
  appearance: none;
  width: 18px;
  height: 18px;
  background: #00ffff;
  border-radius: 50%;
  border: 2px solid #001529;
  cursor: pointer;
  box-shadow: 0 0 5px #00ffff;
}

.timeline-slider::-moz-range-thumb {
  width: 16px;
  height: 16px;
  background: #00ffff;
  border-radius: 50%;
  border: 2px solid #001529;
  cursor: pointer;
  box-shadow: 0 0 5px #00ffff;
}

.hour-scale {
    position: relative;
    width: 100%;
    height: 25px; /* Height for the scale marks */
    margin-top: 5px;
    color: #aaa;
    font-size: 0.7em;
}

.hour-mark {
    position: absolute;
    bottom: 0;
    transform: translateX(-50%); /* Center the text */
    white-space: nowrap;
    padding-top: 5px;
}

.hour-mark::before {
    content: '';
    position: absolute;
    top: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 1px;
    height: 5px;
    background-color: #666;
}

/* Adjust first/last mark alignment */
.hour-mark:first-child { /* Handles the 00:00 mark */
   /* transform: translateX(0); /* Align first mark left slightly better */ 
}
/* Last manually added mark (24:00) handles its own alignment */

</style> 