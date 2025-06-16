<template>
  <div class="rag-chat-container">
    <div class="chat-history" ref="chatHistoryRef">
      <div v-for="(message, index) in messages" :key="index" :class="['message', message.sender]">
        <div class="bubble">
          <template v-if="message.type === 'text' || message.type === 'error' || message.sender === 'user'">
            {{ message.text }}
          </template>
          <template v-else-if="message.type === 'structured' && message.data">
            <div class="structured-response">
              <div v-if="message.data.相关日志 && message.data.相关日志.length > 0" class="section related-documents">
                <strong>相关日志:</strong>
                <ul>
                  <li v-for="(logEntry, logIndex) in message.data.相关日志" :key="logIndex">
                    <button class="log-link" @click="requestVideoPlayback(getLogDetails(logEntry).textFilename)">
                      {{ getLogDetails(logEntry).label.replace('日志', '日志 ') }}
                    </button>
                    <p class="doc-overview">{{ getLogDetails(logEntry).summary }}</p>
                  </li>
                </ul>
              </div>
              <div v-if="message.data.总结报告" class="section summary-report">
                <strong>总结报告:</strong>
                <p>{{ message.data.总结报告 }}</p>
              </div>
            </div>
          </template>
          <template v-else>
            {{ message.text || '无效消息格式' }}
          </template>
        </div>
      </div>
      <div v-if="isLoading" class="message assistant loading">
        <div class="bubble">正在思考中...</div>
      </div>
    </div>
    <div class="input-area">
      <textarea
        v-model="userInput"
        placeholder="请输入您的问题..."
        @keydown.enter.prevent="sendMessage"
        :disabled="isLoading"
      ></textarea>
      <button @click="sendMessage" :disabled="isLoading || !userInput.trim()">
        发送
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick } from 'vue'

const emit = defineEmits(['play-video-from-chat']);

const userInput = ref('')
const messages = ref([]) // 存储聊天记录，格式: { sender: 'user' | 'assistant', type: 'text' | 'structured' | 'error', text?: '...', data?: {...} }
const isLoading = ref(false)
const chatHistoryRef = ref(null) // 用于自动滚动

const scrollToBottom = () => {
  nextTick(() => {
    const chatHistory = chatHistoryRef.value
    if (chatHistory) {
      chatHistory.scrollTop = chatHistory.scrollHeight
    }
  })
}

const sendMessage = async () => {
  const query = userInput.value.trim()
  if (!query || isLoading.value) return

  // 1. 添加用户消息到聊天记录
  messages.value.push({ sender: 'user', type: 'text', text: query })
  userInput.value = '' // 清空输入框
  isLoading.value = true // 显示加载状态
  scrollToBottom() // 滚动到底部

  try {
    // 2. 调用后端 API
    console.log('Sending query to backend:', query)
    const response = await fetch('/api/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query: query }),
    })

    console.log('Received response from backend:', response)
    let assistantMessage = {} // Prepare assistant message object

    if (!response.ok) {
      // 处理 API 错误
      let errorText = `抱歉，处理请求时出错 (${response.status})`
      try {
        const errorData = await response.json()
        console.error('API Error Data:', errorData)
        // Use error message from backend if available
        errorText = errorData.error_message || errorData.error || errorText
      } catch (e) {
        console.error('Failed to parse error response body')
        errorText = `${errorText} (${response.statusText})`
      }
      assistantMessage = { sender: 'assistant', type: 'error', text: errorText }
    } else {
      // 3. 添加 RAG 的回复到聊天记录
      const responseData = await response.json()
      console.log('API Success Data:', responseData)

      // Check the status field from backend
      if (responseData.status === 'success' && responseData.data) {
        // Successfully parsed JSON from LLM
        // The actual LLM output is in responseData.data
        assistantMessage = { 
          sender: 'assistant', 
          type: 'structured', 
          data: responseData.data // This should be the object with "相关日志" and "总结报告"
        };
      } else if (responseData.status === 'error') { // Simplified error handling from backend
        // LLM output was not valid JSON, or other backend error
        console.warn('Backend reported an error or LLM output was not valid JSON.');
        assistantMessage = { 
          sender: 'assistant', 
          type: 'error', // Display as an error
          text: responseData.error_message || responseData.raw_answer || '发生未知错误' 
        };
      } else {
        // Unexpected success response structure from backend
        console.error('Unexpected success response structure:', responseData)
        assistantMessage = { 
          sender: 'assistant', 
          type: 'error', 
          text: '收到意外的响应格式，无法解析回答。' 
        }
      }
    }
    messages.value.push(assistantMessage) // Add the processed assistant message
  } catch (error) {
    // 处理网络或其他 fetch 错误
    console.error('Fetch Error:', error)
    messages.value.push({
      sender: 'assistant',
      type: 'error',
      text: '抱歉，无法连接到问答服务。请检查后端服务是否运行以及网络连接。',
    })
  } finally {
    isLoading.value = false // 结束加载状态
    scrollToBottom() // 再次滚动到底部
  }
}

const getLogDetails = (logEntry) => {
  let textFilename = '';
  let label = '';
  let summary = '';
  
  console.log('Processing log entry:', logEntry);
  
  if (logEntry && typeof logEntry === 'object') {
    for (const [key, value] of Object.entries(logEntry)) {
      if (typeof key === 'string' && key.endsWith('概述')) {
        summary = value;
      } else { 
        label = key;
        textFilename = value;
      }
    }
  }
  
  console.log('Extracted details:', { label, textFilename, summary });
  return { label, textFilename, summary };
};

const requestVideoPlayback = (textFilename) => {
  if (!textFilename || typeof textFilename !== 'string') {
    console.error('Invalid text filename for video playback:', textFilename);
    return;
  }
  
  console.log('Requesting video playback for:', textFilename);
  
  // 如果包含扩展名，去掉扩展名
  let baseName = textFilename;
  if (textFilename.includes('.')) {
    baseName = textFilename.substring(0, textFilename.lastIndexOf('.'));
  }
  
  // 构造视频文件名
  const videoFilename = `${baseName}.mp4`;
  console.log('Emitting play-video-from-chat with:', videoFilename);
  emit('play-video-from-chat', videoFilename);
};

</script>

<style scoped>
.rag-chat-container {
  display: flex;
  flex-direction: column;
  height: 100%; /* Fill parent's fixed height */
  width: 100%;
  background-color: rgba(0, 21, 41, 0.85);
  border: 1px solid rgba(0, 255, 255, 0.1);
  border-radius: 4px;
  padding: 15px;
  box-sizing: border-box;
}

.chat-history {
  flex: 1; /* Take up remaining vertical space */
  overflow-y: auto; /* Enable internal scrolling */
  margin-bottom: 15px;
  padding-right: 10px;
  min-height: 100px; /* Prevent collapsing */
}

.message {
  display: flex;
  margin-bottom: 10px;
}

.message.user {
  justify-content: flex-end; /* 用户消息靠右 */
}

.message.assistant {
  justify-content: flex-start; /* 助手消息靠左 */
}

.bubble {
  max-width: 85%; /* Allow bubble to be a bit wider for structured content */
  padding: 10px 15px;
  border-radius: 15px;
  word-wrap: break-word;
  white-space: pre-wrap; /* 保留换行符 */
  line-height: 1.5;
}

.message.user .bubble {
  background-color: #007bff; /* 蓝色气泡 */
  color: white;
  border-bottom-right-radius: 5px;
}

.message.assistant .bubble {
  background-color: #333; /* 深灰色气泡 */
  color: #f1f1f1;
  border-bottom-left-radius: 5px;
}

.message.loading .bubble {
  background-color: #444; /* 加载中气泡颜色 */
  color: #aaa;
  font-style: italic;
}

.input-area {
  display: flex;
  gap: 10px;
  border-top: 1px solid rgba(0, 255, 255, 0.1);
  padding-top: 15px;
  flex-shrink: 0; /* Prevent input area from shrinking */
}

.input-area textarea {
  flex: 1;
  padding: 10px;
  border-radius: 4px;
  border: 1px solid rgba(0, 255, 255, 0.3);
  background-color: rgba(0, 21, 41, 0.9);
  color: #fff;
  font-size: 16px;
  resize: none; /* 禁止调整大小 */
  min-height: 40px; /* 最小高度 */
  max-height: 120px; /* 最大高度 */
}

.input-area textarea:focus {
  outline: none;
  border-color: #00ffff;
}

.input-area button {
  padding: 10px 20px;
  background-color: #00ffff;
  color: #001529;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-weight: bold;
  font-size: 16px;
  transition: background-color 0.3s;
}

.input-area button:hover:not(:disabled) {
  background-color: #fff;
}

.input-area button:disabled {
  background-color: #555;
  color: #aaa;
  cursor: not-allowed;
}

/* 滚动条美化 */
.chat-history::-webkit-scrollbar {
  width: 6px;
}

.chat-history::-webkit-scrollbar-track {
  background: rgba(0, 21, 41, 0.5);
  border-radius: 3px;
}

.chat-history::-webkit-scrollbar-thumb {
  background: rgba(0, 255, 255, 0.3);
  border-radius: 3px;
}

.chat-history::-webkit-scrollbar-thumb:hover {
  background: rgba(0, 255, 255, 0.5);
}

/* Styles for structured response */
.structured-response {
  text-align: left; /* Ensure text aligns left */
}

.structured-response .section {
  margin-bottom: 12px;
}
.structured-response .section:last-child {
  margin-bottom: 0;
}

.structured-response strong {
  display: block;
  margin-bottom: 4px;
  color: #00ffff; /* Highlight section titles */
  font-size: 0.95em;
}

.structured-response p {
  margin: 0;
  white-space: pre-wrap; /* Respect newlines in paragraphs */
}

.structured-response ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

.structured-response .related-documents li {
  margin-bottom: 8px;
  padding-left: 10px;
  border-left: 2px solid rgba(0, 255, 255, 0.3);
}

.structured-response .related-documents .doc-name {
  font-weight: bold;
  color: #f0f0f0;
  display: block; /* Put name on its own line */
  margin-bottom: 3px;
  font-family: monospace;
}

.structured-response .related-documents .doc-overview {
  font-size: 0.9em;
  color: #ccc;
  display: block;
}

/* Style for the clickable log link */
.log-link {
  background: none;
  border: none;
  color: #00ffff; /* Or your link color */
  text-decoration: underline;
  cursor: pointer;
  padding: 0;
  font-weight: bold; /* Match current style of doc-name strong */
  font-family: monospace; /* Match current style of doc-name */
  display: block; /* Put name on its own line */
  margin-bottom: 3px; /* Match current style of doc-name */
}

.log-link:hover {
  color: #fff; /* Lighter color on hover */
}
</style> 