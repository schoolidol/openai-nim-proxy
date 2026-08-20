// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '100mb' })); 
app.use(express.urlencoded({ limit: '100mb', extended: true }));

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

const SHOW_REASONING = true;
const ENABLE_THINKING_MODE = true;

// Verified active NVIDIA NIM endpoint names
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'meta/llama-3.1-8b-instruct',
  'gpt-4': 'meta/llama-3.1-70b-instruct',
  'gpt-4-turbo': 'meta/llama-3.1-405b-instruct',
  'gpt-4o': 'deepseek-ai/deepseek-r1',
  'claude-3-opus': 'deepseek-ai/deepseek-r1',
  'claude-3-sonnet': 'meta/llama-3.1-70b-instruct',
  'gemini-pro': 'qwen/qwen2.5-coder-32b-instruct'
};

// ----------------------------------------------------
// SIMPLE RATE LIMITER & QUEUE MANAGER TO PREVENT 429
// ----------------------------------------------------
const MIN_REQUEST_INTERVAL_MS = 2000; // Force 2 seconds delay between API calls (~30 RPM)
let lastRequestTime = 0;

async function waitForRateLimitSlot() {
  const now = Date.now();
  const timeSinceLast = now - lastRequestTime;
  if (timeSinceLast < MIN_REQUEST_INTERVAL_MS) {
    const waitTime = MIN_REQUEST_INTERVAL_MS - timeSinceLast;
    await new Promise(res => setTimeout(res, waitTime));
  }
  lastRequestTime = Date.now();
}

app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  
  res.json({ object: 'list', data: models });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    let nimModel = MODEL_MAPPING[model] || model;

    let processedMessages = Array.isArray(messages) ? [...messages] : [];

    if (ENABLE_THINKING_MODE) {
      const systemMsgIndex = processedMessages.findIndex(m => m.role === 'system');
      const forceReasoningPrompt = "CRITICAL OPERATIONAL MANDATE: Meticulously break down every request step-by-step using internal reasoning (<think> block) before providing the final answer. Keep explanations and reasoning strictly in English.";

      if (systemMsgIndex !== -1) {
        processedMessages[systemMsgIndex] = {
          ...processedMessages[systemMsgIndex],
          content: processedMessages[systemMsgIndex].content + `\n\n${forceReasoningPrompt}`
        };
      } else {
        processedMessages.unshift({ role: 'system', content: forceReasoningPrompt });
      }
    }
    
    const nimRequest = {
      model: nimModel,
      messages: processedMessages,
      temperature: temperature ?? 0.6,
      max_tokens: max_tokens ?? 4096,
      stream: stream ?? false
    };

    if (ENABLE_THINKING_MODE) {
      nimRequest.chat_template_kwargs = {
        enable_thinking: true,
        clear_thinking: false
      };
      nimRequest.reasoning_effort = "high";
    }

    // ----------------------------------------------------
    // RETRY LOGIC FOR 429 RATE LIMIT ERRORS
    // ----------------------------------------------------
    let response;
    let retries = 5;
    let backoffDelay = 2000; // Start with 2 second delay

    while (retries > 0) {
      try {
        // Space out calls to respect global 30 RPM window
        await waitForRateLimitSlot();

        response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
          headers: {
            'Authorization': `Bearer ${NIM_API_KEY}`,
            'Content-Type': 'application/json'
          },
          responseType: stream ? 'stream' : 'json'
        });
        break; // Success! Exit retry loop
      } catch (err) {
        const statusCode = err.response?.status;
        if ((statusCode === 429 || statusCode === 503) && retries > 1) {
          console.warn(`[429 Rate Limit] NVIDIA busy. Retrying in ${backoffDelay / 1000}s... (${retries - 1} retries left)`);
          await new Promise(res => setTimeout(res, backoffDelay));
          backoffDelay *= 2; // Exponential backoff (2s -> 4s -> 8s -> 16s)
          retries--;
        } else {
          throw err; // Fail on non-429 or exhausted retries
        }
      }
    }
    
    // Handle Streaming
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let reasoningStarted = false;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\n');
              return;
            }
            
            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content || 
                                  data.choices[0].delta.reasoning || 
                                  data.choices[0].delta.thinking || '';
                const content = data.choices[0].delta.content || '';
                
                if (SHOW_REASONING) {
                  let combinedContent = '';
                  
                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }
                  
                  if (content && reasoningStarted) {
                    combinedContent += '\n</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }
                  
                  data.choices[0].delta.content = combinedContent;
                } else {
                  data.choices[0].delta.content = content;
                }

                delete data.choices[0].delta.reasoning_content;
                delete data.choices[0].delta.reasoning;
                delete data.choices[0].delta.thinking;
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n');
            }
          }
        });
      });
      
      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });
    } else {
      // Handle Non-Streaming
      const choice = response.data.choices?.[0];
      let fullContent = choice?.message?.content || '';
      const reasoning = choice?.message?.reasoning_content || choice?.message?.reasoning || choice?.message?.thinking;
      
      if (SHOW_REASONING && reasoning) {
        fullContent = '<think>\n' + reasoning + '\n</think>\n\n' + fullContent;
      }

      res.json({
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: [{
          index: 0,
          message: {
            role: choice?.message?.role || 'assistant',
            content: fullContent
          },
          finish_reason: choice?.finish_reason || 'stop'
        }],
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      });
    }
    
  } catch (error) {
    const status = error.response?.status || 500;
    const errorDetails = error.response?.data || error.message;
    console.error(`Proxy error [${status}]:`, errorDetails);
    
    res.status(status).json({
      error: {
        message: typeof errorDetails === 'object' ? JSON.stringify(errorDetails) : errorDetails,
        type: 'rate_limit_error',
        code: status
      }
    });
  }
});

app.listen(PORT, () => {
  console.log(`Proxy active on port ${PORT}`);
});
