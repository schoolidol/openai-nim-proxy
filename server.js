// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '100mb' })); 
app.use(express.urlencoded({ limit: '100mb', extended: true }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = true;

// 🔥 THINKING MODE TOGGLE - Enables reasoning parameters
const ENABLE_THINKING_MODE = true;

// Model mapping
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-r1',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking' 
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// List models endpoint
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  
  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    // Smart model selection with fallback
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      try {
        await axios.post(`${NIM_API_BASE}/chat/completions`, {
          model: model,
          messages: [{ role: 'user', content: 'test' }],
          max_tokens: 1
        }, {
          headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
          validateStatus: (status) => status < 500
        }).then(res => {
          if (res.status >= 200 && res.status < 300) {
            nimModel = model;
          }
        });
      } catch (e) {}
      
      if (!nimModel) {
        const modelLower = model.toLowerCase();
        if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
          nimModel = 'meta/llama-3.1-405b-instruct';
        } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
          nimModel = 'meta/llama-3.1-70b-instruct';
        } else {
          nimModel = 'meta/llama-3.1-8b-instruct';
        }
      }
    }

    // 1. Inject System Prompt Hack to force Chain-of-Thought in English
    let processedMessages = Array.isArray(messages) ? [...messages] : [];

    if (ENABLE_THINKING_MODE) {
      const systemMsgIndex = processedMessages.findIndex(m => m.role === 'system');
      const forceReasoningPrompt = "CRITICAL OPERATIONAL MANDATE: You must use your internal chain-of-thought (<think>) block to meticulously break down every single request step-by-step before answering, regardless of how simple or brief the query is. You must conduct all internal reasoning steps and final responses strictly in English. Never skip the reasoning phase.";

      if (systemMsgIndex !== -1) {
        processedMessages[systemMsgIndex] = {
          ...processedMessages[systemMsgIndex],
          content: processedMessages[systemMsgIndex].content + `\n\n${forceReasoningPrompt}`
        };
      } else {
        processedMessages.unshift({ role: 'system', content: forceReasoningPrompt });
      }
    }
    
    // 2. Build NVIDIA Request Payload according to model requirements
    const nimRequest = {
      model: nimModel,
      messages: processedMessages, // FIXED: Now properly uses processedMessages
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 9024,
      stream: stream || false
    };

    if (ENABLE_THINKING_MODE) {
      const lowerModel = nimModel.toLowerCase();
      
      // GLM and Qwen models require chat_template_kwargs
      if (lowerModel.includes('glm') || lowerModel.includes('qwen')) {
        nimRequest.chat_template_kwargs = {
          enable_thinking: true,
          thinking: true
        };
      } else {
        // DeepSeek, Kimi, and other NIM models require extra_body or top-level thinking
        nimRequest.extra_body = {
          thinking: { type: "enabled" }
        };
        nimRequest.chat_template_kwargs = { thinking: true };
      }
    }
    
    // 3. Request with Retry Logic
    let response;
    let retries = 3;
    let delay = 1000;

    while (retries > 0) {
      try {
        response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
          headers: {
            'Authorization': `Bearer ${NIM_API_KEY}`,
            'Content-Type': 'application/json'
          },
          responseType: stream ? 'stream' : 'json'
        });
        break;
      } catch (error) {
        if ((error.response?.status === 503 || error.response?.status === 429) && retries > 1) {
          console.warn(`NVIDIA API busy (Status ${error.response.status}). Retrying in ${delay}ms...`);
          retries--;
          await new Promise(res => setTimeout(res, delay));
          delay *= 1.5;
        } else {
          throw error;
        }
      }
    }
    
    // 4. Handle Streaming
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
                // Catch all possible keys NVIDIA uses for reasoning
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
                  
                  if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                  } else {
                    data.choices[0].delta.content = '';
                  }
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
      // 5. Handle Non-Streaming
      const choice = response.data.choices?.[0];
      let fullContent = choice?.message?.content || '';
      const reasoning = choice?.message?.reasoning_content || choice?.message?.reasoning || choice?.message?.thinking;
      
      if (SHOW_REASONING && reasoning) {
        fullContent = '<think>\n' + reasoning + '\n</think>\n\n' + fullContent;
      }

      const openaiResponse = {
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
      };
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error('Proxy error:', error.message);
    
    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

// Catch-all
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});
