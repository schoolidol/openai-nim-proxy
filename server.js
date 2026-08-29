// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

// 🔥 FIX: A stream-mode axios request that times out can leave its
// underlying socket emitting an 'error' event *after* the request promise
// already rejected. If nothing is listening for that event, Node treats it
// as fatal and crashes the whole process - which is why Render was showing
// a fresh "Detected service running on port 10000" boot line right after a
// retry: the process had actually died and been auto-restarted, taking every
// other in-flight conversation down with it. These handlers turn that into
// a logged warning instead of a full crash.
process.on('unhandledRejection', (reason) => {
  console.error('Unhandled Rejection:', reason);
});

process.on('uncaughtException', (err) => {
  console.error('Uncaught Exception:', err);
});

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '100mb' })); 
app.use(express.urlencoded({ limit: '100mb', extended: true }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

const SHOW_REASONING = true;

const ENABLE_THINKING_MODE = true;

const DEFAULT_FALLBACK_MODEL = 'meta/llama-3.1-405b-instruct';

const RETRYABLE_STATUSES = [503, 429, 410];

const REQUEST_TIMEOUT_MS = 120000;

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v4-flash-0731',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking',
  'deepseek-ai/deepseek-v4-flash-0731': 'deepseek-ai/deepseek-v4-flash-0731',
  'deepseek-ai/deepseek-v4-pro-0813': 'deepseek-ai/deepseek-v4-pro-0813',
  'moonshotai/kimi-k3': 'moonshotai/kimi-k3'
};


function getReasoningPayload(nimModel) {
  if (!ENABLE_THINKING_MODE) return {};

  // Matches any DeepSeek V4 variant (Flash, Pro, and future dated releases)
  // using NVIDIA's documented request shape - chat_template_kwargs.thinking
  // + reasoning_effort. 'xhigh' maps to DeepSeek's max reasoning depth.
  if (nimModel.startsWith('deepseek-ai/deepseek-v4')) {
    return {
      chat_template_kwargs: {
        thinking: true,
        reasoning_effort: 'low'
      }
    };
  }

  // 🔥 FIX: Moonshot/Kimi models (K2.5, K2-Thinking, K2-Instruct, K3, etc.)
  // use a minimal payload per NVIDIA's own documented examples - just
  // chat_template_kwargs.thinking, no enable_thinking/clear_thinking, and
  // no top-level reasoning_effort. Sending the generic Nemotron-style kwargs
  // below to this family risks the same invalid_request/hang issues DeepSeek
  // had with a mismatched payload shape.
  if (nimModel.startsWith('moonshotai/')) {
    return {
      reasoning_effort: 'low',
      chat_template_kwargs: {
        thinking: true
      }
    };
  }

  return {
    reasoning_effort: 'high',
    chat_template_kwargs: {
      enable_thinking: true,
      thinking: true,
      clear_thinking: false
    }
  };
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
    id: model, object: 'model', created: Date.now(), owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

app.post('/v1/chat/completions', async (req, res) => {
  // Log the exact model string the client sends, so we can see whether
  // it's matching MODEL_MAPPING, being used directly as a NIM ID, or
  // falling through to the probe/default logic below.
  console.log('Incoming model requested:', JSON.stringify(req.body.model));

  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    console.log('stream:', stream);
    
    let nimModel = MODEL_MAPPING[model];

  
    if (!nimModel && model.includes('/')) {
      nimModel = model;
    }

    if (!nimModel) {
      try {
        const probe = await axios.post(`${NIM_API_BASE}/chat/completions`, {
          model: model, messages: [{ role: 'user', content: 'test' }], max_tokens: 1
        }, {
          headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
          validateStatus: (status) => status < 500,
          timeout: REQUEST_TIMEOUT_MS
        });
        if (probe.status >= 200 && probe.status < 300) {
          nimModel = model;
        } else {
          console.warn(`Probe for model "${model}" returned status ${probe.status}:`, JSON.stringify(probe.data));
        }
      } catch (e) {
        console.warn(`Probe request failed for model "${model}":`, e.response?.data ? JSON.stringify(e.response.data) : e.message);
      }
      
      if (!nimModel) {
        const modelLower = model.toLowerCase();
        if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
          nimModel = 'meta/llama-3.1-405b-instruct';
        } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
          nimModel = 'meta/llama-3.1-70b-instruct';
        } else {
          // meta/llama-3.1-8b-instruct is end-of-life; use the configured
          // default instead, and log that we hit this path.
          console.warn(`No mapping/probe match for "${model}". Falling back to default: ${DEFAULT_FALLBACK_MODEL}`);
          nimModel = DEFAULT_FALLBACK_MODEL;
        }
      }
    }

    console.log(`Resolved "${model}" -> NIM model "${nimModel}"`);

    // Clone messages safely
    let processedMessages = JSON.parse(JSON.stringify(messages));

    if (ENABLE_THINKING_MODE) {
      // Per NVIDIA docs: Use "detailed thinking on" to trigger native reasoning mode
      // (Nemotron-style models). Harmless no-op for models that don't use this
      // convention, like DeepSeek, which is driven entirely by chat_template_kwargs below.
      const systemMsgIndex = processedMessages.findIndex(m => m.role === 'system');
      const forceReasoningPrompt = "detailed thinking on. You must conduct all internal reasoning steps strictly in English.";

      if (systemMsgIndex !== -1) {
        processedMessages[systemMsgIndex].content = `${forceReasoningPrompt}\n\n${processedMessages[systemMsgIndex].content}`;
      } else {
        processedMessages.unshift({ role: 'system', content: forceReasoningPrompt });
      }
    }
    
    // Comprehensive parameter payload for NVIDIA NIM endpoints
    const nimRequest = {
      model: nimModel,
      messages: processedMessages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 8192,
      stream: stream || false,
      ...getReasoningPayload(nimModel)
    };

    let response;
    // 🔥 Retries disabled: retries = 1 means the loop makes a single attempt
    // and throws immediately on any error instead of looping, so failures
    // (410, timeout, etc.) surface right away. Bump this back up (e.g. 4) if
    // you want automatic retries again later.
    let retries = 1;
    let delay = 1000;

    while (retries > 0) {
      try {
        response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
          headers: {
            'Authorization': `Bearer ${NIM_API_KEY}`,
            'Content-Type': 'application/json'
          },
          responseType: stream ? 'stream' : 'json',
          timeout: REQUEST_TIMEOUT_MS
        });
        break;
      } catch (error) {
        
        const isRetryable =
          RETRYABLE_STATUSES.includes(error.response?.status) ||
          error.code === 'ECONNABORTED';

        if (isRetryable && retries > 1) {
          console.warn(`NVIDIA API error (${error.response?.status || error.code}) on ${nimModel}. Retrying...`);
          retries--;
          await new Promise(res => setTimeout(res, delay));
          delay *= 1.5;
        } else {
          throw error;
        }
      }
    }
    
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let reasoningStarted = false;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        // FIX: Must use single slash for newline parsing in SSE streams
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          const trimmed = line.trim();
          if (trimmed.startsWith('data: ')) {
            if (trimmed.includes('[DONE]')) {
              res.write(`${trimmed}\n\n`);
              return;
            }
            
            try {
              const data = JSON.parse(trimmed.slice(6));
              const delta = data.choices?.[0]?.delta;

              if (delta) {
                // Handle different engines returning reasoning under slightly different keys
                const reasoning = delta.reasoning_content || delta.reasoning || null;
                const content = delta.content || null;
                
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
                  delete data.choices[0].delta.reasoning_content;
                  delete data.choices[0].delta.reasoning;
                } else {
                  data.choices[0].delta.content = content || '';
                  delete data.choices[0].delta.reasoning_content;
                  delete data.choices[0].delta.reasoning;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(`${trimmed}\n\n`);
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
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          const reasoning = choice.message?.reasoning_content || choice.message?.reasoning;
          
          if (SHOW_REASONING && reasoning) {
            fullContent = `<think>\n${reasoning}\n</think>\n\n${fullContent}`;
          }
          
          return {
            index: choice.index,
            message: {
              role: choice.message?.role || 'assistant',
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    // 🔥 Log rate-limit headers when present - these aren't part of the
    // circular stream body, so they're always safe to read even when the
    // response data itself isn't. Helps distinguish a per-minute burst
    // limit from an hourly/daily quota, and shows exactly when it resets.
    const rateLimitHeaders = error.response?.headers
      ? {
          limit: error.response.headers['x-ratelimit-limit'],
          remaining: error.response.headers['x-ratelimit-remaining'],
          reset: error.response.headers['x-ratelimit-reset'],
          retryAfter: error.response.headers['retry-after']
        }
      : null;
    if (error.response?.status === 429 && rateLimitHeaders) {
      console.warn('Rate limit headers:', JSON.stringify(rateLimitHeaders));
    }

    // 🔥 FIX: When a streaming request (responseType: 'stream') errors out,
    // error.response.data is a raw Node stream/socket object, not JSON - it
    // contains circular references (agent -> sockets -> ... -> agent) that
    // crash JSON.stringify outright. Detect that case and fall back to a
    // plain status-based message instead of trying to serialize the stream.
    const rawData = error.response?.data;
    const isStreamBody = rawData && typeof rawData.pipe === 'function';

    let nvidiaDetail;
    let loggedBody;

    if (isStreamBody) {
      nvidiaDetail = `NVIDIA request failed with status ${error.response.status} (streamed response, no readable body)`;
      loggedBody = '[stream body - not serializable]';
    } else if (rawData !== undefined) {
      nvidiaDetail = rawData.detail || rawData.error?.message || rawData;
      try {
        loggedBody = JSON.stringify(rawData);
      } catch (e) {
        loggedBody = '[unserializable response body]';
      }
    }

    console.error(
      'Proxy error:',
      error.message,
      '| NVIDIA response body:',
      loggedBody
    );

    res.status(error.response?.status || 500).json({
      error: {
        message: nvidiaDetail || error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

app.all('*', (req, res) => res.status(404).json({ error: { message: `Not found`, code: 404 } }));

app.listen(PORT, () => {
  console.log(`NVIDIA NIM Proxy running on port ${PORT}`);
});
