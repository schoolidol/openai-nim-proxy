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

// 🔥 REASONING DISPLAY TOGGLE
const SHOW_REASONING = true;

// 🔥 THINKING MODE TOGGLE
const ENABLE_THINKING_MODE = true;

// 🔥 FIX: meta/llama-3.1-8b-instruct reached end-of-life 2026-08-26 and is no
// longer valid. Use this as the last-resort fallback instead - swap it for
// whatever known-good, currently-live model you want as the safety net.
const DEFAULT_FALLBACK_MODEL = 'meta/llama-3.1-405b-instruct';

// Statuses worth retrying automatically. 410 was added after observing that
// DeepSeek/Nemotron NIM deployments sometimes return 410 on a stale routed
// instance, but succeed immediately on a fresh retry.
const RETRYABLE_STATUSES = [503, 429, 410];

// 🔥 FIX: How long to wait for NVIDIA before giving up on a single attempt.
// Previously there was no timeout at all (axios default = 0 = wait forever),
// so a stalled request would just hang until JanitorAI/SillyTavern or Render
// eventually killed it on their own end (~5 min later) instead of your own
// retry logic ever getting a chance to run. Tune this based on how long a
// normal generation takes for your longest responses.
const REQUEST_TIMEOUT_MS = 30000;

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v4-flash-0731',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking',
  'deepseek-ai/deepseek-v4-flash-0731': 'deepseek-ai/deepseek-v4-flash-0731'
};

// 🔥 FIX: Per-model reasoning/thinking config.
// Different backends expect different trigger conventions - a single
// one-size-fits-all payload was causing invalid_request errors on
// DeepSeek's chat template, which only accepts `thinking` and
// `reasoning_effort` nested inside `chat_template_kwargs`.
function getReasoningPayload(nimModel) {
  if (!ENABLE_THINKING_MODE) return {};

  if (nimModel === 'deepseek-ai/deepseek-v4-flash-0731') {
    // Matches NVIDIA's documented request shape for this model exactly.
    return {
      chat_template_kwargs: {
        thinking: true,
        reasoning_effort: 'high'
      }
    };
  }

  // Default: Nemotron/vLLM-style reasoning kwargs used by the other
  // mapped models. NOTE: models added later via the includes('/') fallback
  // below will also hit this default branch - if a new model starts
  // throwing 400/410s that retries don't fix, this mismatch is the first
  // place to check and give its own branch, same as DeepSeek got.
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
    
    let nimModel = MODEL_MAPPING[model];

    // 🔥 FIX: JanitorAI/SillyTavern may send the actual NIM model ID directly
    // (e.g. "stepfun-ai/step-3.7-flash") rather than an OpenAI-style alias
    // like "gpt-4o". Previously this fell through to a "probe" request that
    // silently rerouted to a fallback model whenever the probe itself hit a
    // transient error (like the 410s we've been seeing) - meaning the real
    // request never even reached the intended model on those attempts.
    // Since a string containing "/" is already a well-formed NIM model ID,
    // use it directly and let the real request (with retry logic below)
    // absorb any flakiness, instead of gambling on a second, throwaway call
    // first. This also means new models can be tried from the client side
    // without needing a matching proxy deploy each time.
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
    let retries = 4;
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
        // 🔥 FIX: A stalled/hung connection surfaces as an axios timeout
        // (error.code === 'ECONNABORTED'), not as an HTTP status code, so it
        // wasn't being caught by the RETRYABLE_STATUSES check before. Now a
        // timeout is retried the same way a 410/503/429 is, instead of the
        // request just dying silently after ~5 minutes on someone else's clock.
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
    // Surface the real NVIDIA error body instead of the generic
    // axios "Request failed with status code XXX" message.
    const nvidiaDetail = error.response?.data?.detail
      || error.response?.data?.error?.message
      || error.response?.data;

    console.error(
      'Proxy error:',
      error.message,
      '| NVIDIA response body:',
      JSON.stringify(error.response?.data)
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
