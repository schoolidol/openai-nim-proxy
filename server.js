// server.js - Standardized NVIDIA NIM Proxy
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

// Exact model identifiers as hosted on build.nvidia.com
const MODEL_MAPPING = {
  'gpt-4o': 'deepseek-ai/deepseek-r1',
  'gpt-4': 'deepseek-ai/deepseek-r1',
  'gpt-3.5-turbo': 'meta/llama-3.1-8b-instruct',
  'deepseek-r1': 'deepseek-ai/deepseek-r1'
};

app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'NVIDIA NIM Proxy' });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    // Standardize model name
    const nimModel = MODEL_MAPPING[model] || model;
    let processedMessages = JSON.parse(JSON.stringify(messages));

    // Force step-by-step reasoning using standard prompt injection
    const systemIndex = processedMessages.findIndex(m => m.role === 'system');
    const cotDirective = "Respond using step-by-step internal reasoning enclosed in <think>...</think> tags prior to your final answer.";

    if (systemIndex !== -1) {
      processedMessages[systemIndex].content = `${cotDirective}\n\n${processedMessages[systemIndex].content}`;
    } else {
      processedMessages.unshift({ role: 'system', content: cotDirective });
    }

    // STRICT OPENAI SPEC - No custom keys that break NVIDIA's API gateway
    const nimRequest = {
      model: nimModel,
      messages: processedMessages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 4096,
      stream: stream || false
    };

    console.log(`[Proxy] Sending request for model: ${nimModel}`);

    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      response.data.pipe(res);
    } else {
      res.json(response.data);
    }
  } catch (error) {
    const status = error.response?.status || 500;
    const errorData = error.response?.data || error.message;
    console.error(`[NVIDIA Error ${status}]:`, JSON.stringify(errorData));

    res.status(status).json({
      error: {
        message: errorData.message || 'NVIDIA API Request Failed',
        type: 'invalid_request_error',
        code: status
      }
    });
  }
});

app.listen(PORT, () => {
  console.log(`Proxy listening on port ${PORT}`);
});
