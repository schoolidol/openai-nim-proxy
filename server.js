// server.js - OpenAI to NVIDIA NIM API Proxy with Explicit DeepSeek Reasoning
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

const MODEL_MAPPING = {
  'gpt-4o': 'deepseek-ai/deepseek-r1',
  'gpt-4': 'deepseek-ai/deepseek-r1',
  'gpt-3.5-turbo': 'deepseek-ai/deepseek-r1',
  'deepseek-r1': 'deepseek-ai/deepseek-r1'
};

app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'NVIDIA NIM Reasoning Proxy', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    let nimModel = MODEL_MAPPING[model] || model;
    let processedMessages = JSON.parse(JSON.stringify(messages));

    // Force explicit Chain-of-Thought delimiter prompt for DeepSeek engine
    if (ENABLE_THINKING_MODE) {
      const systemMsgIndex = processedMessages.findIndex(m => m.role === 'system');
      const cotDirective = "Respond using step-by-step reasoning enclosed within <think>...</think> tags before providing the final answer.";

      if (systemMsgIndex !== -1) {
        processedMessages[systemMsgIndex].content = `${cotDirective}\n\n${processedMessages[systemMsgIndex].content}`;
      } else {
        processedMessages.unshift({ role: 'system', content: cotDirective });
      }
    }
    
    // Construct standardized payload compatible with NVIDIA NIM DeepSeek endpoint
    const nimRequest = {
      model: nimModel,
      messages: processedMessages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 8192,
      stream: stream || false,
      // Pass thinking activation via extra_body format for DeepSeek engine compatibility
      ...(ENABLE_THINKING_MODE && {
        reasoning_effort: "high",
        thinking: { type: "enabled" }
      })
    };

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
      
      let buffer = '';
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
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
                // Parse standard content as well as reasoning_content parameter
                const reasoning = delta.reasoning_content || delta.reasoning || null;
                const content = delta.content || null;
                
                let combinedContent = '';
                if (reasoning) {
                  combinedContent += reasoning;
                }
                if (content) {
                  combinedContent += content;
                }
                
                data.choices[0].delta.content = combinedContent;
                delete data.choices[0].delta.reasoning_content;
                delete data.choices[0].delta.reasoning;
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(`${trimmed}\n\n`);
            }
          }
        });
      });
      
      response.data.on('end', () => res.end());
      response.data.on('error', () => res.end());
    } else {
      const choice = response.data.choices[0];
      let fullContent = choice.message?.content || '';
      const reasoning = choice.message?.reasoning_content || choice.message?.reasoning;
      
      if (reasoning && !fullContent.includes('<think>')) {
        fullContent = `<think>\n${reasoning}\n</think>\n\n${fullContent}`;
      }
      
      res.json({
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: [{
          index: 0,
          message: { role: 'assistant', content: fullContent },
          finish_reason: choice.finish_reason
        }],
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      });
    }
  } catch (error) {
    console.error('Proxy Error:', error.response?.data || error.message);
    res.status(500).json({ error: { message: error.message } });
  }
});

app.listen(PORT, () => {
  console.log(`Server executing on port ${PORT}`);
});
