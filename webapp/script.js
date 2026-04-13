/**
 * Prof. Bhagwan Chowdhry — Knowledge Chatbot
 * Smooth streaming with token queue, debounced markdown, KaTeX math
 */

const chatContainer = document.getElementById("chat-container");
const messageInput = document.getElementById("message-input");
const sendBtn = document.getElementById("send-btn");
const statusText = document.getElementById("status-text");
const statusDot = document.querySelector(".status-dot");
const themeToggle = document.getElementById("theme-toggle");

// ---- Theme Logic ----
const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
const storedTheme = localStorage.getItem('theme');
if (storedTheme === 'dark' || (!storedTheme && prefersDark)) {
  document.documentElement.classList.add('dark-mode');
}

themeToggle.addEventListener('click', () => {
  document.documentElement.classList.toggle('dark-mode');
  const isDark = document.documentElement.classList.contains('dark-mode');
  localStorage.setItem('theme', isDark ? 'dark' : 'light');
});

let conversationHistory = [];
let isStreaming = false;

marked.setOptions({ breaks: true, gfm: true });

function stripWikiBlocks(text) {
  return text.replace(/<wiki_update>[\s\S]*?<\/wiki_update>/g, "").trimEnd();
}

// Math-safe markdown rendering:
// 1. Extract math blocks ($$...$$ and $...$) and replace with placeholders
// 2. Run marked on the math-free text
// 3. Re-insert the math blocks
// 4. Run KaTeX on the result
// This prevents marked from eating backslashes in LaTeX.

function renderMarkdown(el, text) {
  const mathBlocks = [];

  let safe = text;

  // Display math: $$...$$ 
  safe = safe.replace(/\$\$([\s\S]+?)\$\$/g, (_, math) => {
    const id = `%%MATH_${mathBlocks.length}%%`;
    mathBlocks.push({ id, math: math.trim(), display: true });
    return id;
  });

  // Display math: \[...\]
  safe = safe.replace(/\\\[([\s\S]+?)\\\]/g, (_, math) => {
    const id = `%%MATH_${mathBlocks.length}%%`;
    mathBlocks.push({ id, math: math.trim(), display: true });
    return id;
  });

  // Inline math: \(...\) ONLY — no $...$ to avoid matching dollar amounts
  safe = safe.replace(/\\\(([\s\S]+?)\\\)/g, (_, math) => {
    const id = `%%MATH_${mathBlocks.length}%%`;
    mathBlocks.push({ id, math: math.trim(), display: false });
    return id;
  });

  let html = marked.parse(safe);

  for (const block of mathBlocks) {
    let rendered;
    if (typeof katex !== "undefined") {
      try {
        rendered = katex.renderToString(block.math, {
          displayMode: block.display,
          throwOnError: false,
        });
      } catch {
        rendered = `<span class="math-fallback">${block.display ? "\\[" : "\\("}${block.math}${block.display ? "\\]" : "\\)"}</span>`;
      }
    } else {
      rendered = `<span class="math-fallback">${block.display ? "\\[" : "\\("}${block.math}${block.display ? "\\]" : "\\)"}</span>`;
    }
    html = html.replace(block.id, rendered);
  }

  el.innerHTML = html;
}

// ---- Textarea auto-resize ----
messageInput.addEventListener("input", () => {
  messageInput.style.height = "auto";
  messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + "px";
});

messageInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!isStreaming) handleSubmit();
  }
});

function sendSuggestion(btn) {
  if (isStreaming) return;
  messageInput.value = btn.textContent;
  handleSubmit();
}

// ---- Submit ----
function handleSubmit(e) {
  if (e) e.preventDefault();
  const text = messageInput.value.trim();
  if (!text || isStreaming) return;

  const welcome = chatContainer.querySelector(".welcome-message");
  if (welcome) welcome.remove();

  appendMessage("user", text);
  conversationHistory.push({ role: "user", content: text });
  messageInput.value = "";
  messageInput.style.height = "auto";
  streamResponse(text);
}

// ---- Append message ----
function appendMessage(role, content) {
  const msg = document.createElement("div");
  msg.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "msg-avatar";
  avatar.textContent = role === "user" ? "You" : "Prof";

  const bubble = document.createElement("div");
  bubble.className = "msg-body";

  if (role === "user") {
    bubble.textContent = content;
  } else if (content) {
    renderMarkdown(bubble, content);
  }

  msg.appendChild(avatar);
  msg.appendChild(bubble);
  chatContainer.appendChild(msg);
  scrollToBottom();
  return bubble;
}

// ---- Thinking indicator ----
function showThinking() {
  const msg = document.createElement("div");
  msg.className = "message bot";
  msg.id = "thinking-msg";
  const avatar = document.createElement("div");
  avatar.className = "msg-avatar";
  avatar.textContent = "Prof";
  const bubble = document.createElement("div");
  bubble.className = "msg-body thinking";
  bubble.innerHTML =
    '<span class="dot-pulse"><span></span><span></span><span></span></span>';
  msg.appendChild(avatar);
  msg.appendChild(bubble);
  chatContainer.appendChild(msg);
  scrollToBottom();
}

function removeThinking() {
  const el = document.getElementById("thinking-msg");
  if (el) el.remove();
}

// =====================================================
// Smooth streaming engine
// =====================================================
// Tokens arrive from SSE in bursts. Instead of rendering
// them all at once (chunky), we queue them and drain at
// a steady rate using requestAnimationFrame. Markdown is
// re-rendered at most every ~80ms (debounced).
// =====================================================

function hasOpenMath(text) {
  // Check for unclosed $$
  const displayCount = (text.match(/\$\$/g) || []).length;
  if (displayCount % 2 !== 0) return true;

  // Check for unclosed \[
  const openDisplay = (text.match(/\\\[/g) || []).length;
  const closeDisplay = (text.match(/\\\]/g) || []).length;
  if (openDisplay !== closeDisplay) return true;

  // Check for unclosed \(
  const openInline = (text.match(/\\\(/g) || []).length;
  const closeInline = (text.match(/\\\)/g) || []).length;
  if (openInline !== closeInline) return true;

  return false;
}

function createStreamRenderer(bubble) {
  let tokenQueue = "";
  let revealedText = "";
  let drainRAF = null;
  let renderTimer = null;
  let finished = false;

  function scheduleRender() {
    if (renderTimer) return;
    renderTimer = setTimeout(() => {
      renderTimer = null;
      const clean = stripWikiBlocks(revealedText);
      // Don't render if math delimiters are unclosed — avoids flash of broken LaTeX
      if (clean && !hasOpenMath(clean)) {
        renderMarkdown(bubble, clean);
        scrollToBottom();
      }
    }, 80);
  }

  function drain() {
    if (tokenQueue.length === 0) {
      drainRAF = null;
      if (finished) finalRender();
      return;
    }

    const chars = Math.min(Math.max(1, Math.ceil(tokenQueue.length / 8)), 12);
    revealedText += tokenQueue.slice(0, chars);
    tokenQueue = tokenQueue.slice(chars);

    scheduleRender();
    drainRAF = requestAnimationFrame(drain);
  }

  function finalRender() {
    if (renderTimer) {
      clearTimeout(renderTimer);
      renderTimer = null;
    }
    bubble.classList.remove("streaming");
    const clean = stripWikiBlocks(revealedText);
    if (clean) {
      renderMarkdown(bubble, clean);
      scrollToBottom();
    }
  }

  return {
    push(text) {
      tokenQueue += text;
      if (!drainRAF) drainRAF = requestAnimationFrame(drain);
    },
    finish() {
      finished = true;
      if (tokenQueue.length === 0 && !drainRAF) finalRender();
    },
    getText() { return revealedText + tokenQueue; },
    getCleanText() { return stripWikiBlocks(revealedText + tokenQueue); },
    destroy() {
      if (drainRAF) cancelAnimationFrame(drainRAF);
      if (renderTimer) clearTimeout(renderTimer);
    },
  };
}

// ---- Stream response ----
async function streamResponse(userMessage) {
  isStreaming = true;
  sendBtn.disabled = true;
  setStatus("Thinking...", true);
  showThinking();

  let renderer = null;

  try {
    const response = await fetch("/api/chat-v2", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: userMessage,
        history: conversationHistory.slice(-10),
      }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.error || `Server error (${response.status})`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let sseBuffer = "";
    let streamDone = false;
    let fullJsonResponse = "";  // Collect raw JSON chunks

    removeThinking();
    setStatus("Responding...", true);
    const bubble = appendMessage("bot", "");
    bubble.classList.add("streaming");

    // ========== STAGE 1: Collect all JSON chunks (don't render yet!) ==========
    while (!streamDone) {
      const { value, done } = await reader.read();
      if (done) break;

      sseBuffer += decoder.decode(value, { stream: true });
      const lines = sseBuffer.split("\n");
      sseBuffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6).trim();

        if (data === "[DONE]") {
          streamDone = true;
          break;
        }

        try {
          const parsed = JSON.parse(data);
          if (parsed.text) {
            fullJsonResponse += parsed.text;  // Accumulate chunks
          } else if (parsed.error) {
            fullJsonResponse += `\n\n**Error:** ${parsed.error}`;
          }
        } catch {
          // skip malformed
        }
      }
    }

    try { reader.cancel(); } catch { }

    // ========== STAGE 2: Parse JSON and extract answer ==========
    let finalAnswer = "";
    try {
      // Strip markdown code fences in case the model wrapped its JSON
      const cleaned = fullJsonResponse
        .replace(/^```(?:json)?\s*/i, "")
        .replace(/\s*```$/, "")
        .trim();
      const responseObj = JSON.parse(cleaned);
      finalAnswer = responseObj?.answer ?? fullJsonResponse;
    } catch {
      // Not valid JSON — show raw response as fallback
      finalAnswer = fullJsonResponse;
    }
    // ========== STAGE 3: NOW render to UI (clean, no flashing) ==========
    bubble.innerHTML = "";  // Clear the bubble
    renderer = createStreamRenderer(bubble);
    renderer.push(finalAnswer);
    renderer.finish();

    const cleanText = finalAnswer;
    if (cleanText.trim()) {
      conversationHistory.push({ role: "assistant", content: cleanText });
    }
  } catch (err) {
    removeThinking();
    if (renderer) renderer.destroy();
    appendMessage("bot", `**Something went wrong:** ${err.message}`);
  } finally {
    isStreaming = false;
    sendBtn.disabled = false;
    setStatus("Ready", false);
    messageInput.focus();
  }
}

// ---- Helpers ----
function scrollToBottom() {
  requestAnimationFrame(() => {
    chatContainer.scrollTop = chatContainer.scrollHeight;
  });
}

function setStatus(text, thinking) {
  statusText.textContent = text;
  statusDot.classList.toggle("thinking", thinking);
}

async function checkHealth() {
  try {
    const resp = await fetch("/api/health");
    const data = await resp.json();
    if (data.wiki_pages === 0 && data.rag_chunks === 0) {
      setStatus("No data loaded", false);
    }
  } catch { }
}

checkHealth();
