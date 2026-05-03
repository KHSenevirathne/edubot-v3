// EduBot v3 frontend (chat UI only).
//
// Trust-tier note: end users can ONLY (a) chat and (b) flag answers
// 👍/👎. The thumbs-down picker submits a SUGGESTION via /feedback;
// the suggestion stays in 'pending review' until an admin approves
// it on /admin. There is no direct "teach" path from this UI.
// Capabilities here:
//   - sending messages with input validation
//   - XSS-safe rendering of user messages
//   - feedback (thumbs-up / thumbs-down) buttons under every bot reply
//   - on thumbs-down: inline correction picker -> POST /feedback

const MAX_MESSAGE_LEN = 500;
const MIN_TEXT_LEN = 1;

const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const quickActions = document.getElementById('quickActions');

// Cache of intent tags - populated once on first need from /api/intents.
let intentCache = null;

// Track the last bot turn so we know what feedback refers to.
let lastBotTurn = null;

// ---------- Input validation helpers ----------

function trimAndCheck(raw, minLen, maxLen) {
    if (typeof raw !== 'string') return { ok: false, reason: 'invalid input' };
    // Strip control + zero-width chars on the client too so the API gets
    // the same value the user actually saw in the box.
    const cleaned = raw
        .replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏  ﻿]/g, '')
        .trim();
    if (cleaned.length < minLen) return { ok: false, reason: 'too short' };
    if (cleaned.length > maxLen) {
        return { ok: false, reason: `too long (max ${maxLen} chars)` };
    }
    return { ok: true, value: cleaned };
}

// ---------- Send / receive ----------

function refreshSendButton() {
    const v = trimAndCheck(userInput.value, MIN_TEXT_LEN, MAX_MESSAGE_LEN);
    sendBtn.disabled = !v.ok;
    sendBtn.classList.toggle('is-disabled', !v.ok);
}

userInput.addEventListener('input', refreshSendButton);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

function sendMessage() {
    const v = trimAndCheck(userInput.value, MIN_TEXT_LEN, MAX_MESSAGE_LEN);
    if (!v.ok) {
        // Provide a tiny shake animation so the failure is visible.
        userInput.classList.remove('input-error');
        // eslint-disable-next-line no-void
        void userInput.offsetWidth;
        userInput.classList.add('input-error');
        return;
    }
    const message = v.value;

    const welcome = document.querySelector('.welcome-card');
    if (welcome) welcome.style.display = 'none';
    if (quickActions) quickActions.style.display = 'none';

    appendMessage(message, 'user');
    userInput.value = '';
    refreshSendButton();
    userInput.focus();
    showTypingIndicator();

    fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
    })
        .then((r) => r.json())
        .then((data) => {
            removeTypingIndicator();
            lastBotTurn = {
                user_message: message,
                bot_response: data.response,
                predicted_intent: data.tag,
                confidence: data.confidence,
            };
            appendMessage(data.response, 'bot', data.confidence, data.tag, data.source);
        })
        .catch((err) => {
            console.error(err);
            removeTypingIndicator();
            appendMessage("Sorry, I'm having trouble connecting. Please try again.", 'bot', 0, 'error', 'fallback');
        });
}

function sendQuickMessage(message) {
    userInput.value = message;
    sendMessage();
}

function clearChat() {
    chatMessages.innerHTML = `
        <div class="welcome-card">
            <div class="welcome-emoji">&#127891;</div>
            <h2>Welcome to EduBot v3</h2>
            <p>Ask me anything about courses, admissions, fees, exams, or campus life. If I get it wrong, click the thumbs-down and teach me.</p>
        </div>
        <div class="quick-actions" id="quickActions">
            <button class="quick-btn" onclick="sendQuickMessage('What courses do you offer?')"><span class="quick-icon">&#128218;</span> Courses</button>
            <button class="quick-btn" onclick="sendQuickMessage('How do I apply for admission?')"><span class="quick-icon">&#128221;</span> Admission</button>
            <button class="quick-btn" onclick="sendQuickMessage('What are the fees?')"><span class="quick-icon">&#128176;</span> Fees</button>
            <button class="quick-btn" onclick="sendQuickMessage('Tell me about scholarships')"><span class="quick-icon">&#127891;</span> Scholarships</button>
            <button class="quick-btn" onclick="sendQuickMessage('When are the exams?')"><span class="quick-icon">&#128203;</span> Exams</button>
            <button class="quick-btn" onclick="sendQuickMessage('Library hours')"><span class="quick-icon">&#128214;</span> Library</button>
            <button class="quick-btn" onclick="sendQuickMessage('Is hostel available?')"><span class="quick-icon">&#127968;</span> Hostel</button>
            <button class="quick-btn" onclick="sendQuickMessage('How to contact admin?')"><span class="quick-icon">&#128222;</span> Contact</button>
        </div>
    `;
    lastBotTurn = null;
}

// ---------- Message rendering ----------

// HTML-escape a string so it's safe to inject as innerHTML. Used for
// the bot bubble where we WANT line breaks rendered but still need to
// neutralise anything the user (or DB seed data) might contain.
function escapeHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function formatBotText(text) {
    // 1) Escape first, 2) then add the visual line breaks/bullets.
    // Never the other way around.
    return escapeHtml(text)
        .replace(/\n  - /g, '<br>&nbsp;&nbsp;&#8226; ')
        .replace(/\n- /g, '<br>&nbsp;&nbsp;&#8226; ')
        .replace(/\n(\d+)\. /g, '<br>&nbsp;&nbsp;$1. ')
        .replace(/\n/g, '<br>');
}

function appendMessage(text, sender, confidence, tag, source) {
    const wrapper = document.createElement('div');
    wrapper.className = `message ${sender}-message`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = sender === 'bot' ? 'E' : 'U';

    const content = document.createElement('div');
    content.className = 'message-content';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    if (sender === 'user') {
        // XSS-safe: render user input as plain text only.
        bubble.textContent = text;
    } else {
        bubble.innerHTML = formatBotText(text);
    }
    content.appendChild(bubble);

    if (sender === 'bot' && confidence !== undefined) {
        const meta = document.createElement('div');
        meta.className = 'message-meta';

        const time = document.createElement('span');
        time.className = 'message-time';
        time.textContent = currentTime();
        meta.appendChild(time);

        const badge = document.createElement('span');
        badge.className = 'confidence-badge ' + confidenceClass(confidence);
        badge.textContent = confidence > 0
            ? `${Math.round(confidence * 100)}% match`
            : 'Low confidence';
        meta.appendChild(badge);

        if (source) {
            const srcBadge = document.createElement('span');
            srcBadge.className = `source-badge source-${source}`;
            srcBadge.textContent = source;
            srcBadge.title = source === 'database'
                ? 'Answer built from live SQLite data'
                : (source === 'static' ? 'Answer from intents.json template' : 'Fallback response');
            meta.appendChild(srcBadge);
        }

        content.appendChild(meta);

        if (tag !== 'error' && tag !== 'fallback') {
            content.appendChild(buildFeedbackRow());
        }
    } else {
        const time = document.createElement('div');
        time.className = 'message-time';
        time.textContent = currentTime();
        content.appendChild(time);
    }

    wrapper.appendChild(avatar);
    wrapper.appendChild(content);
    chatMessages.appendChild(wrapper);
    scrollToBottom();
}

function confidenceClass(c) {
    if (c >= 0.7) return 'confidence-high';
    if (c >= 0.4) return 'confidence-medium';
    return 'confidence-low';
}

// ---------- Feedback row ----------

function buildFeedbackRow() {
    const row = document.createElement('div');
    row.className = 'feedback-row';

    const prompt = document.createElement('span');
    prompt.className = 'feedback-prompt';
    prompt.textContent = 'Was this helpful?';
    row.appendChild(prompt);

    const up = document.createElement('button');
    up.className = 'feedback-btn';
    up.textContent = '👍';
    up.title = 'Yes, this was helpful';
    up.onclick = () => sendFeedback(true, row);
    row.appendChild(up);

    const down = document.createElement('button');
    down.className = 'feedback-btn';
    down.textContent = '👎';
    down.title = 'No, this was not helpful';
    down.onclick = () => sendFeedback(false, row);
    row.appendChild(down);

    return row;
}

async function sendFeedback(helpful, row) {
    if (!lastBotTurn) return;
    if (helpful) {
        await postFeedback({ ...lastBotTurn, helpful: true });
        replaceFeedbackRow(row, 'Thanks for the feedback!');
    } else {
        // Inline correction picker - the user suggests an intent; the
        // suggestion goes into the admin review queue, NOT directly
        // into training. This is the trust-tier split in action.
        row.innerHTML = '';
        const label = document.createElement('span');
        label.className = 'feedback-prompt';
        label.textContent = 'What topic should I have used?';
        row.appendChild(label);

        const select = document.createElement('select');
        select.className = 'feedback-select';
        for (const tag of await getIntents()) {
            const opt = document.createElement('option');
            opt.value = tag;
            opt.textContent = tag;
            select.appendChild(opt);
        }
        row.appendChild(select);

        const save = document.createElement('button');
        save.className = 'feedback-btn feedback-save';
        save.textContent = 'Send suggestion';
        save.onclick = async () => {
            await postFeedback({
                ...lastBotTurn,
                helpful: false,
                expected_intent: select.value,
            });
            replaceFeedbackRow(
                row,
                `Thanks - your suggestion ("${select.value}") has been sent for review.`
            );
        };
        row.appendChild(save);
    }
}

function replaceFeedbackRow(row, message) {
    row.innerHTML = `<span class="feedback-thanks">${message}</span>`;
}

async function postFeedback(payload) {
    try {
        const r = await fetch('/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        return await r.json();
    } catch (e) {
        console.error('feedback failed', e);
    }
}

// ---------- Intent list (used by the inline 👎 correction picker) ----------

async function getIntents() {
    if (intentCache) return intentCache;
    try {
        const r = await fetch('/api/intents');
        const data = await r.json();
        intentCache = data.intents.map((i) => i.tag).sort();
    } catch (e) {
        console.error('failed to load intents', e);
        intentCache = [];
    }
    return intentCache;
}

document.addEventListener('DOMContentLoaded', () => {
    refreshSendButton();
});

// ---------- Typing indicator ----------

function showTypingIndicator() {
    const div = document.createElement('div');
    div.className = 'message bot-message';
    div.id = 'typingIndicator';
    div.innerHTML = `
        <div class="message-avatar">E</div>
        <div class="message-content">
            <div class="typing-indicator"><span></span><span></span><span></span></div>
        </div>
    `;
    chatMessages.appendChild(div);
    scrollToBottom();
}

function removeTypingIndicator() {
    const el = document.getElementById('typingIndicator');
    if (el) el.remove();
}

// ---------- Helpers ----------

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function currentTime() {
    return new Date().toLocaleTimeString('en-US', {
        hour: 'numeric', minute: '2-digit', hour12: true,
    });
}
