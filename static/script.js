// EduBot v3 frontend (chat UI only).
//
// Trust-tier note: end users can ONLY (a) chat and (b) flag answers
// thumbs-up / thumbs-down. The thumbs-down picker submits a SUGGESTION
// via /feedback; the suggestion stays in 'pending review' until an admin
// approves it on /admin. There is no direct "teach" path from this UI.
//
// Capabilities here:
//   - sending messages with input validation
//   - per-browser session ID so the server can do multi-turn dialogue
//     (resolving "it" / "this course" to a previously mentioned entity)
//   - mood-changing avatar driven by the bot's confidence + answer source
//   - optional voice input (Web Speech API) and read-aloud (speechSynthesis)
//   - XSS-safe rendering of user messages
//   - feedback (thumbs-up / thumbs-down) buttons under every bot reply
//   - on thumbs-down: inline correction picker -> POST /feedback

const MAX_MESSAGE_LEN = 500;
const MIN_TEXT_LEN = 1;

const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const quickActions = document.getElementById('quickActions');
const headerAvatar = document.getElementById('headerAvatar');
const headerFace = document.getElementById('botAvatarFace');
const moodLabel = document.getElementById('botMoodLabel');
const micBtn = document.getElementById('micBtn');
const ttsToggle = document.getElementById('ttsToggle');

// Cache of intent tags - populated once on first need from /api/intents.
let intentCache = null;

// Track the last bot turn so we know what feedback refers to.
let lastBotTurn = null;

// ---------- Session ID (multi-turn dialogue) ----------
// We persist a single opaque session token per browser so the backend
// can remember the entity ("BSc Computer Science") that was last
// discussed and resolve "it" / "this one" to it on the next turn.
const SESSION_KEY = 'edubot.sessionId';

function newSessionId() {
    if (window.crypto && typeof window.crypto.randomUUID === 'function') {
        return window.crypto.randomUUID().replace(/-/g, '');
    }
    // Fallback for older browsers - good enough as a session token.
    return 'sid' + Date.now().toString(36) + Math.random().toString(36).slice(2, 10);
}

function getSessionId() {
    let id = null;
    try {
        id = localStorage.getItem(SESSION_KEY);
    } catch (_) { /* private mode */ }
    if (!id) {
        id = newSessionId();
        try { localStorage.setItem(SESSION_KEY, id); } catch (_) { /* ignore */ }
    }
    return id;
}

function resetSessionId() {
    const old = (() => { try { return localStorage.getItem(SESSION_KEY); } catch (_) { return null; } })();
    try { localStorage.removeItem(SESSION_KEY); } catch (_) { /* ignore */ }
    if (old) {
        // Best-effort - server-side reset, but don't block the UI on it.
        fetch('/session/reset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: old }),
        }).catch(() => { /* ignore */ });
    }
}

// ---------- Input validation helpers ----------

// Bad characters: ASCII control codes (NUL, BEL, VT, FF, etc.) plus
// invisible / direction-override / line-separator Unicode chars and the
// byte-order mark. We use \u escape sequences (NOT literal Unicode) on
// purpose: U+2028 / U+2029 are valid line terminators in JavaScript and
// having them appear LITERALLY inside a regex literal can break parsing.
const BAD_CHARS_RE = new RegExp(
    '[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f' +
    '\\u200B-\\u200F' +    // zero-width / direction override
    '\\u2028\\u2029' +     // line / paragraph separators
    '\\uFEFF' +            // byte-order mark
    ']',
    'g'
);

// Quality thresholds - kept in sync with app/validate.py so client and
// server agree on what counts as junk input.
const MAX_ALPHA_RUN   = 30;     // longest unbroken letter run
const MAX_DIGIT_RUN   = 9;      // longest unbroken digit run
const MIN_LETTER_RATIO = 0.30;  // min proportion of letters in the message

const ALPHA_RUN_RE = new RegExp(`[A-Za-z]{${MAX_ALPHA_RUN + 1},}`);
const DIGIT_RUN_RE = new RegExp(`\\d{${MAX_DIGIT_RUN + 1},}`);
const LETTER_RE    = /[A-Za-z]/g;

function trimAndCheck(raw, minLen, maxLen) {
    if (typeof raw !== 'string') return { ok: false, reason: 'invalid input' };
    const cleaned = raw.replace(BAD_CHARS_RE, '').trim();
    if (cleaned.length < minLen) return { ok: false, reason: 'too short' };
    if (cleaned.length > maxLen) {
        return { ok: false, reason: `too long (max ${maxLen} chars)` };
    }
    return { ok: true, value: cleaned };
}

// Heuristic quality gate. Returns { ok: true } or { ok: false, reason }.
// Mirrors app/validate.py:check_message_quality.
function checkQuality(text) {
    if (ALPHA_RUN_RE.test(text)) {
        return {
            ok: false,
            reason: "That looks like gibberish - try asking a real question, e.g. 'what courses do you offer?'",
        };
    }
    if (DIGIT_RUN_RE.test(text)) {
        return {
            ok: false,
            reason: "Long number sequences (phone numbers, account numbers) aren't valid questions. Ask in words.",
        };
    }
    if (text.length >= 4) {
        const letters = (text.match(LETTER_RE) || []).length;
        if (letters / text.length < MIN_LETTER_RATIO) {
            return {
                ok: false,
                reason: 'Please ask a question in words - that input is mostly symbols or numbers.',
            };
        }
    }
    return { ok: true };
}

// ---------- Send / receive ----------

function refreshSendButton() {
    const v = trimAndCheck(userInput.value, MIN_TEXT_LEN, MAX_MESSAGE_LEN);
    let ok = v.ok;
    let reason = v.reason;
    if (ok) {
        const q = checkQuality(v.value);
        if (!q.ok) { ok = false; reason = q.reason; }
    }
    sendBtn.disabled = !ok;
    sendBtn.classList.toggle('is-disabled', !ok);
    // Stash the reason on the button so the click handler can show it.
    sendBtn.dataset.reason = ok ? '' : (reason || 'invalid input');
    if (userInput) {
        userInput.title = ok ? '' : (reason || '');
    }
}

userInput.addEventListener('input', refreshSendButton);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

function sendMessage() {
    const v = trimAndCheck(userInput.value, MIN_TEXT_LEN, MAX_MESSAGE_LEN);
    const failedClient = !v.ok;
    const q = v.ok ? checkQuality(v.value) : { ok: false, reason: v.reason };
    if (failedClient || !q.ok) {
        // Shake the input and surface the reason as a chat message so
        // the user knows WHY their input was rejected (not just "send
        // is disabled").
        userInput.classList.remove('input-error');
        // eslint-disable-next-line no-void
        void userInput.offsetWidth;
        userInput.classList.add('input-error');
        const reason = q.reason || v.reason || 'Invalid input.';
        appendMessage(reason, 'bot', 0, 'error', 'fallback');
        updateMood(0, 'fallback', 'error');
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
        body: JSON.stringify({ message, session_id: getSessionId() }),
    })
        .then(async (r) => {
            const data = await r.json().catch(() => ({}));
            return { ok: r.ok, status: r.status, data };
        })
        .then(({ ok, data }) => {
            removeTypingIndicator();
            if (!ok) {
                // Server-side validation rejection (400) - show the
                // reason inline rather than a generic error.
                const reason = data.error || 'Sorry, that input was rejected by the server.';
                appendMessage(reason, 'bot', 0, 'error', 'fallback');
                updateMood(0, 'fallback', 'error');
                return;
            }
            lastBotTurn = {
                user_message: message,
                bot_response: data.response,
                predicted_intent: data.tag,
                confidence: data.confidence,
            };
            appendMessage(data.response, 'bot', data.confidence, data.tag, data.source);
            updateMood(data.confidence, data.source, data.tag);
            speak(data.response);
        })
        .catch((err) => {
            console.error(err);
            removeTypingIndicator();
            appendMessage("Sorry, I'm having trouble connecting. Please try again.", 'bot', 0, 'error', 'fallback');
            updateMood(0, 'fallback', 'error');
        });
}

function sendQuickMessage(message) {
    userInput.value = message;
    sendMessage();
}

function clearChat() {
    resetSessionId();
    if (typeof window.speechSynthesis !== 'undefined') {
        window.speechSynthesis.cancel();
    }
    updateMood(1.0, 'static', 'greeting');
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
    refreshSendButton();
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
    if (sender === 'bot') {
        const mood = moodFor(confidence, source, tag);
        avatar.classList.add(`mood-${mood.name}`);
        const face = document.createElement('span');
        face.className = 'mood-face';
        face.textContent = mood.face;
        avatar.appendChild(face);
    } else {
        avatar.textContent = 'U';
    }

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
    up.textContent = '👍';   // thumbs up
    up.title = 'Yes, this was helpful';
    up.onclick = () => sendFeedback(true, row);
    row.appendChild(up);

    const down = document.createElement('button');
    down.className = 'feedback-btn';
    down.textContent = '👎'; // thumbs down
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
    row.innerHTML = `<span class="feedback-thanks">${escapeHtml(message)}</span>`;
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

// ---------- Intent list (used by the inline thumbs-down picker) ----------

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

// ---------- Boot ----------

// Run once on initial page load. Without this the send button would
// stay in its HTML default (disabled) until the user types something.
refreshSendButton();
document.addEventListener('DOMContentLoaded', refreshSendButton);

// ---------- Typing indicator ----------

function showTypingIndicator() {
    setMood('thinking', '\u{1F914}', 'Thinking...');     // thinking face
    const div = document.createElement('div');
    div.className = 'message bot-message';
    div.id = 'typingIndicator';
    div.innerHTML = `
        <div class="message-avatar mood-thinking"><span class="mood-face">&#129300;</span></div>
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

// ---------- Mood (emotional intelligence) ----------
//
// The avatar's expression reflects how confident we were in the last
// answer and where it came from. Visible feedback for the user, plus
// a low-cost ticked box on the brief's "emotional intelligence" trait.

function moodFor(confidence, source, tag) {
    if (tag === 'fallback' || source === 'fallback' || tag === 'error') {
        return { name: 'confused', face: '\u{1F615}', label: "Hmm, I'm not sure" };  // confused face
    }
    if (tag === 'clarify') {
        return { name: 'thinking', face: '\u{1F914}', label: 'Need a bit more info' };
    }
    const c = Number(confidence) || 0;
    if (c >= 0.7 || source === 'database') {
        return { name: 'happy', face: '\u{1F642}', label: 'Happy to help' };  // slight smile
    }
    if (c >= 0.4) {
        return { name: 'neutral', face: '\u{1F610}', label: 'Best guess answer' };  // neutral face
    }
    return { name: 'confused', face: '\u{1F615}', label: 'Low confidence' };
}

function setMood(name, face, label) {
    if (!headerAvatar || !headerFace) return;
    headerAvatar.classList.remove('mood-happy', 'mood-neutral', 'mood-confused', 'mood-thinking');
    headerAvatar.classList.add(`mood-${name}`);
    headerFace.textContent = face;
    headerAvatar.classList.add('is-bouncing');
    setTimeout(() => headerAvatar.classList.remove('is-bouncing'), 250);
    if (moodLabel && label) moodLabel.textContent = label;
}

function updateMood(confidence, source, tag) {
    const m = moodFor(confidence, source, tag);
    setMood(m.name, m.face, m.label);
}

// ---------- Voice input (Web Speech API) ----------

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let isListening = false;

if (SpeechRecognition && micBtn) {
    recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        userInput.value = transcript;
        refreshSendButton();
        // A short delay before sending lets the user see what was heard.
        setTimeout(() => {
            if (!sendBtn.disabled) sendMessage();
        }, 250);
    };
    recognition.onerror = (event) => {
        console.warn('Speech recognition error:', event.error);
        stopListening();
    };
    recognition.onend = () => stopListening();
} else if (micBtn) {
    // Browser doesn't support speech recognition - dim the button.
    micBtn.classList.add('is-disabled');
    micBtn.title = 'Voice input not supported in this browser';
}

function toggleVoiceInput() {
    if (!recognition) return;
    if (isListening) {
        try { recognition.stop(); } catch (_) { /* ignore */ }
        return;
    }
    try {
        recognition.start();
        isListening = true;
        micBtn.classList.add('is-listening');
        micBtn.setAttribute('aria-pressed', 'true');
        userInput.placeholder = 'Listening...';
    } catch (e) {
        console.warn('Could not start recognition:', e);
        stopListening();
    }
}

function stopListening() {
    isListening = false;
    if (micBtn) {
        micBtn.classList.remove('is-listening');
        micBtn.setAttribute('aria-pressed', 'false');
    }
    if (userInput) {
        userInput.placeholder = 'Ask me anything about the university...';
    }
}

// ---------- Text-to-speech (read replies aloud) ----------

const TTS_KEY = 'edubot.ttsEnabled';
let ttsEnabled = false;

function loadTtsPref() {
    try {
        ttsEnabled = localStorage.getItem(TTS_KEY) === '1';
    } catch (_) {
        ttsEnabled = false;
    }
    if (ttsToggle) {
        ttsToggle.classList.toggle('is-active', ttsEnabled);
        ttsToggle.setAttribute('aria-pressed', ttsEnabled ? 'true' : 'false');
    }
}

function toggleTts() {
    ttsEnabled = !ttsEnabled;
    try { localStorage.setItem(TTS_KEY, ttsEnabled ? '1' : '0'); } catch (_) { /* ignore */ }
    if (ttsToggle) {
        ttsToggle.classList.toggle('is-active', ttsEnabled);
        ttsToggle.setAttribute('aria-pressed', ttsEnabled ? 'true' : 'false');
        ttsToggle.title = ttsEnabled ? 'Stop reading replies aloud' : 'Read replies aloud';
    }
    if (!ttsEnabled && typeof window.speechSynthesis !== 'undefined') {
        window.speechSynthesis.cancel();
    }
}

function speak(text) {
    if (!ttsEnabled) return;
    if (typeof window.speechSynthesis === 'undefined') return;
    if (!text) return;
    // Strip the bullet/list markup so the synthesiser reads naturally.
    const spoken = String(text)
        .replace(/^\s*-\s+/gm, '')
        .replace(/\n+/g, '. ')
        .replace(/\s{2,}/g, ' ');
    const utt = new SpeechSynthesisUtterance(spoken.slice(0, 600));
    utt.lang = 'en-US';
    utt.rate = 1.0;
    utt.pitch = 1.0;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utt);
}

loadTtsPref();
