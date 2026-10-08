const themeToggleBtn = document.getElementById('themeToggleBtn');
const themeLabel = document.getElementById('themeLabel');
const telStatus = document.getElementById('telStatus');
const telTotalFrames = document.getElementById('telTotalFrames');
const mainVideoPlayer = document.getElementById('mainVideoPlayer');
const videoTimeline = document.getElementById('videoTimeline');
const btnPlayPause = document.getElementById('btnPlayPause');
const btnStepBack = document.getElementById('btnStepBack');
const btnStepFwd = document.getElementById('btnStepFwd');
const playbackRateSelect = document.getElementById('playbackRateSelect');
const btnUploadNew = document.getElementById('btnUploadNew');
const hudSmpte = document.getElementById('hudSmpte');
const hudResolution = document.getElementById('hudResolution');
const hudTimeSec = document.getElementById('hudTimeSec');
const hudFrameIndex = document.getElementById('hudFrameIndex');
const dropzonePrompt = document.getElementById('dropzonePrompt');
const dropzoneBox = document.getElementById('dropzoneBox');
const videoFileInput = document.getElementById('videoFileInput');
const ingestProgressRow = document.getElementById('ingestProgressRow');
const ingestLabel = document.getElementById('ingestLabel');
const ingestPct = document.getElementById('ingestPct');
const ingestBar = document.getElementById('ingestBar');
const activeFrameImg = document.getElementById('activeFrameImg');
const frameEmptyState = document.getElementById('frameEmptyState');
const activeFrameId = document.getElementById('activeFrameId');
const metaSec = document.getElementById('metaSec');
const metaSmpte = document.getElementById('metaSmpte');
const chatConsoleMessages = document.getElementById('chatConsoleMessages');
const consoleForm = document.getElementById('consoleForm');
const consoleInput = document.getElementById('consoleInput');
const btnConsoleSend = document.getElementById('btnConsoleSend');
const btnResetChat = document.getElementById('btnResetChat');

let isVideoIngested = false;
let totalDuration = 0;
let lastRenderedSec = -1;

function initTheme() {
    const savedTheme = localStorage.getItem('app-theme') || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    applyTheme(savedTheme);
}

function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('app-theme', theme);
    themeLabel.textContent = theme.toUpperCase();
}

themeToggleBtn.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    applyTheme(current === 'dark' ? 'light' : 'dark');
});

initTheme();

function setTelemetryStatus(state, label) {
    telStatus.className = `tel-value status-${state}`;
    telStatus.textContent = label;
}

dropzoneBox.addEventListener('click', () => videoFileInput.click());

btnUploadNew.addEventListener('click', () => {
    dropzonePrompt.classList.remove('hidden');
    videoFileInput.click();
});

['dragenter', 'dragover'].forEach(name => {
    dropzoneBox.addEventListener(name, (e) => {
        e.preventDefault();
        dropzoneBox.classList.add('dragover');
    });
});

['dragleave', 'drop'].forEach(name => {
    dropzoneBox.addEventListener(name, (e) => {
        e.preventDefault();
        dropzoneBox.classList.remove('dragover');
    });
});

dropzoneBox.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        uploadSurveillanceVideo(e.dataTransfer.files[0]);
    }
});

videoFileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
        uploadSurveillanceVideo(e.target.files[0]);
    }
});

function uploadSurveillanceVideo(file) {
    ingestProgressRow.classList.remove('hidden');
    ingestBar.style.width = '15%';
    ingestPct.textContent = '15%';
    ingestLabel.textContent = 'INGESTING FOOTAGE & EXTRACTING 1.0 FPS FRAMES...';
    setTelemetryStatus('processing', 'SAMPLING 1 FPS');

    const formData = new FormData();
    formData.append('file', file);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/upload', true);

    xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
            const pct = Math.min(85, Math.round((e.loaded / e.total) * 75) + 10);
            ingestBar.style.width = `${pct}%`;
            ingestPct.textContent = `${pct}%`;
        }
    };

    xhr.onload = () => {
        if (xhr.status === 200) {
            const res = JSON.parse(xhr.responseText);
            ingestBar.style.width = '100%';
            ingestPct.textContent = '100%';
            ingestLabel.textContent = `SYNC COMPLETE: ${res.frames_extracted} FRAMES SAMPLED @ 1.0 FPS`;

            totalDuration = res.duration;
            telTotalFrames.textContent = res.frames_extracted;
            hudResolution.textContent = res.resolution || '-- x --';
            videoTimeline.max = totalDuration;
            isVideoIngested = true;

            setTimeout(() => {
                ingestProgressRow.classList.add('hidden');
                dropzonePrompt.classList.add('hidden');
            }, 600);

            setTelemetryStatus('active', 'STREAM SYNCHRONIZED');
            mainVideoPlayer.src = '/api/video?ts=' + Date.now();
            consoleInput.disabled = false;
            btnConsoleSend.disabled = false;
            consoleInput.focus();

            updateFrameTelemetry(0);
            logSystemEvent(`Footage ingested: ${res.duration}s total duration, ${res.frames_extracted} frames sampled at 1.0 FPS. Multi-turn reasoning initialized on Gemma 4.`);
        } else {
            setTelemetryStatus('idle', 'INGEST ERROR');
            ingestLabel.textContent = 'ERROR INGESTING FOOTAGE';
            ingestBar.style.background = '#ef4444';
        }
    };

    xhr.onerror = () => {
        setTelemetryStatus('idle', 'NETWORK FAIL');
        ingestLabel.textContent = 'COMMUNICATION ERROR';
    };

    xhr.send(formData);
}

function formatSmpte(sec) {
    const s = Math.floor(sec);
    const hrs = Math.floor(s / 3600).toString().padStart(2, '0');
    const mins = Math.floor((s % 3600) / 60).toString().padStart(2, '0');
    const secs = Math.floor(s % 60).toString().padStart(2, '0');
    return `${hrs}:${mins}:${secs}`;
}

function parseTimestampToSeconds(str) {
    const clean = str.replace(/[\[\]]/g, '').trim();
    const parts = clean.split(':').map(Number);
    if (parts.length === 2) {
        return (parts[0] * 60) + parts[1];
    }
    if (parts.length === 3) {
        return (parts[0] * 3600) + (parts[1] * 60) + parts[2];
    }
    const val = parseInt(clean, 10);
    return isNaN(val) ? 0 : val;
}

mainVideoPlayer.addEventListener('timeupdate', () => {
    const cur = mainVideoPlayer.currentTime;
    const curSec = Math.floor(cur);

    videoTimeline.value = curSec;
    hudSmpte.textContent = formatSmpte(cur);
    hudTimeSec.textContent = `T+ ${cur.toFixed(2)}s`;
    hudFrameIndex.textContent = `FRAME #${curSec.toString().padStart(4, '0')}`;

    if (curSec !== lastRenderedSec) {
        lastRenderedSec = curSec;
        updateFrameTelemetry(curSec);
    }
});

mainVideoPlayer.addEventListener('play', () => {
    btnPlayPause.textContent = '❚❚ PAUSE';
});

mainVideoPlayer.addEventListener('pause', () => {
    btnPlayPause.textContent = '▶ PLAY';
});

btnPlayPause.addEventListener('click', () => {
    if (!isVideoIngested) return;
    if (mainVideoPlayer.paused) {
        mainVideoPlayer.play();
    } else {
        mainVideoPlayer.pause();
    }
});

btnStepBack.addEventListener('click', () => {
    if (!isVideoIngested) return;
    mainVideoPlayer.currentTime = Math.max(0, mainVideoPlayer.currentTime - 1);
});

btnStepFwd.addEventListener('click', () => {
    if (!isVideoIngested) return;
    mainVideoPlayer.currentTime = Math.min(totalDuration, mainVideoPlayer.currentTime + 1);
});

playbackRateSelect.addEventListener('change', () => {
    mainVideoPlayer.playbackRate = parseFloat(playbackRateSelect.value);
});

videoTimeline.addEventListener('input', () => {
    if (!isVideoIngested) return;
    const sec = parseInt(videoTimeline.value, 10);
    mainVideoPlayer.currentTime = sec;
    updateFrameTelemetry(sec);
});

function updateFrameTelemetry(sec) {
    const s = Math.max(0, sec);
    activeFrameId.textContent = `FRAME #${s.toString().padStart(4, '0')}`;
    metaSec.textContent = `${s}s`;
    metaSmpte.textContent = formatSmpte(s);

    activeFrameImg.src = `/api/frames/${s}?ts=` + Date.now();
    activeFrameImg.classList.remove('hidden');
    frameEmptyState.classList.add('hidden');
}

function seekToTimestamp(sec) {
    if (!isVideoIngested) return;
    mainVideoPlayer.currentTime = sec;
    mainVideoPlayer.play();
    updateFrameTelemetry(sec);
}

function logSystemEvent(msg) {
    const entry = document.createElement('div');
    entry.className = 'terminal-entry system-entry';
    entry.innerHTML = `
        <div class="entry-meta font-mono">[FORENSIC KERNEL] ${new Date().toLocaleTimeString()}</div>
        <div class="entry-content">${escapeHtml(msg)}</div>
    `;
    chatConsoleMessages.appendChild(entry);
    chatConsoleMessages.scrollTop = chatConsoleMessages.scrollHeight;
}

document.querySelectorAll('.preset-chip').forEach(btn => {
    btn.addEventListener('click', () => {
        if (!isVideoIngested) return;
        consoleInput.value = btn.getAttribute('data-query');
        consoleForm.dispatchEvent(new Event('submit'));
    });
});

chatConsoleMessages.addEventListener('click', (e) => {
    const badge = e.target.closest('.timestamp-badge');
    if (badge) {
        const sec = parseInt(badge.getAttribute('data-sec'), 10);
        seekToTimestamp(sec);
    }
});

consoleForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = consoleInput.value.trim();
    if (!query || !isVideoIngested) return;

    consoleInput.value = '';
    consoleInput.disabled = true;
    btnConsoleSend.disabled = true;
    setTelemetryStatus('processing', 'QUERY STREAMING');

    const userEntry = document.createElement('div');
    userEntry.className = 'terminal-entry user-entry';
    userEntry.innerHTML = `
        <div class="entry-meta font-mono"><span>[OPERATOR QUERY]</span> <span>${new Date().toLocaleTimeString()}</span></div>
        <div class="entry-content">${escapeHtml(query)}</div>
    `;
    chatConsoleMessages.appendChild(userEntry);

    const asstEntry = document.createElement('div');
    asstEntry.className = 'terminal-entry assistant-entry';
    const contentBox = document.createElement('div');
    contentBox.className = 'entry-content';
    contentBox.innerHTML = `<span class="font-mono text-muted">GENERATING EVIDENCE STREAM</span><span class="streaming-cursor"></span>`;

    asstEntry.innerHTML = `
        <div class="entry-meta font-mono"><span>[GEMMA 4 REASONING ENGINE]</span> <span>${new Date().toLocaleTimeString()}</span></div>
    `;
    asstEntry.appendChild(contentBox);
    chatConsoleMessages.appendChild(asstEntry);
    chatConsoleMessages.scrollTop = chatConsoleMessages.scrollHeight;

    let accumulatedText = '';

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: query })
        });

        if (!res.ok) {
            const err = await res.json();
            contentBox.innerHTML = `<span style="color:#ef4444">[ERROR] ${escapeHtml(err.detail || 'Inference execution failed')}</span>`;
            return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let streamBuffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            streamBuffer += decoder.decode(value, { stream: true });
            const segments = streamBuffer.split('\n\n');
            streamBuffer = segments.pop();

            for (const seg of segments) {
                if (seg.startsWith('data: ')) {
                    const rawJson = seg.substring(6).trim();
                    try {
                        const payload = JSON.parse(rawJson);
                        if (payload.token) {
                            accumulatedText += payload.token;
                            contentBox.innerHTML = renderFormattedOutput(accumulatedText) + '<span class="streaming-cursor"></span>';
                            chatConsoleMessages.scrollTop = chatConsoleMessages.scrollHeight;
                        }
                    } catch (e) {}
                }
            }
        }

        contentBox.innerHTML = renderFormattedOutput(accumulatedText);
    } catch (err) {
        contentBox.innerHTML = `<span style="color:#ef4444">[FATAL] Connection severed during streaming.</span>`;
    } finally {
        setTelemetryStatus('active', 'STREAM SYNCHRONIZED');
        consoleInput.disabled = false;
        btnConsoleSend.disabled = false;
        consoleInput.focus();
        chatConsoleMessages.scrollTop = chatConsoleMessages.scrollHeight;
    }
});

function renderFormattedOutput(raw) {
    let text = escapeHtml(raw);

    text = text.replace(/### (.*?)(?:\n|$)/g, '<h4 style="font-size:0.86rem;font-weight:700;margin:8px 0 4px 0;color:var(--accent-active);">$1</h4>');
    text = text.replace(/## (.*?)(?:\n|$)/g, '<h3 style="font-size:0.92rem;font-weight:700;margin:10px 0 6px 0;color:var(--text-primary);">$1</h3>');
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
    text = text.replace(/`(.*?)`/g, '<code style="background:var(--code-bg);padding:1px 4px;border-radius:2px;font-family:var(--font-mono);font-size:0.75rem;">$1</code>');
    text = text.replace(/\n/g, '<br>');

    const tsRegex = /\[?(\d{1,2}:\d{2}(?::\d{2})?)\s*(?:-|to)?\s*(\d{1,2}:\d{2}(?::\d{2})?)?\]?/gi;
    text = text.replace(tsRegex, (match, t1, t2) => {
        if (!t1) return match;
        const sec1 = parseTimestampToSeconds(t1);
        if (t2) {
            const sec2 = parseTimestampToSeconds(t2);
            return `<button type="button" class="timestamp-badge" data-sec="${sec1}">&#9658; [${t1} - ${t2}]</button>`;
        }
        return `<button type="button" class="timestamp-badge" data-sec="${sec1}">&#9658; [${t1}]</button>`;
    });

    return text;
}

function escapeHtml(str) {
    const d = document.createElement('div');
    d.innerText = str;
    return d.innerHTML;
}

btnResetChat.addEventListener('click', () => {
    chatConsoleMessages.innerHTML = '';
    logSystemEvent('Console cleared by operator. Historical buffer reset.');
});

consoleInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        consoleForm.dispatchEvent(new Event('submit'));
    }
});
