// (() => {
    const LOG = document.getElementById('log');
    const datasetSelect = document.getElementById('datasetSelect');
    const modelSelect = document.getElementById('modelSelect');
    const loadDatasetBtn = document.getElementById('loadDatasetBtn');
    const loadModelBtn = document.getElementById('loadModelBtn');
    const editParamsBtn = document.getElementById('editParamsBtn');
    const saveParamsBtn = document.getElementById('saveParamsBtn');
    const paramsList = document.getElementById('paramsList');
    const startSimBtn = document.getElementById('startSimBtn');
    const stopSimBtn = document.getElementById('stopSimBtn');
    const slidersDiv = document.getElementById('sliders');
    const mapDiv = document.getElementById('map');
    const trailSlider = document.getElementById('trailSlider');
    const simDurationSlider = document.getElementById('simDurationSlider');
    const trailValue = document.getElementById('trailValue');
    const simDurationValue = document.getElementById('simDurationValue');
    const autoViewCheckbox = document.getElementById('autoViewCheckbox');
    const playPauseBtn = document.getElementById('playPauseBtn');
    const loopCheckbox = document.getElementById('loopCheckbox');
    const showDestinationsCheckbox = document.getElementById('showDestinationsCheckbox');
    const playbackSpeedSlider = document.getElementById('playbackSpeedSlider');
    const playbackSpeedValue = document.getElementById('playbackSpeedValue');
    const showHighResMapCheckbox = document.getElementById('showHighResMapCheckbox');
    const highResMapOpacitySlider = document.getElementById('highResMapOpacitySlider');
    const highResMapOpacityValue = document.getElementById('highResMapOpacityValue');
    const highResMapOpacityRow = document.getElementById('highResMapOpacityRow');
    const waymoFormatCheckbox = document.getElementById('waymoFormatCheckbox');
    const uploadFormatHint = document.getElementById('uploadFormatHint');
    const uploadDropZone = document.getElementById('uploadDropZone');
    const uploadFileInput = document.getElementById('uploadFileInput');
    const uploadFileList = document.getElementById('uploadFileList');
    const uploadStatus = document.getElementById('uploadStatus');
    const confirmUploadBtn = document.getElementById('confirmUploadBtn');
    const connectCarlaBtn = document.getElementById('connectCarlaBtn');
    const syncCarlaBtn = document.getElementById('syncCarlaBtn');
    const carlaStatus = document.getElementById('carlaStatus');
    const carlaHostInput = document.getElementById('carlaHostInput');
    const carlaPortInput = document.getElementById('carlaPortInput');
    const carlaModalStatus = document.getElementById('carlaModalStatus');
    const confirmCarlaBtn = document.getElementById('confirmCarlaBtn');
    let selectedUploadFiles = [];

    // Context Menu & Modal Elements
    const contextMenu = document.getElementById('contextMenu');
    const ctxSaveTraj = document.getElementById('ctxSaveTraj');
    const saveTrajModalEl = document.getElementById('saveTrajModal');
    const saveTrajModal = new bootstrap.Modal(saveTrajModalEl);
    const saveModalDatasetName = document.getElementById('saveModalDatasetName');
    const saveRangeMin = document.getElementById('saveRangeMin');
    const saveRangeMax = document.getElementById('saveRangeMax');
    const saveFrameRangeVal = document.getElementById('saveFrameRangeVal');
    const confirmSaveTrajBtn = document.getElementById('confirmSaveTrajBtn');
    let contextMenuTargetName = null; // Target dataset name for the right-click action.

    // WebSocket state.
    let ws = null;
    let wsConnected = false;
    let reconnectAttempts = 0;
    let reconnectTimer = null;
    let pingIntervalId = null;

    // Playback state.
    let playTimer = null;
    let isPlaying = false;
    let playbackSpeed = parseFloat(playbackSpeedValue.textContent);

    // Cached data and runtime state.
    const DATA_CACHE = {};  // { name: { name, fps, map, frames: { frameNumber: { id: {type, x, y}, ... } }, destinations, currentFrame, sliderId } }
    let ACTIVE_NAME = null; // Currently selected name.
    let ARGS_LOADED = null; // Currently loaded model parameters.
    let MODEL_LOADED = null; // Currently loaded model.
    let SIMULATION_RUNNING = false; // Whether a simulation is currently running.

    // Plotly plot instance.
    let myPlot;

    // Destination-dragging state.
    let dragData = {
        isDragging: false,
        targetPedId: null,
        startX: 0,
        startY: 0
    };

    // Trail Control
    trailSlider.addEventListener('input', (e) => {
        trailValue.textContent = e.target.value;
        if (ACTIVE_NAME) render(ACTIVE_NAME);
    });

    // Simulation Duration Control
    simDurationSlider.addEventListener('input', (e) => {
        simDurationValue.textContent = e.target.value;
    });

    playbackSpeedSlider.addEventListener('input', (e) => {
        playbackSpeedValue.textContent = e.target.value;
        playbackSpeed = parseFloat(e.target.value);
    });

    // Show Destinations Control
    showDestinationsCheckbox.addEventListener('change', () => {
        if (ACTIVE_NAME) render(ACTIVE_NAME);
    });

    // High-res Map Control
    showHighResMapCheckbox.addEventListener('change', () => {
        if (showHighResMapCheckbox.checked) {
            highResMapOpacitySlider.value = 1.0;
            highResMapOpacityValue.textContent = '1.0';
        } else {
            highResMapOpacitySlider.value = 0.0;
            highResMapOpacityValue.textContent = '0.0';
        }
        if (ACTIVE_NAME) render(ACTIVE_NAME);
    });

    highResMapOpacitySlider.addEventListener('input', (e) => {
        highResMapOpacityValue.textContent = e.target.value;
        if (ACTIVE_NAME) render(ACTIVE_NAME);
    });

    // Auto View Control: redraw immediately when the state changes so the setting takes effect right away.
    autoViewCheckbox.addEventListener('change', () => {
        if (ACTIVE_NAME) render(ACTIVE_NAME);
    });

    // Log output.
    function log(...args) {
        const t = new Date().toLocaleString();
        const s = args.map(a => (typeof a === 'object' ? JSON.stringify(a) : String(a))).join(' ');
        LOG.textContent += `[${t}] ${s}\n`;
        LOG.scrollTop = LOG.scrollHeight;
        console.debug(...args);
    }

    function setUploadFiles(files) {
        selectedUploadFiles = Array.from(files || []);
        uploadFileList.textContent = selectedUploadFiles.length
            ? selectedUploadFiles.map(file => file.name).join(', ')
            : 'No files selected.';
        confirmUploadBtn.disabled = selectedUploadFiles.length === 0;
        uploadStatus.textContent = '';
    }

    uploadDropZone.addEventListener('click', () => uploadFileInput.click());
    uploadDropZone.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            uploadFileInput.click();
        }
    });
    uploadFileInput.addEventListener('change', () => setUploadFiles(uploadFileInput.files));
    for (const eventName of ['dragenter', 'dragover']) {
        uploadDropZone.addEventListener(eventName, event => {
            event.preventDefault();
            uploadDropZone.classList.add('drag-over');
        });
    }
    for (const eventName of ['dragleave', 'drop']) {
        uploadDropZone.addEventListener(eventName, event => {
            event.preventDefault();
            uploadDropZone.classList.remove('drag-over');
        });
    }
    uploadDropZone.addEventListener('drop', event => setUploadFiles(event.dataTransfer.files));
    waymoFormatCheckbox.addEventListener('change', () => {
        if (waymoFormatCheckbox.checked) {
            uploadFormatHint.textContent = 'Upload one or more Waymo Motion TFRecord files. Every scenario will be converted to data.csv.gz, demo.png, map_range.txt, and map.png.';
            // Waymo shard names often end in "tfrecord-00000-of-01000", so do not apply an extension filter.
            uploadFileInput.accept = '';
        } else {
            uploadFormatHint.textContent = 'Upload a CSV or CSV.GZ file containing the five columns f, x, y, id, type. Optional map.png and map_range.txt files are supported.';
            uploadFileInput.accept = '.csv,.gz,.png,.txt';
        }
    });

    confirmUploadBtn.addEventListener('click', async () => {
        if (!selectedUploadFiles.length) return;
        const body = new FormData();
        selectedUploadFiles.forEach(file => body.append('files', file));
        body.append('is_waymo', waymoFormatCheckbox.checked ? 'true' : 'false');
        confirmUploadBtn.disabled = true;
        uploadStatus.className = 'small mt-2 text-primary';
        uploadStatus.textContent = waymoFormatCheckbox.checked ? 'Uploading and processing Waymo data…' : 'Uploading…';
        try {
            const response = await fetch('/api/upload_dataset', {method: 'POST', body});
            const message = await response.json();
            if (!response.ok || message.status !== 'ok') throw new Error(message.msg || 'Upload failed.');
            for (const dataset of message.datasets || []) {
                const option = document.createElement('option');
                option.value = dataset.index;
                option.textContent = `[Upload] ${dataset.name}`;
                datasetSelect.appendChild(option);
                datasetSelect.value = String(dataset.index);
            }
            uploadStatus.className = 'small mt-2 text-success';
            uploadStatus.textContent = message.msg;
            log('Server:', message.msg);
        } catch (error) {
            uploadStatus.className = 'small mt-2 text-danger';
            uploadStatus.textContent = error.message;
            log('Dataset upload failed:', error.message);
        } finally {
            confirmUploadBtn.disabled = false;
        }
    });

    function updateCarlaControls(status) {
        const connected = Boolean(status.connected);
        const syncing = connected && Boolean(status.sync_enabled);
        syncCarlaBtn.disabled = !connected;
        syncCarlaBtn.classList.toggle('active', syncing);
        syncCarlaBtn.setAttribute('aria-pressed', syncing ? 'true' : 'false');
        carlaStatus.textContent = connected ? '已连接到 CARLA' : '未找到 CARLA';
        carlaStatus.className = `small mb-2 ${connected ? 'text-success' : 'text-muted'}`;
        connectCarlaBtn.classList.toggle('btn-outline-secondary', !connected);
        connectCarlaBtn.classList.toggle('btn-outline-success', connected);
    }

    confirmCarlaBtn.addEventListener('click', async () => {
        const host = carlaHostInput.value.trim();
        const port = Number(carlaPortInput.value);
        if (!host || !Number.isInteger(port) || port < 1 || port > 65535) {
            carlaModalStatus.className = 'small mt-3 text-danger';
            carlaModalStatus.textContent = 'Please enter a valid IP address and port.';
            return;
        }
        confirmCarlaBtn.disabled = true;
        carlaModalStatus.className = 'small mt-3 text-primary';
        carlaModalStatus.textContent = 'Connecting…';
        try {
            const response = await fetch('/api/carla/connect', {
                method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({host, port})
            });
            const message = await response.json();
            updateCarlaControls(message);
            carlaModalStatus.className = `small mt-3 ${message.status === 'ok' ? 'text-success' : 'text-danger'}`;
            carlaModalStatus.textContent = message.status === 'ok' ? '已连接到 CARLA' : '未找到 CARLA';
            log(carlaModalStatus.textContent, `${host}:${port}`);
        } catch (error) {
            updateCarlaControls({connected: false, sync_enabled: false});
            carlaModalStatus.className = 'small mt-3 text-danger';
            carlaModalStatus.textContent = '未找到 CARLA';
        } finally {
            confirmCarlaBtn.disabled = false;
        }
    });

    syncCarlaBtn.addEventListener('click', async () => {
        const enabled = !syncCarlaBtn.classList.contains('active');
        try {
            const response = await fetch('/api/carla/sync', {
                method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({enabled})
            });
            const message = await response.json();
            updateCarlaControls(message);
            if (!response.ok) throw new Error(message.msg || 'Unable to change CARLA synchronization.');
            log(`CARLA synchronization ${message.sync_enabled ? 'enabled' : 'disabled'}.`);
        } catch (error) {
            log('CARLA synchronization failed:', error.message);
        }
    });

    fetch('/api/carla/status').then(response => response.json()).then(updateCarlaControls).catch(() => {});

    // WebSocket connection and message handling.
    function connectWebsocket() {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const url = `${proto}//${location.host}/ws`;
        log('Attempting to connect WebSocket:', url);
        ws = new WebSocket(url);
        ws.onopen = () => {
            wsConnected = true;
            reconnectAttempts = 0;
            log('WebSocket connected');
            startPing();
            // updateButtonsState();
        };
        ws.onclose = (ev) => {
            wsConnected = false;
            log('WebSocket disconnected', ev.code, ev.reason || '');
            // updateButtonsState();
            scheduleReconnect();
            stopPing();
        };
        ws.onerror = (err) => {
            log('WebSocket error', err && err.message ? err.message : err);
            // updateButtonsState();
            scheduleReconnect();
            stopPing();
        };
        ws.onmessage = (ev) => {
            try {
                const msg = JSON.parse(ev.data); // The backend usually sends `{status: 'ok'|'error', data: response, msg: '...'}`.
                if (msg.msg) { log('Server:', msg.msg); }
                if (msg.data) { mergeAndHandleResponse(msg.data); }
                if (msg.carla) { updateCarlaControls(msg.carla); }
            } catch (e) {
                log('Received an unparseable WebSocket message:', ev.data);
            }
        };
    }

    // Automatic reconnect mechanism.
    function scheduleReconnect() {
        if (reconnectTimer) return;
        reconnectAttempts += 1;
        const delay = Math.min(30000, 1000 * Math.pow(1.6, Math.min(reconnectAttempts, 10))); // Exponential backoff, capped at 30s.
        log(`WebSocket will retry in ${Math.round(delay / 1000)}s (attempt ${reconnectAttempts})`);
        reconnectTimer = setTimeout(() => {
            reconnectTimer = null;
            connectWebsocket();
        }, delay);
    }

    // Simple heartbeat: send `ping` to the server and expect `pong`.
    function startPing() {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        if (pingIntervalId) clearInterval(pingIntervalId);
        pingIntervalId = setInterval(() => {
            try {
                ws.send(JSON.stringify({ action: 'ping' }));
            } catch (e) {
                // ignore
            }
        }, 15000);
    }
    function stopPing() {
        if (pingIntervalId) {
            clearInterval(pingIntervalId);
            pingIntervalId = null;
        }
    }

    // Merge the backend response into the cache and create/update sliders and the map.
    function mergeAndHandleResponse(response) {
        const name = response.name;
        // New cache entry.
        if (!DATA_CACHE[name]) {
            DATA_CACHE[name] = {
                name: response.name,
                map: null,
                frames: {},
                destinations: {},  // Store destination data.
                currentFrame: null,
                sliderId: null,
                fps: response.fps || 10,
                has_high_res_map: false,  // Whether a high-resolution source image is available.
            };
        }
        // Update map.
        if (response.map) {
            DATA_CACHE[name].map = response.map;
        }
        if (response.fps) {
            DATA_CACHE[name].fps = response.fps;
        }
        // Update `high_res_map_path`.
        if (response.has_high_res_map !== undefined) {
            DATA_CACHE[name].has_high_res_map = response.has_high_res_map;
        }
        // Update destinations if provided by the backend.
        if (response.destinations) {
            DATA_CACHE[name].destinations = response.destinations;
        }
        // Update `frames` and `currentFrame`.
        const frames = response.frames || {};
        for (const fkey of Object.keys(frames)) {
            const fnum = Number(fkey);
            DATA_CACHE[name].frames[fnum] = frames[fkey];
            DATA_CACHE[name].currentFrame = fnum; // Update the current frame to the latest frame received.
        }
        // Cache the high-resolution map URL.
        const item = DATA_CACHE[name];
        if (item.has_high_res_map) {
            item.high_res_map_url = `/api/get_high_res_map?dataset_name=${encodeURIComponent(name)}`;
        }
        // Create / update the slider.
        if (!DATA_CACHE[name].sliderId) { // Create the slider if it does not exist yet.
            createSliderForResponse(name);
            // Check whether the original image exists and enable the checkbox accordingly.
            updateHighResMapCheckbox(name);
        } else { // Update the slider max if needed and set it to the latest currentFrame.
            updateSliderRangeAndValue(name);
        }
        // Re-render the map.
        render(name);
    }

    // Update high-res map checkbox state based on current dataset
    function updateHighResMapCheckbox(name) {
        const item = DATA_CACHE[name];
        if (item && item.has_high_res_map) {
            showHighResMapCheckbox.disabled = false;
            showHighResMapCheckbox.title = 'High-resolution original image available';
            highResMapOpacityRow.style.display = 'flex';
            highResMapOpacityRow.style.alignItems = 'center';
        } else {
            showHighResMapCheckbox.disabled = true;
            showHighResMapCheckbox.checked = false;
            showHighResMapCheckbox.title = 'This dataset has no original image';
            highResMapOpacityRow.style.display = 'none';
        }
    }

    // UI: create sliders and manage slider events.
    function createSliderForResponse(name) {
        const item = DATA_CACHE[name];
        const frameKeys = Object.keys(item.frames).map(k => Number(k)).sort((a, b) => a - b);
        const min = frameKeys.length ? frameKeys[0] : 0;
        const max = frameKeys.length ? frameKeys[frameKeys.length - 1] : min;
        const cur = item.currentFrame != null ? item.currentFrame : min;

        // DOM
        const wrap = document.createElement('div');
        wrap.className = 'slider-wrap';
        wrap.dataset.name = name; // Bind the dataset name for the context menu.
        wrap.draggable = false;  // Do not allow dragging on the wrapper itself.

        const header = document.createElement('div');
        header.style.display = 'flex';
        header.style.justifyContent = 'space-between';
        header.style.alignItems = 'center';

        const label = document.createElement('div');
        label.className = 'slider-label';
        label.textContent = name;
        label.draggable = true; // Allow dragging from the label.

        const closeBtn = document.createElement('button');
        closeBtn.textContent = '✖';
        closeBtn.className = 'btn btn-sm btn-outline-danger';
        closeBtn.style.padding = '0 6px';
        closeBtn.style.lineHeight = '1';
        closeBtn.title = 'Remove this slider';

        header.appendChild(label);
        header.appendChild(closeBtn);
        wrap.appendChild(header);

        const slider = document.createElement('input');
        slider.type = 'range';
        slider.min = min;
        slider.max = max;
        slider.value = cur;
        slider.step = 1;
        slider.style.width = '100%';
        slider.id = `slider-${name.replace(/\s+/g, '_')}-${Date.now()}`;
        item.sliderId = slider.id;

        const valueSpan = document.createElement('span');
        valueSpan.textContent = `Current frame: ${cur} (${min}~${max})`;
        valueSpan.style.marginLeft = '8px';

        wrap.appendChild(slider);
        wrap.appendChild(valueSpan);
        slidersDiv.appendChild(wrap);

        // === Remove button event ===
        closeBtn.addEventListener('click', () => {
            wrap.remove();
            delete DATA_CACHE[name]; // Optional.
            if (ACTIVE_NAME === name) ACTIVE_NAME = null;
        });

        // === Drag events ===
        label.addEventListener('dragstart', (ev) => {
            ev.dataTransfer.setData('text/plain', name);
            wrap.classList.add('dragging');
        });
        label.addEventListener('dragend', () => wrap.classList.remove('dragging'));
        // slider.addEventListener('mousedown', (ev) => ev.stopPropagation());
        // slider.addEventListener('touchstart', (ev) => ev.stopPropagation());

        // === Context menu event ===
        wrap.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            contextMenuTargetName = name;
            // Simple menu positioning.
            contextMenu.style.display = 'block';
            contextMenu.style.left = e.pageX + 'px';
            contextMenu.style.top = e.pageY + 'px';
        });

        // === Slider event ===
        slider.addEventListener('input', (ev) => {
            const v = Number(ev.target.value);
            item.currentFrame = v;
            valueSpan.textContent = `Current frame: ${v} (${min}~${max})`;
            render(name);
        });
        slider.addEventListener('mousedown', () => render(name));
        slider.addEventListener('touchstart', () => render(name));
    }

    // Close the context menu on global click.
    document.addEventListener('click', () => {
        contextMenu.style.display = 'none';
    });

    // Context menu item click.
    ctxSaveTraj.addEventListener('click', () => {
        if (contextMenuTargetName && DATA_CACHE[contextMenuTargetName]) {
            openSaveModal(contextMenuTargetName);
        }
    });

    // === Open save modal logic ===
    function openSaveModal(name) {
        saveModalDatasetName.textContent = name;
        const item = DATA_CACHE[name];
        const frameKeys = Object.keys(item.frames).map(k => Number(k)).sort((a, b) => a - b);
        const min = frameKeys.length ? frameKeys[0] : 0;
        const max = frameKeys.length ? frameKeys[frameKeys.length - 1] : min;
        
        // Set range slider attributes.
        saveRangeMin.min = min; saveRangeMin.max = max;
        saveRangeMax.min = min; saveRangeMax.max = max;
        saveRangeMin.value = min;
        saveRangeMax.value = max;
        
        updateDualSliderUI(min, max);
        
        // Activate the current dataset view for preview.
        render(name);
        
        saveTrajModal.show();
    }

    // Dual-handle slider logic.
    function updateDualSliderUI(min, max) {
        let vMin = parseInt(saveRangeMin.value);
        let vMax = parseInt(saveRangeMax.value);
        
        // Prevent the two handles from crossing.
        if (vMin > vMax) {
             // Simple mutual exclusion logic: adjust whichever handle moved.
             // It is cleaner to handle this in the input event.
        }
        
        saveFrameRangeVal.textContent = `${vMin} - ${vMax}`;
    }

    // Listen for dual-handle slider changes.
    function handleDualSliderInput(e) {
        const item = DATA_CACHE[contextMenuTargetName];
        let vMin = parseInt(saveRangeMin.value);
        let vMax = parseInt(saveRangeMax.value);

        if (vMin > vMax) {
            if (e.target === saveRangeMin) {
                saveRangeMin.value = vMax;
                vMin = vMax;
            } else {
                saveRangeMax.value = vMin;
                vMax = vMin;
            }
        }
        
        saveFrameRangeVal.textContent = `${vMin} - ${vMax}`;
        
        // Update the main view in real time.
        // If the min handle moved, show the min frame; if the max handle moved, show the max frame.
        if (item) {
            item.currentFrame = (e.target === saveRangeMin) ? vMin : vMax;
            // Update the main slider UI for this dataset to keep the state consistent,
            // even though it is not visible inside the modal.
            const mainSlider = document.getElementById(item.sliderId);
            if(mainSlider) mainSlider.value = item.currentFrame;
            render(contextMenuTargetName);
        }
    }

    saveRangeMin.addEventListener('input', handleDualSliderInput);
    saveRangeMax.addEventListener('input', handleDualSliderInput);

    // Confirm save.
    confirmSaveTrajBtn.addEventListener('click', async () => {
        const name = saveModalDatasetName.textContent;
        const start = parseInt(saveRangeMin.value);
        const end = parseInt(saveRangeMax.value);
        const dest = document.querySelector('input[name="saveDest"]:checked').value;
        const compress = document.getElementById('saveCompress').checked;

        const payload = {
            name: name,
            start_frame: start,
            end_frame: end,
            destination: dest,
            compress: compress
        };

        // Close the modal.
        saveTrajModal.hide();
        
        log(`Requesting trajectory save: ${name} [${start}-${end}] -> ${dest}`);

        try {
            const res = await fetch('/api/save_trajectory', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            if (dest === 'local') {
                if (res.ok) {
                    const blob = await res.blob();
                    // Get the filename from Content-Disposition.
                    const disposition = res.headers.get('Content-Disposition');
                    let filename = `trajectory.csv${compress ? '.tz' : ''}`;
                    if (disposition && disposition.indexOf('attachment') !== -1) {
                        const matches = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/.exec(disposition);
                        if (matches != null && matches[1]) { 
                            filename = matches[1].replace(/['"]/g, '');
                        }
                    }
                    // Trigger the download.
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = filename;
                    document.body.appendChild(a);
                    a.click();
                    a.remove();
                    window.URL.revokeObjectURL(url);
                    log('Download started.');
                } else {
                    const err = await res.json();
                    alert('Download failed: ' + (err.msg || 'Unknown error'));
                }
            } else {
                const msg = await res.json();
                if (msg.status === 'ok') {
                    log('Save succeeded:', msg.msg);
                    alert('Save succeeded: ' + msg.msg);
                } else {
                    alert('Save failed: ' + msg.msg);
                }
            }
        } catch (e) {
            console.error(e);
            alert('Failed to send request');
        }
    });


    // === Global: enable drag-and-drop sorting for slidersDiv ===
    slidersDiv.addEventListener('dragover', (ev) => {
        ev.preventDefault();
        const dragging = document.querySelector('.dragging');
        const afterElement = getDragAfterElement(slidersDiv, ev.clientY);
        if (afterElement == null) {
            slidersDiv.appendChild(dragging);
        } else {
            slidersDiv.insertBefore(dragging, afterElement);
        }
    });

    function getDragAfterElement(container, y) {
        const draggableElements = [...container.querySelectorAll('.slider-wrap:not(.dragging)')];
        return draggableElements.reduce((closest, child) => {
            const box = child.getBoundingClientRect();
            const offset = y - box.top - box.height / 2;
            if (offset < 0 && offset > closest.offset) {
                return { offset: offset, element: child };
            } else {
                return closest;
            }
        }, { offset: Number.NEGATIVE_INFINITY }).element;
    }

    function updateSliderRangeAndValue(name) {
        const item = DATA_CACHE[name];
        const slider = document.getElementById(item.sliderId);
        if (!slider) return;
        const frameKeys = Object.keys(item.frames).map(k => Number(k)).sort((a, b) => a - b);
        const min = frameKeys.length ? frameKeys[0] : 0;
        const max = frameKeys.length ? frameKeys[frameKeys.length - 1] : min;
        const cur = item.currentFrame != null ? item.currentFrame : min;
        slider.min = min;
        slider.max = max;
        slider.value = cur;
        // update label text (parent .slider-wrap first child)
        const wrap = slider.parentElement;
        const label = wrap.querySelector('.slider-label');
        if (label) label.textContent = `${name}`;
        const span = wrap.querySelector('span');
        if (span) span.textContent = `Current frame: ${cur} (${min}~${max})`;
    }

    // UI: highlight the currently active slider.
    function highlightSlider() {
        const item = DATA_CACHE[ACTIVE_NAME];
        if(!item) return;
        const sliderWraps = slidersDiv.querySelectorAll('.slider-wrap');
        sliderWraps.forEach(wrap => {
            if (wrap.contains(document.getElementById(item.sliderId))) {
                // Highlight the current slider.
                wrap.style.border = '2px solid #007bff';
                wrap.style.backgroundColor = '#e7f1ff';
            } else {
                // Remove highlight.
                wrap.style.border = '1px solid #dee2e6';
                wrap.style.backgroundColor = '#ffffff';
            }
        });
    }

    // Render function: render the map and trajectories based on ACTIVE_NAME.
    function render(name) {
        if (ACTIVE_NAME !== name) {
            ACTIVE_NAME = name;
            log(`Switched to dataset: ${name}`);
            highlightSlider();
        }
        renderTrace();
    }

    // Render entity trajectories (Plotly).
    function renderTrace() {
        if (!ACTIVE_NAME || !DATA_CACHE[ACTIVE_NAME]) {
            Plotly.react(mapDiv, [], { title: "No data available" });
            return;
        }
        const plotData = [];
        const item = DATA_CACHE[ACTIVE_NAME];
        const currentFrame = Number(item.currentFrame);
        const fps = item.fps || 10;
        const trailSec = Number(trailSlider.value);
        const trailFrames = Math.round(trailSec * fps);
        // if (!entities) {
        //     Plotly.react(mapDiv, [], { title: `${ACTIVE_NAME} - No data for the current frame` });
        //     return;
        // }

        // 1. Map Layer (Black/White, Sharp)
        // 0=White (Road), 1=Black (Obstacle)
        let layout = {
            margin: { t: 30, b: 30, l: 30, r: 30 },
            xaxis: { title: 'x', scaleratio: 1, showgrid: false },
            yaxis: { title: 'y', scaleanchor: "x", showgrid: false },
            plot_bgcolor: '#ffffff',
            hovermode: 'closest',
        };
        const mapInfo = item.map;
        if (mapInfo && mapInfo.grid) {
            // transpose grid so rows become columns
            let z;
            const g = mapInfo.grid;
            if (!Array.isArray(g) || g.length === 0 || !Array.isArray(g[0])) {
                z = g;
            } else {
                const rows = g.length;
                const cols = Math.max(...g.map(r => Array.isArray(r) ? r.length : 0));
                z = Array.from({ length: cols }, (_, c) =>
                    Array.from({ length: rows }, (_, r) => {
                        const row = g[r];
                        return Array.isArray(row) ? row[c] : undefined;
                    })
                );
            }
            const xmin = mapInfo.xmin, xmax = mapInfo.xmax;
            const ymin = mapInfo.ymin, ymax = mapInfo.ymax;
            const ny = z.length, nx = Array.isArray(z[0]) ? z[0].length : 0;
            const xcoords = [], ycoords = [];
            for (let i = 0; i < nx; i++) xcoords.push(xmin + (i + 0.5) * (xmax - xmin) / nx);
            for (let j = 0; j < ny; j++) ycoords.push(ymin + (j + 0.5) * (ymax - ymin) / ny);
            plotData.push({
                z: z,
                type: 'heatmap',
                colorscale: 'Greys',
                reversescale: true, // 0(Low)=White, 1(High)=Black
                showscale: false,
                zsmooth: false, // Sharp pixels
                x: xcoords,
                y: ycoords,
                hoverinfo: 'none',
                opacity: 1.0 - parseFloat(highResMapOpacitySlider.value),
            });
        }
        
        const currentEntities = item.frames[currentFrame] || [];

        // 1. Find all active entities in the current frame and maintain a trace for each one.
        const activeTraces = {};
        for (const e in currentEntities) {
            activeTraces[e] = { x: [], y: [] };
        }

        // 2. Enumerate backward from the current frame over the specified trail length.
        const startF = Math.max(0, currentFrame - trailFrames);
        for (let f = currentFrame; f >= startF; f--) {
            const frameData = item.frames[f];
            if (!frameData) continue;
            // 3. For each selected historical frame, check whether active entities appear in that frame.
            for (const e in frameData) {
                if (activeTraces[e]) {
                    activeTraces[e].x.push(frameData[e].x);
                    activeTraces[e].y.push(frameData[e].y);
                }
            }
        }

        // 4. After enumeration, add each entity trace to the plot.
        for (const e in currentEntities) {
            const trace = activeTraces[e];
            if (trace.x.length > 1) {
                const color = (currentEntities[e].type === 'vehicle') ? 'rgba(200, 80, 0, 0.4)' : 'rgba(0, 100, 255, 0.4)';
                plotData.push({
                    x: trace.x,
                    y: trace.y,
                    mode: 'lines',
                    line: { color, width: 2 },
                    name: `${currentEntities[e].type} ID: ${e}`,
                    showlegend: false,
                });
            }
        }

        // 3. Entity Layer (Vehicles as Rects, Pedestrians as Dots)
        const vehX = [], vehY = []; // Vehicle Polygons
        const pedX = [], pedY = [], pedText = [], pedIds = []; // Pedestrians
        const vehMarkersX = [], vehMarkersY = [], vehText = [], vehIds = []; // Vehicle Centers (for ID)

        for (const id in currentEntities) {
            const e = currentEntities[id];
            if (e.type === 'vehicle' && e.length && e.width && e.heading != null) {
                // Calculate rectangle corners
                // Heading: angle in radians. 
                const cosT = Math.cos(e.heading);
                const sinT = Math.sin(e.heading);
                const l2 = e.length / 2;
                const w2 = e.width / 2;
                
                // Corners relative to center (unrotated): (l, w), (l, -w), (-l, -w), (-l, w)
                // Rotated: x' = x cos - y sin, y' = x sin + y cos
                const corners = [
                    { x: l2, y: w2 },
                    { x: l2, y: -w2 },
                    { x: -l2, y: -w2 },
                    { x: -l2, y: w2 },
                    { x: l2, y: w2 } // Close loop
                ];
                
                corners.forEach(c => {
                    const rx = e.x + (c.x * cosT - c.y * sinT);
                    const ry = e.y + (c.x * sinT + c.y * cosT);
                    vehX.push(rx);
                    vehY.push(ry);
                });
                vehX.push(null);
                vehY.push(null);

                // Add center for hover ID
                vehMarkersX.push(e.x);
                vehMarkersY.push(e.y);
                vehIds.push(id);
                vehText.push(`ID: ${id}<br>Veh`);

            } else if (e.type === 'vehicle') {
                // Fallback for vehicle without dims -> Dot
                vehMarkersX.push(e.x);
                vehMarkersY.push(e.y);
                vehIds.push(id);
                vehText.push(`ID: ${id}<br>Veh (No Dim)`);
            } else {
                // Pedestrian without dims -> Dot
                pedX.push(e.x);
                pedY.push(e.y);
                pedIds.push(id);
                pedText.push(`ID: ${id}<br>Ped`);
            }
        }

        // Vehicle Polygons
        if (vehX.length > 0) {
            plotData.push({
                x: vehX,
                y: vehY,
                ids: vehIds,
                mode: 'lines',
                fill: 'toself',
                fillcolor: 'rgba(255, 100, 0, 0.5)',
                line: { color: 'rgb(200, 80, 0)', width: 1 },
                hoverinfo: 'none',
                name: 'Vehicle',
                showlegend: true,
            });
        }

        // Vehicle Markers (Centers)
        if (vehMarkersX.length > 0) {
            plotData.push({
                x: vehMarkersX,
                y: vehMarkersY,
                ids: vehIds,
                mode: 'markers',
                marker: { size: 6, color: 'rgb(200, 80, 0)', symbol: 'square' },
                text: vehText,
                hoverinfo: 'text',
                name: 'Vehicle',
                showlegend: true,
            });
        }

        // Compute pedestrian marker size in pixels at the current zoom level (fixed 0.3m diameter).
        const PEDESTRIAN_DIAMETER_METERS = 0.3;
        let pedMarkerSize = 6;  // Default minimum pixel size.
        if (myPlot) {
            const xaxis = myPlot._fullLayout?.xaxis;
            const yaxis = myPlot._fullLayout?.yaxis;
            const gs = myPlot._fullLayout?._size;
            if (xaxis?.range && yaxis?.range && gs) {
                const xSpan = Math.abs(xaxis.range[1] - xaxis.range[0]);
                const plotWidth = gs.w || 1;
                const metersPerPixel = xSpan / plotWidth;
                pedMarkerSize = Math.max(4, PEDESTRIAN_DIAMETER_METERS / metersPerPixel);
            }
        }

        // Pedestrian dots (fixed 0.3m diameter, changes with zoom).
        if (pedX.length > 0) {
            plotData.push({
                x: pedX,
                y: pedY,
                ids: pedIds,
                mode: 'markers',
                marker: {
                    size: pedMarkerSize,
                    sizemode: 'diameter',
                    color: 'rgb(0, 100, 255)'
                },
                text: pedText,
                hoverinfo: 'text',
                name: 'Pedestrian',
                showlegend: true,
            });
        }

        // 4. Destinations and Connection Lines (if enabled)
        if (showDestinationsCheckbox.checked && item.destinations) {
            const desX = [], desY = [], desText = [];
            const lineX = [], lineY = [], lineIds = [];

            for (const pedId in currentEntities) {
                if (currentEntities[pedId].type === 'pedestrian' && pedId in item.destinations) {
                    const ped = currentEntities[pedId];
                    const des = item.destinations[pedId];

                    // Add connection line (gray dashed)
                    lineX.push(ped.x, des.x, null);  // null to separate lines
                    lineY.push(ped.y, des.y, null);
                    lineIds.push(pedId);

                    // Add destination marker (red cross)
                    desX.push(des.x);
                    desY.push(des.y);
                    desText.push(`Destination ID: ${pedId}`);
                }
            }

            // Add connection lines
            if (lineX.length > 0) {
                plotData.push({
                    x: lineX,
                    y: lineY,
                    ids: lineIds,
                    mode: 'lines',
                    line: {
                        color: 'rgba(128, 128, 128, 0.6)',
                        width: 1.5,
                        dash: 'solid'  // Gray solid line.
                    },
                    hoverinfo: 'none',
                    name: 'To Destination',
                    showlegend: false,
                });
            }

            // Add destination markers (red crosses)
            if (desX.length > 0) {
                plotData.push({
                    x: desX,
                    y: desY,
                    ids: Object.keys(item.destinations),  // Use destination IDs.
                    mode: 'markers',
                    marker: {
                        size: 12,
                        color: 'rgb(255, 0, 0)',
                        symbol: 'x'  // Red cross marker.
                    },
                    text: desText,
                    hoverinfo: 'text',
                    name: 'Destination',
                    showlegend: true,
                });
            }
        }

        if (autoViewCheckbox.checked) {
            layout.uirevision = undefined;
            layout.xaxis.range = undefined;
            layout.yaxis.range = undefined;
        } else {
            layout.uirevision = 'constant';
            layout.xaxis.range = myPlot ? myPlot.layout.xaxis.range : undefined;
            layout.yaxis.range = myPlot ? myPlot.layout.yaxis.range : undefined;
        }

        // Add high-resolution map image overlay if enabled
        if (showHighResMapCheckbox.checked && !showHighResMapCheckbox.disabled && item.has_high_res_map && item.high_res_map_url) {
            const opacity = parseFloat(highResMapOpacitySlider.value);
            const mapInfo = item.map;
            layout.images = [{
                source: item.high_res_map_url,
                xref: 'x',
                yref: 'y',
                x: mapInfo.xmin,          // Left boundary of the image.
                y: mapInfo.ymax,          // Upper boundary of the image (with yanchor: 'top').
                sizex: mapInfo.xmax - mapInfo.xmin,
                sizey: mapInfo.ymax - mapInfo.ymin,
                xanchor: 'left',          // Anchor at the left edge.
                yanchor: 'top',           // Anchor at the top edge, so the y coordinate matches the top of the image.
                sizing: 'stretch',
                opacity: opacity,
                layer: 'below'            // Display below the trajectories.
            }];
        } else {
            layout.images = [];  // Clear the image when unchecked.
        }
        
        // const smooth = true;
        // if (smooth) {
        //     layout.transition = {
        //         duration: 1000 / fps / playbackSpeed, // Animation duration equals the frame interval, e.g. 2.5fps -> 400ms
        //         easing: 'linear'      // Linear motion, simulating constant velocity
        //     };
        // } else {
        //     layout.transition = { duration: 0 }; // Respond immediately during manual dragging.
        // }

        if (myPlot) {
            Plotly.react(myPlot, plotData, layout);
            // Always disable Plotly drag-zoom and use custom mouse events instead.
            Plotly.relayout(myPlot, { 'dragmode': false });
        } else {
            Plotly.newPlot(mapDiv, plotData, layout, {
                responsive: true,
                scrollZoom: true,
                dragmode: false  // Disable drag-zoom.
            })
            .then((plotElement) => {
                myPlot = plotElement;
                // Listen for zoom events and re-render to update pedestrian marker size.
                myPlot.on('plotly_relayout', (eventData) => {
                    // Re-render only when xaxis.range or yaxis.range changes, indicating zoom or pan.
                    if (eventData['xaxis.range'] || eventData['yaxis.range'] ||
                        eventData['xaxis.range[0]'] || eventData['xaxis.range[1]'] ||
                        eventData['yaxis.range[0]'] || eventData['yaxis.range[1]']) {
                        renderTrace();
                    }
                });
            });
        }
    }

    // Fetch dataset_list / model_list and populate the dropdowns.
    async function loadLists() {
        try {
            const [dsr, mr] = await Promise.all([
                fetch('/api/dataset_list').then(r => r.json()).catch(e => ({})),
                fetch('/api/model_list').then(r => r.json()).catch(e => ({}))
            ]);
            // dataset_list returns an object mapping index -> full_name.
            datasetSelect.innerHTML = '';
            for (const k of Object.keys(dsr)) {
                const opt = document.createElement('option');
                opt.value = k;
                opt.textContent = dsr[k];
                datasetSelect.appendChild(opt);
            }
            modelSelect.innerHTML = '';
            for (const k of Object.keys(mr)) {
                const opt = document.createElement('option');
                opt.value = k;
                opt.textContent = mr[k];
                modelSelect.appendChild(opt);
            }
            log('Loaded dataset_list and model_list');
        } catch (e) {
            log('Failed to load dataset/model lists:', e);
        }
    }

    // KEEP_ARGS ordering fetched from the backend.
    let KEEP_ARGS_ORDER = null;

    // Load dataset.
    loadDatasetBtn.addEventListener('click', async () => {
        const idx = datasetSelect.value;
        if (idx == null) { alert('Please select a dataset first'); return; }
        if (ARGS_LOADED == null) { alert('Please load a model first'); return; }
        try {
            // Set name to a unique identifier; the backend uses it as the key in DATASET_DICT[name].
            const name = datasetSelect.options[datasetSelect.selectedIndex].text || 'dataset';
            log(`Loading dataset ${name} (idx=${idx}) ...`);
            const res = await fetch(`/api/load_dataset?idx=${encodeURIComponent(idx)}&name=${encodeURIComponent(name)}`)
            const msg = await res.json();
            if (msg.status === 'ok') {
                if (msg.response) {
                    log('Server:', msg.msg || 'Dataset loaded.');
                    mergeAndHandleResponse(msg.response);
                } else {
                    log('Warning: load_dataset response does not include the response field. Please check the backend.', msg);
                }
            } else {
                log('Failed to load dataset:', msg.msg || JSON.stringify(msg));
            }
        } catch (e) {
            log('Dataset load request failed:', e);
        }
    });

    // Load model.
    loadModelBtn.addEventListener('click', async () => {
        const idx = modelSelect.value;
        if (idx == null) { log('Please select a model first'); return; }
        try {
            const name = modelSelect.options[modelSelect.selectedIndex].text || 'model';
            log(`Loading model weights ${name} (idx=${idx}) ...`);
            const res = await fetch(`/api/load_model?idx=${encodeURIComponent(idx)}`);
            const msg = await res.json();
            if (msg.status === 'ok') {
                log('Server:', msg.msg || 'Model loaded.');
                MODEL_LOADED = name;
                ARGS_LOADED = msg.response;
                KEEP_ARGS_ORDER = msg.keep_args_order || null;  // Save the parameter order.
                log('Current model parameters:', ARGS_LOADED);
                editParamsBtn.classList.remove('d-none');
                renderParamsEditor(ARGS_LOADED);
            } else {
                log('Failed to load model:', msg.msg || msg);
            }
        } catch (e) {
            log('Model load request failed:', e);
        }
    });

    // Render the parameter list in KEEP_ARGS_ORDER order.
    function renderParamsEditor(args) {
        paramsList.innerHTML = '';

        // Use the order defined by KEEP_ARGS_ORDER, and place missing keys at the end.
        const orderedKeys = [];
        const remainingKeys = [];

        if (KEEP_ARGS_ORDER && Array.isArray(KEEP_ARGS_ORDER)) {
            // Collect existing keys in KEEP_ARGS_ORDER order.
            for (const key of KEEP_ARGS_ORDER) {
                if (key in args) {
                    orderedKeys.push(key);
                }
            }
            // Collect the remaining keys that are not in KEEP_ARGS_ORDER.
            for (const key of Object.keys(args)) {
                if (!orderedKeys.includes(key)) {
                    remainingKeys.push(key);
                }
            }
            remainingKeys.sort();  // Sort the remaining keys alphabetically.
        } else {
            // Fall back to alphabetical order when no ordering info is available.
            Object.keys(args).sort().forEach(k => orderedKeys.push(k));
        }

        const allKeys = [...orderedKeys, ...remainingKeys];

        allKeys.forEach(key => {
            const val = args[key];
            // Skip complex objects and only allow editing primitive types.
            if (val !== null && typeof val === 'object') return;
            const row = document.createElement('div');
            row.className = 'mb-2 row g-1 align-items-center';

            const labelCol = document.createElement('div');
            labelCol.className = 'col-5 text-break';
            labelCol.textContent = key;
            labelCol.title = key; // Show the full key on hover.
            
            const inputCol = document.createElement('div');
            inputCol.className = 'col-7';
            
            const input = document.createElement('input');
            input.className = 'form-control form-control-sm param-input';
            input.dataset.key = key;
            input.dataset.original = val; // Store the original value.
            input.value = val;
            
            // Configure input attributes based on type.
            if (typeof val === 'number') {
                input.type = 'number';
                input.step = 'any'; // Allow decimals.
            } else if (typeof val === 'boolean') {
                // For booleans, this could be a dropdown or checkbox.
                // Use a select here for a slightly better UI.
                const select = document.createElement('select');
                select.className = 'form-select form-select-sm param-input';
                select.dataset.key = key;
                select.dataset.original = val;
                
                const optTrue = document.createElement('option');
                optTrue.value = 'true'; optTrue.text = 'True';
                const optFalse = document.createElement('option');
                optFalse.value = 'false'; optFalse.text = 'False';
                
                select.appendChild(optTrue);
                select.appendChild(optFalse);
                select.value = val.toString();
                
                // Replace the input with a select.
                inputCol.appendChild(select);
                
                // Select event.
                select.addEventListener('change', (e) => {
                    const currentVal = (e.target.value === 'true');
                    const originalVal = (e.target.dataset.original === 'true');
                    if (currentVal !== originalVal) {
                        e.target.classList.add('text-danger', 'fw-bold');
                        e.target.style.borderColor = '#dc3545';
                    } else {
                        e.target.classList.remove('text-danger', 'fw-bold');
                        e.target.style.borderColor = '';
                    }
                });
                
                row.appendChild(labelCol);
                row.appendChild(inputCol);
                paramsList.appendChild(row);
                return; // End the current loop iteration.
            } else {
                input.type = 'text';
            }
            
            // Input event: detect modifications and highlight them in red.
            input.addEventListener('input', (e) => {
                const currentVal = e.target.value;
                const originalVal = String(e.target.dataset.original);
                
                // Simple string comparison.
                if (currentVal !== originalVal) {
                    e.target.classList.add('text-danger', 'fw-bold'); // Bootstrap red + bold.
                    e.target.style.borderColor = '#dc3545'; // Make the border red as well.
                } else {
                    e.target.classList.remove('text-danger', 'fw-bold');
                    e.target.style.borderColor = '';
                }
            });

            inputCol.appendChild(input);
            row.appendChild(labelCol);
            row.appendChild(inputCol);
            paramsList.appendChild(row);
        });
    }

    saveParamsBtn.addEventListener('click', async () => {
        const inputs = document.querySelectorAll('.param-input');
        const newArgs = {};
        let hasChanges = false;
        
        inputs.forEach(el => {
            const key = el.dataset.key;
            let val = el.value;
            const originalStr = String(el.dataset.original);
            
            // Type conversion.
            if (el.tagName === 'SELECT') {
                val = (val === 'true');
            } else if (el.type === 'number') {
                val = Number(val);
            }
            
            // Only modified fields need special handling.
            // Note: when null is rendered as an input, it may become an empty string,
            // which should not count as a modification.
            const isNullToEmpty = (originalStr === 'null' && val === '');
            if (String(val) !== originalStr && !isNullToEmpty) {
                hasChanges = true;
                newArgs[key] = val;
            }
        });
            
        if (!hasChanges) {
            alert("No parameter changes detected.");
            return;
        }
        
        try {
            const res = await fetch('/api/update_args', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(newArgs)
            });
            const msg = await res.json();
            
            if (msg.status === 'ok') {
                log('Parameters saved successfully:', msg.msg);
                
                // Update the local ARGS_LOADED cache.
                ARGS_LOADED = { ...ARGS_LOADED, ...newArgs };
                
                // Reset the UI state and remove the red highlights.
                inputs.forEach(el => {
                    // Update dataset.original to the current value.
                    if (el.tagName === 'SELECT') {
                        el.dataset.original = (el.value === 'true');
                    } else {
                        el.dataset.original = el.value;
                    }
                    el.classList.remove('text-danger', 'fw-bold');
                    el.style.borderColor = '';
                });
                // Mark this button as not clickable.
                // saveParamsBtn.disabled = true;
                    
                // Close the collapsible panel.
                const bsCollapse = new bootstrap.Collapse(document.getElementById('paramsCollapse'), {toggle: false});
                bsCollapse.hide();
                
            } else {
                alert('Save failed: ' + msg.msg);
            }
        } catch (e) {
            console.error(e);
            alert('Failed to send save request');
        }
    });

    // Start simulation / stop simulation.
    startSimBtn.addEventListener('click', async () => {
        if (!wsConnected) {
            alert('WebSocket is not connected, cannot start simulation!');
            return;
        }
        if (!ACTIVE_NAME || !DATA_CACHE[ACTIVE_NAME]) {
            alert('Please load data before starting the simulation!');
            return;
        }
        if (!MODEL_LOADED) {
            alert('Please load a model before starting the simulation!');
            return;
        }

        // Wait for all destination update requests to complete.
        if (destinationUpdatePromises.length > 0) {
            log('Waiting for destination update requests to complete...');
            await Promise.all(destinationUpdatePromises);
            log('All destination updates completed');
        }

        const datasetName = ACTIVE_NAME;
        const item = DATA_CACHE[datasetName];
        const startFrame = item.currentFrame != null ? Number(item.currentFrame) : Number(Object.keys(item.frames)[0] || 0);
        const totalFrame = Math.round(Number(simDurationValue.textContent) * item.fps);
        try {
            log(`Sending command to start simulation: starting from frame ${startFrame} of ${datasetName}...`);
            ws.send(JSON.stringify({ action: 'start', dataset_name: datasetName, frame_idx: startFrame, frame_num: totalFrame }));
            SIMULATION_RUNNING = true;
        } catch (e) {
            log('Failed to send start simulation command:', e);
        }
    });

    stopSimBtn.addEventListener('click', () => {
        try {
            ws.send(JSON.stringify({ action: 'stop' }));
            log('Sending command to stop simulation...');
            SIMULATION_RUNNING = false;
        } catch (e) {
            log('Failed to send stop simulation command:', e);
        }
    });

    // === Play/pause functionality ===
    function togglePlay() {
        if (isPlaying) {
            stopPlayback();
        } else {
            startPlayback();
        }
    }

    function startPlayback() {
        if (!ACTIVE_NAME || !DATA_CACHE[ACTIVE_NAME]) {
            alert("Please load data first");
            return;
        }

        const item = DATA_CACHE[ACTIVE_NAME];
        // If the current frame is already the last one and looping is disabled,
        // reset to the first frame before starting.
        const slider = document.getElementById(item.sliderId);
        if (slider) {
            const maxFrame = parseInt(slider.max);
            const currentFrame = item.currentFrame;
            if (currentFrame >= maxFrame && !loopCheckbox.checked) {
                // If we are at the end and looping is disabled, reset to the beginning.
                item.currentFrame = parseInt(slider.min);
                render(ACTIVE_NAME);
                updateSliderRangeAndValue(ACTIVE_NAME);
            }
        }

        isPlaying = true;
        playPauseBtn.textContent = "Pause";
        playPauseBtn.classList.replace('btn-success', 'btn-warning');

        // Get FPS, defaulting to 10.
        const fps = item.fps || 10;
        const interval = 1000 / fps / playbackSpeed; // Interval in milliseconds.

        if (playTimer) clearInterval(playTimer);
        playTimer = setInterval(playNextFrame, interval);
    }

    function stopPlayback() {
        isPlaying = false;
        playPauseBtn.textContent = "Play";
        playPauseBtn.classList.replace('btn-warning', 'btn-success');
        if (playTimer) {
            clearInterval(playTimer);
            playTimer = null;
        }
    }

    function playNextFrame() {
        if (!ACTIVE_NAME || !DATA_CACHE[ACTIVE_NAME]) {
            stopPlayback();
            return;
        }

        const item = DATA_CACHE[ACTIVE_NAME];
        const slider = document.getElementById(item.sliderId);
        
        if (!slider) {
            stopPlayback();
            return;
        }

        let current = parseInt(item.currentFrame);
        const max = parseInt(slider.max);
        const min = parseInt(slider.min);

        let next = current + 1;

        if (next > max) {
            if (loopCheckbox.checked) {
                next = min; // Looping: go back to the start.
            } else {
                stopPlayback(); // No loop: stop playback.
                return;
            }
        }

        // Update state.
        item.currentFrame = next;
        
        // Update the slider UI directly instead of recreating it, for better performance.
        slider.value = next;
        // Update the text next to the slider.
        const wrap = slider.parentElement;
        const valueSpan = wrap.querySelector('span');
        if (valueSpan) {
            valueSpan.textContent = `Current frame: ${next} (${min}~${max})`;
        }

        // Render the map.
        render(ACTIVE_NAME);
    }

    // Event listeners.
    playPauseBtn.addEventListener('click', togglePlay);

    // When the user manually drags the slider during playback, should playback pause or continue?
    // Keep playback continuous here: playback continues from the frame the user drags to.
    // We still need to keep item.currentFrame synchronized with slider.value,
    // which is already handled in the input event of createSliderForResponse.

    // ======== Destination dragging functionality ========

    // Add mouse event listeners to the map container, using the capture phase
    // to ensure they are not intercepted by Plotly.
    mapDiv.addEventListener('mousedown', (e) => handleMapMouseDown(e), true);
    mapDiv.addEventListener('mousemove', (e) => handleMapMouseMove(e), true);
    mapDiv.addEventListener('mouseup', (e) => handleMapMouseUp(e), true);
    mapDiv.addEventListener('mouseleave', (e) => handleMapMouseUp(e), true);

    // Handle mouse-down events on the map.
    function handleMapMouseDown(e) {
        // Only allow dragging when destinations are visible and playback is not running.
        if (!showDestinationsCheckbox.checked || isPlaying) return;

        // Check whether a destination marker was clicked.
        const rect = mapDiv.getBoundingClientRect();
        const pixelX = e.clientX - rect.left;
        const pixelY = e.clientY - rect.top;

        const pedId = findPedestrianAtPosition(pixelX, pixelY);
        if (pedId) {
            dragData = {
                isDragging: true,
                targetPedId: pedId,
                startX: pixelX,
                startY: pixelY
            };
            e.preventDefault();
            e.stopPropagation();
            mapDiv.style.cursor = 'grabbing';
            log(`👆 Clicked the destination of pedestrian ${pedId}; start dragging`);
        }
    }

    // Handle mouse-move events on the map.
    function handleMapMouseMove(e) {
        if (!dragData.isDragging) return;

        // Prevent Plotly's default behavior.
        e.preventDefault();
        e.stopPropagation();

        const rect = mapDiv.getBoundingClientRect();
        const pixelX = e.clientX - rect.left;
        const pixelY = e.clientY - rect.top;

        // Compute the new position by converting pixel coordinates to data coordinates.
        const coords = pixelToDataCoords(pixelX, pixelY);
        if (coords) {
            // Update local destination data and re-render.
            const item = DATA_CACHE[ACTIVE_NAME];
            if (item && item.destinations && item.destinations[dragData.targetPedId]) {
                item.destinations[dragData.targetPedId] = coords;
                render(ACTIVE_NAME);
            }
        }
    }

    // Handle mouse-up events on the map.
    function handleMapMouseUp(e) {
        if (dragData.isDragging) {
            const rect = mapDiv.getBoundingClientRect();
            const pixelX = e.clientX - rect.left;
            const pixelY = e.clientY - rect.top;

            const coords = pixelToDataCoords(pixelX, pixelY);
            if (coords) {
                sendDestinationUpdate(dragData.targetPedId, coords);
            }
            log(`👇 Mouse released; destination updated`);
            dragData = { isDragging: false, targetPedId: null };
            // Restore cursor style.
            mapDiv.style.cursor = '';
        }
    }

    // Coordinate conversion: pixel coordinates -> data coordinates.
    function pixelToDataCoords(pixelX, pixelY) {
        if (!myPlot) return null;

        const xaxis = myPlot._fullLayout.xaxis;
        const yaxis = myPlot._fullLayout.yaxis;

        if (!xaxis || !yaxis || !xaxis.range || !yaxis.range) return null;

        // Use the plot area size computed internally by Plotly.
        const gs = myPlot._fullLayout._size;
        if (!gs) {
            // Try to locate the plot-area rectangle from the SVG.
            const plotRect = myPlot.querySelector('.nsewdrag.drag[data-subplot="xy"]');
            if (!plotRect) return null;

            const marginLeft = parseFloat(plotRect.getAttribute('x'));
            const marginTop = parseFloat(plotRect.getAttribute('y'));
            const plotWidth = parseFloat(plotRect.getAttribute('width'));
            const plotHeight = parseFloat(plotRect.getAttribute('height'));

            const plotX = pixelX - marginLeft;
            const plotY = pixelY - marginTop;

            const xRange = xaxis.range;
            const yRange = yaxis.range;

            const dataX = xRange[0] + (plotX / plotWidth) * (xRange[1] - xRange[0]);
            const dataY = yRange[1] - (plotY / plotHeight) * (yRange[1] - yRange[0]);

            return { x: dataX, y: dataY };
        }

        const marginLeft = gs.l || 0;
        const marginTop = gs.t || 0;
        const plotWidth = gs.w || 1;
        const plotHeight = gs.h || 1;

        if (plotWidth <= 0 || plotHeight <= 0) return null;

        const plotX = pixelX - marginLeft;
        const plotY = pixelY - marginTop;

        const xRange = xaxis.range;
        const yRange = yaxis.range;
        const dataX = xRange[0] + (plotX / plotWidth) * (xRange[1] - xRange[0]);
        const dataY = yRange[1] - (plotY / plotHeight) * (yRange[1] - yRange[0]);

        return { x: dataX, y: dataY };
    }

    // Find the pedestrian destination near the mouse position.
    function findPedestrianAtPosition(pixelX, pixelY) {
        const coords = pixelToDataCoords(pixelX, pixelY);
        if (!coords) return null;

        const item = DATA_CACHE[ACTIVE_NAME];
        if (!item || !item.destinations) return null;

        const xaxis = myPlot?._fullLayout?.xaxis;
        const yaxis = myPlot?._fullLayout?.yaxis;
        const xRange = xaxis?.range || [0, 100];
        const yRange = yaxis?.range || [0, 100];
        const xSpan = xRange[1] - xRange[0];
        const ySpan = yRange[1] - yRange[0];
        const threshold = Math.min(xSpan, ySpan) * 0.08;

        let closestPedId = null;
        let minDist = Infinity;

        for (const pedId in item.destinations) {
            const des = item.destinations[pedId];
            const dist = Math.sqrt(
                Math.pow(des.x - coords.x, 2) +
                Math.pow(des.y - coords.y, 2)
            );
            if (dist < threshold && dist < minDist) {
                minDist = dist;
                closestPedId = pedId;
            }
        }

        return closestPedId;
    }

    // Destination update request queue.
    let destinationUpdatePromises = [];

    // Send a destination update request.
    async function sendDestinationUpdate(pedestrianId, newCoords) {
        const item = DATA_CACHE[ACTIVE_NAME];
        if (!item) return;

        const payload = {
            dataset_name: ACTIVE_NAME,
            pedestrian_id: pedestrianId,
            destination: {
                x: newCoords.x,
                y: newCoords.y
            }
        };

        // Create the Promise and add it to the queue.
        const updatePromise = fetch('/api/update_destination', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(res => res.json())
        .then(msg => {
            if (msg.status === 'ok') {
                log(`🎯 Destination update succeeded: Pedestrian ${pedestrianId} -> (${newCoords.x.toFixed(2)}, ${newCoords.y.toFixed(2)})`);
            } else {
                log('❌ Destination update failed:', msg.msg);
                alert('Update failed: ' + msg.msg);
                render(ACTIVE_NAME);
            }
        })
        .catch(e => {
            console.error(e);
            alert('Failed to send update request: ' + e.message);
            render(ACTIVE_NAME);
        });

        // Add to the pending queue.
        destinationUpdatePromises.push(updatePromise);

        // Wait for the current request to finish, then remove it from the queue.
        await updatePromise;
        destinationUpdatePromises = destinationUpdatePromises.filter(p => p !== updatePromise);
    }

    // ------- Initialization -------
    async function init() {
        connectWebsocket();
        loadLists();
        renderTrace();
    }

    // Start.
    init();

    // Close the WebSocket when the page unloads.
    window.addEventListener('beforeunload', () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            try { ws.close(); } catch (e) { }
        }
    });

// })();
