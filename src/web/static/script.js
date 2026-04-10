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
    let contextMenuTargetName = null; // 右键点击的目标 dataset name

    // WebSocket 相关状态
    let ws = null;
    let wsConnected = false;
    let reconnectAttempts = 0;
    let reconnectTimer = null;
    let pingIntervalId = null;

    // 播放状态
    let playTimer = null;
    let isPlaying = false;
    let playbackSpeed = parseFloat(playbackSpeedValue.textContent);

    // 数据缓存与运行状态
    const DATA_CACHE = {};  // { name: { name, fps, map, frames: { frameNumber: { id: {type, x, y}, ... } }, destinations, currentFrame, sliderId } }
    let ACTIVE_NAME = null; // 当前选中的 name
    let ARGS_LOADED = null; // 当前加载的模型参数
    let MODEL_LOADED = null; // 当前加载的模型
    let SIMULATION_RUNNING = false; // 是否有模拟在运行中

    // Plotly 图表实例
    let myPlot;

    // 目的地拖动状态
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

    // Auto View Control: 状态改变时立即重绘以应用设置（例如取消勾选时立即复位视图）
    autoViewCheckbox.addEventListener('change', () => {
        if (ACTIVE_NAME) render(ACTIVE_NAME);
    });

    // 日志输出
    function log(...args) {
        const t = new Date().toLocaleString();
        const s = args.map(a => (typeof a === 'object' ? JSON.stringify(a) : String(a))).join(' ');
        LOG.textContent += `[${t}] ${s}\n`;
        LOG.scrollTop = LOG.scrollHeight;
        console.debug(...args);
    }

    // WebSocket 连接与消息处理
    function connectWebsocket() {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const url = `${proto}//${location.host}/ws`;
        log('尝试连接 WebSocket:', url);
        ws = new WebSocket(url);
        ws.onopen = () => {
            wsConnected = true;
            reconnectAttempts = 0;
            log('WebSocket 已连接');
            startPing();
            // updateButtonsState();
        };
        ws.onclose = (ev) => {
            wsConnected = false;
            log('WebSocket 已断开', ev.code, ev.reason || '');
            // updateButtonsState();
            scheduleReconnect();
            stopPing();
        };
        ws.onerror = (err) => {
            log('WebSocket 错误', err && err.message ? err.message : err);
            // updateButtonsState();
            scheduleReconnect();
            stopPing();
        };
        ws.onmessage = (ev) => {
            try {
                const msg = JSON.parse(ev.data); // 后端发送的格式通常是 {status: 'ok'|'error', data: response, msg: '...'}
                if (msg.msg) { log('Server:', msg.msg); }
                if (msg.data) { mergeAndHandleResponse(msg.data); }
            } catch (e) {
                log('收到无法解析的 WebSocket 消息:', ev.data);
            }
        };
    }

    // 自动重连机制
    function scheduleReconnect() {
        if (reconnectTimer) return;
        reconnectAttempts += 1;
        const delay = Math.min(30000, 1000 * Math.pow(1.6, Math.min(reconnectAttempts, 10))); // 指数回退，最大30s
        log(`WebSocket 将在 ${Math.round(delay / 1000)}s 后重试连接 (第 ${reconnectAttempts} 次)`);
        reconnectTimer = setTimeout(() => {
            reconnectTimer = null;
            connectWebsocket();
        }, delay);
    }

    // 简单的心跳 (向服务器发送 ping，服务器会回复 pong)
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

    // 将后端的 response 合并缓存并创建/更新滑块与地图
    function mergeAndHandleResponse(response) {
        const name = response.name;
        // 新的缓存项
        if (!DATA_CACHE[name]) {
            DATA_CACHE[name] = {
                name: response.name,
                map: null,
                frames: {},
                destinations: {},  // 存储目的地数据
                currentFrame: null,
                sliderId: null,
                fps: response.fps || 10,
                has_high_res_map: false,  // 是否有高分辨率原图
            };
        }
        // 更新 map
        if (response.map) {
            DATA_CACHE[name].map = response.map;
        }
        if (response.fps) {
            DATA_CACHE[name].fps = response.fps;
        }
        // 更新 high_res_map_path
        if (response.has_high_res_map !== undefined) {
            DATA_CACHE[name].has_high_res_map = response.has_high_res_map;
        }
        // 更新 destinations (如果后端提供)
        if (response.destinations) {
            DATA_CACHE[name].destinations = response.destinations;
        }
        // 更新 frames 和 currentFrame
        const frames = response.frames || {};
        for (const fkey of Object.keys(frames)) {
            const fnum = Number(fkey);
            DATA_CACHE[name].frames[fnum] = frames[fkey];
            DATA_CACHE[name].currentFrame = fnum; // 更新当前帧到最新传来的帧
        }
        // 缓存高分辨率地图 URL
        const item = DATA_CACHE[name];
        if (item.has_high_res_map) {
            item.high_res_map_url = `/api/get_high_res_map?dataset_name=${encodeURIComponent(name)}`;
        }
        // 创建 / 更新滑块
        if (!DATA_CACHE[name].sliderId) { // 如果没有 slider，则创建
            createSliderForResponse(name);
            // 检查是否有原图，启用 checkbox
            updateHighResMapCheckbox(name);
        } else { // 更新 slider 的 max (如果需要) 并把滑块值设置到最新 currentFrame
            updateSliderRangeAndValue(name);
        }
        // 重新渲染地图
        render(name);
    }

    // Update high-res map checkbox state based on current dataset
    function updateHighResMapCheckbox(name) {
        const item = DATA_CACHE[name];
        if (item && item.has_high_res_map) {
            showHighResMapCheckbox.disabled = false;
            showHighResMapCheckbox.title = '可用高分辨率原图';
            highResMapOpacityRow.style.display = 'flex';
            highResMapOpacityRow.style.alignItems = 'center';
        } else {
            showHighResMapCheckbox.disabled = true;
            showHighResMapCheckbox.checked = false;
            showHighResMapCheckbox.title = '该数据集没有原图';
            highResMapOpacityRow.style.display = 'none';
        }
    }

    // UI: 创建滑块、管理滑块事件
    function createSliderForResponse(name) {
        const item = DATA_CACHE[name];
        const frameKeys = Object.keys(item.frames).map(k => Number(k)).sort((a, b) => a - b);
        const min = frameKeys.length ? frameKeys[0] : 0;
        const max = frameKeys.length ? frameKeys[frameKeys.length - 1] : min;
        const cur = item.currentFrame != null ? item.currentFrame : min;

        // DOM
        const wrap = document.createElement('div');
        wrap.className = 'slider-wrap';
        wrap.dataset.name = name; // 绑定 dataset name 用于右键菜单
        wrap.draggable = false;  // 允许拖动  

        const header = document.createElement('div');
        header.style.display = 'flex';
        header.style.justifyContent = 'space-between';
        header.style.alignItems = 'center';

        const label = document.createElement('div');
        label.className = 'slider-label';
        label.textContent = name;
        label.draggable = true; // 允许拖动 

        const closeBtn = document.createElement('button');
        closeBtn.textContent = '✖';
        closeBtn.className = 'btn btn-sm btn-outline-danger';
        closeBtn.style.padding = '0 6px';
        closeBtn.style.lineHeight = '1';
        closeBtn.title = '删除此滑块';

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
        valueSpan.textContent = `当前帧: ${cur} (${min}~${max})`;
        valueSpan.style.marginLeft = '8px';

        wrap.appendChild(slider);
        wrap.appendChild(valueSpan);
        slidersDiv.appendChild(wrap);

        // === 删除按钮事件 ===
        closeBtn.addEventListener('click', () => {
            wrap.remove();
            delete DATA_CACHE[name]; // 可选
            if (ACTIVE_NAME === name) ACTIVE_NAME = null;
        });

        // === 拖动事件 ===
        label.addEventListener('dragstart', (ev) => {
            ev.dataTransfer.setData('text/plain', name);
            wrap.classList.add('dragging');
        });
        label.addEventListener('dragend', () => wrap.classList.remove('dragging'));
        // slider.addEventListener('mousedown', (ev) => ev.stopPropagation());
        // slider.addEventListener('touchstart', (ev) => ev.stopPropagation());

        // === 右键菜单事件 ===
        wrap.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            contextMenuTargetName = name;
            // 简单的菜单定位
            contextMenu.style.display = 'block';
            contextMenu.style.left = e.pageX + 'px';
            contextMenu.style.top = e.pageY + 'px';
        });

        // === 滑动事件 ===
        slider.addEventListener('input', (ev) => {
            const v = Number(ev.target.value);
            item.currentFrame = v;
            valueSpan.textContent = `当前帧: ${v} (${min}~${max})`;
            render(name);
        });
        slider.addEventListener('mousedown', () => render(name));
        slider.addEventListener('touchstart', () => render(name));
    }

    // 全局点击关闭右键菜单
    document.addEventListener('click', () => {
        contextMenu.style.display = 'none';
    });

    // 右键菜单项点击
    ctxSaveTraj.addEventListener('click', () => {
        if (contextMenuTargetName && DATA_CACHE[contextMenuTargetName]) {
            openSaveModal(contextMenuTargetName);
        }
    });

    // === 打开保存模态框逻辑 ===
    function openSaveModal(name) {
        saveModalDatasetName.textContent = name;
        const item = DATA_CACHE[name];
        const frameKeys = Object.keys(item.frames).map(k => Number(k)).sort((a, b) => a - b);
        const min = frameKeys.length ? frameKeys[0] : 0;
        const max = frameKeys.length ? frameKeys[frameKeys.length - 1] : min;
        
        // 设置范围滑块属性
        saveRangeMin.min = min; saveRangeMin.max = max;
        saveRangeMax.min = min; saveRangeMax.max = max;
        saveRangeMin.value = min;
        saveRangeMax.value = max;
        
        updateDualSliderUI(min, max);
        
        // 激活当前 dataset 视图以便预览
        render(name);
        
        saveTrajModal.show();
    }

    // 双柄滑块逻辑
    function updateDualSliderUI(min, max) {
        let vMin = parseInt(saveRangeMin.value);
        let vMax = parseInt(saveRangeMax.value);
        
        // 限制交叉
        if (vMin > vMax) {
             // 简单的互斥逻辑：谁动了改谁，这里简单处理
             // 我们在 input 事件里处理更合适
        }
        
        saveFrameRangeVal.textContent = `${vMin} - ${vMax}`;
    }

    // 监听双柄滑块变化
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
        
        // 实时更新主视图
        // 如果动的是 min，显示 min 帧；动的是 max，显示 max 帧
        if (item) {
            item.currentFrame = (e.target === saveRangeMin) ? vMin : vMax;
            // 更新该 dataset 对应的 slider UI（虽然在模态框里看不到，但保持状态一致）
            const mainSlider = document.getElementById(item.sliderId);
            if(mainSlider) mainSlider.value = item.currentFrame;
            render(contextMenuTargetName);
        }
    }

    saveRangeMin.addEventListener('input', handleDualSliderInput);
    saveRangeMax.addEventListener('input', handleDualSliderInput);

    // 确定保存
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

        // 关闭模态框
        saveTrajModal.hide();
        
        log(`正在请求保存轨迹: ${name} [${start}-${end}] -> ${dest}`);

        try {
            const res = await fetch('/api/save_trajectory', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            if (dest === 'local') {
                if (res.ok) {
                    const blob = await res.blob();
                    // 从 Content-Disposition 获取文件名
                    const disposition = res.headers.get('Content-Disposition');
                    let filename = `trajectory.csv${compress ? '.tz' : ''}`;
                    if (disposition && disposition.indexOf('attachment') !== -1) {
                        const matches = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/.exec(disposition);
                        if (matches != null && matches[1]) { 
                            filename = matches[1].replace(/['"]/g, '');
                        }
                    }
                    // 触发下载
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = filename;
                    document.body.appendChild(a);
                    a.click();
                    a.remove();
                    window.URL.revokeObjectURL(url);
                    log('下载已开始。');
                } else {
                    const err = await res.json();
                    alert('下载失败: ' + (err.msg || 'Unknown error'));
                }
            } else {
                const msg = await res.json();
                if (msg.status === 'ok') {
                    log('保存成功:', msg.msg);
                    alert('保存成功: ' + msg.msg);
                } else {
                    alert('保存失败: ' + msg.msg);
                }
            }
        } catch (e) {
            console.error(e);
            alert('请求发送失败');
        }
    });


    // === 全局：为 slidersDiv 启用拖拽排序 ===
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
        if (span) span.textContent = `当前帧: ${cur} (${min}~${max})`;
    }

    // UI: 高亮当前活动滑块
    function highlightSlider() {
        const item = DATA_CACHE[ACTIVE_NAME];
        if(!item) return;
        const sliderWraps = slidersDiv.querySelectorAll('.slider-wrap');
        sliderWraps.forEach(wrap => {
            if (wrap.contains(document.getElementById(item.sliderId))) {
                // 当前滑块高亮
                wrap.style.border = '2px solid #007bff';
                wrap.style.backgroundColor = '#e7f1ff';
            } else {
                // 取消高亮
                wrap.style.border = '1px solid #dee2e6';
                wrap.style.backgroundColor = '#ffffff';
            }
        });
    }

    // 渲染函数：根据 ACTIVE_NAME 渲染地图和轨迹
    function render(name) {
        if (ACTIVE_NAME !== name) {
            ACTIVE_NAME = name;
            log(`切换至数据集：${name}`);
            highlightSlider();
        }
        renderTrace();
    }

    // 渲染实体轨迹 (Plotly)
    function renderTrace() {
        if (!ACTIVE_NAME || !DATA_CACHE[ACTIVE_NAME]) {
            Plotly.react(mapDiv, [], { title: "暂无数据" });
            return;
        }
        const plotData = [];
        const item = DATA_CACHE[ACTIVE_NAME];
        const currentFrame = Number(item.currentFrame);
        const fps = item.fps || 10;
        const trailSec = Number(trailSlider.value);
        const trailFrames = Math.round(trailSec * fps);
        // if (!entities) {
        //     Plotly.react(mapDiv, [], { title: `${ACTIVE_NAME} - 当前帧无数据` });
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

        // 1. 找到当前帧所有活动的个体，为每个个体维护一个 trace
        const activeTraces = {};
        for (const e in currentEntities) {
            activeTraces[e] = { x: [], y: [] };
        }

        // 2. 从当前帧向历史枚举指定长度的帧
        const startF = Math.max(0, currentFrame - trailFrames);
        for (let f = currentFrame; f >= startF; f--) {
            const frameData = item.frames[f];
            if (!frameData) continue;
            // 3. 对于选定的历史帧，检查活动个体在这一帧是否出现
            for (const e in frameData) {
                if (activeTraces[e]) {
                    activeTraces[e].x.push(frameData[e].x);
                    activeTraces[e].y.push(frameData[e].y);
                }
            }
        }

        // 4. 枚举完后，将每个个体的 trace 添加到绘图
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

        // 计算行人点在当前缩放下的像素大小（固定 0.3m 直径）
        const PEDESTRIAN_DIAMETER_METERS = 0.3;
        let pedMarkerSize = 6;  // 默认最小像素大小
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

        // Pedestrian Dots (固定 0.3m 直径，随缩放变化)
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
                        dash: 'solid'  // 灰色实线
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
                    ids: Object.keys(item.destinations),  // 使用目的地的 ID
                    mode: 'markers',
                    marker: {
                        size: 12,
                        color: 'rgb(255, 0, 0)',
                        symbol: 'x'  // 红色叉号
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
                x: mapInfo.xmin,          // 图像左边界
                y: mapInfo.ymax,          // 图像上边界（使用 yanchor: 'top'）
                sizex: mapInfo.xmax - mapInfo.xmin,
                sizey: mapInfo.ymax - mapInfo.ymin,
                xanchor: 'left',          // 锚点在左边缘
                yanchor: 'top',           // 锚点在上边缘（关键：使 y 坐标对应图像顶部）
                sizing: 'stretch',
                opacity: opacity,
                layer: 'below'            // 显示在轨迹下方
            }];
        } else {
            layout.images = [];  // 未勾选时清除图片
        }
        
        // const smooth = true;
        // if (smooth) {
        //     layout.transition = {
        //         duration: 1000 / fps / playbackSpeed, // 动画时长等于帧间隔，例如 2.5fps -> 400ms
        //         easing: 'linear'      // 线性移动，模拟匀速运动
        //     };
        // } else {
        //     layout.transition = { duration: 0 }; // 手动拖拽时立即响应
        // }

        if (myPlot) {
            Plotly.react(myPlot, plotData, layout);
            // 永远禁用 Plotly 的拖动缩放，使用自定义的鼠标事件
            Plotly.relayout(myPlot, { 'dragmode': false });
        } else {
            Plotly.newPlot(mapDiv, plotData, layout, {
                responsive: true,
                scrollZoom: true,
                dragmode: false  // 禁用拖动缩放
            })
            .then((plotElement) => {
                myPlot = plotElement;
                // 监听缩放事件，重新渲染以更新行人点大小
                myPlot.on('plotly_relayout', (eventData) => {
                    // 只在 xaxis.range 或 yaxis.range 变化时重新渲染（表示缩放/平移）
                    if (eventData['xaxis.range'] || eventData['yaxis.range'] ||
                        eventData['xaxis.range[0]'] || eventData['xaxis.range[1]'] ||
                        eventData['yaxis.range[0]'] || eventData['yaxis.range[1]']) {
                        renderTrace();
                    }
                });
            });
        }
    }

    // 拉取 dataset_list / model_list 并填充下拉框
    async function loadLists() {
        try {
            const [dsr, mr] = await Promise.all([
                fetch('/api/dataset_list').then(r => r.json()).catch(e => ({})),
                fetch('/api/model_list').then(r => r.json()).catch(e => ({}))
            ]);
            // dataset_list 返回的是一个对象 index->full_name
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
            log('已加载 dataset_list 和 model_list');
        } catch (e) {
            log('加载 dataset/model 列表失败:', e);
        }
    }

    // 从后端获取的 KEEP_ARGS 顺序
    let KEEP_ARGS_ORDER = null;

    // 加载 dataset
    loadDatasetBtn.addEventListener('click', async () => {
        const idx = datasetSelect.value;
        if (idx == null) { alert('请先选择数据集'); return; }
        if (ARGS_LOADED == null) { alert('请先加载模型'); return; }
        try {
            // 将 name 设为一个唯一标识，后端会把它作为 key 保存在 DATASET_DICT[name]
            const name = datasetSelect.options[datasetSelect.selectedIndex].text || 'dataset';
            log(`正在加载数据集 ${name} (idx=${idx}) ...`);
            const res = await fetch(`/api/load_dataset?idx=${encodeURIComponent(idx)}&name=${encodeURIComponent(name)}`)
            const msg = await res.json();
            if (msg.status === 'ok') {
                if (msg.response) {
                    log('Server:', msg.msg || 'Dataset loaded.');
                    mergeAndHandleResponse(msg.response);
                } else {
                    log('警告: load_dataset 返回格式未包含 response 字段，请检查后端。', msg);
                }
            } else {
                log('加载数据集失败:', msg.msg || JSON.stringify(msg));
            }
        } catch (e) {
            log('加载数据集请求失败:', e);
        }
    });

    // 加载 model
    loadModelBtn.addEventListener('click', async () => {
        const idx = modelSelect.value;
        if (idx == null) { log('请先选择模型'); return; }
        try {
            const name = modelSelect.options[modelSelect.selectedIndex].text || 'model';
            log(`正在加载模型权重 ${name} (idx=${idx}) ...`);
            const res = await fetch(`/api/load_model?idx=${encodeURIComponent(idx)}`);
            const msg = await res.json();
            if (msg.status === 'ok') {
                log('Server:', msg.msg || 'Model loaded.');
                MODEL_LOADED = name;
                ARGS_LOADED = msg.response;
                KEEP_ARGS_ORDER = msg.keep_args_order || null;  // 保存参数顺序
                log('当前模型参数:', ARGS_LOADED);
                editParamsBtn.classList.remove('d-none');
                renderParamsEditor(ARGS_LOADED);
            } else {
                log('加载模型失败:', msg.msg || msg);
            }
        } catch (e) {
            log('加载模型请求失败:', e);
        }
    });

    // 渲染参数列表函数（按 KEEP_ARGS_ORDER 顺序显示）
    function renderParamsEditor(args) {
        paramsList.innerHTML = '';

        // 使用 KEEP_ARGS_ORDER 定义的顺序，没有的 key 放在最后
        const orderedKeys = [];
        const remainingKeys = [];

        if (KEEP_ARGS_ORDER && Array.isArray(KEEP_ARGS_ORDER)) {
            // 按 KEEP_ARGS_ORDER 顺序收集存在的 key
            for (const key of KEEP_ARGS_ORDER) {
                if (key in args) {
                    orderedKeys.push(key);
                }
            }
            // 收集剩余的 key（不在 KEEP_ARGS_ORDER 中的）
            for (const key of Object.keys(args)) {
                if (!orderedKeys.includes(key)) {
                    remainingKeys.push(key);
                }
            }
            remainingKeys.sort();  // 剩余的按字母排序
        } else {
            // 没有顺序信息时按字母排序（向后兼容）
            Object.keys(args).sort().forEach(k => orderedKeys.push(k));
        }

        const allKeys = [...orderedKeys, ...remainingKeys];

        allKeys.forEach(key => {
            const val = args[key];
            // 跳过复杂对象，只允许编辑基础类型
            if (val !== null && typeof val === 'object') return;
            const row = document.createElement('div');
            row.className = 'mb-2 row g-1 align-items-center';

            const labelCol = document.createElement('div');
            labelCol.className = 'col-5 text-break';
            labelCol.textContent = key;
            labelCol.title = key; // hover 显示完整 key
            
            const inputCol = document.createElement('div');
            inputCol.className = 'col-7';
            
            const input = document.createElement('input');
            input.className = 'form-control form-control-sm param-input';
            input.dataset.key = key;
            input.dataset.original = val; // 存储原始值
            input.value = val;
            
            // 根据类型设置 input 属性
            if (typeof val === 'number') {
                input.type = 'number';
                input.step = 'any'; // 允许小数
            } else if (typeof val === 'boolean') {
                // 对于布尔值，可以做成下拉框或者 checkbox，这里简单用 text 模拟，或者 input type=text
                // 为了 fancy 一点，我们用 select
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
                
                // 替换 input 为 select
                inputCol.appendChild(select);
                
                // Select 事件
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
                return; // 结束当前循环
            } else {
                input.type = 'text';
            }
            
            // Input 事件：检测修改并标红
            input.addEventListener('input', (e) => {
                const currentVal = e.target.value;
                const originalVal = String(e.target.dataset.original);
                
                // 简单比较字符串
                if (currentVal !== originalVal) {
                    e.target.classList.add('text-danger', 'fw-bold'); // Bootstrap 红色 + 加粗
                    e.target.style.borderColor = '#dc3545'; // 边框也变红
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
            
            // 类型转换
            if (el.tagName === 'SELECT') {
                val = (val === 'true');
            } else if (el.type === 'number') {
                val = Number(val);
            }
            
            // 只有修改过的才需要特别关注
            // 注意：null 被渲染为 input 时可能变成空字符串，这不算修改
            const isNullToEmpty = (originalStr === 'null' && val === '');
            if (String(val) !== originalStr && !isNullToEmpty) {
                hasChanges = true;
                newArgs[key] = val;
            }
        });
            
        if (!hasChanges) {
            alert("未检测到任何参数修改。");
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
                log('参数保存成功:', msg.msg);
                
                // 更新本地缓存 ARGS_LOADED
                ARGS_LOADED = { ...ARGS_LOADED, ...newArgs };
                
                // 重置 UI 状态（去掉红色）
                inputs.forEach(el => {
                    // 更新 dataset.original 为当前新值
                    if (el.tagName === 'SELECT') {
                        el.dataset.original = (el.value === 'true');
                    } else {
                        el.dataset.original = el.value;
                    }
                    el.classList.remove('text-danger', 'fw-bold');
                    el.style.borderColor = '';
                });
                // 将这个按钮标记为不可点击的
                // saveParamsBtn.disabled = true;
                    
                // 关闭折叠面板
                const bsCollapse = new bootstrap.Collapse(document.getElementById('paramsCollapse'), {toggle: false});
                bsCollapse.hide();
                
            } else {
                alert('保存失败: ' + msg.msg);
            }
        } catch (e) {
            console.error(e);
            alert('保存请求发送失败');
        }
    });

    // 开始模拟 / 结束模拟
    startSimBtn.addEventListener('click', () => {
        if (!wsConnected) {
            alert('WebSocket 未连接，无法开始模拟!');
            return;
        }
        if (!ACTIVE_NAME || !DATA_CACHE[ACTIVE_NAME]) {
            alert('请先加载数据后再开始模拟!');
            return;
        }
        if (!MODEL_LOADED) {
            alert('请先加载模型后再开始模拟!');
            return;
        }
        const datasetName = ACTIVE_NAME;
        const item = DATA_CACHE[datasetName];
        const startFrame = item.currentFrame != null ? Number(item.currentFrame) : Number(Object.keys(item.frames)[0] || 0);
        const totalFrame = Math.round(Number(simDurationValue.textContent) * item.fps);
        try {
            log(`发送指令以开始模拟: 从 ${datasetName} 的第 ${startFrame} 帧开始...`);
            ws.send(JSON.stringify({ action: 'start', dataset_name: datasetName, frame_idx: startFrame, frame_num: totalFrame }));
            SIMULATION_RUNNING = true;
        } catch (e) {
            log('发送开始模拟指令失败:', e);
        }
    });

    stopSimBtn.addEventListener('click', () => {
        try {
            ws.send(JSON.stringify({ action: 'stop' }));
            log('发送指令以停止模拟...');
            SIMULATION_RUNNING = false;
        } catch (e) {
            log('发送停止模拟指令失败:', e);
        }
    });

    // === 播放/暂停功能 ===
    function togglePlay() {
        if (isPlaying) {
            stopPlayback();
        } else {
            startPlayback();
        }
    }

    function startPlayback() {
        if (!ACTIVE_NAME || !DATA_CACHE[ACTIVE_NAME]) {
            alert("请先加载数据");
            return;
        }

        const item = DATA_CACHE[ACTIVE_NAME];
        // 如果当前已经在最后一帧，且没有开启循环，则重置到第一帧再开始
        const slider = document.getElementById(item.sliderId);
        if (slider) {
            const maxFrame = parseInt(slider.max);
            const currentFrame = item.currentFrame;
            if (currentFrame >= maxFrame && !loopCheckbox.checked) {
                // 如果在末尾且不循环，重置到开头
                item.currentFrame = parseInt(slider.min);
                render(ACTIVE_NAME);
                updateSliderRangeAndValue(ACTIVE_NAME);
            }
        }

        isPlaying = true;
        playPauseBtn.textContent = "暂停";
        playPauseBtn.classList.replace('btn-success', 'btn-warning');

        // 获取 FPS，默认为 10
        const fps = item.fps || 10;
        const interval = 1000 / fps / playbackSpeed; // 毫秒间隔

        if (playTimer) clearInterval(playTimer);
        playTimer = setInterval(playNextFrame, interval);
    }

    function stopPlayback() {
        isPlaying = false;
        playPauseBtn.textContent = "播放";
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
                next = min; // 循环：回到起点
            } else {
                stopPlayback(); // 不循环：停止
                return;
            }
        }

        // 更新状态
        item.currentFrame = next;
        
        // 更新滑块 UI (不重新创建，直接修改值以提高性能)
        slider.value = next;
        // 更新滑块旁边的文本 (span)
        const wrap = slider.parentElement;
        const valueSpan = wrap.querySelector('span');
        if (valueSpan) {
            valueSpan.textContent = `当前帧: ${next} (${min}~${max})`;
        }

        // 渲染地图
        render(ACTIVE_NAME);
    }

    // 事件监听
    playPauseBtn.addEventListener('click', togglePlay);

    // 当用户手动拖动滑块时，如果正在播放，建议暂时停止或保持播放？
    // 这里保持播放逻辑：用户拖到哪，就从哪继续播。
    // 但我们需要确保 item.currentFrame 与 slider.value 同步，这在 createSliderForResponse 的 input 事件中已经处理了。

    // ======== 目的地拖动功能 ========

    // 添加地图容器的鼠标事件监听（使用 capture 阶段确保不被 Plotly 拦截）
    mapDiv.addEventListener('mousedown', (e) => handleMapMouseDown(e), true);
    mapDiv.addEventListener('mousemove', (e) => handleMapMouseMove(e), true);
    mapDiv.addEventListener('mouseup', (e) => handleMapMouseUp(e), true);
    mapDiv.addEventListener('mouseleave', (e) => handleMapMouseUp(e), true);

    // 处理地图鼠标按下事件
    function handleMapMouseDown(e) {
        // 只有在显示目的地且没有正在播放时才允许拖动
        if (!showDestinationsCheckbox.checked || isPlaying) return;

        // 检测是否点击了目的地标记
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
            log(`👆 点击了行人 ${pedId} 的目的地，开始拖动`);
        }
    }

    // 处理地图鼠标移动事件
    function handleMapMouseMove(e) {
        if (!dragData.isDragging) return;

        // 阻止 Plotly 的默认行为
        e.preventDefault();
        e.stopPropagation();

        const rect = mapDiv.getBoundingClientRect();
        const pixelX = e.clientX - rect.left;
        const pixelY = e.clientY - rect.top;

        // 计算新位置（需要将像素坐标转换为数据坐标）
        const coords = pixelToDataCoords(pixelX, pixelY);
        if (coords) {
            // 更新本地目的地数据并重绘
            const item = DATA_CACHE[ACTIVE_NAME];
            if (item && item.destinations && item.destinations[dragData.targetPedId]) {
                item.destinations[dragData.targetPedId] = coords;
                render(ACTIVE_NAME);
            }
        }
    }

    // 处理地图鼠标释放事件
    function handleMapMouseUp(e) {
        if (dragData.isDragging) {
            const rect = mapDiv.getBoundingClientRect();
            const pixelX = e.clientX - rect.left;
            const pixelY = e.clientY - rect.top;

            const coords = pixelToDataCoords(pixelX, pixelY);
            if (coords) {
                sendDestinationUpdate(dragData.targetPedId, coords);
            }
            log(`👇 释放鼠标，目的地已更新`);
            dragData = { isDragging: false, targetPedId: null };
            // 恢复 cursor 样式
            mapDiv.style.cursor = '';
        }
    }

    // 坐标转换：像素坐标 -> 数据坐标
    function pixelToDataCoords(pixelX, pixelY) {
        if (!myPlot) return null;

        const xaxis = myPlot._fullLayout.xaxis;
        const yaxis = myPlot._fullLayout.yaxis;

        if (!xaxis || !yaxis || !xaxis.range || !yaxis.range) return null;

        // 使用 Plotly 内部计算的绘图区域尺寸
        const gs = myPlot._fullLayout._size;
        if (!gs) {
            // 尝试从 SVG 中查找绘图区域 rect
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

    // 查找鼠标位置附近的行人目的地
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

    // 发送目的地更新请求
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

        try {
            const res = await fetch('/api/update_destination', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const msg = await res.json();

            if (msg.status === 'ok') {
                log(`🎯 目的地更新成功: Pedestrian ${pedestrianId} -> (${newCoords.x.toFixed(2)}, ${newCoords.y.toFixed(2)})`);
            } else {
                log('❌ 目的地更新失败:', msg.msg);
                alert('更新失败: ' + msg.msg);
                // 恢复原始位置
                render(ACTIVE_NAME);
            }
        } catch (e) {
            console.error(e);
            alert('更新请求发送失败: ' + e.message);
            // 恢复原始位置
            render(ACTIVE_NAME);
        }
    }

    // ------- 初始化 -------
    async function init() {
        connectWebsocket();
        loadLists();
        renderTrace();
    }

    // 启动
    init();

    // 页面卸载时关闭 ws
    window.addEventListener('beforeunload', () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            try { ws.close(); } catch (e) { }
        }
    });

// })();