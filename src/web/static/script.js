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

    // WebSocket 相关状态
    let ws = null;
    let wsConnected = false;
    let reconnectAttempts = 0;
    let reconnectTimer = null;
    let pingIntervalId = null;
    
    // 数据缓存与运行状态
    const DATA_CACHE = {};  // { name: { name, fps, map, frames: { frameNumber: { id: {type, x, y}, ... } }, currentFrame, sliderId } }
    let ACTIVE_NAME = null; // 当前选中的 name
    let ARGS_LOADED = null; // 当前加载的模型参数
    let MODEL_LOADED = null; // 当前加载的模型
    let SIMULATION_RUNNING = false; // 是否有模拟在运行中

    // Plotly 图表实例
    let myPlot;

    // Trail Control
    trailSlider.addEventListener('input', (e) => {
        trailValue.textContent = e.target.value;
        if (ACTIVE_NAME) render(ACTIVE_NAME);
    });

    // Simulation Duration Control
    simDurationSlider.addEventListener('input', (e) => {
        simDurationValue.textContent = e.target.value;
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
                currentFrame: null,
                sliderId: null,
                fps: response.fps || 10, 
            };
        }
        // 更新 map
        if (response.map) {
            DATA_CACHE[name].map = response.map;
        }
        if (response.fps) {
            DATA_CACHE[name].fps = response.fps;
        }
        // 更新 frames 和 currentFrame
        const frames = response.frames || {};
        for (const fkey of Object.keys(frames)) {
            const fnum = Number(fkey);
            DATA_CACHE[name].frames[fnum] = frames[fkey];
            DATA_CACHE[name].currentFrame = fnum; // 更新当前帧到最新传来的帧
        }
        // 创建 / 更新滑块
        if (!DATA_CACHE[name].sliderId) { // 如果没有 slider，则创建
            createSliderForResponse(name);
        } else { // 更新 slider 的 max (如果需要) 并把滑块值设置到最新 currentFrame
            updateSliderRangeAndValue(name);
        }
        // 重新渲染地图
        render(name);
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
                opacity: 1.0
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
        const pedX = [], pedY = [], pedText = []; // Pedestrians
        const vehMarkersX = [], vehMarkersY = [], vehText = []; // Vehicle Centers (for ID)

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
                vehText.push(`ID: ${id}<br>Veh`);

            } else if (e.type === 'vehicle') {
                // Fallback for vehicle without dims -> Dot
                vehMarkersX.push(e.x);
                vehMarkersY.push(e.y);
                vehText.push(`ID: ${id}<br>Veh (No Dim)`);
            } else {
                // Pedestrian without dims -> Dot
                pedX.push(e.x);
                pedY.push(e.y);
                pedText.push(`ID: ${id}<br>Ped`);
            }
        }

        // Vehicle Polygons
        if (vehX.length > 0) {
            plotData.push({
                x: vehX,
                y: vehY,
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
                mode: 'markers',
                marker: { size: 6, color: 'rgb(200, 80, 0)', symbol: 'square' },
                text: vehText,
                hoverinfo: 'text',
                name: 'Vehicle',
                showlegend: true,
            });
        }

        // Pedestrian Dots
        if (pedX.length > 0) {
            plotData.push({
                x: pedX,
                y: pedY,
                mode: 'markers',
                marker: { size: 6, color: 'rgb(0, 100, 255)' },
                text: pedText,
                hoverinfo: 'text',
                name: 'Pedestrian',
                showlegend: true,
            });
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
        if (myPlot) {
            Plotly.react(myPlot, plotData, layout);
        } else {
            Plotly.newPlot(mapDiv, plotData, layout, { responsive: true })
                .then((plotElement) => {myPlot = plotElement;});
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

    // 渲染参数列表函数
    function renderParamsEditor(args) {
        paramsList.innerHTML = '';
        const keys = Object.keys(args).sort();
        keys.forEach(key => {
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
            if (String(val) !== originalStr) {
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