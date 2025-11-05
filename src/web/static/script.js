(() => {
    const LOG = document.getElementById('log');
    const datasetSelect = document.getElementById('datasetSelect');
    const modelSelect = document.getElementById('modelSelect');
    const loadDatasetBtn = document.getElementById('loadDatasetBtn');
    const loadModelBtn = document.getElementById('loadModelBtn');
    const startSimBtn = document.getElementById('startSimBtn');
    const stopSimBtn = document.getElementById('stopSimBtn');
    const slidersDiv = document.getElementById('sliders');
    const mapDiv = document.getElementById('map');

    // WebSocket 相关状态
    let ws = null;
    let wsConnected = false;
    let reconnectAttempts = 0;
    let reconnectTimer = null;
    let pingIntervalId = null;
    
    // 数据缓存与运行状态
    const DATA_CACHE = {};  // { name: { name, map, frames: { frameNumber: { id: {type, x, y}, ... } }, currentFrame, sliderId } }
    let ACTIVE_NAME = null; // 当前选中的 name
    let ARGS_LOADED = null; // 当前加载的模型参数
    let MODEL_LOADED = null; // 当前加载的模型
    let SIMULATION_RUNNING = false; // 是否有模拟在运行中

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
            };
        }
        // 更新 map
        if (response.map) {
            DATA_CACHE[name].map = response.map;
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
        const entities = item.frames[Number(item.currentFrame)];
        // if (!entities) {
        //     Plotly.react(mapDiv, [], { title: `${ACTIVE_NAME} - 当前帧无数据` });
        //     return;
        // }

        const byType = {};
        entities.forEach(e => {
            const t = e.type || 'pedestrian';
            if (!byType[t]) byType[t] = [];
            byType[t].push(e);
        });

        const traces = [];
        Object.keys(byType).forEach(t => {
            const arr = byType[t];
            // const TYPE_STYLE = {
            //     pedestrian: { marker: { size: 8, symbol: 'circle' } },
            //     vehicle: { marker: { size: 12, symbol: 'square' } },
            // };
            // const type = TYPE_STYLE[t];
            traces.push({
                x: arr.map(e => e.x),
                y: arr.map(e => e.y),
                mode: 'markers+text',
                name: `${t}`,
                text: arr.map(e => e.id || ''),
                textposition: 'top center',
                marker: {
                    size: (t === 'vehicle' ? 12 : 8),
                    symbol: (t === 'vehicle' ? 'square' : 'circle'),
                },
                hoverinfo: 'text+name',
                hovertext: arr.map(e => `ID: ${e.id || ''}<br>Type: ${t}<br>x: ${e.x}<br>y: ${e.y}`),
            });
        });
        traces.forEach(t => plotData.push(t));
    //     Plotly.react(mapDiv, plotData, { responsive: true });
    // }

    // // 渲染背景地图 (Plotly)
    // function renderMap() {
    //     if (!ACTIVE_NAME || !DATA_CACHE[ACTIVE_NAME]) {
    //         Plotly.react(mapDiv, [], { title: "暂无数据" });
    //         return;
    //     }
    //     const plotData = [];
    //     const item = DATA_CACHE[ACTIVE_NAME];
        let layout = {
            margin: { t: 20, b: 40, l: 40, r: 10 },
            xaxis: { title: 'x', autorange: true, scaleratio: 1 },
            yaxis: { title: 'y', autorange: true, scaleanchor: "x" },
            legend: { orientation: 'h', x: 0, y: 1.15 },
            plot_bgcolor: '#fafafa'
        };
        const mapInfo = item.map;
        if (mapInfo && mapInfo.grid) {
            const z = mapInfo.grid;
            const xmin = mapInfo.xmin, xmax = mapInfo.xmax;
            const ymin = mapInfo.ymin, ymax = mapInfo.ymax;
            const ny = z.length, nx = Array.isArray(z[0]) ? z[0].length : 0;
            const xcoords = [], ycoords = [];
            for (let i = 0; i < nx; i++) xcoords.push(xmin + (i + 0.5) * (xmax - xmin) / nx);
            for (let j = 0; j < ny; j++) ycoords.push(ymin + (j + 0.5) * (ymax - ymin) / ny);
            plotData.push({
                z: z,
                type: 'heatmap',
                showscale: false,
                zsmooth: 'fast',
                x: xcoords,
                y: ycoords,
                hoverinfo: 'none',
                opacity: 0.7
            });
            layout.xaxis.range = [xmin, xmax];
            layout.yaxis.range = [ymin, ymax];
        }
        Plotly.react(mapDiv, plotData, layout, { responsive: true });
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
            } else {
                log('加载模型失败:', msg.msg || msg);
            }
        } catch (e) {
            log('加载模型请求失败:', e);
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
        try {
            log(`发送指令以开始模拟: 从 ${datasetName} 的第 ${startFrame} 帧开始...`);
            ws.send(JSON.stringify({ action: 'start', dataset_name: datasetName, frame_idx: startFrame }));
            SIMULATION_RUNNING = true;
        } catch (e) {
            log('发送开始模拟指令失败:', e);
        }
    });

    stopSimBtn.addEventListener('click', () => {
        if (!SIMULATION_RUNNING) {
            alert('当前无运行中的模拟!');
            return;
        }
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

})();
