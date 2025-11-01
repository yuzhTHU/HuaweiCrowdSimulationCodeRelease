// frontend/static/app.js
(() => {
  // Elements
  const btnLoadDS = document.getElementById("btn_load_dataset");
  const datasetPathInput = document.getElementById("dataset_path");
  const datasetInfo = document.getElementById("dataset_info");
  const btnLoadCK = document.getElementById("btn_load_checkpoint");
  const checkpointInput = document.getElementById("checkpoint_path");
  const checkpointInfo = document.getElementById("checkpoint_info");
  const frameSlider = document.getElementById("frame_slider");
  const frameIdxSpan = document.getElementById("frame_idx");
  const btnRequestFrame = document.getElementById("btn_request_frame");
  const btnStartSim = document.getElementById("btn_start_sim");
  const btnStopSim = document.getElementById("btn_stop_sim");
  const rollStepsInput = document.getElementById("roll_steps");
  const sampleNumInput = document.getElementById("sample_num");
  const futureSpeedInput = document.getElementById("future_5s_speed");
  const simSlidersContainer = document.getElementById("sim_sliders_container");
  const wsStatus = document.getElementById("ws_status");

  // Canvas
  const canvas = document.getElementById("map_canvas");
  const ctx = canvas.getContext("2d");

  // State
  let datasetId = null;
  let checkpointId = null;
  let mapImg = null;
  let mapMeta = null;
  let currentFrameData = null; // payload from backend
  let sims = {}; // sim_id -> { sliderEl, simData, currentStep }
  let ws = null;

  // Colors
  const pedColor = "#1f78b4";
  const vehColor = "#33a02c";
  const simColor = "#e31a1c";
  const highlightColor = "#ff7f00";

  function connectWS() {
    const protocol = (location.protocol === "https:") ? "wss" : "ws";
    ws = new WebSocket(`${protocol}://${location.host}/ws`);
    ws.onopen = () => {
      wsStatus.textContent = "connected";
      console.log("ws opened");
    };
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      handleWSMessage(msg);
    };
    ws.onclose = () => {
      wsStatus.textContent = "disconnected";
      console.log("ws closed, reconnect in 1s");
      setTimeout(connectWS, 1000);
    };
  }

  function handleWSMessage(msg) {
    const type = msg.type;
    if (type === "sim_started") {
      console.log("sim started:", msg.sim_id);
      // create slider UI for this sim
      addSimSlider(msg.sim_id);
    } else if (type === "sim_step") {
      // update slider progress & draw simulated trajectories
      const sim_id = msg.sim_id;
      const step = msg.step;
      const sim_done = msg.sim_done || false;
      const simulated_future = msg.simulated_future || {};
      if (!sims[sim_id]) addSimSlider(sim_id);
      sims[sim_id].currentStep = step;
      sims[sim_id].simData = simulated_future;
      updateSimSliderUI(sim_id, step);
      // draw overlay
      drawCurrentView();
    } else if (type === "sim_finished") {
      console.log("sim finished", msg);
    } else if (type === "frame_info") {
      currentFrameData = msg.payload || msg;
      // draw the frame (history + current)
      drawCurrentView();
    } else if (type === "sim_seek") {
      // show previously simulated state
      const sim_id = msg.sim_id;
      if (!sims[sim_id]) addSimSlider(sim_id);
      sims[sim_id].simData = msg.simulated_future || {};
      sims[sim_id].currentStep = msg.step || 0;
      updateSimSliderUI(sim_id, sims[sim_id].currentStep);
      drawCurrentView();
    }
    // other types ignored for now
  }

  // --- REST helpers ---
  async function postJSON(url, payload) {
    const r = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    return r.json();
  }
  async function getJSON(url) {
    const r = await fetch(url);
    return r.json();
  }

  // --- UI actions ---
  btnLoadDS.addEventListener("click", async () => {
    const path = datasetPathInput.value;
    const res = await postJSON("/api/load_dataset", {dataset_path: path});
    datasetId = res.dataset_id;
    datasetInfo.textContent = `id=${res.dataset_id} frames=${res.n_frames}`;
    // decode map image
    if (res.map_png_b64) {
      const img = new Image();
      img.src = "data:image/png;base64," + res.map_png_b64;
      img.onload = () => {
        mapImg = img;
        mapMeta = res.map_meta;
        drawCurrentView();
      };
    }
    // set slider max
    if (res.n_frames) {
      frameSlider.max = Math.max(0, res.n_frames - 1);
    }
  });

  btnLoadCK.addEventListener("click", async () => {
    const path = checkpointInput.value;
    const res = await postJSON("/api/load_checkpoint", {checkpoint_path: path});
    checkpointId = res.checkpoint_id;
    checkpointInfo.textContent = `id=${checkpointId}`;
  });

  frameSlider.addEventListener("input", () => {
    frameIdxSpan.textContent = frameSlider.value;
  });

  btnRequestFrame.addEventListener("click", async () => {
    if (!datasetId) { alert("请先加载数据集"); return; }
    const frameIdx = parseInt(frameSlider.value);
    // request via websocket (so we get the ws 'frame_info' message) or REST
    ws.send(JSON.stringify({type: "request_frame", dataset_id: datasetId, frame_idx: frameIdx}));
  });

  btnStartSim.addEventListener("click", async () => {
    if (!datasetId || !checkpointId) { alert("需先加载 dataset 与 checkpoint"); return; }
    const payload = {
      dataset_id: datasetId,
      frame_idx: parseInt(frameSlider.value),
      checkpoint_id: checkpointId,
      sample_num: parseInt(sampleNumInput.value),
      roll_step: parseInt(rollStepsInput.value),
      pred_step: 1,
      future_5s_speed: futureSpeedInput.value ? parseFloat(futureSpeedInput.value) : null
    };
    ws.send(JSON.stringify({type: "start_sim", payload}));
  });

  btnStopSim.addEventListener("click", () => {
    // stop the latest sim (or all)
    Object.keys(sims).forEach(sim_id => {
      ws.send(JSON.stringify({type: "stop_sim", sim_id}));
    });
  });

  // --- sim slider UI helpers ---
  function addSimSlider(sim_id) {
    if (sims[sim_id]) return;
    const wrapper = document.createElement("div");
    wrapper.className = "sim_slider_row";
    wrapper.id = `sim_row_${sim_id}`;
    const label = document.createElement("span");
    label.textContent = `Sim ${sim_id}: `;
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = 0;
    slider.max = 100; // update when sim data arrives
    slider.value = 0;
    slider.addEventListener("input", () => {
      // user seeking the simulation
      const step = parseInt(slider.value);
      ws.send(JSON.stringify({type: "seek_sim", sim_id: sim_id, step}));
    });
    const btn = document.createElement("button");
    btn.textContent = "重新从这里开始模拟";
    btn.addEventListener("click", () => {
      // start a new simulation from this step (frontend instructs backend)
      // we will tell backend to start a sim with same dataset & frame but starting at new relative offset.
      // For simplicity we just request backend to start a new sim with same params and using this step as frame offset if you adapted backend.
      const payload = {
        dataset_id,
        frame_idx: parseInt(frameSlider.value) + parseInt(slider.value),
        checkpoint_id,
        roll_step: parseInt(rollStepsInput.value),
        sample_num: parseInt(sampleNumInput.value)
      };
      ws.send(JSON.stringify({type: "start_sim", payload}));
    });
    wrapper.appendChild(label);
    wrapper.appendChild(slider);
    wrapper.appendChild(btn);
    simSlidersContainer.appendChild(wrapper);

    sims[sim_id] = { sliderEl: slider, wrapper, simData: null, currentStep: 0 };
  }

  function updateSimSliderUI(sim_id, step) {
    const s = sims[sim_id];
    if (!s) return;
    // If simData includes total_steps, update max:
    if (s.simData && s.simData.total_steps) {
      s.sliderEl.max = s.simData.total_steps;
    } else {
      // fallback
      s.sliderEl.max = Math.max(s.sliderEl.max, step + 1);
    }
    s.sliderEl.value = step;
  }

  // --- drawing ---
  function clearCanvas() {
    ctx.clearRect(0,0,canvas.width,canvas.height);
  }
  function worldToCanvas(x,y) {
    // Convert world coords (mapMeta.xmin..xmax, ymin..ymax) to canvas pixels
    if (!mapMeta) return [0,0];
    const mx = mapMeta.xmin, Mx = mapMeta.xmax, my = mapMeta.ymin, My = mapMeta.ymax;
    const px = (x - mx) / (Mx - mx) * canvas.width;
    const py = canvas.height - (y - my) / (My - my) * canvas.height; // invert y
    return [px, py];
  }

  function drawMapBg() {
    if (mapImg) {
      ctx.drawImage(mapImg, 0, 0, canvas.width, canvas.height);
    } else {
      ctx.fillStyle = "#cfd8dc";
      ctx.fillRect(0,0,canvas.width,canvas.height);
    }
  }

  function drawCurrentView() {
    clearCanvas();
    drawMapBg();
    // draw history and current from currentFrameData
    if (currentFrameData) {
      // history: {id: [[x,y],...], ...}
      const history = currentFrameData.history || {};
      const current = currentFrameData.current || {};
      const ped_list = currentFrameData.ped_list || [];
      const veh_list = currentFrameData.veh_list || [];

      // draw agent histories
      Object.keys(history).forEach(id => {
        const points = history[id];
        if (!points || points.length===0) return;
        ctx.beginPath();
        for (let i=0;i<points.length;i++){
          const [x,y] = points[i];
          const [px,py] = worldToCanvas(x,y);
          if (i===0) ctx.moveTo(px,py); else ctx.lineTo(px,py);
        }
        ctx.strokeStyle = pedColor;
        ctx.lineWidth = 1;
        ctx.stroke();
      });

      // draw current positions
      Object.keys(current).forEach(id => {
        const [x,y,type] = current[id]; // type maybe "pedestrian"/"vehicle"
        const [px,py] = worldToCanvas(x,y);
        ctx.fillStyle = (type && type.startsWith("veh")) ? vehColor : pedColor;
        ctx.beginPath();
        ctx.arc(px,py,6,0,Math.PI*2);
        ctx.fill();
      });

      // draw ground-truth future (dashed)
      const gt_future = currentFrameData.gt_future || {};
      Object.keys(gt_future).forEach(id =>{
        const pts = gt_future[id];
        if (!pts || pts.length===0) return;
        ctx.setLineDash([4,4]);
        ctx.beginPath();
        for (let i=0;i<pts.length;i++){
          const [x,y] = pts[i];
          const [px,py] = worldToCanvas(x,y);
          if (i===0) ctx.moveTo(px,py); else ctx.lineTo(px,py);
        }
        ctx.strokeStyle = "#666";
        ctx.stroke();
        ctx.setLineDash([]);
      });
    }

    // draw all simulations on top (different color)
    Object.keys(sims).forEach(sim_id => {
      const s = sims[sim_id];
      const simData = s.simData || {};
      // Assumed format: simData: { sample_index: { id: [[x,y],...] } }
      // If you change server format, update here accordingly.
      Object.keys(simData).forEach(sampleIdx => {
        const sample = simData[sampleIdx];
        Object.keys(sample).forEach(id => {
          const pts = sample[id];
          if (!pts || pts.length===0) return;
          ctx.beginPath();
          for (let i=0;i<pts.length;i++){
            const [x,y] = pts[i];
            const [px,py] = worldToCanvas(x,y);
            if (i===0) ctx.moveTo(px,py); else ctx.lineTo(px,py);
          }
          ctx.strokeStyle = simColor;
          ctx.lineWidth = 1.5;
          ctx.stroke();
        });
      });
    });
  }

  // init
  connectWS();
  drawCurrentView();

  // allow clicking map to set destination for one pedestrian (frontend will send dest to backend)
  canvas.addEventListener("click", (ev) => {
    if (!mapMeta) return;
    const rect = canvas.getBoundingClientRect();
    const cx = ev.clientX - rect.left;
    const cy = ev.clientY - rect.top;
    // invert canvas->world
    const mx = mapMeta.xmin, Mx = mapMeta.xmax, my = mapMeta.ymin, My = mapMeta.ymax;
    const x = mx + (cx / canvas.width) * (Mx - mx);
    const y = my + ((canvas.height - cy) / canvas.height) * (My - my);
    // tell backend this is a manual destination (example format)
    const payload = { type: "set_destination", payload: { x, y } };
    // we didn't implement set_destination on server — you can add it. For demo, just print
    console.log("clicked world", x, y);
    // Optionally send to backend:
    // ws.send(JSON.stringify({type: "set_destination", payload: {dataset_id: datasetId, frame_idx: parseInt(frameSlider.value), x, y}}));
  });

})();
