let currentFrame = 0;
let numFrames = 0;
let H=0, W=0;
let simTrace = [];
let socket = null;

async function init() {
  document.getElementById("load").onclick = async () => {
    const meta = await (await fetch("/load_dataset")).json();
    await fetch("/load_model");
    numFrames = meta.num_frames;
    H = meta.H; W = meta.W;
    document.getElementById("frame").max = numFrames - 1;
    await updateFrame(0);
  };

  document.getElementById("frame").oninput = async (e) => {
    currentFrame = parseInt(e.target.value);
    await updateFrame(currentFrame);
  };

  document.getElementById("simulate").onclick = startSim;
  document.getElementById("stop").onclick = stopSim;

  // Plotly click事件
  document.getElementById("plot").on('plotly_click', async (data)=>{
    if(data.points.length > 0){
      let trace = data.points[0].data;
      let pointIndex = data.points[0].pointIndex;
      let agent_id = trace.text ? trace.text[pointIndex] : null;
      if(agent_id){
        const res = await fetch(`/agent/${agent_id}`);
        const agentData = await res.json();
        drawAgentTrajectory(agentData);
      }
    }
  });

  Plotly.newPlot("plot", [], {xaxis:{}, yaxis:{scaleanchor:"x", scaleratio:1}});
}

async function updateFrame(f) {
  const res = await fetch(`/frame/${f}`);
  const data = await res.json();
  drawFrame(data);
}

function drawFrame(data) {
  const agents = data.agents;
  const mapArr = data.map;
  const mapLayer = {
      type:'heatmap',
      z: mapArr,
      showscale:false,
      colorscale:[[0,'white'],[1,'black']],
      xgap:0, ygap:0
  };

  const ped = {x:[], y:[], mode:'markers', type:'scattergl', marker:{color:'blue', size:5}, text:[], name:'ped'};
  const veh = {x:[], y:[], mode:'markers', type:'scattergl', marker:{color:'green', size:6}, text:[], name:'veh'};
  for (const [id,info] of Object.entries(agents)) {
    const [x,y] = info.positions;
    if (info.type==='ped') { ped.x.push(x); ped.y.push(y); ped.text.push(id); }
    else { veh.x.push(x); veh.y.push(y); veh.text.push(id); }
  }
  const sim = simTrace.length>0 ? {x:simTrace.map(p=>p.x), y:simTrace.map(p=>p.y),
                                   mode:'lines', type:'scattergl', line:{color:'magenta'}, name:'sim'} : null;
  const traces = sim ? [mapLayer, ped, veh, sim] : [mapLayer, ped, veh];
  Plotly.react("plot", traces, {xaxis:{range:[0,W]}, yaxis:{range:[H,0]}, margin:{t:0}});
}

function drawAgentTrajectory(agentData){
  let x=[], y=[];
  for(const pos of agentData.positions){
    x.push(pos[0]);
    y.push(pos[1]);
  }
  const trace = {x:x, y:y, mode:'lines+markers', type:'scattergl', line:{color:'red'}, marker:{size:4}, name:`${agentData.agent_id}`};
  Plotly.addTraces("plot", trace);
  document.getElementById("agent-info").innerHTML = `<b>Agent:</b> ${agentData.agent_id} (${agentData.type})`;
}

async function startSim() {
  simTrace = [];
  if (!socket) {
    socket = new WebSocket(`ws://101.6.69.111:8000/ws`);
    socket.onmessage = (msg) => {
      const data = JSON.parse(msg.data);
      for (const [id, pos] of Object.entries(data.future_step)) {
        simTrace.push({x:pos[0], y:pos[1]});
      }
      drawFrame({agents:{}, map:Array(H).fill(Array(W).fill(0))}); // map简单画空白，可根据需要优化
    };
  }
  await fetch("/simulate/start", {method:"POST"});
}

async function stopSim() {
  await fetch("/simulate/stop", {method:"POST"});
}

window.onload = init;
