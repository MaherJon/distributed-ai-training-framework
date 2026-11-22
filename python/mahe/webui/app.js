/* MAHE Web 终端前端：启动字幕 → ui_ready → SSE 事件渲染。
   仅使用原生 API，无外部依赖，离线可用。 */

(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  // 刷新页面时恢复“同一标签页”的 UI 状态；训练数据本身来自后端快照。
  const STORE = {
    choice: "mahe.choice",
    view: "mahe.view",
    form: "mahe.form",
    preset: "mahe.preset",
  };

  function readStore(key, fallback) {
    try {
      const raw = sessionStorage.getItem(key);
      return raw === null ? fallback : JSON.parse(raw);
    } catch (_) {
      return fallback;
    }
  }

  function writeStore(key, value) {
    try {
      sessionStorage.setItem(key, JSON.stringify(value));
    } catch (_) {
      /* 隐私模式禁写时忽略 */
    }
  }

  const STATUS_TEXT = {
    booting: "BOOTING",
    initializing: "INIT",
    ready: "READY",
    training: "TRAINING",
    done: "DONE",
    stopped: "STOPPED",
    stopping: "STOPPING",
    error: "ERROR",
  };

  // 训练预设：只改训练量相关的三项，学习率与最少节点保持用户设置
  const PRESETS = {
    quick: { rounds: 10, k_steps: 10, batch: 64 },
    normal: { rounds: 20, k_steps: 20, batch: 64 },
    deep: { rounds: 40, k_steps: 30, batch: 128 },
  };

  const state = {
    status: "booting",
    kernel: "",
    masterHost: "",
    repo: "",
    branch: "",
    token: "",
    rounds: [],
    nodes: new Map(),
    devices: new Map(),
    selected: new Set(),
    infoCount: 0,
    scanning: false,
    choiceShown: readStore(STORE.choice, false),
    scriptPlatform: "posix",
    settings: {},
    datasets: [],
    community: {},
    train: { running: false, rounds_total: 0 },
    agent: { enabled: false },
    preset: readStore(STORE.preset, "normal"),
    uiReady: false,
    pickPending: false,
    showAll: false,
    query: "",
    view: null,
  };

  let linkOpen = false;
  let currentScript = "";

  // ---------------- 基础工具 ----------------
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  async function post(payload) {
    const res = await fetch("/api/command", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    let data = {};
    try {
      data = await res.json();
    } catch (_) {
      data = {};
    }
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return data;
  }

  function nowText() {
    return new Date().toLocaleTimeString("zh-CN", { hour12: false });
  }

  function setText(sel, text) {
    const el = $(sel);
    if (el) el.textContent = text;
  }

  // ---------------- 启动字幕 ----------------
  const BOOT_LINES = [
    ["MAHE", "WEB TERMINAL BUNDLE / 静态资源加载完成"],
    ["CHECK", "loopback channel 127.0.0.1:47830"],
    ["CHECK", "EVENT STREAM / SSE 信道建立"],
    ["CHECK", "COMMAND CHANNEL / JSON 指令通道"],
    ["CHECK", "RENDERER / 仪表盘与信息流"],
    ["CHECK", "UI STATE / 启动状态机就绪"],
    ["WAIT", "等待训练模块初始化信号…"],
    ["DONE", "界面初始化完成，移交训练模块"],
  ];
  BOOT_LINES[1][1] = `loopback channel ${location.host}`;

  async function runBoot() {
    const box = $("#boot-lines");
    const bar = $("#boot-bar");
    const total = BOOT_LINES.length;

    for (let i = 0; i < total; i++) {
      const [tag, msg] = BOOT_LINES[i];
      const row = document.createElement("div");
      row.className = "boot-line";
      const mark = document.createElement("span");
      mark.className = tag === "WAIT" ? "wait" : "ok";
      mark.textContent = tag === "WAIT" ? "[ .. ]" : "[ OK ]";
      const tagEl = document.createElement("span");
      tagEl.className = "tag";
      tagEl.textContent = tag.padEnd(6, " ");
      const msgEl = document.createElement("span");
      msgEl.className = "msg";
      msgEl.textContent = msg;
      row.append(mark, tagEl, msgEl);
      box.appendChild(row);
      bar.style.width = `${Math.round(((i + 1) / total) * 100)}%`;
      await sleep(i === 0 ? 180 : 150);
    }

    setText("#boot-hint", "等待控制面确认…");
    await waitForLink(4000);

    try {
      await post({ cmd: "ui_ready" });
      state.uiReady = true;
      setText("#boot-hint", "已通知训练模块初始化");
    } catch (err) {
      setText("#boot-hint", `通知失败：${err.message}（后端仍会在超时后继续）`);
    }

    await sleep(320);
    const boot = $("#boot");
    if (boot) boot.classList.add("fade");
    showApp();
    setTimeout(() => {
      const el = $("#boot");
      if (el) el.remove();
    }, 420);
  }

  function showApp() {
    $("#app").classList.remove("hidden");
    requestAnimationFrame(() => drawChart());
  }

  async function fetchSnapshot() {
    try {
      const res = await fetch("/api/status", { cache: "no-store" });
      if (!res.ok) return null;
      return await res.json();
    } catch (_) {
      return null;
    }
  }

  /* 首次打开播放启动字幕并触发后端初始化；刷新/重连时直接恢复快照，
     不再重放 init，也不会重置训练状态。 */
  async function boot() {
    bind();
    connect();
    renderNodes();
    renderDevices();
    renderTrainState();

    const snap = await fetchSnapshot();
    if (snap && snap.ui_ready) {
      state.uiReady = true;
      hydrate(snap);
      restoreUiState();
      const el = $("#boot");
      if (el) el.remove();
      showApp();
      addInfo("UI", "已恢复上次会话：训练状态与图表来自引擎快照", false);
      return;
    }
    await runBoot();
    restoreUiState();
  }

  function waitForLink(timeout) {
    return new Promise((resolve) => {
      if (linkOpen) return resolve(true);
      const t0 = Date.now();
      const timer = setInterval(() => {
        if (linkOpen || Date.now() - t0 > timeout) {
          clearInterval(timer);
          resolve(linkOpen);
        }
      }, 100);
    });
  }

  // ---------------- 事件流 ----------------
  function connect() {
    const es = new EventSource("/api/events");
    es.onopen = () => setLink(true);
    es.onerror = () => setLink(false);
    es.onmessage = (e) => {
      let ev;
      try {
        ev = JSON.parse(e.data);
      } catch (_) {
        return;
      }
      handleEvent(ev);
    };
  }

  function setLink(open) {
    linkOpen = open;
    const el = $("#link");
    el.textContent = open ? "◉ LINKED" : "◌ OFFLINE";
    el.classList.toggle("online", open);
    el.classList.toggle("offline", !open);
  }

  function handleEvent(ev) {
    switch (ev.type) {
      case "snapshot":
        hydrate(ev);
        break;
      case "status":
        setStatus(ev.status);
        break;
      case "info":
        addInfo(ev.kind || "INFO", ev.text || "", ev.ts);
        break;
      case "round":
        applyRound(ev);
        break;
      case "node_join":
        upsertNode(ev, true);
        break;
      case "node_dead":
        upsertNode(ev, false);
        break;
      case "node_removed":
        removeNode(ev.id);
        break;
      case "train_start":
        state.train = { running: true, rounds_total: ev.rounds || 0 };
        setStatus("training");
        renderTrainState();
        addEvent("[TRAIN] 开始训练");
        break;
      case "train_end":
        state.train.running = false;
        renderTrainState();
        addEvent("[TRAIN] 训练结束");
        break;
      case "train_state":
        state.train = {
          running: !!ev.running,
          rounds_total: ev.rounds_total || state.train.rounds_total,
        };
        renderTrainState();
        break;
      case "settings":
        applySettings(ev.settings || {}, false);
        break;
      case "datasets":
        applyDatasets(ev.datasets || [], ev.community || {});
        break;
      case "agent":
        applyAgent(ev.agent || ev);
        break;
      case "model_path":
        onModelPathEvent(ev);
        break;
      case "app_ready":
        onAppReady(ev);
        break;
      case "scan":
        onScanEvent(ev);
        break;
      case "deploy":
        onDeployEvent(ev);
        break;
      default:
        break;
    }
  }

  function hydrate(snap) {
    state.status = snap.status || "booting";
    state.kernel = snap.kernel || "";
    state.masterHost = snap.master_host || state.masterHost;
    setStatus(state.status);
    setText("#meta-kernel", state.kernel || "—");
    setText("#meta-master", state.masterHost || "—");

    state.rounds = Array.isArray(snap.rounds) ? snap.rounds.slice() : [];
    state.nodes = new Map();
    (snap.nodes || []).forEach((n) => state.nodes.set(Number(n.id), n));
    renderNodes();
    renderMetrics();
    drawChart();

    state.devices = new Map();
    const keepSelected = new Set(state.selected);
    state.selected.clear();
    (snap.devices || []).forEach((d) => state.devices.set(d.ip, d));
    for (const ip of keepSelected) {
      if (state.devices.has(ip)) state.selected.add(ip);
    }
    renderDevices();

    applySettings(snap.settings || {});
    applyDatasets(snap.datasets || [], snap.community || {});
    state.train = snap.train || { running: false, rounds_total: 0 };
    renderTrainState();
    applyAgent(snap.agent || {});
    applyPickState(snap.pick || { status: "idle" });
    restoreFormDraft();

    // 重连快照只补最近的信息，避免重复堆积
    const stream = $("#info-stream");
    stream.innerHTML = "";
    state.infoCount = 0;
    (snap.info || []).forEach((item) => {
      addInfo(item.kind || "INFO", item.text || "", item.ts, false);
    });

    if (snap.scanning) setScanState("扫描中…", true);

    // 页面在训练模块初始化完成后才打开时，也补弹部署面板
    if (snap.ui_ready && ["ready", "training", "done"].includes(snap.status)) {
      showChoiceOnce();
    }
  }

  // 部署方式选择面板只在“第一次就绪”时弹出；刷新后不再骚扰
  function showChoiceOnce() {
    if (state.choiceShown) return;
    state.choiceShown = true;
    writeStore(STORE.choice, true);
    openModal("modal-choice");
  }

  function setStatus(status) {
    state.status = status || "booting";
    const pill = $("#status-pill");
    pill.textContent = STATUS_TEXT[state.status] || state.status.toUpperCase();
    pill.className = `pill ${state.status}`;
    renderTrainState();
  }

  // ---------------- 信息流 / 事件日志 ----------------
  function addInfo(kind, text, ts, writeEvent = true) {
    const stream = $("#info-stream");
    const row = document.createElement("div");
    row.className = "info-line";
    const tag = document.createElement("span");
    tag.className = `info-kind ${kind}`;
    tag.textContent = `[${kind}]`;
    const body = document.createElement("span");
    body.className = "info-text";
    body.textContent = ` ${text}`;
    row.append(tag, body);
    stream.appendChild(row);

    state.infoCount += 1;
    while (stream.childElementCount > 400) stream.removeChild(stream.firstChild);
    setText("#info-count", String(state.infoCount));
    stream.scrollTop = stream.scrollHeight;

    const time = ts || nowText();
    const line = `[${kind}] ${text}`;
    if (writeEvent) addEvent(line, time);
  }

  function addEvent(text, ts) {
    const time = ts || nowText();
    for (const id of ["#event-log", "#event-log-full"]) {
      const box = $(id);
      if (!box) continue;
      const row = document.createElement("div");
      row.className = "event-row";
      const t = document.createElement("span");
      t.className = "event-ts";
      t.textContent = time;
      const body = document.createElement("span");
      body.className = "event-body";
      body.textContent = text;
      row.append(t, body);
      box.appendChild(row);
      while (box.childElementCount > 200) box.removeChild(box.firstChild);
      box.scrollTop = box.scrollHeight;
    }
  }

  // ---------------- 指标 / 曲线 / 节点 ----------------
  function applyRound(r) {
    state.rounds.push(r);
    if (state.rounds.length > 300) state.rounds.shift();
    renderMetrics();
    drawChart();
    renderTrainState();
    addEvent(
      `[ROUND] ${r.round} · workers=${r.workers} · loss=${fmt(r.eval_loss)} · acc=${fmt(r.eval_acc)}`
    );
  }

  function renderMetrics() {
    const last = state.rounds[state.rounds.length - 1];
    const alive = Array.from(state.nodes.values()).filter((n) => n.alive !== false).length;
    setText("#m-round", last ? String(last.round) : "0");
    setText("#m-loss", last ? fmt(last.eval_loss) : "—");
    setText("#m-acc", last ? fmt(last.eval_acc) : "—");
    setText("#m-nodes", String(alive));
    setText("#meta-round", last ? String(last.round) : "0");
    setText("#meta-nodes", String(alive));
  }

  function fmt(v) {
    return typeof v === "number" ? v.toFixed(4) : "—";
  }

  function upsertNode(ev, alive) {
    const id = Number(ev.id);
    const prev = state.nodes.get(id) || {};
    state.nodes.set(id, {
      ...prev,
      id,
      name: ev.name || prev.name || `node-${id}`,
      cores: ev.cores ?? prev.cores ?? 0,
      score: ev.score ?? prev.score ?? 1,
      alive,
    });
    renderNodes();
    renderMetrics();
    addEvent(`[NODE] ${alive ? "入网" : "离线"} · ${ev.name || `node-${id}`}`);
  }

  function removeNode(id) {
    const nid = Number(id);
    if (!state.nodes.has(nid)) return;
    state.nodes.delete(nid);
    renderNodes();
    renderMetrics();
    addEvent(`[NODE] 移出分组 · node-${nid}`);
  }

  async function kickNode(id) {
    if (!window.confirm(`把 node-${id} 移出当前分组？该节点会收到退出指令。`)) return;
    try {
      await post({ cmd: "kick_node", id });
      removeNode(id);
      addInfo("NODE", `已请求移出 node-${id}`);
    } catch (err) {
      addInfo("ERROR", `移出失败：${err.message}`);
    }
  }

  async function clearNodes() {
    if (!window.confirm("清空当前分组的全部节点？")) return;
    try {
      await post({ cmd: "clear_nodes" });
      addInfo("NODE", "已请求清空节点");
    } catch (err) {
      addInfo("ERROR", `清空失败：${err.message}`);
    }
  }

  function renderNodes() {
    const nodes = Array.from(state.nodes.values()).sort((a, b) => a.id - b.id);
    const alive = nodes.filter((n) => n.alive !== false).length;
    setText("#node-badge", String(alive));
    setText("#cluster-summary", `${alive} 个在线 / 共 ${nodes.length} 个`);

    for (const id of ["#node-list", "#node-list-full"]) {
      const box = $(id);
      if (!box) continue;
      box.innerHTML = "";
      if (!nodes.length) {
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "尚未有节点入网：可在「部署」页扫描局域网，或手动部署子端。";
        box.appendChild(empty);
        continue;
      }
      for (const n of nodes) {
        const row = document.createElement("div");
        row.className = `node-row${n.alive === false ? " dead" : ""}`;
        const dot = document.createElement("span");
        dot.className = "dot";
        dot.textContent = n.alive === false ? "○" : "●";
        const name = document.createElement("div");
        const main = document.createElement("div");
        main.className = "node-name";
        main.textContent = n.name || `node-${n.id}`;
        const sub = document.createElement("div");
        sub.className = "node-sub";
        sub.textContent = `id=${n.id} · cores=${n.cores || 0} · score=${Number(n.score || 0).toFixed(2)}`;
        name.append(main, sub);
        const role = document.createElement("span");
        role.className = "node-role";
        role.textContent = n.alive === false ? "DEAD" : "ALIVE";
        const kick = document.createElement("button");
        kick.className = "icon-btn node-kick";
        kick.textContent = "✕";
        kick.title = "移出分组";
        kick.addEventListener("click", () => kickNode(n.id));
        row.append(dot, name, role, kick);
        box.appendChild(row);
      }
    }
  }

  function drawChart() {
    const canvas = $("#chart");
    if (!canvas) return;
    const wrap = canvas.parentElement;
    const w = Math.max(10, wrap.clientWidth);
    const h = Math.max(10, wrap.clientHeight);
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const pad = { l: 10, r: 10, t: 12, b: 12 };
    const iw = w - pad.l - pad.r;
    const ih = h - pad.t - pad.b;

    ctx.strokeStyle = "rgba(88,224,182,0.12)";
    ctx.lineWidth = 1;
    for (let i = 1; i <= 4; i++) {
      const y = pad.t + (ih * i) / 5;
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(w - pad.r, y);
      ctx.stroke();
    }

    const rounds = state.rounds.filter(
      (r) => typeof r.eval_loss === "number" || typeof r.eval_acc === "number"
    );
    if (rounds.length < 2) {
      setText("#chart-note", "等待训练数据");
      return;
    }
    setText("#chart-note", `最近 ${rounds.length} 轮`);

    const losses = rounds.map((r) => (typeof r.eval_loss === "number" ? r.eval_loss : null));
    const accs = rounds.map((r) => (typeof r.eval_acc === "number" ? r.eval_acc : null));
    const maxLoss = Math.max(1e-6, ...losses.filter((v) => v !== null));

    const px = (i) => pad.l + (iw * i) / (rounds.length - 1);

    const drawSeries = (values, color, scale) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      let started = false;
      values.forEach((v, i) => {
        if (v === null) return;
        const x = px(i);
        const y = pad.t + ih - Math.min(1, Math.max(0, scale(v))) * ih;
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
    };

    drawSeries(losses, "#e0a75a", (v) => v / maxLoss);
    drawSeries(accs, "#58e0b6", (v) => v);
  }

  // ---------------- 训练设置 ----------------
  function applySettings(settings, full = true) {
    state.settings = { ...state.settings, ...settings };
    const path = $("#model-path");
    if (path && state.settings.model_path) path.value = state.settings.model_path;
    if (!full) return;

    const map = {
      "#p-rounds": "rounds",
      "#p-ksteps": "k_steps",
      "#p-batch": "batch",
      "#p-lr": "lr",
      "#p-workers": "min_workers",
    };
    for (const [sel, key] of Object.entries(map)) {
      const el = $(sel);
      if (el && state.settings[key] !== undefined) el.value = state.settings[key];
    }
    renderDatasetSelect();
    renderEstimate();
  }

  function applyDatasets(datasets, community) {
    state.datasets = Array.isArray(datasets) ? datasets : [];
    state.community = community || {};
    let badge = "内置目录";
    if (state.community.configured) {
      badge = state.community.count
        ? `社区已连接 · ${state.community.count} 个`
        : "社区目录未取到，使用内置";
    }
    setText("#community-badge", badge);
    renderDatasetSelect();
    renderDatasetDetail();
    renderEstimate();
  }

  function renderDatasetSelect() {
    const select = $("#dataset-select");
    if (!select) return;
    const current = select.value || state.settings.dataset || "";
    select.innerHTML = "";
    if (!state.datasets.length) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "目录加载中…";
      select.appendChild(opt);
      select.disabled = true;
      return;
    }
    select.disabled = false;
    for (const ds of state.datasets) {
      const opt = document.createElement("option");
      opt.value = ds.id;
      const tag = ds.source === "community" ? "社区" : "内置";
      opt.textContent = `${ds.name} · ${ds.n_classes} 类 / ${ds.size} 样本（${tag}）`;
      select.appendChild(opt);
    }
    const keep = current || state.settings.dataset;
    if (keep && state.datasets.some((d) => d.id === keep)) select.value = keep;
  }

  function currentSpec() {
    const id = $("#dataset-select")?.value || state.settings.dataset;
    return state.datasets.find((d) => d.id === id) || state.datasets[0] || null;
  }

  function renderDatasetDetail() {
    const box = $("#dataset-detail");
    if (!box) return;
    const ds = currentSpec();
    if (!ds) {
      box.textContent = "暂无可用数据集。";
      return;
    }
    const rec = ds.recommended || {};
    box.innerHTML = "";
    const desc = document.createElement("p");
    desc.className = "dataset-desc";
    desc.textContent = ds.description || "无说明";
    const meta = document.createElement("p");
    meta.className = "dataset-meta";
    const author = ds.author ? ` · 作者 ${ds.author}` : "";
    meta.textContent =
      `维度 ${ds.n_features} · ${ds.n_classes} 类 · ${ds.size} 样本${author}` +
      (rec.rounds ? ` · 推荐 ${rec.rounds} 轮 × ${rec.k_steps} 步` : "");
    box.append(desc, meta);
  }

  function readTrainForm() {
    const num = (sel, fallback) => {
      const v = Number($(sel).value);
      return Number.isFinite(v) ? v : fallback;
    };
    return {
      dataset: $("#dataset-select").value,
      model_path: ($("#model-path").value || "").trim(),
      rounds: num("#p-rounds", 20),
      k_steps: num("#p-ksteps", 20),
      batch: num("#p-batch", 64),
      lr: num("#p-lr", 0.5),
      min_workers: num("#p-workers", 1),
    };
  }

  function estimateClient() {
    const ds = currentSpec();
    if (!ds) return null;
    const f = readTrainForm();
    const workers = Math.max(1, f.min_workers);
    const size = Math.max(1, ds.size);
    const perRound = Math.min(size, f.k_steps * f.batch * workers);
    const epochs = (f.rounds * perRound) / size;
    return {
      epochs: Math.round(epochs * 100) / 100,
      steps: f.rounds * f.k_steps,
      perRound,
      size,
    };
  }

  function renderEstimate() {
    const box = $("#estimate");
    if (!box) return;
    const est = estimateClient();
    if (!est) {
      box.textContent = "";
      return;
    }
    let verdict = "训练量适中";
    let cls = "ok";
    if (est.epochs < 1) {
      verdict = "训练量偏少：模型可能还没开始收敛";
      cls = "warn";
    } else if (est.epochs > 200) {
      verdict = "训练量偏大：小数据集上容易过拟合";
      cls = "warn";
    } else if (est.epochs > 50) {
      verdict = "训练量充足：留意验证指标是否还在提升";
      cls = "ok";
    }
    box.className = `estimate ${cls}`;
    box.textContent =
      `总步数 ${est.steps} · 每轮处理 ${est.perRound} 样本 · ` +
      `等效 ${est.epochs} epoch（数据集 ${est.size} 样本）—— ${verdict}`;
  }

  function renderTrainState() {
    const running = state.train.running;
    const last = state.rounds[state.rounds.length - 1];
    const done = last ? last.round : 0;
    const total = state.train.rounds_total || 0;
    const progress = `训练中 ${done}${total ? ` / ${total}` : ""} 轮`;
    setText("#train-state-text",
      running ? progress : (done ? `已结束：${done} 轮` : "未运行"));
    setText("#action-state",
      running ? progress : (STATUS_TEXT[state.status] || "就绪"));
    const start = $("#btn-start");
    const ready = state.uiReady &&
      ["ready", "training", "done", "stopped", "error"].includes(state.status);
    if (start) start.disabled = running || !ready;
    for (const id of ["#btn-stop", "#btn-stop-2"]) {
      const btn = $(id);
      if (btn) btn.disabled = !running;
    }
  }

  async function startTraining() {
    const form = readTrainForm();
    if (!form.model_path) {
      addInfo("WARN", "请先选择模型保存位置");
      switchView("train");
      return;
    }
    saveFormDraft();
    const spec = currentSpec();
    if (spec) form.dataset = spec;
    try {
      const res = await post({ cmd: "start_training", ...form });
      if (res.ok === false) {
        addInfo("ERROR", res.error || "启动训练失败");
        return;
      }
      state.train.running = true;
      state.train.rounds_total = form.rounds;
      setStatus("training");
      renderTrainState();
      addInfo("TRAIN", "训练指令已下发");
      switchView("dashboard");
    } catch (err) {
      addInfo("ERROR", `启动训练失败：${err.message}`);
    }
  }

  async function stopTraining() {
    try {
      await post({ cmd: "stop_training" });
    } catch (err) {
      addInfo("ERROR", `停止训练失败：${err.message}`);
    }
  }

  let pickTimer = null;

  function setPickPending(pending) {
    state.pickPending = pending;
    const btn = $("#btn-pick-path");
    if (btn) {
      btn.disabled = pending;
      btn.textContent = pending ? "等待对话框…" : "浏览…";
    }
    if (pickTimer) {
      clearTimeout(pickTimer);
      pickTimer = null;
    }
    if (pending) {
      // 兜底：对话框卡住时不让按钮永远转圈
      pickTimer = setTimeout(() => {
        setPickPending(false);
        addInfo("WARN", "系统对话框长时间无响应：请查看引擎主机任务栏，或直接输入路径");
      }, 180000);
    }
  }

  function applyPickState(pick) {
    if (!pick) return;
    if (pick.status === "pending") {
      setPickPending(true);
    } else {
      setPickPending(false);
      if (pick.status === "ok" && pick.path) {
        $("#model-path").value = pick.path;
        state.settings.model_path = pick.path;
      }
    }
  }

  function onModelPathEvent(ev) {
    if (ev.status === "pending") {
      setPickPending(true);
      return;
    }
    setPickPending(false);
    if (ev.status === "ok" && ev.path) {
      $("#model-path").value = ev.path;
      state.settings.model_path = ev.path;
      saveFormDraft();
    } else if (ev.status === "cancelled") {
      $("#model-path")?.focus();
    } else {
      $("#model-path")?.focus();
    }
    // 具体提示由后端 emit_info 统一输出，避免前端重复一条
  }

  async function pickModelPath() {
    if (state.pickPending) return;
    setPickPending(true);
    try {
      const res = await post({ cmd: "pick_model_path",
                               current: $("#model-path").value.trim() });
      if (res.ok === false || res.status === "busy" ||
          res.status === "unavailable" || res.status === "error") {
        setPickPending(false);
        addInfo("WARN", res.detail || res.error ||
          "无法打开系统对话框，请直接输入路径");
        $("#model-path")?.focus();
      }
      // status === "pending"：结果稍后通过 model_path 事件回传
    } catch (err) {
      setPickPending(false);
      addInfo("WARN", `无法打开文件对话框：${err.message}（可直接输入路径）`);
      $("#model-path")?.focus();
    }
  }

  async function refreshDatasets() {
    const btn = $("#btn-refresh-datasets");
    if (btn) btn.disabled = true;
    try {
      const res = await post({ cmd: "refresh_datasets" });
      if (res.ok === false) {
        addInfo("WARN", res.error || "目录刷新失败");
        return;
      }
      applyDatasets(res.datasets || [], res.community || {});
      addInfo("INFO", `数据集目录已刷新：${(res.datasets || []).length} 个`);
    } catch (err) {
      addInfo("ERROR", `目录刷新失败：${err.message}`);
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  const FORM_FIELDS = {
    dataset: "#dataset-select",
    model_path: "#model-path",
    rounds: "#p-rounds",
    k_steps: "#p-ksteps",
    batch: "#p-batch",
    lr: "#p-lr",
    min_workers: "#p-workers",
  };

  function markPreset(name) {
    $$(".preset").forEach((b) =>
      b.classList.toggle("active", b.dataset.preset === name)
    );
  }

  function saveFormDraft() {
    const draft = {};
    for (const [key, sel] of Object.entries(FORM_FIELDS)) {
      const el = $(sel);
      if (el) draft[key] = el.value;
    }
    writeStore(STORE.form, draft);
  }

  /* 刷新后恢复标签页自己的输入草稿；训练数据由后端快照恢复。
     快照重连也会覆盖表单，所以 hydrate 里同样要再套一次草稿。 */
  function restoreFormDraft() {
    const draft = readStore(STORE.form, null);
    if (draft && typeof draft === "object") {
      for (const [key, sel] of Object.entries(FORM_FIELDS)) {
        const el = $(sel);
        if (el && typeof draft[key] === "string" && draft[key] !== "") {
          el.value = draft[key];
        }
      }
      const select = $("#dataset-select");
      if (select && !Array.from(select.options).some((o) => o.value === select.value)) {
        select.selectedIndex = 0;
      }
      renderDatasetDetail();
      renderEstimate();
    }
    const preset = readStore(STORE.preset, "");
    if (preset && PRESETS[preset]) {
      state.preset = preset;
      markPreset(preset);
    }
  }

  function restoreUiState() {
    restoreFormDraft();
    const view = readStore(STORE.view, "dashboard");
    const known = ["dashboard", "train", "cluster", "deploy", "agent", "logs"];
    switchView(known.includes(view) ? view : "dashboard");
  }

  function applyPreset(name) {
    const preset = PRESETS[name];
    if (!preset) return;
    state.preset = name;
    writeStore(STORE.preset, name);
    $("#p-rounds").value = preset.rounds;
    $("#p-ksteps").value = preset.k_steps;
    $("#p-batch").value = preset.batch;
    markPreset(name);
    saveFormDraft();
    renderEstimate();
  }

  // ---------------- 智能助手 ----------------
  function applyAgent(info) {
    state.agent = info || {};
    const enabled = !!state.agent.enabled;
    const input = $("#agent-input");
    const btn = $("#btn-agent");
    if (input) {
      input.disabled = !enabled;
      input.placeholder = enabled
        ? "例如：我的训练量够吗？"
        : "智能助手待启用";
    }
    if (btn) btn.disabled = !enabled;
    setText("#agent-state", enabled
      ? `已启用 · ${state.agent.model || "deepseek"}`
      : "待启用");
    setText("#agent-hint", state.agent.hint ||
      "设置环境变量 DEEPSEEK_API_KEY 并重启引擎后启用。");
  }

  function appendChat(role, text) {
    const box = $("#agent-log");
    if (!box) return;
    const row = document.createElement("div");
    row.className = `chat-row ${role}`;
    row.textContent = text;
    box.appendChild(row);
    box.scrollTop = box.scrollHeight;
  }

  async function sendAgent() {
    const input = $("#agent-input");
    const text = (input.value || "").trim();
    if (!text) return;
    input.value = "";
    appendChat("user", text);
    const btn = $("#btn-agent");
    if (btn) btn.disabled = true;
    try {
      const history = Array.from($$("#agent-log .chat-row"))
        .slice(-9, -1)
        .map((el) => ({
          role: el.classList.contains("user") ? "user" : "assistant",
          content: el.textContent,
        }));
      const res = await post({ cmd: "agent_ask", text, history });
      if (res.ok === false) {
        appendChat("bot error", res.error || "助手暂不可用");
        return;
      }
      appendChat("bot", res.reply || "（空回复）");
    } catch (err) {
      appendChat("bot error", `请求失败：${err.message}`);
    } finally {
      if (btn && state.agent.enabled) btn.disabled = false;
    }
  }

  // ---------------- 扫描 / 部署 ----------------
  function onAppReady(ev) {
    state.masterHost = ev.master_host || state.masterHost;
    state.repo = ev.repo || state.repo;
    state.branch = ev.branch || state.branch;
    state.token = ev.token || state.token;
    setText("#meta-master", state.masterHost || "—");
    showChoiceOnce();
  }

  function setScanState(text, busy = false) {
    state.scanning = busy;
    setText("#scan-state", text);
    for (const id of ["#btn-scan", "#btn-scan-2"]) {
      const btn = $(id);
      if (btn) btn.disabled = busy;
    }
  }

  const OS_LABELS = { windows: "Windows", linux: "Linux / Unix", macos: "macOS" };

  function onScanEvent(ev) {
    if (ev.phase === "start") {
      state.selected.clear();
      state.devices = new Map();
      renderDevices();
      setScanState("扫描中…", true);
      setText("#device-summary", "");
      return;
    }
    if (ev.phase === "device" && ev.device) {
      state.devices.set(ev.device.ip, ev.device);
      renderDevices();
      return;
    }
    if (ev.phase === "done") {
      const devices = ev.devices || [];
      state.devices = new Map();
      state.selected.clear();
      devices.forEach((d) => state.devices.set(d.ip, d));
      renderDevices();
      const candidates = devices.filter((d) => d.candidate).length;
      const ready = devices.filter((d) => d.deployable).length;
      setText("#device-summary",
        `${candidates} 台候选 · ${ready} 台可自动部署 · ` +
        `${devices.length - candidates} 台非目标已折叠`);
      setScanState("扫描完成");
    }
    if (ev.phase === "error") {
      setScanState(`扫描失败：${ev.text || ""}`);
    }
  }

  function allDevices() {
    return Array.from(state.devices.values()).sort((a, b) =>
      a.ip.localeCompare(b.ip, undefined, { numeric: true })
    );
  }

  function deviceSearchText(d) {
    return [d.name, d.hostname, d.netbios, d.ip, d.mac, d.vendor, d.type, d.os]
      .filter(Boolean).join(" ").toLowerCase();
  }

  function visibleDevices() {
    const q = state.query.trim().toLowerCase();
    return allDevices().filter((d) => {
      if (!state.showAll && !d.candidate) return false;
      return !q || deviceSearchText(d).includes(q);
    });
  }

  function syncCheckAll(list) {
    const box = $("#check-all");
    if (!box) return;
    const targets = (list || visibleDevices()).filter((d) => d.deployable);
    const selected = targets.filter((d) => state.selected.has(d.ip)).length;
    box.disabled = targets.length === 0;
    box.checked = targets.length > 0 && selected === targets.length;
    box.indeterminate = selected > 0 && selected < targets.length;
  }

  function deviceRow(d) {
    const tr = document.createElement("tr");
    tr.className = d.candidate ? "device-row" : "device-row not-target";

    const checkCell = document.createElement("td");
    checkCell.className = "col-check";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.disabled = !d.deployable;
    cb.checked = state.selected.has(d.ip);
    cb.title = d.deployable
      ? "选择该设备以自动部署"
      : "该设备不支持 SSH 自动部署，可下载手动脚本";
    cb.addEventListener("change", () => {
      if (cb.checked) state.selected.add(d.ip);
      else state.selected.delete(d.ip);
      syncCheckAll(visibleDevices());
    });
    checkCell.appendChild(cb);

    const nameCell = document.createElement("td");
    nameCell.className = "device-name";
    const main = document.createElement("span");
    main.className = "device-main";
    main.textContent = d.name || "未命名设备";
    if (d.mahe) {
      const tag = document.createElement("span");
      tag.className = "device-tag";
      tag.textContent = "MAHE";
      main.appendChild(tag);
    }
    const sub = document.createElement("span");
    sub.className = "device-sub";
    const bits = [d.vendor, OS_LABELS[d.os],
                  d.candidate ? "" : "非训练目标"].filter(Boolean);
    sub.textContent = bits.length ? bits.join(" · ") : "厂商未知";
    nameCell.append(main, sub);
    nameCell.title = d.reason || d.note || "";

    const ipCell = document.createElement("td");
    ipCell.className = "device-ip";
    ipCell.textContent = d.ip;
    const mac = document.createElement("span");
    mac.className = "device-mac";
    mac.textContent = d.mac || "MAC 未知";
    ipCell.appendChild(mac);

    const typeCell = document.createElement("td");
    const typeBadge = document.createElement("span");
    typeBadge.className = "badge";
    typeBadge.textContent = d.type || "未知设备";
    typeCell.appendChild(typeBadge);

    const portsCell = document.createElement("td");
    portsCell.className = "ports";
    portsCell.textContent = (d.ports || []).length ? d.ports.join(", ") : "—";

    const condCell = document.createElement("td");
    const cond = document.createElement("span");
    let cls = "warn";
    let label = "需手动部署";
    if (d.mahe) {
      cls = "ok";
      label = "已运行 MAHE";
    } else if (d.deployable) {
      cls = "ok";
      label = "可自动部署（SSH）";
    } else if (d.channel === "winrm") {
      label = "需手动（WinRM）";
    } else if (!["computer", "nas"].includes(d.kind)) {
      label = "非训练目标";
    }
    cond.className = `badge ${cls}`;
    cond.textContent = label;
    cond.title = d.note || "";
    condCell.appendChild(cond);

    tr.append(checkCell, nameCell, ipCell, typeCell, portsCell, condCell);
    return tr;
  }

  function renderDevices() {
    const body = $("#device-body");
    if (!body) return;
    body.innerHTML = "";
    const all = allDevices();
    const list = visibleDevices();
    const candidates = all.filter((d) => d.candidate).length;
    const ready = all.filter((d) => d.deployable).length;
    setText("#device-summary", all.length
      ? `${candidates} 台候选 · ${ready} 台可自动部署 · ` +
        `${all.length - candidates} 台非目标已折叠`
      : "");
    setText("#filter-count", all.length ? `显示 ${list.length} / ${all.length} 台` : "");

    if (!list.length) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 6;
      td.className = "empty";
      td.textContent = all.length
        ? "当前筛选没有匹配设备。"
        : "暂无设备：点击“扫描”发现局域网主机；默认只显示可部署的候选设备。";
      tr.appendChild(td);
      body.appendChild(tr);
    } else {
      for (const d of list) body.appendChild(deviceRow(d));
    }
    syncCheckAll(list);
  }

  async function startScan() {
    if (state.scanning) return;
    switchView("deploy");
    const subnet = ($("#subnet").value || "").trim();
    setScanState("扫描中…", true);
    try {
      await post({ cmd: "scan_start", subnet });
    } catch (err) {
      setScanState(`扫描请求失败：${err.message}`, false);
    }
  }

  function deployLog(text) {
    const el = $("#deploy-log");
    el.textContent += `${text}\n`;
    el.scrollTop = el.scrollHeight;
  }

  async function deploySelected() {
    const user = ($("#ssh-user").value || "").trim();
    const targets = Array.from(state.devices.values()).filter((d) =>
      state.selected.has(d.ip)
    );
    if (!targets.length) {
      deployLog("[WARN] 请先勾选可自动部署的设备");
      return;
    }
    deployLog(`[INFO] 部署 ${targets.length} 台设备…`);
    for (const dev of targets) {
      if (dev.channel !== "ssh") {
        deployLog(`[SKIP] ${dev.ip} 未开放 SSH，请使用手动部署脚本`);
        continue;
      }
      try {
        await post({ cmd: "deploy", device: dev, user });
      } catch (err) {
        deployLog(`[ERROR] ${dev.ip} 指令失败：${err.message}`);
      }
    }
  }

  function onDeployEvent(ev) {
    if (ev.phase === "start") {
      deployLog(`[DEPLOY] ${ev.text || "开始部署"}`);
      switchView("deploy");
    } else if (ev.phase === "log") {
      deployLog(ev.text || "");
    } else if (ev.phase === "done") {
      deployLog(`[OK] ${ev.text || "部署完成"}`);
    } else if (ev.phase === "error") {
      deployLog(`[ERROR] ${ev.text || "部署失败"}`);
    }
  }

  // ---------------- 手动部署 ----------------
  async function openManual() {
    openModal("modal-manual");
    await loadScript(state.scriptPlatform);
    const meta = [
      `Master ${state.masterHost || "—"}`,
      `仓库 ${state.repo || "默认"}`,
      `分支 ${state.branch || "main"}`,
    ].join(" · ");
    setText("#manual-meta", `${meta}。脚本会在目标机浅克隆源码、执行 build.py 并后台启动 Worker。`);
  }

  async function loadScript(platform) {
    state.scriptPlatform = platform;
    $$(".tab").forEach((t) =>
      t.classList.toggle("active", t.dataset.platform === platform)
    );
    setText("#script-view", "加载中…");
    try {
      const res = await fetch(`/api/deploy/script?platform=${platform}`);
      currentScript = await res.text();
      setText("#script-view", currentScript);
    } catch (err) {
      setText("#script-view", `脚本获取失败：${err.message}`);
    }
  }

  async function copyScript() {
    try {
      await navigator.clipboard.writeText(currentScript);
      setText("#btn-copy", "已复制");
    } catch (_) {
      const ta = document.createElement("textarea");
      ta.value = currentScript;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
      setText("#btn-copy", "已复制");
    }
    setTimeout(() => setText("#btn-copy", "复制"), 1200);
  }

  function downloadScript() {
    const ext = state.scriptPlatform === "windows" ? "ps1" : "sh";
    const blob = new Blob([currentScript], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `mahe-worker-${state.scriptPlatform}.${ext}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  // ---------------- 视图 / 弹窗 ----------------
  function switchView(name) {
    state.view = name;
    writeStore(STORE.view, name);
    $$(".view").forEach((v) =>
      v.classList.toggle("hidden", v.id !== `view-${name}`)
    );
    $$(".nav-item").forEach((b) =>
      b.classList.toggle("active", b.dataset.view === name)
    );
    if (name === "dashboard") drawChart();
    if (name === "train") renderEstimate();
    if (name === "deploy") renderDevices();
  }

  function openModal(id) {
    $(`#${id}`).classList.remove("hidden");
  }

  function closeModal(id) {
    $(`#${id}`).classList.add("hidden");
  }

  function bind() {
    $$(".nav-item").forEach((btn) =>
      btn.addEventListener("click", () => switchView(btn.dataset.view))
    );
    $$("[data-close]").forEach((btn) =>
      btn.addEventListener("click", () => closeModal(btn.dataset.close))
    );
    $$(".modal").forEach((m) =>
      m.addEventListener("click", (e) => {
        if (e.target === m) m.classList.add("hidden");
      })
    );
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") $$(".modal").forEach((m) => m.classList.add("hidden"));
    });

    for (const id of ["#btn-scan", "#btn-scan-2"]) {
      const btn = $(id);
      if (btn) btn.addEventListener("click", startScan);
    }
    for (const id of ["#btn-manual", "#btn-manual-2"]) {
      const btn = $(id);
      if (btn) btn.addEventListener("click", openManual);
    }
    $("#btn-deploy").addEventListener("click", deploySelected);
    $("#btn-copy").addEventListener("click", copyScript);
    $("#btn-download").addEventListener("click", downloadScript);
    for (const id of ["#btn-stop", "#btn-stop-2"]) {
      const btn = $(id);
      if (btn) btn.addEventListener("click", stopTraining);
    }

    $("#btn-go-train").addEventListener("click", () => switchView("train"));
    $("#btn-start").addEventListener("click", startTraining);
    $("#btn-pick-path").addEventListener("click", pickModelPath);
    $("#btn-refresh-datasets").addEventListener("click", refreshDatasets);
    $("#btn-clear-nodes").addEventListener("click", clearNodes);
    $("#dataset-select").addEventListener("change", () => {
      renderDatasetDetail();
      renderEstimate();
      saveFormDraft();
    });
    $$(".preset").forEach((btn) =>
      btn.addEventListener("click", () => applyPreset(btn.dataset.preset))
    );
    for (const sel of ["#p-rounds", "#p-ksteps", "#p-batch", "#p-lr", "#p-workers"]) {
      $(sel).addEventListener("input", () => {
        renderEstimate();
        saveFormDraft();
      });
    }
    for (const id of ["#model-path"]) {
      const el = $(id);
      if (el) {
        el.addEventListener("input", saveFormDraft);
        el.addEventListener("change", saveFormDraft);
      }
    }

    const showAll = $("#show-all");
    if (showAll) {
      showAll.addEventListener("change", () => {
        state.showAll = showAll.checked;
        renderDevices();
      });
    }
    const search = $("#device-search");
    if (search) {
      search.addEventListener("input", () => {
        state.query = search.value || "";
        renderDevices();
      });
    }
    const checkAll = $("#check-all");
    if (checkAll) {
      checkAll.addEventListener("change", () => {
        for (const d of visibleDevices()) {
          if (!d.deployable) continue;
          if (checkAll.checked) state.selected.add(d.ip);
          else state.selected.delete(d.ip);
        }
        renderDevices();
      });
    }

    $("#btn-agent").addEventListener("click", sendAgent);
    $("#agent-input").addEventListener("keydown", (e) => {
      if (e.key === "Enter") sendAgent();
    });

    $$(".tab").forEach((tab) =>
      tab.addEventListener("click", () => loadScript(tab.dataset.platform))
    );

    $("#choice-scan").addEventListener("click", () => {
      closeModal("modal-choice");
      startScan();
    });
    $("#choice-manual").addEventListener("click", () => {
      closeModal("modal-choice");
      openManual();
    });
    $("#choice-later").addEventListener("click", () => closeModal("modal-choice"));

    const ro = new ResizeObserver(() => drawChart());
    const wrap = document.querySelector(".chart-wrap");
    if (wrap) ro.observe(wrap);
  }

  // ---------------- 启动 ----------------
  document.addEventListener("DOMContentLoaded", () => {
    boot();
  });
})();
