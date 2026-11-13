/* MAHE Web 终端前端：启动字幕 → ui_ready → SSE 事件渲染。
   仅使用原生 API，无外部依赖，离线可用。 */

(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

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
    choiceShown: false,
    scriptPlatform: "posix",
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
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
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
      setText("#boot-hint", "已通知训练模块初始化");
    } catch (err) {
      setText("#boot-hint", `通知失败：${err.message}（后端仍会在超时后继续）`);
    }

    await sleep(320);
    $("#boot").classList.add("fade");
    $("#app").classList.remove("hidden");
    setTimeout(() => $("#boot").remove(), 420);
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
      case "train_start":
        setStatus("training");
        addEvent("[TRAIN] 开始训练");
        break;
      case "train_end":
        addEvent("[TRAIN] 训练结束");
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
    state.selected.clear();
    (snap.devices || []).forEach((d) => state.devices.set(d.ip, d));
    renderDevices();

    // 重连快照只补最近的信息，避免重复堆积
    const stream = $("#info-stream");
    stream.innerHTML = "";
    state.infoCount = 0;
    (snap.info || []).forEach((item) => {
      addInfo(item.kind || "INFO", item.text || "", item.ts, false);
    });

    if (snap.scanning) setScanState("扫描中…", true);

    // 页面在训练模块初始化完成后才打开时，也补弹部署面板
    if (snap.ui_ready && !state.choiceShown &&
        ["ready", "training"].includes(snap.status)) {
      state.choiceShown = true;
      openModal("modal-choice");
    }
  }

  function setStatus(status) {
    state.status = status || "booting";
    const pill = $("#status-pill");
    pill.textContent = STATUS_TEXT[state.status] || state.status.toUpperCase();
    pill.className = `pill ${state.status}`;
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
        empty.textContent = "尚未有节点入网…";
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
        row.append(dot, name, role);
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

  // ---------------- 扫描 / 部署 ----------------
  function onAppReady(ev) {
    state.masterHost = ev.master_host || state.masterHost;
    state.repo = ev.repo || state.repo;
    state.branch = ev.branch || state.branch;
    state.token = ev.token || state.token;
    setText("#meta-master", state.masterHost || "—");
    if (!state.choiceShown) {
      state.choiceShown = true;
      openModal("modal-choice");
    }
  }

  function setScanState(text, busy = false) {
    state.scanning = busy;
    setText("#scan-state", text);
    for (const id of ["#btn-scan", "#btn-scan-2"]) {
      const btn = $(id);
      if (btn) btn.disabled = busy;
    }
  }

  function onScanEvent(ev) {
    if (ev.phase === "start") {
      state.selected.clear();
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
      state.devices = new Map();
      (ev.devices || []).forEach((d) => state.devices.set(d.ip, d));
      renderDevices();
      const ready = (ev.devices || []).filter((d) => d.deployable).length;
      setText("#device-summary", `${state.devices.size} 台在线 · ${ready} 台可自动部署`);
      setScanState("扫描完成");
      return;
    }
    if (ev.phase === "error") {
      setScanState(`扫描失败：${ev.text || ""}`);
    }
  }

  function renderDevices() {
    const body = $("#device-body");
    if (!body) return;
    body.innerHTML = "";
    const devices = Array.from(state.devices.values()).sort((a, b) =>
      a.ip.localeCompare(b.ip, undefined, { numeric: true })
    );
    if (!devices.length) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 6;
      td.className = "empty";
      td.textContent = "暂无设备：点击“扫描”开始发现局域网主机";
      tr.appendChild(td);
      body.appendChild(tr);
      return;
    }

    for (const d of devices) {
      const tr = document.createElement("tr");

      const checkCell = document.createElement("td");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.disabled = !d.deployable;
      cb.checked = state.selected.has(d.ip);
      cb.addEventListener("change", () => {
        if (cb.checked) state.selected.add(d.ip);
        else state.selected.delete(d.ip);
      });
      checkCell.appendChild(cb);

      const nameCell = document.createElement("td");
      nameCell.className = "device-name";
      nameCell.textContent = d.name || "未命名设备";
      nameCell.title = d.note || "";

      const ipCell = document.createElement("td");
      ipCell.className = "device-ip";
      ipCell.textContent = d.ip;

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
      cond.className = `badge ${d.deployable ? "ok" : "warn"}`;
      cond.textContent = d.deployable
        ? `可自动部署（${d.channel}）`
        : d.channel === "winrm"
          ? "需手动（WinRM）"
          : "需手动";
      condCell.appendChild(cond);

      tr.append(checkCell, nameCell, ipCell, typeCell, portsCell, condCell);
      body.appendChild(tr);
    }
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
    $$(".view").forEach((v) =>
      v.classList.toggle("hidden", v.id !== `view-${name}`)
    );
    $$(".nav-item").forEach((b) =>
      b.classList.toggle("active", b.dataset.view === name)
    );
    if (name === "dashboard") drawChart();
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
    $("#btn-stop").addEventListener("click", async () => {
      try {
        await post({ cmd: "stop_training" });
      } catch (err) {
        addInfo("ERROR", `停止训练失败：${err.message}`);
      }
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
    bind();
    connect();
    renderNodes();
    renderDevices();
    runBoot();
  });
})();
