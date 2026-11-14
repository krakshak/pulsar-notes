/* Zero-Trust Local Meeting Assistant — Frontend Application (Zero CDN) */
(function () {
  "use strict";

  const tokenMeta = document.querySelector('meta[name="local-session-token"]');
  const SESSION_TOKEN = tokenMeta ? tokenMeta.getAttribute("content") || "" : "";

  // DOM References
  const themeToggleBtn = document.getElementById("themeToggleBtn");
  const searchInput = document.getElementById("searchInput");
  const tabAll = document.getElementById("tabAll");
  const tabOpen = document.getElementById("tabOpen");
  const meetingList = document.getElementById("meetingList");

  const micSelect = document.getElementById("micSelect");
  const sysSelect = document.getElementById("sysSelect");
  const llmSelect = document.getElementById("llmSelect");
  const startBtn = document.getElementById("startBtn");
  const stopBtn = document.getElementById("stopBtn");
  const discardBtn = document.getElementById("discardBtn");

  const liveTelemetry = document.getElementById("liveTelemetry");
  const micMeter = document.getElementById("micMeter");
  const sysMeter = document.getElementById("sysMeter");
  const timerPill = document.getElementById("timerPill");
  const ramCounter = document.getElementById("ramCounter");
  const toggleDrawerBtn = document.getElementById("toggleDrawerBtn");
  const liveDrawer = document.getElementById("liveDrawer");
  const liveSnippetText = document.getElementById("liveSnippetText");

  const summarizingBanner = document.getElementById("summarizingBanner");
  const summarizingText = document.getElementById("summarizingText");
  const failureBanner = document.getElementById("failureBanner");
  const failureText = document.getElementById("failureText");
  const retryBtn = document.getElementById("retryBtn");
  const retryFallbackBtn = document.getElementById("retryFallbackBtn");
  const discardFailedBtn = document.getElementById("discardFailedBtn");

  const emptyState = document.getElementById("emptyState");
  const meetingCanvas = document.getElementById("meetingCanvas");
  const canvasTitle = document.getElementById("canvasTitle");
  const canvasMeta = document.getElementById("canvasMeta");
  const canvasSummary = document.getElementById("canvasSummary");
  const takeawaysList = document.getElementById("takeawaysList");
  const addTakeawayBtn = document.getElementById("addTakeawayBtn");
  const actionItemsList = document.getElementById("actionItemsList");
  const addActionForm = document.getElementById("addActionForm");
  const newActionOwner = document.getElementById("newActionOwner");
  const newActionTask = document.getElementById("newActionTask");
  const newActionDue = document.getElementById("newActionDue");
  const copyMarkdownBtn = document.getElementById("copyMarkdownBtn");
  const deleteMeetingBtn = document.getElementById("deleteMeetingBtn");
  const toastEl = document.getElementById("toast");

  // Application State
  let meetings = [];
  let selectedMeetingId = null;
  let currentFilter = "all"; // "all" | "open_tasks"
  let currentSearch = "";
  let sessionState = "IDLE";
  let availableModels = [];
  let ws = null;
  let drawerOpen = false;
  let toastTimer = null;

  // =========================================================================
  // Icons & Theme Setup
  // =========================================================================
  function mountIcons() {
    if (!window.LocalIcons) return;
    const { createIcon } = window.LocalIcons;
    document.getElementById("searchIconSlot").appendChild(createIcon("search", 14));
    document.getElementById("micIconSlot").appendChild(createIcon("mic", 14));
    document.getElementById("sysIconSlot").appendChild(createIcon("speaker", 14));
    document.getElementById("emptyShieldSlot").appendChild(createIcon("shield", 40));
    updateThemeIcon();
  }

  function updateThemeIcon() {
    if (!window.LocalIcons) return;
    const isLight = document.documentElement.getAttribute("data-theme") === "light";
    themeToggleBtn.replaceChildren(window.LocalIcons.createIcon(isLight ? "moon" : "sun", 15));
  }

  function initTheme() {
    const saved = localStorage.getItem("local_meeting_theme") || "dark";
    if (saved === "light") {
      document.documentElement.setAttribute("data-theme", "light");
    }
    themeToggleBtn.addEventListener("click", function () {
      const current = document.documentElement.getAttribute("data-theme");
      const next = current === "light" ? "dark" : "light";
      if (next === "light") {
        document.documentElement.setAttribute("data-theme", "light");
      } else {
        document.documentElement.removeAttribute("data-theme");
      }
      localStorage.setItem("local_meeting_theme", next);
      updateThemeIcon();
    });
  }

  function showToast(message) {
    toastEl.textContent = message;
    toastEl.classList.remove("hidden");
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(function () {
      toastEl.classList.add("hidden");
    }, 3000);
  }

  // =========================================================================
  // Authenticated API Client
  // =========================================================================
  async function apiFetch(path, options) {
    const opts = options || {};
    const headers = Object.assign(
      {
        "X-Local-Session-Token": SESSION_TOKEN,
        "Content-Type": "application/json",
      },
      opts.headers || {}
    );
    const resp = await fetch(path, Object.assign({}, opts, { headers }));
    if (!resp.ok) {
      let detail = "Request failed (" + resp.status + ")";
      try {
        const errData = await resp.json();
        if (errData && errData.detail) detail = errData.detail;
      } catch (_) {}
      throw new Error(detail);
    }
    return resp.json();
  }

  // =========================================================================
  // Date / Time Helpers (Local Timezone)
  // =========================================================================
  function formatDuration(seconds) {
    const s = Math.max(0, Math.floor(seconds || 0));
    const h = String(Math.floor(s / 3600)).padStart(2, "0");
    const m = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
    const rem = String(s % 60).padStart(2, "0");
    return h + ":" + m + ":" + rem;
  }

  function formatMeetingDate(isoStr) {
    const dt = new Date(isoStr);
    if (Number.isNaN(dt.getTime())) return isoStr;
    return new Intl.DateTimeFormat(undefined, {
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
    }).format(dt);
  }

  function getDateGroupLabel(isoStr) {
    const dt = new Date(isoStr);
    if (Number.isNaN(dt.getTime())) return "Older";

    const now = new Date();
    const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const startOfTarget = new Date(dt.getFullYear(), dt.getMonth(), dt.getDate());
    const diffDays = Math.round((startOfToday - startOfTarget) / (1000 * 60 * 60 * 24));

    if (diffDays <= 0) return "Today";
    if (diffDays === 1) return "Yesterday";
    if (diffDays < 7) return "Earlier This Week";
    return "Older";
  }

  // =========================================================================
  // Health, Devices & State Machine UI
  // =========================================================================
  async function loadHealth() {
    try {
      const health = await apiFetch("/api/health");
      updateSessionStateUI(health.session_state);
      populateDevices(health.audio_devices || []);
      populateModels(health.available_models || [], health.ollama_reachable);
    } catch (err) {
      showToast("Health check failed: " + err.message);
    }
  }

  function populateDevices(devices) {
    micSelect.replaceChildren();
    sysSelect.replaceChildren();

    const defaultMicOpt = document.createElement("option");
    defaultMicOpt.value = "";
    defaultMicOpt.textContent = "Auto-detect Mic";
    micSelect.appendChild(defaultMicOpt);

    const defaultSysOpt = document.createElement("option");
    defaultSysOpt.value = "";
    defaultSysOpt.textContent = "Auto-detect BlackHole";
    sysSelect.appendChild(defaultSysOpt);

    devices.forEach(function (dev) {
      const opt1 = document.createElement("option");
      opt1.value = String(dev.id);
      opt1.textContent = dev.name + (dev.is_default ? " (Default)" : "");
      if (dev.is_default && !dev.is_blackhole) opt1.selected = true;
      micSelect.appendChild(opt1);

      const opt2 = document.createElement("option");
      opt2.value = String(dev.id);
      opt2.textContent = dev.name + (dev.is_blackhole ? " [System]" : "");
      if (dev.is_blackhole) opt2.selected = true;
      sysSelect.appendChild(opt2);
    });
  }

  function populateModels(models, reachable) {
    availableModels = models;
    llmSelect.replaceChildren();

    const preferred = ["qwen2.5:14b", "llama3.1:8b"];
    const merged = Array.from(new Set(models.length > 0 ? models : preferred));

    merged.forEach(function (name) {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name + (reachable ? "" : " (Offline)");
      llmSelect.appendChild(opt);
    });

    if (models.includes("qwen2.5:14b")) {
      llmSelect.value = "qwen2.5:14b";
    } else if (models.length > 0) {
      llmSelect.value = models[0];
    }
  }

  function updateSessionStateUI(state) {
    sessionState = state;
    const isIdle = state === "IDLE";
    const isRec = state === "RECORDING";
    const isSum = state === "SUMMARIZING";
    const isFail = state === "SUMMARIZATION_FAILED";

    startBtn.classList.toggle("hidden", !isIdle);
    stopBtn.classList.toggle("hidden", !isRec);
    discardBtn.classList.toggle("hidden", !isRec);

    micSelect.disabled = !isIdle;
    sysSelect.disabled = !isIdle;
    llmSelect.disabled = isSum;

    liveTelemetry.classList.toggle("hidden", !isRec);
    liveDrawer.classList.toggle("hidden", !(isRec && drawerOpen));
    summarizingBanner.classList.toggle("hidden", !isSum);
    failureBanner.classList.toggle("hidden", !isFail);

    if (isSum) {
      summarizingText.textContent =
        "Unloading ASR & summarizing locally with " + (llmSelect.value || "local LLM") + " (0 bytes written to disk)...";
    }

    if (isRec) {
      connectWebSocket();
    } else {
      disconnectWebSocket();
    }
  }

  // =========================================================================
  // Protected Live WebSocket Streamer
  // =========================================================================
  function connectWebSocket() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      return;
    }
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = protocol + "//" + window.location.host + "/ws/live?token=" + encodeURIComponent(SESSION_TOKEN);
    ws = new WebSocket(wsUrl);

    ws.onmessage = function (event) {
      try {
        const data = JSON.parse(event.data);
        const micPct = Math.min(100, Math.round((data.mic_rms || 0) * 350));
        const sysPct = Math.min(100, Math.round((data.system_rms || 0) * 350));
        micMeter.style.width = micPct + "%";
        sysMeter.style.width = sysPct + "%";
        timerPill.textContent = formatDuration(data.elapsed_seconds);
        ramCounter.textContent =
          (data.word_count_in_ram || 0) + " words • " + (data.estimated_tokens || 0) + " tokens in RAM";
        if (data.latest_snippet) {
          liveSnippetText.textContent = data.latest_snippet;
        }
      } catch (_) {}
    };

    ws.onclose = function () {
      ws = null;
    };
  }

  function disconnectWebSocket() {
    if (ws) {
      ws.close();
      ws = null;
    }
    micMeter.style.width = "0%";
    sysMeter.style.width = "0%";
    timerPill.textContent = "00:00:00";
    ramCounter.textContent = "0 words • 0 tokens in RAM";
    liveSnippetText.textContent = "Listening for speech...";
  }

  // =========================================================================
  // Meeting List & Canvas Rendering (XSS-Safe DOM APIs)
  // =========================================================================
  async function loadMeetings() {
    const params = new URLSearchParams();
    if (currentSearch.trim()) params.set("search", currentSearch.trim());
    params.set("filter", currentFilter);

    try {
      meetings = await apiFetch("/api/meetings?" + params.toString());
      renderSidebarList();
      if (selectedMeetingId !== null) {
        const current = meetings.find(function (m) { return m.id === selectedMeetingId; });
        if (current) {
          renderMeetingCanvas(current);
        } else if (meetings.length > 0) {
          selectMeeting(meetings[0].id);
        } else {
          selectedMeetingId = null;
          renderEmptyCanvas();
        }
      } else if (meetings.length > 0) {
        selectMeeting(meetings[0].id);
      } else {
        renderEmptyCanvas();
      }
    } catch (err) {
      showToast("Failed to load meetings: " + err.message);
    }
  }

  function renderSidebarList() {
    meetingList.replaceChildren();
    if (meetings.length === 0) {
      const emptyMsg = document.createElement("div");
      emptyMsg.className = "group-header";
      emptyMsg.textContent = "No matching meetings";
      meetingList.appendChild(emptyMsg);
      return;
    }

    const groups = { Today: [], Yesterday: [], "Earlier This Week": [], Older: [] };
    meetings.forEach(function (m) {
      const label = getDateGroupLabel(m.created_at);
      groups[label].push(m);
    });

    ["Today", "Yesterday", "Earlier This Week", "Older"].forEach(function (groupName) {
      const items = groups[groupName];
      if (!items || items.length === 0) return;

      const header = document.createElement("div");
      header.className = "group-header";
      header.textContent = groupName;
      meetingList.appendChild(header);

      items.forEach(function (m) {
        const card = document.createElement("button");
        card.type = "button";
        card.className = "meeting-card" + (m.id === selectedMeetingId ? " active" : "");
        card.addEventListener("click", function () {
          selectMeeting(m.id);
        });

        const titleEl = document.createElement("div");
        titleEl.className = "meeting-card-title";
        titleEl.textContent = m.title;

        const metaEl = document.createElement("div");
        metaEl.className = "meeting-card-meta";

        const dateSpan = document.createElement("span");
        dateSpan.textContent = formatMeetingDate(m.created_at);
        metaEl.appendChild(dateSpan);

        const openCount = (m.action_items || []).filter(function (a) { return !a.done; }).length;
        if (openCount > 0) {
          const badge = document.createElement("span");
          badge.className = "open-badge";
          badge.textContent = openCount + " open";
          metaEl.appendChild(badge);
        }

        card.appendChild(titleEl);
        card.appendChild(metaEl);
        meetingList.appendChild(card);
      });
    });
  }

  function selectMeeting(id) {
    selectedMeetingId = id;
    const meeting = meetings.find(function (m) { return m.id === id; });
    renderSidebarList();
    if (meeting) {
      renderMeetingCanvas(meeting);
    } else {
      renderEmptyCanvas();
    }
  }

  function renderEmptyCanvas() {
    emptyState.classList.remove("hidden");
    meetingCanvas.classList.add("hidden");
  }

  function renderMeetingCanvas(meeting) {
    emptyState.classList.add("hidden");
    meetingCanvas.classList.remove("hidden");

    canvasTitle.value = meeting.title || "";
    canvasSummary.value = meeting.summary || "";

    // Metadata badges
    canvasMeta.replaceChildren();
    const chips = [
      formatMeetingDate(meeting.created_at),
      "Duration: " + formatDuration(meeting.duration_seconds),
      "ASR: " + meeting.asr_model,
      "LLM: " + meeting.llm_model,
    ];
    chips.forEach(function (text) {
      const span = document.createElement("span");
      span.className = "meta-chip";
      span.textContent = text;
      canvasMeta.appendChild(span);
    });

    // Key Takeaways
    renderTakeaways(meeting);

    // Action Items Checklist
    renderActionItems(meeting);
  }

  function renderTakeaways(meeting) {
    takeawaysList.replaceChildren();
    const list = meeting.takeaways || [];
    if (list.length === 0) {
      const none = document.createElement("p");
      none.className = "ram-counter";
      none.textContent = "No key takeaways yet. Add a topic below.";
      takeawaysList.appendChild(none);
      return;
    }

    list.forEach(function (entry, index) {
      if (entry && typeof entry === "object" && Array.isArray(entry.bullets)) {
        const groupLi = document.createElement("li");
        groupLi.className = "takeaway-topic-group";

        const topicRow = document.createElement("div");
        topicRow.className = "takeaway-topic-row";

        const topicInput = document.createElement("input");
        topicInput.type = "text";
        topicInput.className = "takeaway-topic-input";
        topicInput.value = entry.topic || "";
        topicInput.placeholder = "Topic heading";
        topicInput.addEventListener("change", async function () {
          const updated = list.map(function (item, idx) {
            if (idx !== index) return item;
            return {
              topic: topicInput.value.trim() || "Topic",
              bullets: (item.bullets || []).slice(),
            };
          });
          await saveMeetingPatch(meeting.id, { takeaways: updated });
        });

        const addBulletBtn = document.createElement("button");
        addBulletBtn.type = "button";
        addBulletBtn.className = "btn btn-ghost btn-sm";
        addBulletBtn.textContent = "+ Bullet";
        addBulletBtn.title = "Add sub-bullet";
        addBulletBtn.addEventListener("click", async function () {
          const updated = list.map(function (item, idx) {
            if (idx !== index) return item;
            return {
              topic: item.topic,
              bullets: (item.bullets || []).concat(["New point"]),
            };
          });
          await saveMeetingPatch(meeting.id, { takeaways: updated });
        });

        const delTopicBtn = document.createElement("button");
        delTopicBtn.type = "button";
        delTopicBtn.className = "btn btn-ghost btn-sm";
        delTopicBtn.textContent = "×";
        delTopicBtn.title = "Remove topic";
        delTopicBtn.addEventListener("click", async function () {
          const updated = list.filter(function (_, idx) { return idx !== index; });
          await saveMeetingPatch(meeting.id, { takeaways: updated });
        });

        topicRow.appendChild(topicInput);
        topicRow.appendChild(addBulletBtn);
        topicRow.appendChild(delTopicBtn);
        groupLi.appendChild(topicRow);

        const subList = document.createElement("ul");
        subList.className = "takeaway-sublist";

        (entry.bullets || []).forEach(function (bulletText, bIndex) {
          const subLi = document.createElement("li");
          subLi.className = "takeaway-subitem";

          const bulletInput = document.createElement("input");
          bulletInput.type = "text";
          bulletInput.className = "takeaway-input";
          bulletInput.value = bulletText;
          bulletInput.addEventListener("change", async function () {
            const val = bulletInput.value.trim();
            const updated = list.map(function (item, idx) {
              if (idx !== index) return item;
              const nextBullets = (item.bullets || []).slice();
              if (val) {
                nextBullets[bIndex] = val;
              } else {
                nextBullets.splice(bIndex, 1);
              }
              return { topic: item.topic, bullets: nextBullets };
            });
            await saveMeetingPatch(meeting.id, { takeaways: updated });
          });

          const delBulletBtn = document.createElement("button");
          delBulletBtn.type = "button";
          delBulletBtn.className = "btn btn-ghost btn-sm";
          delBulletBtn.textContent = "×";
          delBulletBtn.title = "Remove bullet";
          delBulletBtn.addEventListener("click", async function () {
            const updated = list.map(function (item, idx) {
              if (idx !== index) return item;
              return {
                topic: item.topic,
                bullets: (item.bullets || []).filter(function (_, bi) { return bi !== bIndex; }),
              };
            });
            await saveMeetingPatch(meeting.id, { takeaways: updated });
          });

          subLi.appendChild(bulletInput);
          subLi.appendChild(delBulletBtn);
          subList.appendChild(subLi);
        });

        groupLi.appendChild(subList);
        takeawaysList.appendChild(groupLi);
        return;
      }

      // Legacy flat string takeaway fallback
      const li = document.createElement("li");
      li.className = "takeaway-item";

      const input = document.createElement("input");
      input.type = "text";
      input.className = "takeaway-input";
      input.value = String(entry || "");
      input.addEventListener("change", async function () {
        const updated = list.slice();
        updated[index] = input.value.trim();
        await saveMeetingPatch(meeting.id, { takeaways: updated.filter(Boolean) });
      });

      const delBtn = document.createElement("button");
      delBtn.type = "button";
      delBtn.className = "btn btn-ghost btn-sm";
      delBtn.textContent = "×";
      delBtn.title = "Remove takeaway";
      delBtn.addEventListener("click", async function () {
        const updated = list.filter(function (_, idx) { return idx !== index; });
        await saveMeetingPatch(meeting.id, { takeaways: updated });
      });

      li.appendChild(input);
      li.appendChild(delBtn);
      takeawaysList.appendChild(li);
    });
  }

  function renderActionItems(meeting) {
    actionItemsList.replaceChildren();
    const items = meeting.action_items || [];
    if (items.length === 0) {
      const none = document.createElement("p");
      none.className = "ram-counter";
      none.textContent = "No action items yet. Add one below.";
      actionItemsList.appendChild(none);
      return;
    }

    items.forEach(function (item) {
      const row = document.createElement("div");
      row.className = "action-row" + (item.done ? " done" : "");

      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.className = "action-checkbox";
      checkbox.checked = Boolean(item.done);
      checkbox.addEventListener("change", async function () {
        try {
          const updated = await apiFetch("/api/action-items/" + encodeURIComponent(item.id), {
            method: "PATCH",
            body: JSON.stringify({ done: checkbox.checked }),
          });
          item.done = updated.done;
          row.classList.toggle("done", item.done);
          renderSidebarList();
        } catch (err) {
          checkbox.checked = !checkbox.checked;
          showToast("Failed to update task: " + err.message);
        }
      });

      const rawOwner = (item.owner || "Unassigned").trim();
      const ownerDisplay = rawOwner.startsWith("@") ? rawOwner : "@" + rawOwner;
      const ownerPill = document.createElement("span");
      const isMe = rawOwner.replace(/^@/, "").toLowerCase() === "me";
      ownerPill.className = "owner-pill" + (isMe ? " is-me" : "");
      ownerPill.textContent = "Owner: " + ownerDisplay;

      const taskSpan = document.createElement("span");
      taskSpan.className = "action-task";
      taskSpan.textContent = item.task;

      row.appendChild(checkbox);
      row.appendChild(taskSpan);
      row.appendChild(ownerPill);

      if (item.due_date) {
        const dueEl = document.createElement("span");
        dueEl.className = "due-pill";
        dueEl.textContent = "Deadline: " + item.due_date;
        row.appendChild(dueEl);
      }

      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "btn btn-ghost btn-sm";
      removeBtn.textContent = "×";
      removeBtn.title = "Delete action item";
      removeBtn.addEventListener("click", async function () {
        try {
          await apiFetch("/api/action-items/" + encodeURIComponent(item.id), { method: "DELETE" });
          meeting.action_items = meeting.action_items.filter(function (a) { return a.id !== item.id; });
          renderActionItems(meeting);
          renderSidebarList();
        } catch (err) {
          showToast("Failed to delete action item: " + err.message);
        }
      });

      row.appendChild(removeBtn);
      actionItemsList.appendChild(row);
    });
  }

  async function saveMeetingPatch(meetingId, patch) {
    try {
      const updated = await apiFetch("/api/meetings/" + meetingId, {
        method: "PATCH",
        body: JSON.stringify(patch),
      });
      const idx = meetings.findIndex(function (m) { return m.id === meetingId; });
      if (idx !== -1) meetings[idx] = updated;
      renderSidebarList();
      renderMeetingCanvas(updated);
    } catch (err) {
      showToast("Failed to save changes: " + err.message);
    }
  }

  // =========================================================================
  // Event Listeners
  // =========================================================================
  let searchDebounce = null;
  searchInput.addEventListener("input", function () {
    currentSearch = searchInput.value;
    if (searchDebounce) clearTimeout(searchDebounce);
    searchDebounce = setTimeout(loadMeetings, 180);
  });

  tabAll.addEventListener("click", function () {
    currentFilter = "all";
    tabAll.classList.add("active");
    tabOpen.classList.remove("active");
    loadMeetings();
  });

  tabOpen.addEventListener("click", function () {
    currentFilter = "open_tasks";
    tabOpen.classList.add("active");
    tabAll.classList.remove("active");
    loadMeetings();
  });

  toggleDrawerBtn.addEventListener("click", function () {
    drawerOpen = !drawerOpen;
    liveDrawer.classList.toggle("hidden", !(sessionState === "RECORDING" && drawerOpen));
  });

  // Start Recording
  startBtn.addEventListener("click", async function () {
    try {
      const micId = micSelect.value ? parseInt(micSelect.value, 10) : null;
      const sysId = sysSelect.value ? parseInt(sysSelect.value, 10) : null;
      await apiFetch("/api/meeting/start", {
        method: "POST",
        body: JSON.stringify({ mic_device_id: micId, system_device_id: sysId }),
      });
      updateSessionStateUI("RECORDING");
    } catch (err) {
      showToast("Could not start meeting: " + err.message);
    }
  });

  // Stop & Summarize
  stopBtn.addEventListener("click", async function () {
    updateSessionStateUI("SUMMARIZING");
    try {
      const saved = await apiFetch("/api/meeting/stop", {
        method: "POST",
        body: JSON.stringify({ llm_model: llmSelect.value || "qwen2.5:14b" }),
      });
      updateSessionStateUI("IDLE");
      selectedMeetingId = saved.id;
      await loadMeetings();
      showToast("Meeting saved locally (RAM scrubbed)");
    } catch (err) {
      failureText.textContent = "Summarization failed (transcript safely retained in RAM): " + err.message;
      updateSessionStateUI("SUMMARIZATION_FAILED");
    }
  });

  // Retry Summarization
  async function handleRetry(useFallback) {
    let model = llmSelect.value || "qwen2.5:14b";
    if (useFallback) {
      if (availableModels.includes("llama3.1:8b")) {
        model = "llama3.1:8b";
      } else if (availableModels.length > 0) {
        model = availableModels[availableModels.length - 1];
      } else {
        model = "llama3.1:8b";
      }
      llmSelect.value = model;
    }

    updateSessionStateUI("SUMMARIZING");
    try {
      const saved = await apiFetch("/api/meeting/retry-summarize", {
        method: "POST",
        body: JSON.stringify({ llm_model: model }),
      });
      updateSessionStateUI("IDLE");
      selectedMeetingId = saved.id;
      await loadMeetings();
      showToast("Meeting saved on retry (RAM scrubbed)");
    } catch (err) {
      failureText.textContent = "Retry failed (transcript retained in RAM): " + err.message;
      updateSessionStateUI("SUMMARIZATION_FAILED");
    }
  }

  retryBtn.addEventListener("click", function () { handleRetry(false); });
  retryFallbackBtn.addEventListener("click", function () { handleRetry(true); });

  async function handleDiscard() {
    try {
      await apiFetch("/api/meeting/discard", { method: "POST" });
      updateSessionStateUI("IDLE");
      showToast("Session discarded and RAM buffers scrubbed");
    } catch (err) {
      showToast("Discard failed: " + err.message);
    }
  }

  discardBtn.addEventListener("click", handleDiscard);
  discardFailedBtn.addEventListener("click", handleDiscard);

  // Inline Edits
  canvasTitle.addEventListener("change", function () {
    if (selectedMeetingId !== null && canvasTitle.value.trim()) {
      saveMeetingPatch(selectedMeetingId, { title: canvasTitle.value.trim() });
    }
  });

  canvasSummary.addEventListener("change", function () {
    if (selectedMeetingId !== null) {
      saveMeetingPatch(selectedMeetingId, { summary: canvasSummary.value.trim() });
    }
  });

  addTakeawayBtn.addEventListener("click", function () {
    if (selectedMeetingId === null) return;
    const meeting = meetings.find(function (m) { return m.id === selectedMeetingId; });
    if (!meeting) return;
    const nextList = (meeting.takeaways || []).concat([
      { topic: "New Topic", bullets: ["Key point"] },
    ]);
    saveMeetingPatch(meeting.id, { takeaways: nextList });
  });

  // Add Action Item
  addActionForm.addEventListener("submit", async function (e) {
    e.preventDefault();
    if (selectedMeetingId === null) return;
    const task = newActionTask.value.trim();
    if (!task) return;
    const owner = newActionOwner.value.trim() || "Me";
    const due = newActionDue.value.trim() || null;

    try {
      const created = await apiFetch("/api/meetings/" + selectedMeetingId + "/action-items", {
        method: "POST",
        body: JSON.stringify({ owner: owner, task: task, due_date: due }),
      });
      const meeting = meetings.find(function (m) { return m.id === selectedMeetingId; });
      if (meeting) {
        meeting.action_items.push(created);
        renderActionItems(meeting);
        renderSidebarList();
      }
      newActionTask.value = "";
      newActionDue.value = "";
    } catch (err) {
      showToast("Failed to add action item: " + err.message);
    }
  });

  // Copy as Slack / Markdown
  copyMarkdownBtn.addEventListener("click", async function () {
    const meeting = meetings.find(function (m) { return m.id === selectedMeetingId; });
    if (!meeting) return;

    const lines = [
      "### " + meeting.title,
      "_" + formatMeetingDate(meeting.created_at) + " • Duration: " + formatDuration(meeting.duration_seconds) + "_",
      "",
      "**Summary**",
      meeting.summary,
      "",
    ];

    if (meeting.takeaways && meeting.takeaways.length > 0) {
      lines.push("**Key Takeaways**");
      meeting.takeaways.forEach(function (t) {
        if (t && typeof t === "object" && Array.isArray(t.bullets)) {
          lines.push("• **" + (t.topic || "Topic") + "**");
          t.bullets.forEach(function (b) {
            lines.push("  • " + b);
          });
        } else {
          lines.push("• " + String(t));
        }
      });
      lines.push("");
    }

    if (meeting.action_items && meeting.action_items.length > 0) {
      lines.push("• **Action Items & Owners**");
      meeting.action_items.forEach(function (a) {
        const box = a.done ? "[x] " : "";
        const rawOwner = (a.owner || "Unassigned").trim().replace(/^@/, "");
        const due = a.due_date ? " (Deadline: " + a.due_date + ")" : "";
        lines.push("  • " + box + a.task + " — Owner: @" + rawOwner + due);
      });
    }

    try {
      await navigator.clipboard.writeText(lines.join("\n"));
      showToast("Copied as Markdown to clipboard");
    } catch (_) {
      showToast("Clipboard write failed");
    }
  });

  // Delete Meeting (Secure Delete + WAL Truncate)
  deleteMeetingBtn.addEventListener("click", async function () {
    if (selectedMeetingId === null) return;
    try {
      await apiFetch("/api/meetings/" + selectedMeetingId, { method: "DELETE" });
      selectedMeetingId = null;
      await loadMeetings();
      showToast("Meeting securely deleted from disk");
    } catch (err) {
      showToast("Failed to delete meeting: " + err.message);
    }
  });

  // =========================================================================
  // Boot
  // =========================================================================
  initTheme();
  mountIcons();
  loadHealth();
  loadMeetings();
})();

