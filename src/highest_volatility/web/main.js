import { DataLayer } from "./data-layer.js";

const dataLayer = new DataLayer({ baseUrl: window.location.origin });
window.hvDataLayer = dataLayer;

const connectionStatusEl = document.getElementById("connection-status");
const pendingCountEl = document.getElementById("pending-count");
const universeListEl = document.getElementById("universe-list");
const annotationsContainerEl = document.getElementById("annotations-container");
const auditListEl = document.getElementById("audit-list");
const refreshUniverseButton = document.getElementById("refresh-universe");
const refreshAnnotationsButton = document.getElementById("refresh-annotations");
const syncButton = document.getElementById("sync-button");
const syncFeedbackEl = document.getElementById("sync-feedback");

const formatTimestamp = (value) => {
  if (!value) {
    return "unknown";
  }
  try {
    const date = new Date(value);
    return date.toLocaleString();
  } catch (error) {
    return value;
  }
};

const updateConnectionStatus = () => {
  const online = navigator.onLine;
  connectionStatusEl.textContent = online ? "Online" : "Offline";
  connectionStatusEl.classList.toggle("online", online);
  connectionStatusEl.classList.toggle("offline", !online);
};

const registerServiceWorker = async () => {
  if (!("serviceWorker" in navigator)) {
    return;
  }
  try {
    await navigator.serviceWorker.register("/service-worker.js", { type: "module" });
    navigator.serviceWorker.addEventListener("message", (event) => {
      const { type, count } = event.data ?? {};
      if (type === "mutation-queue" && typeof count === "number") {
        pendingCountEl.textContent = count.toString();
      }
      if (type === "mutation-stored") {
        syncFeedbackEl.textContent = "Stored offline update. It'll sync when online.";
      }
    });
  } catch (error) {
    console.warn("Service worker registration failed", error);
  }
};

const renderUniverse = (tickers) => {
  universeListEl.innerHTML = "";
  annotationsContainerEl.innerHTML = "";
  tickers.forEach((ticker) => {
    const li = document.createElement("li");
    li.textContent = ticker;
    universeListEl.appendChild(li);

    const card = document.createElement("div");
    card.className = "annotation-card";
    card.dataset.ticker = ticker;

    const heading = document.createElement("h3");
    heading.textContent = ticker;

    const textarea = document.createElement("textarea");
    textarea.name = `note-${ticker}`;
    textarea.placeholder = "Add a note";

    const actions = document.createElement("div");
    actions.className = "actions";

    const saveButton = document.createElement("button");
    saveButton.type = "button";
    saveButton.dataset.action = "save";
    saveButton.dataset.ticker = ticker;
    saveButton.textContent = "Save";

    const status = document.createElement("p");
    status.className = "meta";
    status.dataset.role = "status";
    status.textContent = "No annotations yet.";

    actions.appendChild(saveButton);
    card.appendChild(heading);
    card.appendChild(textarea);
    card.appendChild(actions);
    card.appendChild(status);
    annotationsContainerEl.appendChild(card);
  });
};

const renderAnnotations = ({ items, pending, audit }) => {
  pendingCountEl.textContent = pending.length.toString();
  items.forEach((item) => {
    const card = annotationsContainerEl.querySelector(`[data-ticker="${item.ticker}"]`);
    if (!card) {
      return;
    }
    const textarea = card.querySelector("textarea");
    const status = card.querySelector('[data-role="status"]');
    textarea.value = item.note ?? "";
    if (item.pending) {
      status.textContent = `Pending sync · updated ${formatTimestamp(item.updated_at)}`;
    } else {
      status.textContent = `Synced · updated ${formatTimestamp(item.updated_at)}`;
    }
  });

  auditListEl.innerHTML = "";
  audit.forEach((entry) => {
    const li = document.createElement("li");
    li.classList.add(entry.type ?? "");
    li.textContent = `[${formatTimestamp(entry.timestamp)}] ${entry.detail ?? entry.type}`;
    auditListEl.appendChild(li);
  });
};

const refreshUniverse = async () => {
  try {
    const tickers = await dataLayer.loadUniverse();
    renderUniverse(tickers.slice(0, 12));
  } catch (error) {
    syncFeedbackEl.textContent = `Failed to load universe: ${error.message}`;
  }
  await refreshAnnotations();
};

const refreshAnnotations = async () => {
  const merged = await dataLayer.getMergedAnnotations();
  renderAnnotations(merged);
};

const bindEvents = () => {
  refreshUniverseButton.addEventListener("click", () => {
    refreshUniverse();
  });

  refreshAnnotationsButton.addEventListener("click", () => {
    refreshAnnotations();
  });

  syncButton.addEventListener("click", async () => {
    syncButton.disabled = true;
    syncFeedbackEl.textContent = "Syncing…";
    try {
      const result = await dataLayer.syncPendingMutations();
      if (result.conflicts.length) {
        syncFeedbackEl.textContent = `Sync completed with ${result.conflicts.length} conflict(s).`;
      } else {
        syncFeedbackEl.textContent = "Sync completed.";
      }
    } catch (error) {
      syncFeedbackEl.textContent = `Sync failed: ${error.message}`;
    } finally {
      syncButton.disabled = false;
      await refreshAnnotations();
    }
  });

  annotationsContainerEl.addEventListener("click", async (event) => {
    const target = event.target;
    if (!(target instanceof HTMLButtonElement)) {
      return;
    }
    if (target.dataset.action !== "save") {
      return;
    }
    const card = target.closest(".annotation-card");
    if (!card) {
      return;
    }
    const textarea = card.querySelector("textarea");
    const ticker = target.dataset.ticker;
    if (!ticker) {
      return;
    }
    try {
      await dataLayer.saveAnnotation(ticker, textarea.value);
      syncFeedbackEl.textContent = `Queued note for ${ticker}.`;
    } catch (error) {
      syncFeedbackEl.textContent = `Unable to save note: ${error.message}`;
    } finally {
      await refreshAnnotations();
    }
  });
};

const bootstrap = async () => {
  await registerServiceWorker();
  updateConnectionStatus();
  await refreshUniverse();
};

window.addEventListener("online", async () => {
  updateConnectionStatus();
  syncFeedbackEl.textContent = "Connection restored. Syncing…";
  try {
    await dataLayer.syncPendingMutations();
  } catch (error) {
    syncFeedbackEl.textContent = `Sync failed after reconnect: ${error.message}`;
  } finally {
    await refreshAnnotations();
  }
});

window.addEventListener("offline", () => {
  updateConnectionStatus();
  syncFeedbackEl.textContent = "You are offline. Changes will be queued.";
});

bindEvents();
bootstrap();
