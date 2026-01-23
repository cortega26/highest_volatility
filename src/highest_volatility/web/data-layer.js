import {
  addMutation,
  deleteMutation,
  getAllMutations,
  appendAudit,
  getAuditTrail,
} from "./db.js";

const nowIsoString = () => new Date().toISOString();
const API_KEY_STORAGE = "hvApiKey";

const readStoredApiKey = () => {
  try {
    if (typeof window === "undefined" || !window.localStorage) {
      return null;
    }
    return window.localStorage.getItem(API_KEY_STORAGE);
  } catch (_error) {
    return null;
  }
};

export class DataLayer {
  constructor({ baseUrl, apiKey } = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.apiKey = apiKey ?? readStoredApiKey();
  }

  setApiKey(apiKey) {
    this.apiKey = apiKey ?? null;
    try {
      if (typeof window === "undefined" || !window.localStorage) {
        return;
      }
      if (this.apiKey) {
        window.localStorage.setItem(API_KEY_STORAGE, this.apiKey);
      } else {
        window.localStorage.removeItem(API_KEY_STORAGE);
      }
    } catch (_error) {
      // Ignore storage failures (private browsing, policy, etc.).
    }
  }

  async loadUniverse() {
    const payload = await this.#fetchJson("/universe");
    return payload.tickers ?? [];
  }

  async loadAnnotations() {
    try {
      const payload = await this.#fetchJson("/annotations");
      return Array.isArray(payload) ? payload : [];
    } catch (error) {
      await appendAudit({
        timestamp: nowIsoString(),
        type: "annotations_error",
        detail: String(error?.message ?? error),
      });
      return [];
    }
  }

  async saveAnnotation(ticker, note) {
    const sanitizedTicker = String(ticker ?? "").trim().toUpperCase();
    if (!sanitizedTicker) {
      throw new Error("Ticker is required");
    }
    const trimmedNote = String(note ?? "").trim();
    if (!trimmedNote) {
      throw new Error("Note cannot be empty");
    }

    const mutation = await addMutation({
      type: "annotation",
      ticker: sanitizedTicker,
      note: trimmedNote,
      clientTimestamp: nowIsoString(),
    });

    this.#notifyServiceWorker({ type: "STORE_MUTATION", mutation });

    await appendAudit({
      timestamp: nowIsoString(),
      type: "queued",
      detail: `Queued annotation for ${sanitizedTicker}`,
      mutationId: mutation.id,
    });

    if (navigator.onLine) {
      await this.syncPendingMutations();
    }

    return mutation;
  }

  async syncPendingMutations() {
    const pending = await getAllMutations();
    if (!pending.length) {
      return { applied: 0, conflicts: [] };
    }

    let serverAnnotations = [];
    try {
      serverAnnotations = await this.loadAnnotations();
    } catch (error) {
      await appendAudit({
        timestamp: nowIsoString(),
        type: "sync_failed",
        detail: `Failed to load annotations before sync: ${String(error)}`,
      });
      throw error;
    }

    const byTicker = new Map(serverAnnotations.map((item) => [item.ticker, item]));
    const conflicts = [];
    let applied = 0;

    for (const mutation of pending) {
      if (mutation.type !== "annotation") {
        continue;
      }
      const serverRecord = byTicker.get(mutation.ticker);
      const mutationTime = new Date(mutation.clientTimestamp ?? 0).getTime();
      const serverTime = serverRecord ? new Date(serverRecord.updated_at ?? 0).getTime() : 0;

      if (!serverRecord || mutationTime > serverTime) {
        try {
          const response = await this.#fetchJson(`/annotations/${encodeURIComponent(mutation.ticker)}`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              note: mutation.note,
              client_timestamp: mutation.clientTimestamp,
            }),
          });
          applied += 1;
          byTicker.set(response.ticker, response);
          await deleteMutation(mutation.id);
          await appendAudit({
            timestamp: nowIsoString(),
            type: "applied",
            detail: `Applied annotation for ${mutation.ticker}`,
            mutationId: mutation.id,
          });
        } catch (error) {
          await appendAudit({
            timestamp: nowIsoString(),
            type: "sync_failed",
            detail: `Failed to apply mutation for ${mutation.ticker}: ${String(error)}`,
            mutationId: mutation.id,
          });
        }
      } else {
        conflicts.push({
          ticker: mutation.ticker,
          mutationTimestamp: mutation.clientTimestamp,
          serverTimestamp: serverRecord.updated_at,
          serverNote: serverRecord.note,
        });
        await deleteMutation(mutation.id);
        await appendAudit({
          timestamp: nowIsoString(),
          type: "conflict",
          detail: `Server version kept for ${mutation.ticker}`,
          mutationId: mutation.id,
        });
      }
    }

    return { applied, conflicts };
  }

  async getMergedAnnotations() {
    const [server, pending, audit] = await Promise.all([
      this.loadAnnotations(),
      getAllMutations(),
      getAuditTrail(100),
    ]);

    const merged = new Map(server.map((item) => [item.ticker, { ...item, pending: false }]));

    for (const mutation of pending) {
      if (mutation.type !== "annotation") {
        continue;
      }
      const existing = merged.get(mutation.ticker);
      const mutationTime = new Date(mutation.clientTimestamp ?? 0).getTime();
      const existingTime = existing ? new Date(existing.updated_at ?? 0).getTime() : 0;
      if (!existing || mutationTime > existingTime) {
        merged.set(mutation.ticker, {
          ticker: mutation.ticker,
          note: mutation.note,
          updated_at: mutation.clientTimestamp,
          pending: true,
        });
      }
    }

    const items = Array.from(merged.values()).sort((a, b) => a.ticker.localeCompare(b.ticker));
    return { items, pending, audit };
  }

  async #fetchJson(path, options = undefined) {
    const url = `${this.baseUrl}${path}`;
    const headers = new Headers(options?.headers ?? {});
    const apiKey = this.apiKey ?? readStoredApiKey();
    if (apiKey && !headers.has("Authorization") && !headers.has("X-API-Key")) {
      headers.set("Authorization", `Bearer ${apiKey}`);
    }
    const response = await fetch(url, {
      credentials: "same-origin",
      cache: "no-store",
      ...options,
      headers,
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Request failed (${response.status}): ${text}`);
    }
    if (response.status === 204) {
      return null;
    }
    return response.json();
  }

  #notifyServiceWorker(message) {
    if (!navigator.serviceWorker) {
      return;
    }
    navigator.serviceWorker.ready
      .then((registration) => {
        if (registration.active) {
          registration.active.postMessage(message);
        }
      })
      .catch(() => {
        // Ignore service worker registration issues in unsupported environments.
      });
  }
}

