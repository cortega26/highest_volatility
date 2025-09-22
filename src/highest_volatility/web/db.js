const DB_NAME = "hv-pwa";
const DB_VERSION = 1;
export const MUTATION_STORE = "mutations";
export const AUDIT_STORE = "audit";

const uuidv4 = () => {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return ([1e7] + -1e3 + -4e3 + -8e3 + -1e11)
    .replace(/[018]/g, (c) => (c ^ (crypto.getRandomValues(new Uint8Array(1))[0] & (15 >> (c / 4)))).toString(16));
};

function promisifyRequest(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("IndexedDB request failed"));
  });
}

function openDatabase() {
  return new Promise((resolve, reject) => {
    if (!("indexedDB" in globalThis)) {
      reject(new Error("IndexedDB is not available in this environment"));
      return;
    }
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(MUTATION_STORE)) {
        db.createObjectStore(MUTATION_STORE, { keyPath: "id" });
      }
      if (!db.objectStoreNames.contains(AUDIT_STORE)) {
        db.createObjectStore(AUDIT_STORE, { keyPath: "id" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("Failed to open IndexedDB"));
  });
}

async function withStore(storeName, mode, callback) {
  const db = await openDatabase();
  try {
    const tx = db.transaction(storeName, mode);
    const store = tx.objectStore(storeName);
    const result = await callback(store);
    await new Promise((resolve, reject) => {
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error ?? new Error("IndexedDB transaction failed"));
      tx.onabort = () => reject(tx.error ?? new Error("IndexedDB transaction aborted"));
    });
    return result;
  } finally {
    db.close();
  }
}

export async function addMutation(record) {
  const mutation = { ...record, id: record.id ?? uuidv4() };
  await withStore(MUTATION_STORE, "readwrite", (store) => {
    store.put(mutation);
  });
  return mutation;
}

export async function deleteMutation(id) {
  await withStore(MUTATION_STORE, "readwrite", (store) => {
    store.delete(id);
  });
}

export async function getAllMutations() {
  return withStore(MUTATION_STORE, "readonly", (store) => {
    const request = store.getAll();
    return promisifyRequest(request).then((items) =>
      (items ?? []).sort((a, b) => {
        return new Date(a.clientTimestamp ?? 0).getTime() - new Date(b.clientTimestamp ?? 0).getTime();
      })
    );
  });
}

export async function clearMutations() {
  await withStore(MUTATION_STORE, "readwrite", (store) => {
    store.clear();
  });
}

export async function appendAudit(entry) {
  const payload = { ...entry, id: entry.id ?? uuidv4() };
  await withStore(AUDIT_STORE, "readwrite", (store) => {
    store.put(payload);
  });
  return payload;
}

export async function getAuditTrail(limit = 50) {
  return withStore(AUDIT_STORE, "readonly", (store) => {
    const request = store.getAll();
    return promisifyRequest(request).then((items) => {
      const sorted = (items ?? []).sort((a, b) => {
        return new Date(b.timestamp ?? 0).getTime() - new Date(a.timestamp ?? 0).getTime();
      });
      return sorted.slice(0, limit);
    });
  });
}

