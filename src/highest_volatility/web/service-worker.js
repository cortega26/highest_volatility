import { addMutation, getAllMutations } from "./db.js";

const STATIC_CACHE = "hv-static-v1";
const DATA_CACHE = "hv-data-v1";
const CORE_ASSETS = [
  "/",
  "/web/main.js",
  "/web/data-layer.js",
  "/web/db.js",
  "/web/styles.css",
  "/web/icon.svg",
  "/manifest.webmanifest",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(STATIC_CACHE).then((cache) => cache.addAll(CORE_ASSETS)).catch(() => undefined)
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const keys = await caches.keys();
      await Promise.all(
        keys
          .filter((key) => key !== STATIC_CACHE && key !== DATA_CACHE)
          .map((key) => caches.delete(key))
      );
      await self.clients.claim();
    })()
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);

  if (request.method === "GET") {
    if (url.origin === self.location.origin) {
      if (isStaticAsset(url.pathname)) {
        event.respondWith(cacheFirst(request));
        return;
      }
      if (isApiRequest(url.pathname)) {
        event.respondWith(networkWithCacheFallback(request));
        return;
      }
    }
    return;
  }

  if (url.origin === self.location.origin) {
    event.respondWith(handleMutationRequest(event));
  }
});

self.addEventListener("message", (event) => {
  const { type, mutation } = event.data ?? {};
  if (type === "STORE_MUTATION" && mutation) {
    event.waitUntil(storeMutation(mutation));
  }
  if (type === "SYNC_MUTATIONS") {
    event.waitUntil(notifyQueueLength());
  }
});

const isStaticAsset = (pathname) => {
  return (
    pathname === "/" ||
    pathname.startsWith("/web/") ||
    pathname === "/manifest.webmanifest"
  );
};

const isApiRequest = (pathname) => {
  return ["/universe", "/metrics", "/prices", "/annotations"].some((prefix) =>
    pathname.startsWith(prefix)
  );
};

const cacheFirst = async (request) => {
  const cache = await caches.open(STATIC_CACHE);
  const cached = await cache.match(request);
  if (cached) {
    return cached;
  }
  const response = await fetch(request);
  cache.put(request, response.clone());
  return response;
};

const networkWithCacheFallback = async (request) => {
  const cache = await caches.open(DATA_CACHE);
  try {
    const response = await fetch(request);
    cache.put(request, response.clone());
    return response;
  } catch (error) {
    const cached = await cache.match(request);
    if (cached) {
      return cached;
    }
    throw error;
  }
};

const handleMutationRequest = async (event) => {
  const { request } = event;
  try {
    return await fetch(request.clone());
  } catch (error) {
    const mutation = await storeRequest(request);
    await notifyClients({ type: "mutation-stored", mutation });
    return new Response(
      JSON.stringify({ status: "queued", id: mutation.id }),
      {
        status: 202,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
};

const storeRequest = async (request) => {
  let body = null;
  try {
    body = await request.clone().json();
  } catch (error) {
    try {
      body = await request.clone().text();
    } catch (innerError) {
      body = null;
    }
  }
  const mutation = await storeMutation({
    type: "annotation",
    url: new URL(request.url).pathname,
    method: request.method,
    body,
    clientTimestamp: new Date().toISOString(),
  });
  return mutation;
};

const storeMutation = async (payload) => {
  const mutation = await addMutation(payload);
  await notifyQueueLength();
  return mutation;
};

const notifyClients = async (message) => {
  const clients = await self.clients.matchAll({ type: "window" });
  clients.forEach((client) => client.postMessage(message));
};

const notifyQueueLength = async () => {
  const pending = await getAllMutations();
  await notifyClients({ type: "mutation-queue", count: pending.length });
};
