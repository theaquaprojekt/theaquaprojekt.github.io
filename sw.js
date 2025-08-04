// Service Worker for AQUA Project
// Improves performance through caching and offline support

const CACHE_NAME = 'aqua-v1.2';
const STATIC_CACHE = 'aqua-static-v1.2';
const DYNAMIC_CACHE = 'aqua-dynamic-v1.2';

const STATIC_ASSETS = [
    '/',
    '/index.html',
    '/cavity.html',
    '/silicon.html',
    '/assets/landing-styles.css',
    '/assets/styles.css',
    '/optics_assets/Thumbnail.svg',
    '/optics_assets/fig_2.svg',
    '/manifest.json'
];

const DYNAMIC_ASSETS = [
    '/optics_assets/',
    '/silicon_assets/'
];

// Install event - cache static assets
self.addEventListener('install', event => {
    console.log('Service Worker installing...');
    event.waitUntil(
        caches.open(STATIC_CACHE)
            .then(cache => {
                console.log('Caching static assets');
                return cache.addAll(STATIC_ASSETS);
            })
            .then(() => self.skipWaiting())
            .catch(err => console.log('Cache install error:', err))
    );
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
    console.log('Service Worker activating...');
    event.waitUntil(
        caches.keys()
            .then(keys => {
                return Promise.all(
                    keys
                        .filter(key => key !== STATIC_CACHE && key !== DYNAMIC_CACHE)
                        .map(key => caches.delete(key))
                );
            })
            .then(() => self.clients.claim())
    );
});

// Fetch event - serve from cache when possible
self.addEventListener('fetch', event => {
    const { request } = event;
    const url = new URL(request.url);

    // Skip non-GET requests and external URLs
    if (request.method !== 'GET' || !url.origin.includes(self.location.origin)) {
        return;
    }

    // Handle static assets
    if (STATIC_ASSETS.some(asset => url.pathname.endsWith(asset))) {
        event.respondWith(
            caches.match(request)
                .then(response => {
                    return response || fetch(request).then(fetchResponse => {
                        const responseClone = fetchResponse.clone();
                        caches.open(STATIC_CACHE)
                            .then(cache => cache.put(request, responseClone));
                        return fetchResponse;
                    });
                })
                .catch(() => {
                    // Fallback for offline
                    if (request.destination === 'document') {
                        return caches.match('/index.html');
                    }
                })
        );
        return;
    }

    // Handle dynamic assets (images, etc.)
    if (DYNAMIC_ASSETS.some(path => url.pathname.includes(path))) {
        event.respondWith(
            caches.match(request)
                .then(response => {
                    if (response) {
                        return response;
                    }
                    return fetch(request).then(fetchResponse => {
                        const responseClone = fetchResponse.clone();
                        caches.open(DYNAMIC_CACHE)
                            .then(cache => {
                                // Only cache successful responses
                                if (fetchResponse.status === 200) {
                                    cache.put(request, responseClone);
                                }
                            });
                        return fetchResponse;
                    });
                })
                .catch(() => {
                    console.log('Failed to fetch:', request.url);
                })
        );
        return;
    }

    // Network first for other requests
    event.respondWith(
        fetch(request)
            .catch(() => caches.match(request))
    );
});

// Background sync for better offline experience
self.addEventListener('sync', event => {
    if (event.tag === 'background-sync') {
        event.waitUntil(
            // Perform background tasks
            console.log('Background sync triggered')
        );
    }
});

// Push notifications (future feature)
self.addEventListener('push', event => {
    if (event.data) {
        const options = {
            body: event.data.text(),
            icon: '/optics_assets/Thumbnail.svg',
            badge: '/optics_assets/Thumbnail.svg',
            tag: 'aqua-notification'
        };
        
        event.waitUntil(
            self.registration.showNotification('AQUA Project Update', options)
        );
    }
});
