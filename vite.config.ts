// vite.config.ts
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig, type PluginOption } from 'vite'; // Import PluginOption type
import { viteStaticCopy } from 'vite-plugin-static-copy';

// Middleware to set required headers for SharedArrayBuffer (needed for WASM threads)
const viteServerConfig: PluginOption = { // Add type PluginOption
    name: 'configure-server-headers', // Changed name for clarity
    configureServer(server) {
        server.middlewares.use((req, res, next) => {
            res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
            res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
            next();
        });
    }
};

export default defineConfig({
    plugins: [
        sveltekit(),
        viteStaticCopy({
            targets: [
                {
                    // Use *.wasm to copy both standard and .jsep variants if needed
                    src: 'node_modules/onnxruntime-web/dist/*.wasm',
                    // Copy them to '/wasm/' relative to the build output root
                    dest: 'wasm'
                }
            ]
        }),
        // ***** ADD THE PLUGIN TO THE ARRAY *****
        viteServerConfig
        // ***** END ADD *****
    ],
    define: {
        // Keep your existing defines
        APP_VERSION: JSON.stringify(process.env.npm_package_version),
        APP_BUILD_HASH: JSON.stringify(process.env.APP_BUILD_HASH || 'dev-build')
    },
    build: {
        sourcemap: true // Keep sourcemaps if needed
    },
    worker: {
        format: 'es' // Keep worker format
    },
    // ***** ADD THIS ENTIRE server SECTION *****
    server: {
        // Allow Vite's dev server to access files from ORT's dist directory
        fs: {
            allow: [
                './node_modules/onnxruntime-web/dist/'
            ]
        },
        // Proxy API and potentially other backend requests
        proxy: {
            '/api': {
                target: 'http://localhost:8080', // <<< VERIFY YOUR BACKEND PORT
                changeOrigin: true,
                secure: false,
            },
            '/ollama': {
                target: 'http://localhost:8080', // <<< VERIFY YOUR BACKEND PORT
                changeOrigin: true,
                secure: false,
                rewrite: (path) => path.replace(/^\/ollama/, '/ollama')
            }
        }
    },
    // ***** END ADD server SECTION *****

    // ***** ADD THIS optimizeDeps SECTION *****
    optimizeDeps: {
        exclude: ['onnxruntime-web']
    }
    // ***** END ADD optimizeDeps SECTION *****
});