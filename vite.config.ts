import { defineConfig, loadEnv } from 'vite';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');

  return {
    server: {
      proxy: {
        '/api/chat': {
          target: 'https://api.minimaxi.com',
          changeOrigin: true,
          secure: true,
          rewrite: () => '/anthropic/v1/messages',
          configure: (proxy) => {
            proxy.on('proxyReq', (proxyReq) => {
              const apiKey = env.MINIMAX_API_KEY;
              if (!apiKey) {
                return;
              }

              proxyReq.setHeader('x-api-key', apiKey);
              proxyReq.setHeader('anthropic-version', '2023-06-01');
            });
          },
        },
      },
    },
  };
});
