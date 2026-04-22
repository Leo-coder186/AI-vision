import { createServer } from 'node:http';
import { createReadStream, existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const distDir = path.join(__dirname, 'dist');
const publicDir = path.join(__dirname, 'public');
const port = Number(process.env.PORT ?? 3000);
const apiKey = process.env.MINIMAX_API_KEY;

const mimeTypes = new Map([
  ['.html', 'text/html; charset=utf-8'],
  ['.js', 'text/javascript; charset=utf-8'],
  ['.css', 'text/css; charset=utf-8'],
  ['.svg', 'image/svg+xml'],
  ['.png', 'image/png'],
  ['.jpg', 'image/jpeg'],
  ['.jpeg', 'image/jpeg'],
  ['.json', 'application/json; charset=utf-8'],
  ['.ico', 'image/x-icon'],
]);

function sendJson(res, statusCode, payload) {
  res.writeHead(statusCode, { 'Content-Type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify(payload));
}

function getSafeFilePath(urlPath) {
  const normalizedPath = urlPath === '/' ? '/index.html' : urlPath;
  const decodedPath = decodeURIComponent(normalizedPath);
  const candidate = path.normalize(decodedPath).replace(/^(\.\.[/\\])+/, '');
  const distPath = path.join(distDir, candidate);
  const publicPath = path.join(publicDir, candidate);

  if (existsSync(distPath)) {
    return distPath;
  }

  if (existsSync(publicPath)) {
    return publicPath;
  }

  return path.join(distDir, 'index.html');
}

async function readRequestBody(req) {
  const chunks = [];
  for await (const chunk of req) {
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString('utf8');
}

async function handleChat(req, res) {
  if (!apiKey) {
    sendJson(res, 500, { error: '缺少 MINIMAX_API_KEY 环境变量。' });
    return;
  }

  let body;
  try {
    body = JSON.parse(await readRequestBody(req));
  } catch {
    sendJson(res, 400, { error: '请求体不是合法 JSON。' });
    return;
  }

  const question = typeof body.question === 'string' ? body.question.trim() : '';
  if (!question) {
    sendJson(res, 400, { error: 'question 不能为空。' });
    return;
  }

  try {
    const upstreamResponse = await fetch('https://api.minimaxi.com/anthropic/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify({
        model: 'MiniMax-M2.7',
        max_tokens: 1024,
        messages: [{ role: 'user', content: question }],
      }),
    });

    const text = await upstreamResponse.text();
    res.writeHead(upstreamResponse.status, {
      'Content-Type': upstreamResponse.headers.get('content-type') ?? 'application/json; charset=utf-8',
    });
    res.end(text);
  } catch (error) {
    sendJson(res, 502, {
      error: `上游请求失败：${error instanceof Error ? error.message : '未知错误'}`,
    });
  }
}

createServer(async (req, res) => {
  const requestUrl = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`);

  if (req.method === 'POST' && requestUrl.pathname === '/api/chat') {
    await handleChat(req, res);
    return;
  }

  if (req.method !== 'GET' && req.method !== 'HEAD') {
    sendJson(res, 405, { error: 'Method Not Allowed' });
    return;
  }

  const filePath = getSafeFilePath(requestUrl.pathname);
  const ext = path.extname(filePath);
  const contentType = mimeTypes.get(ext) ?? 'application/octet-stream';

  try {
    if (req.method === 'HEAD') {
      const stat = await readFile(filePath);
      res.writeHead(200, {
        'Content-Type': contentType,
        'Content-Length': stat.byteLength,
      });
      res.end();
      return;
    }

    res.writeHead(200, { 'Content-Type': contentType });
    createReadStream(filePath).pipe(res);
  } catch {
    sendJson(res, 404, { error: 'Not Found' });
  }
}).listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
