// Minimal static file server for local testing of the results page (no dependencies).
import http from "http"; import fs from "fs"; import path from "path";
const root = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1")));
const types = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".bin": "application/octet-stream", ".css": "text/css", ".png": "image/png", ".svg": "image/svg+xml" };
const port = +(process.argv[2] || 8765);
http.createServer((req, res) => {
  let p = decodeURIComponent(new URL(req.url, "http://x").pathname); if (p.endsWith("/")) p += "index.html";
  const f = path.join(root, p); if (!f.startsWith(root)) { res.writeHead(403); return res.end(); }
  fs.readFile(f, (err, data) => {
    if (err) { res.writeHead(404); return res.end("not found"); }
    let body = data;
    if (p.endsWith("index.html")) body = Buffer.from(`<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"><style>:root{color-scheme:light}body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style></head><body>` + data.toString() + `</body></html>`);
    res.writeHead(200, { "Content-Type": types[path.extname(f)] || "application/octet-stream", "Cache-Control": "no-store" }); res.end(body);
  });
}).listen(port, () => console.log(`serving ${root} on http://localhost:${port}`));
