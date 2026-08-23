#!/usr/bin/env python3
"""Serve the 8844 static hub with a consistent metric-column order.

The hub is written by several independent dashboard builders.  This wrapper
keeps those builders and their data untouched, while applying the shared UI
ordering at response time: VBench dynamic is the first metric column on every
metric-oriented page.
"""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path("/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub")
UTONIA_DASHBOARD_PATH = "/utonia-scene-weights-no-scene-three-tests/dashboard.json"

UTONIA_ROOT_ENTRY = r"""
<!-- UTONIA_NO_SCENE_THREE_TESTS_ENTRY -->
<section class="entry"><div><h2>PHYRVG-Full-SA · Utonia Scene Weights · No-Scene / Scene Enabled · 三测试集矩阵</h2>
  <div class="meta">全部 Utonia Scene Weights；No-Scene 与 Scene Enabled（online VGGT → Utonia）分色展示，可切换 test_5、PhysicIQ 67-case、physV V2V 0819 test70，未生成显示 pending</div>
  <a href="utonia-scene-weights-no-scene-three-tests/">进入三测试集可视化</a></div>
  <div class="status">动态矩阵<strong>18 个权重列</strong><small>手动刷新 · 每个 case 行可全部重新播放</small></div>
</section>
"""

ORDER_SCRIPT = r"""
<script id="vbench-dynamic-first-column">
(() => {
  "use strict";
  const dynamicLabel = /\bvbench\s+dynamic(?:\s+degree)?\b/i;
  const metricLabel = /(VideoPhy2|Cosmos|Physics-IQ|PMF|WMReward|VBench)/i;
  const metricPage = __METRIC_PAGE__;

  const cellText = (cell) => (cell?.textContent || "")
    .replace(/\s+/g, " ").trim();

  function moveTable(table) {
    const rows = [...table.querySelectorAll("tr")];
    if (!rows.length) return;
    const header = rows.find((row) =>
      [...row.cells].some((cell) => dynamicLabel.test(cellText(cell)))
    );
    if (!header) return;
    const headerCells = [...header.cells];
    const from = headerCells.findIndex((cell) =>
      dynamicLabel.test(cellText(cell))
    );
    if (from < 0) return;
    const firstMetric = headerCells.findIndex((cell) =>
      metricLabel.test(cellText(cell))
    );
    const to = firstMetric >= 0 ? firstMetric : Math.min(2, headerCells.length);
    if (from === to) return;

    // Rows with colspan are group labels rather than metric rows.  Leave them
    // intact; all ordinary header/body rows have the same cell count.
    for (const row of rows) {
      const cells = [...row.cells];
      if (cells.length !== headerCells.length) continue;
      const reordered = cells.slice();
      const [dynamicCell] = reordered.splice(from, 1);
      reordered.splice(to, 0, dynamicCell);
      for (const cell of reordered) row.appendChild(cell);
    }
  }

  function moveMetricCards() {
    if (!metricPage) return;
    const cards = [...document.querySelectorAll("article, figure")];
    for (const card of cards) {
      const label = `${card.textContent || ""} ${[...card.querySelectorAll("img")]
        .map((img) => `${img.alt || ""} ${img.src || ""}`).join(" ")}`;
      if (!dynamicLabel.test(label)) continue;
      const parent = card.parentElement;
      if (!parent) continue;
      const siblings = [...parent.children].filter((child) =>
        child.matches("article, figure")
      );
      if (siblings.length > 1 && siblings[0] !== card) {
        parent.insertBefore(card, siblings[0]);
      }
    }
  }

  function moveMetricOptions() {
    if (!metricPage) return;
    for (const select of document.querySelectorAll("select")) {
      const option = [...select.options].find((item) =>
        dynamicLabel.test(item.textContent || "")
      );
      if (!option) continue;
      const firstMetric = [...select.options].find((item) =>
        metricLabel.test(item.textContent || "")
      );
      if (firstMetric && option !== firstMetric) {
        select.insertBefore(option, firstMetric);
      }
    }
  }

  function applyOrder() {
    if (metricPage) document.querySelectorAll("table").forEach(moveTable);
    moveMetricCards();
    moveMetricOptions();
  }

  let scheduled = false;
  const schedule = () => {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => {
      scheduled = false;
      applyOrder();
    });
  };

  const start = () => {
    applyOrder();
    const observer = new MutationObserver(schedule);
    observer.observe(document.body, {childList: true, subtree: true});
  };
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, {once: true});
  } else {
    start();
  }
})();
</script>
"""


def is_metric_page(path: str) -> bool:
    lowered = path.lower()
    return any(
        token in lowered
        for token in (
            "metric",
            "average",
            "extreme",
            "top3",
            "worst",
            "context-length",
            "lora-ablation",
            "solid-mechanics",
            "curve",
            "motion-analysis",
            "focused-impact",
            "disagreement",
        )
    )


class OrderedHubHandler(SimpleHTTPRequestHandler):
    def _send_html(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, body: bytes, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _serve_utonia_dashboard(self, path: str) -> bool:
        if path != UTONIA_DASHBOARD_PATH:
            return False
        try:
            from build_utonia_no_scene_three_test_dashboard import build_dashboard

            payload = build_dashboard(write=True)
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send_json(body)
        except Exception as exc:  # keep the error visible to the page/client
            body = json.dumps(
                {"error": f"failed to build Utonia dashboard: {exc}"},
                ensure_ascii=False,
            ).encode("utf-8")
            self._send_json(body, status=500)
        return True

    def _serve_html_with_order(self, path: str) -> bool:
        filesystem_path = Path(self.translate_path(path))
        if filesystem_path.is_dir():
            filesystem_path = filesystem_path / "index.html"
        if filesystem_path.suffix.lower() != ".html" or not filesystem_path.is_file():
            return False
        body = filesystem_path.read_bytes()
        if path in ("/", "/index.html") and b"UTONIA_NO_SCENE_THREE_TESTS_ENTRY" not in body:
            root_marker = b"</main>"
            if root_marker in body:
                body = body.replace(root_marker, UTONIA_ROOT_ENTRY.encode("utf-8") + root_marker, 1)
        marker = b"</body>"
        if marker in body:
            script = ORDER_SCRIPT.replace(
                "__METRIC_PAGE__", "true" if is_metric_page(path) else "false"
            ).encode("utf-8")
            body = body.replace(marker, script + marker, 1)
        self._send_html(body)
        return True

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if self._serve_utonia_dashboard(path):
            return
        if self._serve_html_with_order(path):
            return
        super().do_GET()


def main() -> None:
    server = ThreadingHTTPServer(
        ("127.0.0.1", 8844),
        partial(OrderedHubHandler, directory=str(ROOT)),
    )
    print("serving ordered 8844 hub at http://127.0.0.1:8844/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
