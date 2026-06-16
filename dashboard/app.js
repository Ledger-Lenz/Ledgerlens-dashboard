(() => {
  "use strict";

  const API_BASE = window.LEDGERLENS_API || "http://localhost:8000";

  // ── Helpers ─────────────────────────────────────────────────────────────────

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  function scoreClass(score) {
    if (score < 40) return "low";
    if (score < 70) return "medium";
    return "high";
  }

  function scoreLabel(score) {
    if (score < 40) return "Low Risk";
    if (score < 70) return "Medium Risk";
    return "High Risk";
  }

  function pill(score) {
    const cls = scoreClass(score);
    return `<span class="score-pill ${cls}">${score}</span>`;
  }

  function formatTs(iso) {
    return new Date(iso).toLocaleString(undefined, {
      month: "short", day: "numeric",
      hour: "2-digit", minute: "2-digit",
    });
  }

  async function apiFetch(path) {
    const resp = await fetch(`${API_BASE}${path}`);
    if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
    return resp.json();
  }

  // ── Health check ─────────────────────────────────────────────────────────────

  async function checkHealth() {
    const dot = $("#status-dot");
    try {
      const data = await apiFetch("/health");
      dot.classList.toggle("offline", data.status !== "ok");
    } catch {
      dot.classList.add("offline");
    }
  }

  // ── Stats row ─────────────────────────────────────────────────────────────────

  async function loadStats() {
    try {
      const [alerts, assets] = await Promise.all([
        apiFetch("/alerts/recent?limit=200"),
        apiFetch("/assets/risk-ranking?limit=200"),
      ]);
      const flagged = alerts.total;
      const assetCount = assets.total;
      const avgScore = assets.assets.length
        ? Math.round(assets.assets.reduce((s, a) => s + a.avg_score, 0) / assets.assets.length)
        : "—";

      $("#stat-flagged").textContent = flagged;
      $("#stat-assets").textContent = assetCount;
      $("#stat-avg").textContent = avgScore;
    } catch (e) {
      console.warn("Stats load failed:", e);
    }
  }

  // ── Score lookup ──────────────────────────────────────────────────────────────

  async function lookupScore() {
    const wallet = $("#wallet-input").value.trim();
    const pair = $("#pair-input").value.trim();
    const errEl = $("#lookup-error");
    const resultEl = $("#score-result");

    errEl.style.display = "none";
    resultEl.classList.remove("visible");

    if (!wallet || !pair) {
      errEl.textContent = "Please enter both wallet address and asset pair.";
      errEl.style.display = "block";
      return;
    }

    const btn = $("#lookup-btn");
    btn.disabled = true;
    btn.textContent = "Scoring…";

    try {
      const encodedPair = encodeURIComponent(pair);
      const data = await apiFetch(`/score/${wallet}/${encodedPair}`);

      // Gauge
      const gauge = $("#score-gauge");
      gauge.className = `gauge ${scoreClass(data.score)}`;
      $("#score-num").textContent = data.score;
      $("#score-risk-label").textContent = scoreLabel(data.score);

      // Flags
      const flagRow = $("#flag-row");
      flagRow.innerHTML = "";
      if (data.benford_flag) flagRow.innerHTML += `<span class="badge benford">Benford anomaly</span>`;
      if (data.ml_flag) flagRow.innerHTML += `<span class="badge ml">ML flagged</span>`;
      if (!data.benford_flag && !data.ml_flag) flagRow.innerHTML += `<span class="badge clean">Clean signals</span>`;

      // SHAP features
      const list = $("#features-list");
      list.innerHTML = "";
      const features = data.explanation?.top_features || [];
      features.forEach((f) => {
        const pct = Math.min(100, Math.round(Math.abs(f.contribution) * 300));
        const neg = f.direction === "decreases_risk";
        list.innerHTML += `
          <li>
            <span class="feature-name">${f.feature.replace(/_/g, " ")}</span>
            <span class="feature-bar-wrap">
              <span class="feature-bar"><span class="feature-bar-fill${neg ? " neg" : ""}" style="width:${pct}%"></span></span>
              <span style="font-size:11px;color:var(--text-muted);width:42px;text-align:right">${f.contribution > 0 ? "+" : ""}${f.contribution.toFixed(3)}</span>
            </span>
          </li>`;
      });
      if (!features.length) {
        list.innerHTML = `<li style="color:var(--text-muted);font-size:12px">No SHAP explanation available (model not loaded)</li>`;
      }

      resultEl.classList.add("visible");
    } catch (err) {
      errEl.textContent = `Scoring failed: ${err.message}`;
      errEl.style.display = "block";
    } finally {
      btn.disabled = false;
      btn.textContent = "Score";
    }
  }

  // ── Alerts table ──────────────────────────────────────────────────────────────

  async function loadAlerts() {
    const tbody = $("#alerts-body");
    try {
      const data = await apiFetch("/alerts/recent?limit=50&min_score=75");
      if (!data.alerts.length) {
        tbody.innerHTML = `<tr><td colspan="6"><div class="empty-state">No alerts in the last 24h</div></td></tr>`;
        return;
      }
      tbody.innerHTML = data.alerts.map((a) => `
        <tr>
          <td title="${a.wallet}">${a.wallet.slice(0, 12)}…${a.wallet.slice(-4)}</td>
          <td>${a.asset_pair}</td>
          <td>${pill(a.score)}</td>
          <td>${a.benford_flag ? "⚠ Yes" : "—"}</td>
          <td>${a.ml_flag ? "⚠ Yes" : "—"}</td>
          <td>${formatTs(a.flagged_at)}</td>
        </tr>`).join("");
    } catch (err) {
      tbody.innerHTML = `<tr><td colspan="6"><div class="empty-state">Failed to load alerts: ${err.message}</div></td></tr>`;
    }
  }

  // ── Asset ranking ─────────────────────────────────────────────────────────────

  async function loadAssets(window = "24h") {
    const grid = $("#asset-grid");
    try {
      const data = await apiFetch(`/assets/risk-ranking?limit=30&window=${window}`);
      if (!data.assets.length) {
        grid.innerHTML = `<div class="empty-state">No asset data yet</div>`;
        return;
      }
      grid.innerHTML = data.assets.map((a) => `
        <div class="asset-card">
          <div class="asset-code">${a.asset_code}</div>
          <div class="asset-avg" style="color:var(--${scoreClass(Math.round(a.avg_score))})">${Math.round(a.avg_score)}</div>
          <div class="asset-meta">
            Max ${a.max_score} · ${a.flagged_wallet_count} flagged wallet${a.flagged_wallet_count !== 1 ? "s" : ""}
          </div>
        </div>`).join("");
    } catch (err) {
      grid.innerHTML = `<div class="empty-state">Failed to load assets: ${err.message}</div>`;
    }
  }

  // ── Boot ──────────────────────────────────────────────────────────────────────

  function init() {
    checkHealth();
    loadStats();
    loadAlerts();
    loadAssets("24h");

    $("#lookup-btn").addEventListener("click", lookupScore);
    $("#wallet-input").addEventListener("keydown", (e) => { if (e.key === "Enter") lookupScore(); });
    $("#pair-input").addEventListener("keydown", (e) => { if (e.key === "Enter") lookupScore(); });

    $("#window-select").addEventListener("change", (e) => loadAssets(e.target.value));

    // Refresh every 60s
    setInterval(() => {
      checkHealth();
      loadStats();
      loadAlerts();
      loadAssets($("#window-select").value);
    }, 60_000);
  }

  document.readyState === "loading"
    ? document.addEventListener("DOMContentLoaded", init)
    : init();
})();
