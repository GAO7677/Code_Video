(() => {
  const state = { data: null, search: "", family: "all", taxonomy: "all" };
  const $ = (selector) => document.querySelector(selector);
  const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
  const pretty = (value) => String(value ?? "").replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
  const formatValue = (value) => value == null ? "—" : typeof value === "number" ? Number.isInteger(value) ? String(value) : value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "") : String(value);
  const assetUrl = (value) => {
    const path = String(value ?? "");
    if (window.location.pathname.startsWith("/physv-v2v-0819-strict-cycles-atlas") && path.startsWith("../")) {
      return `/strict-cycles-assets/${path.slice(3)}`;
    }
    return path;
  };

  function renderOverview() {
    const d = state.data.dataset;
    $("#dataset-description").textContent = d.description;
    $("#stat-grid").innerHTML = [
      [d.sample_count, "reference case"], [d.family_count, "控制类别"], ["896×512", "原生 CYCLES"], ["90", "每 case 帧数"],
    ].map(([value, label]) => `<div class="stat"><strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></div>`).join("");
    const max = Math.max(...Object.values(d.taxonomy_counts));
    $("#taxonomy-list").innerHTML = Object.entries(d.taxonomy_counts).map(([key, count]) => `<div class="taxonomy-row"><b>${escapeHtml(key)}</b><span>${key === "Scene" ? "静态场景几何变化" : key === "Object" ? "物体几何或状态变化" : "相对位置、方向或支撑关系变化"}</span><strong>${count}</strong><div class="taxonomy-bar"><i style="width:${count / max * 100}%"></i></div></div>`).join("");
    $("#source-summary").textContent = `${Object.keys(d.source_counts).length} 个 source group · ${d.official_rigidbench ? "官方 RigidBench" : "RigidBench-style / official=false"}`;
    $("#truth-table").innerHTML = `<table><thead><tr><th>真值类别</th><th>文件 / 数组</th><th>shape</th><th>dtype</th><th>用途与说明</th></tr></thead><tbody>${d.truth_table.map(row => `<tr><td><strong>${escapeHtml(row.category)}</strong></td><td><code>${escapeHtml(row.file)}</code></td><td><code>${escapeHtml(row.shape)}</code></td><td>${escapeHtml(row.dtype)}</td><td>${escapeHtml(row.use)}</td></tr>`).join("")}</tbody></table>`;
  }

  function renderFamilies() {
    const families = state.data.families;
    $("#family-filter").innerHTML = `<option value="all">全部类别</option>` + families.map(f => `<option value="${escapeHtml(f.key)}">${escapeHtml(f.label)}</option>`).join("");
    $("#family-strip").innerHTML = families.map(f => `<button class="family-tile" type="button" data-family="${escapeHtml(f.key)}"><div class="family-tile-code">${escapeHtml(f.key)}</div><h3>${escapeHtml(f.label)}</h3><small>${f.case_count} 个 case · ${escapeHtml(f.taxonomy)}</small></button>`).join("");
    document.querySelectorAll(".family-tile").forEach(button => button.addEventListener("click", () => { state.family = button.dataset.family; $("#family-filter").value = state.family; renderCases(); document.querySelectorAll(".family-tile").forEach(b => b.classList.toggle("active", b === button)); $("#cases").scrollIntoView({ behavior: "smooth" }); }));
  }

  function renderMaintenance() {
    const archive = state.data.archive || {};
    const categories = archive.categories || [];
    const updated = archive.updated_at ? new Date(archive.updated_at).toLocaleString() : "未记录";
    $("#archive-updated").textContent = `${categories.reduce((n, row) => n + Number(row.count || 0), 0)} 项归档记录 · ${updated}`;
    $("#archive-path").textContent = archive.directory || "archive 路径未记录";
    $("#archive-cards").innerHTML = categories.map(row => `<article class="archive-card"><div class="archive-card-top"><span class="archive-count">${escapeHtml(row.count)}</span><span class="archive-label">${escapeHtml(row.label)}</span></div><code>${escapeHtml(row.path)}</code><p>${escapeHtml(row.reason)}</p></article>`).join("");
    $("#active-dependencies-list").innerHTML = (archive.active_dependencies || []).map(item => `<span>${escapeHtml(item)}</span>`).join("");
  }

  function matches(c) {
    if (state.family !== "all" && c.family_key !== state.family) return false;
    if (state.taxonomy !== "all" && c.taxonomy !== state.taxonomy) return false;
    if (!state.search) return true;
    const haystack = [c.id, c.title, c.family_label, c.task_type, c.caption_specific, c.caption_abstract, c.control.variable, c.control.value_label, ...c.dynamic_objects].join(" ").toLowerCase();
    return haystack.includes(state.search);
  }

  function actorHtml(c) {
    return c.actors.map(a => `<div class="actor-row"><b>${escapeHtml(a.name)}</b> <span>· ${escapeHtml(a.shape || "物体")} · ${a.mass_kg == null ? "质量 —" : `${formatValue(a.mass_kg)} kg`}</span><br /><span class="actor-role">${escapeHtml(a.role || "未知角色")}${a.dynamic ? " / 动态" : " / 静态"}</span></div>`).join("");
  }

  function cardHtml(c) {
    const p = c.paths;
    const contact = c.contacts;
    const checked = Object.values(c.checks).every(Boolean);
    return `<article class="case-card" data-case="${escapeHtml(c.id)}">
      <div class="case-media"><video controls preload="none" poster="${escapeHtml(assetUrl(p.first_frame))}" src="${escapeHtml(assetUrl(p.reference_video))}"></video><span class="media-tag">CYCLES / RGB</span></div>
      <div class="card-body"><div class="card-top"><span class="family-code">${escapeHtml(c.family_key)}</span><span class="case-id">${escapeHtml(c.id)}</span></div>
      <h4>${escapeHtml(c.title)}</h4><p class="case-caption">${escapeHtml(c.caption_specific)}</p>
      <div class="tag-row"><span class="tag axis-${escapeHtml(c.taxonomy)}">${escapeHtml(c.taxonomy)}</span><span class="tag">${escapeHtml(c.task_type)}</span>${checked ? '<span class="tag">GT 完整 ✓</span>' : '<span class="tag">GT 待核查</span>'}</div>
      <div class="fact-grid"><div class="fact"><span class="fact-label">控制变量</span><span class="fact-value">${escapeHtml(c.control.value_label || c.control.variable || "—")}</span></div><div class="fact"><span class="fact-label">动态物体</span><span class="fact-value">${escapeHtml(c.dynamic_objects.join(", ") || "—")}</span></div><div class="fact"><span class="fact-label">来源组</span><span class="fact-value">${escapeHtml(c.source_group || "—")}</span></div></div>
      <div class="link-row"><a href="${escapeHtml(assetUrl(p.context8))}" target="_blank">context 8 帧 ↗</a><a href="${escapeHtml(assetUrl(p.context16))}" target="_blank">context 16 帧 ↗</a><a href="${escapeHtml(assetUrl(p.dynamic_masks))}" download>mask GT ↓</a><a href="${escapeHtml(assetUrl(p.depth))}" download>depth GT ↓</a><a href="${escapeHtml(assetUrl(p.trajectory_pixels))}" download>2D 轨迹 ↓</a><a href="${escapeHtml(assetUrl(p.rigidbench_metadata))}" target="_blank">RB metadata ↗</a><a href="${escapeHtml(assetUrl(p.test_json))}" target="_blank">测试 JSON ↗</a></div>
      <details class="card-details"><summary>物理物体与接触信息</summary><div class="actor-list">${actorHtml(c)}</div><div class="contact-note">${contact.contact_frames} 个接触帧 · 峰值法向力 ${formatValue(contact.peak_normal_force_n)} N${contact.top_pairs.length ? ` · ${escapeHtml(contact.top_pairs[0].pair)}` : ""}</div></details>
      </div></article>`;
  }

  function renderCases() {
    const filtered = state.data.cases.filter(matches);
    $("#result-summary").textContent = `显示 ${filtered.length} / ${state.data.cases.length} 个 case · 点击类别卡片可跳转并筛选`;
    const groups = new Map();
    filtered.forEach(c => { if (!groups.has(c.family_key)) groups.set(c.family_key, []); groups.get(c.family_key).push(c); });
    $("#case-groups").innerHTML = [...groups.entries()].map(([key, cases]) => { const f = state.data.families.find(x => x.key === key) || {}; return `<section class="family-group" id="group-${escapeHtml(key)}"><div class="group-head"><div class="group-title"><code>${escapeHtml(key)}</code><h3>${escapeHtml(f.label || key)}</h3></div><div class="group-meta">${cases.length} 个 case · ${escapeHtml(f.taxonomy || "")}</div></div><div class="case-grid">${cases.map(cardHtml).join("")}</div></section>`; }).join("");
    $("#empty-state").textContent = "没有符合筛选条件的 case，请清除搜索或更换控制变量类别。";
    $("#empty-state").hidden = filtered.length !== 0;
  }

  function wireControls() {
    $("#search").addEventListener("input", e => { state.search = e.target.value.trim().toLowerCase(); renderCases(); });
    $("#family-filter").addEventListener("change", e => { state.family = e.target.value; document.querySelectorAll(".family-tile").forEach(b => b.classList.toggle("active", b.dataset.family === state.family)); renderCases(); });
    $("#taxonomy-filter").addEventListener("change", e => { state.taxonomy = e.target.value; renderCases(); });
    $("#clear-filters").addEventListener("click", () => { state.search = ""; state.family = "all"; state.taxonomy = "all"; $("#search").value = ""; $("#family-filter").value = "all"; $("#taxonomy-filter").value = "all"; document.querySelectorAll(".family-tile").forEach(b => b.classList.remove("active")); renderCases(); });
  }

  fetch("data.json").then(r => r.json()).then(data => { state.data = data; renderOverview(); renderMaintenance(); renderFamilies(); wireControls(); renderCases(); }).catch(error => { $("#case-groups").innerHTML = `<div class="empty-state">无法加载 data.json。请通过 HTTP 服务访问此页面，而不是直接双击 HTML。<br />${escapeHtml(error)}</div>`; });
})();
