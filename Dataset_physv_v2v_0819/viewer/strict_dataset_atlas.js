(() => {
  const state = { data: null, search: "", family: "all", taxonomy: "all" };
  const $ = (selector) => document.querySelector(selector);
  const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
  const pretty = (value) => String(value ?? "").replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
  const formatValue = (value) => value == null ? "—" : typeof value === "number" ? Number.isInteger(value) ? String(value) : value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "") : String(value);

  function renderOverview() {
    const d = state.data.dataset;
    $("#dataset-description").textContent = d.description;
    $("#stat-grid").innerHTML = [
      [d.sample_count, "reference cases"], [d.family_count, "control families"], ["896×512", "native CYCLES"], ["90", "frames / case"],
    ].map(([value, label]) => `<div class="stat"><strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></div>`).join("");
    const max = Math.max(...Object.values(d.taxonomy_counts));
    $("#taxonomy-list").innerHTML = Object.entries(d.taxonomy_counts).map(([key, count]) => `<div class="taxonomy-row"><b>${escapeHtml(key)}</b><span>${key === "Scene" ? "environment changes" : key === "Object" ? "actor or state changes" : "relative placement changes"}</span><strong>${count}</strong><div class="taxonomy-bar"><i style="width:${count / max * 100}%"></i></div></div>`).join("");
    $("#source-summary").textContent = `${Object.keys(d.source_counts).length} source groups · ${d.official_rigidbench ? "official" : "RigidBench-style / official=false"}`;
  }

  function renderFamilies() {
    const families = state.data.families;
    $("#family-filter").innerHTML = `<option value="all">All families</option>` + families.map(f => `<option value="${escapeHtml(f.key)}">${escapeHtml(f.label)}</option>`).join("");
    $("#family-strip").innerHTML = families.map(f => `<button class="family-tile" type="button" data-family="${escapeHtml(f.key)}"><div class="family-tile-code">${escapeHtml(f.key)}</div><h3>${escapeHtml(f.label)}</h3><small>${f.case_count} cases · ${escapeHtml(f.taxonomy)}</small></button>`).join("");
    document.querySelectorAll(".family-tile").forEach(button => button.addEventListener("click", () => { state.family = button.dataset.family; $("#family-filter").value = state.family; renderCases(); document.querySelectorAll(".family-tile").forEach(b => b.classList.toggle("active", b === button)); $("#cases").scrollIntoView({ behavior: "smooth" }); }));
  }

  function matches(c) {
    if (state.family !== "all" && c.family_key !== state.family) return false;
    if (state.taxonomy !== "all" && c.taxonomy !== state.taxonomy) return false;
    if (!state.search) return true;
    const haystack = [c.id, c.title, c.family_label, c.task_type, c.caption_specific, c.caption_abstract, c.control.variable, c.control.value_label, ...c.dynamic_objects].join(" ").toLowerCase();
    return haystack.includes(state.search);
  }

  function actorHtml(c) {
    return c.actors.map(a => `<div class="actor-row"><b>${escapeHtml(a.name)}</b> <span>· ${escapeHtml(a.shape || "object")} · ${a.mass_kg == null ? "mass —" : `${formatValue(a.mass_kg)} kg`}</span><br /><span class="actor-role">${escapeHtml(a.role || "unknown")}${a.dynamic ? " / dynamic" : " / static"}</span></div>`).join("");
  }

  function cardHtml(c) {
    const p = c.paths;
    const contact = c.contacts;
    const checked = Object.values(c.checks).every(Boolean);
    return `<article class="case-card" data-case="${escapeHtml(c.id)}">
      <div class="case-media"><video controls preload="none" poster="${escapeHtml(p.first_frame)}" src="${escapeHtml(p.reference_video)}"></video><span class="media-tag">CYCLES / RGB</span></div>
      <div class="card-body"><div class="card-top"><span class="family-code">${escapeHtml(c.family_key)}</span><span class="case-id">${escapeHtml(c.id)}</span></div>
      <h4>${escapeHtml(c.title)}</h4><p class="case-caption">${escapeHtml(c.caption_specific)}</p>
      <div class="tag-row"><span class="tag axis-${escapeHtml(c.taxonomy)}">${escapeHtml(c.taxonomy)}</span><span class="tag">${escapeHtml(c.task_type)}</span>${checked ? '<span class="tag">GT ✓</span>' : '<span class="tag">GT check</span>'}</div>
      <div class="fact-grid"><div class="fact"><span class="fact-label">control</span><span class="fact-value">${escapeHtml(c.control.value_label || c.control.variable || "—")}</span></div><div class="fact"><span class="fact-label">dynamic</span><span class="fact-value">${escapeHtml(c.dynamic_objects.join(", ") || "—")}</span></div><div class="fact"><span class="fact-label">source group</span><span class="fact-value">${escapeHtml(c.source_group || "—")}</span></div></div>
      <div class="link-row"><a href="${escapeHtml(p.context8)}" target="_blank">context 8f ↗</a><a href="${escapeHtml(p.context16)}" target="_blank">context 16f ↗</a><a href="${escapeHtml(p.dynamic_masks)}" download>mask GT ↓</a><a href="${escapeHtml(p.depth)}" download>depth GT ↓</a><a href="${escapeHtml(p.trajectory_pixels)}" download>2D track ↓</a><a href="${escapeHtml(p.rigidbench_metadata)}" target="_blank">RB meta ↗</a><a href="${escapeHtml(p.test_json)}" target="_blank">JSON ↗</a></div>
      <details class="card-details"><summary>Physical actors & contacts</summary><div class="actor-list">${actorHtml(c)}</div><div class="contact-note">${contact.contact_frames} contact frames · peak normal force ${formatValue(contact.peak_normal_force_n)} N${contact.top_pairs.length ? ` · ${escapeHtml(contact.top_pairs[0].pair)}` : ""}</div></details>
      </div></article>`;
  }

  function renderCases() {
    const filtered = state.data.cases.filter(matches);
    $("#result-summary").textContent = `${filtered.length} / ${state.data.cases.length} cases visible · click a family tile to jump and filter`;
    const groups = new Map();
    filtered.forEach(c => { if (!groups.has(c.family_key)) groups.set(c.family_key, []); groups.get(c.family_key).push(c); });
    $("#case-groups").innerHTML = [...groups.entries()].map(([key, cases]) => { const f = state.data.families.find(x => x.key === key) || {}; return `<section class="family-group" id="group-${escapeHtml(key)}"><div class="group-head"><div class="group-title"><code>${escapeHtml(key)}</code><h3>${escapeHtml(f.label || key)}</h3></div><div class="group-meta">${cases.length} cases · ${escapeHtml(f.taxonomy || "")}</div></div><div class="case-grid">${cases.map(cardHtml).join("")}</div></section>`; }).join("");
    $("#empty-state").hidden = filtered.length !== 0;
  }

  function wireControls() {
    $("#search").addEventListener("input", e => { state.search = e.target.value.trim().toLowerCase(); renderCases(); });
    $("#family-filter").addEventListener("change", e => { state.family = e.target.value; document.querySelectorAll(".family-tile").forEach(b => b.classList.toggle("active", b.dataset.family === state.family)); renderCases(); });
    $("#taxonomy-filter").addEventListener("change", e => { state.taxonomy = e.target.value; renderCases(); });
    $("#clear-filters").addEventListener("click", () => { state.search = ""; state.family = "all"; state.taxonomy = "all"; $("#search").value = ""; $("#family-filter").value = "all"; $("#taxonomy-filter").value = "all"; document.querySelectorAll(".family-tile").forEach(b => b.classList.remove("active")); renderCases(); });
  }

  fetch("data.json").then(r => r.json()).then(data => { state.data = data; renderOverview(); renderFamilies(); wireControls(); renderCases(); }).catch(error => { $("#case-groups").innerHTML = `<div class="empty-state">Could not load data.json. Serve this folder over HTTP, for example with <code>python -m http.server</code>.<br />${escapeHtml(error)}</div>`; });
})();
