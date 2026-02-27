// predictor/static/predictor/result.js

// ============================
// 1) Full-screen flakes overlay
// ============================
(function () {
  const overlay = document.getElementById("resultConfettiOverlay");
  const canvas = document.getElementById("resultConfettiCanvas");

  // If overlay isn't on the page, just skip flakes (do NOT break the rest)
  if (!overlay || !canvas) return;

  overlay.classList.remove("hidden");
  overlay.style.display = "block";

  const ctx = canvas.getContext("2d");
  let rafId = null;

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = window.innerWidth;
    const h = window.innerHeight;

    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { w, h };
  }

  let { w, h } = resize();

  // Soft “skin-cell flakes” (subtle, health-appropriate)
  const pieces = Array.from({ length: 220 }).map(() => ({
    x: Math.random() * w,
    y: -60 - Math.random() * h,
    r: 3 + Math.random() * 5,
    vx: -1.2 + Math.random() * 2.4,
    vy: 1.8 + Math.random() * 3.8,
    rot: Math.random() * Math.PI,
    vr: -0.18 + Math.random() * 0.36,
    a: 1,
    // soft warm neutrals
    shade: 220 + Math.floor(Math.random() * 30), // 220..250
    tint: 200 + Math.floor(Math.random() * 30),  // 200..230
  }));

  const start = performance.now();

  function frame(now) {
    const t = now - start;
    ctx.clearRect(0, 0, w, h);

    pieces.forEach((p) => {
      p.x += p.vx;
      p.y += p.vy;
      p.rot += p.vr;
      p.vy += 0.02; // gravity
      p.a = Math.max(0, 1 - t / 2200);

      ctx.globalAlpha = p.a;
      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.rotate(p.rot);

      // “cell flake” = rounded pill shape
      ctx.fillStyle = `rgb(${p.shade}, ${p.tint}, ${p.tint})`;
      const ww = p.r * 2.6;
      const hh = p.r * 1.5;

      // rounded rect
      const x = -ww / 2, y = -hh / 2;
      const radius = Math.min(8, hh / 2);
      ctx.beginPath();
      ctx.moveTo(x + radius, y);
      ctx.lineTo(x + ww - radius, y);
      ctx.quadraticCurveTo(x + ww, y, x + ww, y + radius);
      ctx.lineTo(x + ww, y + hh - radius);
      ctx.quadraticCurveTo(x + ww, y + hh, x + ww - radius, y + hh);
      ctx.lineTo(x + radius, y + hh);
      ctx.quadraticCurveTo(x, y + hh, x, y + hh - radius);
      ctx.lineTo(x, y + radius);
      ctx.quadraticCurveTo(x, y, x + radius, y);
      ctx.closePath();
      ctx.fill();

      ctx.restore();
    });

    ctx.globalAlpha = 1;

    if (t < 2400) {
      rafId = requestAnimationFrame(frame);
    } else {
      overlay.style.display = "none";
    }
  }

  function onResize() {
    ({ w, h } = resize());
  }

  window.addEventListener("resize", onResize, { passive: true });
  rafId = requestAnimationFrame(frame);

  window.setTimeout(() => {
    window.removeEventListener("resize", onResize);
    if (rafId) cancelAnimationFrame(rafId);
  }, 2600);
})();


// ============================
// 2) Donut + bar chart animation
// ============================
(function () {
  function clamp(n, a, b) {
    return Math.max(a, Math.min(b, n));
  }

  function parseNumber(raw) {
    if (raw == null) return NaN;
    const s = String(raw).trim().replace(",", ".");
    const n = Number(s);
    return Number.isFinite(n) ? n : NaN;
  }

  const confWrap = document.getElementById("confidenceData");
  const donut = document.getElementById("confDonut");
  const pctEl = document.getElementById("confPct");
  const labelEl = document.getElementById("confLabel");

  // If donut isn't on the page, skip (do NOT error)
  if (!donut) return;

  let c = NaN;

  // Primary: data-confidence attribute
  if (confWrap) {
    c = parseNumber(confWrap.getAttribute("data-confidence"));
  }

  // Fallback: read the visible confidence number (your big text)
  if (!Number.isFinite(c)) {
    const node = document.querySelector("[data-confidence-text]");
    if (node) c = parseNumber(node.textContent);
  }

  // Final fallback: try scanning the page for "0.xxx"
  if (!Number.isFinite(c)) {
    const m = document.body.innerText.match(/\b0\.\d{2,6}\b/);
    if (m) c = parseNumber(m[0]);
  }

  // If still invalid, stop (better than showing fake 0)
  if (!Number.isFinite(c)) {
    if (pctEl) pctEl.textContent = "—";
    if (labelEl) labelEl.textContent = "Confidence unavailable (JS parse issue).";
    return;
  }

  const pct = clamp(c * 100, 0, 100);

  // Donut math: circumference for r=46 ≈ 289
  const CIRC = 289;
  const targetOffset = CIRC * (1 - pct / 100);

  // Label
  let label = "Low confidence — verify carefully.";
  if (pct >= 80) label = "High confidence — still verify with symptoms.";
  else if (pct >= 60) label = "Moderate confidence — consider context.";
  else if (pct >= 40) label = "Low–moderate confidence — be cautious.";
  if (labelEl) labelEl.textContent = label;

  // Animate donut
  let start = null;
  const startOffset = CIRC;
  const duration = 900;

  function animate(ts) {
    if (!start) start = ts;
    const t = clamp((ts - start) / duration, 0, 1);
    const eased = 1 - Math.pow(1 - t, 3);

    const current = startOffset + (targetOffset - startOffset) * eased;
    donut.style.strokeDashoffset = String(current);

    if (pctEl) pctEl.textContent = `${pct.toFixed(0)}%`;

    if (t < 1) requestAnimationFrame(animate);
  }

  requestAnimationFrame(animate);

  // Bars
  const bars = document.querySelectorAll(".topbar[data-p]");
  requestAnimationFrame(() => {
    bars.forEach((bar) => {
      const p = parseNumber(bar.getAttribute("data-p"));
      const w = Number.isFinite(p) ? clamp(p * 100, 0, 100) : 0;
      bar.style.width = w.toFixed(1) + "%";
    });
  });
})();