(function () {
  const isMobile =
    window.matchMedia("(pointer: coarse)").matches || window.innerWidth < 768;

  // Blocks
  const mobileBlock = document.getElementById("mobileCameraBlock");
  const webcamBlock = document.getElementById("webcamBlock");

  // Inputs
  const uploadInput = document.getElementById("uploadInput");
  const mobileCaptureInput = document.getElementById("mobileCaptureInput");
  const capturedField = document.querySelector('input[name="captured_image"]');

  // Previews
  const uploadPreviewWrap = document.getElementById("uploadPreviewWrap");
  const uploadPreviewImg = document.getElementById("uploadPreviewImg");
  const mobilePreviewWrap = document.getElementById("mobilePreviewWrap");
  const mobilePreviewImg = document.getElementById("mobilePreviewImg");
  const webcamPreviewWrap = document.getElementById("webcamPreviewWrap");
  const webcamPreviewImg = document.getElementById("webcamPreviewImg");

  // Webcam elements
  const video = document.getElementById("video");
  const startBtn = document.getElementById("startCamBtn");
  const captureBtn = document.getElementById("captureBtn");
  const stopBtn = document.getElementById("stopCamBtn");

  // Fake progress UI elements
  const form = document.getElementById("scanForm");
  const progressCard = document.getElementById("progressCard");
  const progressBar = document.getElementById("progressBar");
  const progressText = document.getElementById("progressText");
  const progressPct = document.getElementById("progressPct");
  const successOverlay = document.getElementById("successOverlay");
  const confettiCanvas = document.getElementById("confettiCanvas");

  let progressTimer = null;
  let confettiRaf = null;
  let animating = false;

  // ---------- UI toggle ----------
  if (isMobile) {
    if (mobileBlock) mobileBlock.style.display = "block";
    if (webcamBlock) webcamBlock.style.display = "none";
    if (uploadInput) uploadInput.removeAttribute("capture"); // gallery
    if (mobileCaptureInput) mobileCaptureInput.setAttribute("capture", "environment"); // camera
  } else {
    if (mobileBlock) mobileBlock.style.display = "none";
    if (webcamBlock) webcamBlock.style.display = "block";
  }

  // ---------- Helpers ----------
  function showFilePreview(file, imgEl, wrapEl) {
    if (!file) return;
    const url = URL.createObjectURL(file);
    if (imgEl) imgEl.src = url;
    if (wrapEl) wrapEl.style.display = "block";
  }

  function clearWebcamCapture() {
    if (capturedField) capturedField.value = "";
    if (webcamPreviewWrap) webcamPreviewWrap.style.display = "none";
    if (webcamPreviewImg) webcamPreviewImg.src = "";
  }
  function clearUpload() {
    if (uploadInput) uploadInput.value = "";
    if (uploadPreviewWrap) uploadPreviewWrap.style.display = "none";
    if (uploadPreviewImg) uploadPreviewImg.src = "";
  }
  function clearMobileCapture() {
    if (mobileCaptureInput) mobileCaptureInput.value = "";
    if (mobilePreviewWrap) mobilePreviewWrap.style.display = "none";
    if (mobilePreviewImg) mobilePreviewImg.src = "";
  }

  // ---------- Fake progress ----------
  function startFakeProgress() {
    if (!progressCard || !progressBar || !progressText || !progressPct) return;

    progressCard.style.display = "block";
    progressBar.style.width = "0%";
    progressText.textContent = "Preparing image";
    progressPct.textContent = "0%";

    const stages = [
      { p: 12, t: "Preparing image" },
      { p: 30, t: "Checking quality" },
      { p: 55, t: "Running AI model" },
      { p: 76, t: "Generating recommendations" },
      { p: 90, t: "Finalizing result" },
    ];

    let current = 0;
    let stageIndex = 0;
    let target = stages[stageIndex].p;
    let pauseUntil = 0;

    animating = true;

    function tick() {
      if (!animating) return;

      const now = Date.now();
      if (now < pauseUntil) {
        progressTimer = window.setTimeout(tick, 120);
        return;
      }

      // slower easing
      const delta = Math.max(0.25, (target - current) * 0.035);
      current = Math.min(target, current + delta);

      progressBar.style.width = `${current.toFixed(0)}%`;
      progressPct.textContent = `${current.toFixed(0)}%`;

      // stage advance
      if (current >= target - 0.5) {
        progressText.textContent = stages[stageIndex].t;

        if (stageIndex < stages.length - 1) {
          stageIndex += 1;
          target = stages[stageIndex].p;
          pauseUntil = Date.now() + 650; // pause between stages
          progressText.textContent = stages[stageIndex].t;
        } else {
          // hover near the end while server responds
          target = 93;
        }
      }

      progressTimer = window.setTimeout(tick, 90);
    }

    tick();
  }

  function stopFakeProgress() {
    animating = false;
    if (progressTimer) window.clearTimeout(progressTimer);
    progressTimer = null;
  }

  // ---------- Success overlay + confetti (home) ----------
  function showSuccessOverlay() {
    if (!successOverlay || !confettiCanvas) return;
    successOverlay.style.display = "flex";
    runConfetti(confettiCanvas);
  }

    function runConfetti(canvas) {
    const ctx = canvas.getContext("2d");

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

    // Dermatology-appropriate soft skin tones
    const skinTones = [
        "rgba(255, 224, 189, 0.55)",
        "rgba(241, 194, 125, 0.55)",
        "rgba(224, 172, 105, 0.55)",
        "rgba(198, 134, 66, 0.55)",
        "rgba(141, 85, 36, 0.45)"
    ];

    // Create soft irregular flakes
    const flakes = Array.from({ length: 160 }).map(() => ({
        x: Math.random() * w,
        y: -20 - Math.random() * h,
        size: 6 + Math.random() * 8,
        vx: -0.4 + Math.random() * 0.8,   // gentle drift
        vy: 0.5 + Math.random() * 1.2,    // slow fall
        rot: Math.random() * Math.PI,
        vr: -0.01 + Math.random() * 0.02,
        opacity: 0.3 + Math.random() * 0.4,
        tone: skinTones[Math.floor(Math.random() * skinTones.length)]
    }));

    const start = performance.now();

    function drawFlake(p) {
        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.rotate(p.rot);
        ctx.globalAlpha = p.opacity;
        ctx.fillStyle = p.tone;

        // irregular organic skin flake shape
        ctx.beginPath();
        ctx.moveTo(0, -p.size * 0.6);
        ctx.bezierCurveTo(
        p.size * 0.6, -p.size * 0.6,
        p.size * 0.8, p.size * 0.4,
        0, p.size
        );
        ctx.bezierCurveTo(
        -p.size * 0.8, p.size * 0.4,
        -p.size * 0.6, -p.size * 0.6,
        0, -p.size * 0.6
        );
        ctx.fill();

        ctx.restore();
        ctx.globalAlpha = 1;
    }

    function frame(now) {
        const t = now - start;
        ctx.clearRect(0, 0, w, h);

        flakes.forEach(p => {
        p.x += p.vx + Math.sin(p.y * 0.01) * 0.2; // soft floating motion
        p.y += p.vy;
        p.rot += p.vr;

        drawFlake(p);
        });

        // longer gentle duration
        if (t < 3500) {
        confettiRaf = requestAnimationFrame(frame);
        }
    }

    function onResize() {
        ({ w, h } = resize());
    }

    window.addEventListener("resize", onResize, { passive: true });
    confettiRaf = requestAnimationFrame(frame);

    setTimeout(() => {
        window.removeEventListener("resize", onResize);
    }, 3600);
    }

  // ---------- Upload / mobile capture ----------
  if (uploadInput) {
    uploadInput.addEventListener("change", () => {
      if (uploadInput.files && uploadInput.files.length > 0) {
        showFilePreview(uploadInput.files[0], uploadPreviewImg, uploadPreviewWrap);
        clearWebcamCapture();
        clearMobileCapture();
      }
    });
  }

  if (mobileCaptureInput) {
    mobileCaptureInput.addEventListener("change", () => {
      if (mobileCaptureInput.files && mobileCaptureInput.files.length > 0) {
        const file = mobileCaptureInput.files[0];
        showFilePreview(file, mobilePreviewImg, mobilePreviewWrap);

        // Try to copy into upload input (some browsers block it — ok)
        try {
          if (uploadInput) uploadInput.files = mobileCaptureInput.files;
        } catch (e) {}

        clearWebcamCapture();
        if (uploadPreviewWrap) uploadPreviewWrap.style.display = "none";
      }
    });
  }

  // ---------- Webcam ----------
  let stream = null;

  async function startCamera() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      if (video) video.srcObject = stream;
      if (captureBtn) captureBtn.disabled = false;
      if (stopBtn) stopBtn.disabled = false;
      if (startBtn) startBtn.disabled = true;
    } catch (err) {
      alert("Could not access camera. Allow permission or use upload.\n\n" + err);
    }
  }

  function stopCamera() {
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }
    if (video) video.srcObject = null;
    if (captureBtn) captureBtn.disabled = true;
    if (stopBtn) stopBtn.disabled = true;
    if (startBtn) startBtn.disabled = false;
  }

  function captureFrame() {
    if (!video || !video.videoWidth) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
    if (capturedField) capturedField.value = dataUrl;

    clearUpload();
    clearMobileCapture();

    if (webcamPreviewImg) webcamPreviewImg.src = dataUrl;
    if (webcamPreviewWrap) webcamPreviewWrap.style.display = "block";

    // auto-close webcam
    stopCamera();
  }

  if (startBtn) startBtn.addEventListener("click", startCamera);
  if (stopBtn) stopBtn.addEventListener("click", stopCamera);
  if (captureBtn) captureBtn.addEventListener("click", captureFrame);

  // ---------- Submit: start progress + show overlay ----------
  if (form) {
    form.addEventListener("submit", () => {
      const hasUpload = uploadInput && uploadInput.files && uploadInput.files.length > 0;
      const hasMobile = mobileCaptureInput && mobileCaptureInput.files && mobileCaptureInput.files.length > 0;
      const hasWebcam = capturedField && capturedField.value && capturedField.value.startsWith("data:image");

      if (!(hasUpload || hasMobile || hasWebcam)) return;

      startFakeProgress();

      // show success overlay after ~1.2s (long enough to notice)
      window.setTimeout(() => {
        showSuccessOverlay();
      }, 700);
    });
  }

  // Cleanup
  window.addEventListener("beforeunload", () => {
    stopFakeProgress();
    if (confettiRaf) cancelAnimationFrame(confettiRaf);
  });
  window.addEventListener("beforeunload", stopCamera);
})();