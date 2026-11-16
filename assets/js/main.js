/* -------------------------------------------------------------
   "Does CLIP See Place?" Academic Project Website Script
   Interactive Probing Inspector, Dynamic Model Inference, Theme & BibTeX
------------------------------------------------------------- */

let demoData = null;
let currentRegion = "downtown_la";
let currentIntervention = "clean";

document.addEventListener("DOMContentLoaded", async () => {
  initTheme();
  await loadDemoData();
  initLiveDemo();
  initBibtexCopy();
});

/* -------------------------------------------------------------
   1. Theme Management (Dark / Light)
------------------------------------------------------------- */
function initTheme() {
  const toggleBtn = document.getElementById("theme-toggle");
  if (!toggleBtn) return;

  const currentTheme = localStorage.getItem("theme") || 
    (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");

  document.documentElement.setAttribute("data-theme", currentTheme);
  updateThemeIcon(currentTheme);

  toggleBtn.addEventListener("click", () => {
    const activeTheme = document.documentElement.getAttribute("data-theme");
    const newTheme = activeTheme === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", newTheme);
    localStorage.setItem("theme", newTheme);
    updateThemeIcon(newTheme);
  });
}

function updateThemeIcon(theme) {
  const icon = document.getElementById("theme-icon");
  const text = document.getElementById("theme-text");
  if (!icon || !text) return;
  if (theme === "dark") {
    icon.innerHTML = `<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />`;
    text.textContent = "Light";
  } else {
    icon.innerHTML = `<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />`;
    text.textContent = "Dark";
  }
}

/* -------------------------------------------------------------
   2. Live Demo Data Loading & Inspector
------------------------------------------------------------- */
async function loadDemoData() {
  try {
    const res = await fetch("assets/demo_data.json");
    demoData = await res.json();
  } catch (err) {
    console.warn("Could not fetch assets/demo_data.json, trying relative path for docs/", err);
    try {
      const res = await fetch("../assets/demo_data.json");
      demoData = await res.json();
    } catch (e) {
      console.error("Failed to load demo data", e);
    }
  }
}

function initLiveDemo() {
  const regionBtns = document.querySelectorAll(".inspector-region-bar .region-btn");
  const tabBtns = document.querySelectorAll(".inspector-tabs .tab-btn");

  regionBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      regionBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentRegion = btn.getAttribute("data-region");
      updateDemoView();
    });
  });

  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentIntervention = btn.getAttribute("data-intervention");
      updateDemoView();
    });
  });

  updateDemoView();
}

function updateDemoView() {
  if (!demoData || !demoData[currentRegion]) return;

  const regData = demoData[currentRegion];
  const invData = regData.interventions[currentIntervention];
  if (!invData) return;

  const viewerImg = document.getElementById("demo-viewer-img");
  const overlayTag = document.getElementById("demo-overlay-tag");
  const regionTitle = document.getElementById("demo-region-title");
  const regionCoords = document.getElementById("demo-region-coords");

  // Determine image path (handle root vs docs)
  let imgPath = invData.image_url;
  if (window.location.pathname.includes("/docs/")) {
    imgPath = "../" + imgPath;
  }

  viewerImg.style.opacity = "0.25";
  setTimeout(() => {
    viewerImg.src = imgPath;
    overlayTag.textContent = invData.label;
    regionTitle.textContent = `${regData.region_name} (${regData.dist_str})`;
    regionCoords.textContent = `Coordinates: ${regData.coords}`;
    viewerImg.style.opacity = "1";
  }, 120);

  // Update Model Cards
  renderModelCard("zs", invData.zero_shot);
  renderModelCard("lora", invData.lora);
  renderModelCard("ft", invData.full_ft);
}

function renderModelCard(prefix, modelData) {
  const cardElem = document.getElementById(`card-${prefix}`);
  const badgeElem = document.getElementById(`badge-${prefix}`);
  const predText = document.getElementById(`pred-${prefix}`);
  const confText = document.getElementById(`conf-${prefix}`);
  const fillElem = document.getElementById(`fill-${prefix}`);

  if (!cardElem || !modelData) return;

  const isMatch = modelData.match;
  cardElem.className = `model-pred-card ${isMatch ? "correct" : "incorrect"}`;

  badgeElem.className = `pred-badge ${isMatch ? "match" : "mismatch"}`;
  badgeElem.innerHTML = isMatch ? `✓ Match` : `✗ Mispredict`;

  predText.textContent = `Predicted: ${modelData.pred}`;
  confText.textContent = `Confidence: ${modelData.conf.toFixed(1)}%`;

  fillElem.style.width = `${Math.min(100, Math.max(5, modelData.conf))}%`;
  fillElem.className = `pred-conf-fill ${isMatch ? "" : "rose"}`;
}

/* -------------------------------------------------------------
   3. BibTeX One-Click Copy
------------------------------------------------------------- */
function initBibtexCopy() {
  const copyBtn = document.getElementById("copy-bibtex-btn");
  const bibtexText = document.getElementById("bibtex-code");
  if (!copyBtn || !bibtexText) return;

  copyBtn.addEventListener("click", () => {
    const textToCopy = bibtexText.innerText;
    navigator.clipboard.writeText(textToCopy).then(() => {
      const originalText = copyBtn.innerHTML;
      copyBtn.innerHTML = `<svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" /></svg> Copied!`;
      setTimeout(() => {
        copyBtn.innerHTML = originalText;
      }, 2000);
    }).catch(err => {
      console.error("Failed to copy BibTeX: ", err);
    });
  });
}
