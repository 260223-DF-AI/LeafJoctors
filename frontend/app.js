(() => {
  "use strict";

  // ── DOM refs ──────────────────────────────────────────────
  const dropZone       = document.getElementById("dropZone");
  const fileInput      = document.getElementById("fileInput");
  const uploadPrompt   = document.getElementById("uploadPrompt");
  const previewArea    = document.getElementById("previewArea");
  const previewImg     = document.getElementById("previewImg");
  const fileName       = document.getElementById("fileName");
  const removeWrapper  = document.getElementById("removeWrapper");
  const removeBtn      = document.getElementById("removeBtn");
  const analyzeBtn     = document.getElementById("analyzeBtn");
  const analyzeBtnText = document.getElementById("analyzeBtnText");
  const analyzeBtnSpinner = document.getElementById("analyzeBtnSpinner");
  const errorAlert     = document.getElementById("errorAlert");
  const errorText      = document.getElementById("errorText");
  const resultsPanel   = document.getElementById("resultsPanel");
  const classificationBadge = document.getElementById("classificationBadge");
  const confidenceBar  = document.getElementById("confidenceBar");
  const confidenceText = document.getElementById("confidenceText");
  const treatmentText  = document.getElementById("treatmentText");
  const demoBanner     = document.getElementById("demoBanner");

  // ── State ─────────────────────────────────────────────────
  let selectedFile = null;
  let isLoading    = false;

  const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB
  const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/gif"];

  const API_URL = "/vision/analyze";

  // ── Classification styling map ────────────────────────────
  const CLASS_STYLES = {
    "Healthy": {
      bg: "bg-green-100", text: "text-green-800", bar: "bg-green-500",
    },
    "Nutrient Deficient": {
      bg: "bg-amber-100", text: "text-amber-800", bar: "bg-amber-500",
    },
    "Pest-Infested": {
      bg: "bg-red-100", text: "text-red-800", bar: "bg-red-500",
    },
  };

  // ── Mock data for demo mode ───────────────────────────────
  const MOCK_RESPONSES = [
    {
      classification: "Healthy",
      confidence: 0.94,
      treatment: "No treatment needed. Your plant looks great! Continue your current care regimen — consistent watering, adequate sunlight, and proper drainage are keeping this leaf in excellent condition.",
    },
    {
      classification: "Nutrient Deficient",
      confidence: 0.87,
      treatment: "This leaf shows signs of nutrient deficiency, likely nitrogen or potassium. We recommend applying a balanced NPK fertilizer (10-10-10) and testing the soil pH. Ensure the soil pH is between 6.0–7.0 for optimal nutrient uptake. Consider foliar feeding for faster recovery.",
    },
    {
      classification: "Pest-Infested",
      confidence: 0.91,
      treatment: "Pest damage detected on this leaf. Inspect the undersides of leaves for aphids, mites, or whiteflies. Apply neem oil spray (2 tbsp per gallon of water) every 7 days. For severe infestations, consider introducing beneficial insects like ladybugs or lacewings. Remove heavily damaged leaves to prevent spread.",
    },
  ];

  // ── Helpers ───────────────────────────────────────────────
  function showError(msg) {
    errorText.textContent = msg;
    errorAlert.classList.remove("hidden");
  }

  function hideError() {
    errorAlert.classList.add("hidden");
  }

  function setLoading(loading) {
    isLoading = loading;
    analyzeBtn.disabled = loading;
    analyzeBtnText.textContent = loading ? "Analyzing…" : "Analyze Leaf";
    analyzeBtnSpinner.classList.toggle("hidden", !loading);
  }

  function validateFile(file) {
    if (!file) return "No file selected.";
    if (!ALLOWED_TYPES.includes(file.type)) {
      return "Invalid file type. Please upload a PNG, JPG, WebP, or GIF image.";
    }
    if (file.size > MAX_FILE_SIZE) {
      return `File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Maximum size is 10 MB.`;
    }
    return null;
  }

  // ── File selection ────────────────────────────────────────
  function handleFile(file) {
    hideError();

    const err = validateFile(file);
    if (err) {
      showError(err);
      return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      uploadPrompt.classList.add("hidden");
      previewArea.classList.remove("hidden");
      removeWrapper.classList.remove("hidden");
      fileName.textContent = file.name;
      analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  function clearFile() {
    selectedFile = null;
    fileInput.value = "";
    previewImg.src = "";
    previewArea.classList.add("hidden");
    uploadPrompt.classList.remove("hidden");
    removeWrapper.classList.add("hidden");
    resultsPanel.classList.add("hidden");
    analyzeBtn.disabled = true;
    demoBanner.classList.add("hidden");
    hideError();
  }

  // ── Render results ────────────────────────────────────────
  function renderResults(data) {
    const style = CLASS_STYLES[data.classification] || CLASS_STYLES["Healthy"];

    // Badge
    classificationBadge.className = `inline-block px-4 py-1.5 rounded-full text-sm font-bold uppercase tracking-wide ${style.bg} ${style.text}`;
    classificationBadge.textContent = data.classification;

    // Confidence bar
    const pct = Math.round(data.confidence * 100);
    confidenceBar.className = `h-full rounded-full transition-all duration-700 ease-out ${style.bar}`;
    // Trigger reflow for animation
    confidenceBar.style.width = "0%";
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        confidenceBar.style.width = `${pct}%`;
      });
    });
    confidenceText.textContent = `${pct}%`;

    // Treatment
    treatmentText.textContent = data.treatment;

    // Show panel
    resultsPanel.classList.remove("hidden");
    resultsPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // ── API call / mock ────────────────────────────────────────
  async function analyzeLeaf() {
    if (!selectedFile || isLoading) return;

    hideError();
    setLoading(true);
    resultsPanel.classList.add("hidden");

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const body = await response.text();
        throw new Error(`Server error ${response.status}: ${body}`);
      }

      const data = await response.json();
      demoBanner.classList.add("hidden");
      renderResults(data);
    } catch (err) {
      // If API unreachable, fall back to mock
      console.warn("API unavailable, using demo mode:", err.message);
      demoBanner.classList.remove("hidden");

      const mock = MOCK_RESPONSES[Math.floor(Math.random() * MOCK_RESPONSES.length)];
      // Add slight random jitter to confidence for realism
      const jittered = { ...mock, confidence: mock.confidence + (Math.random() * 0.06 - 0.03) };
      jittered.confidence = Math.min(0.99, Math.max(0.5, jittered.confidence));

      // Simulate network delay
      await new Promise((r) => setTimeout(r, 800 + Math.random() * 600));
      renderResults(jittered);
    } finally {
      setLoading(false);
    }
  }

  // ── Event listeners ───────────────────────────────────────

  // Click to browse
  dropZone.addEventListener("click", () => {
    if (!isLoading) fileInput.click();
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
  });

  // Drag & drop
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drop-highlight");
  });

  dropZone.addEventListener("dragleave", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drop-highlight");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drop-highlight");
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
  });

  // Remove file
  removeBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    clearFile();
  });

  // Prevent remove wrapper click from opening file dialog
  removeWrapper.addEventListener("click", (e) => e.stopPropagation());

  // Analyze
  analyzeBtn.addEventListener("click", analyzeLeaf);

  // Keyboard: Enter to analyze
  document.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && selectedFile && !isLoading) analyzeLeaf();
  });
})();
