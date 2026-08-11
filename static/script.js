/**
 * MarathiSense AI — Frontend Interactions
 */

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("review-form");
    const textarea = document.getElementById("review");
    const analyzeBtn = document.getElementById("analyze-btn");
    const charCounter = document.getElementById("char-counter");
    const errorMessage = document.getElementById("error-message");
    const resultPanel = document.getElementById("prediction-result");

    if (!form || !textarea) return;

    /* ----- Character Counter ----- */
    function updateCharCounter() {
        const length = textarea.value.length;
        charCounter.textContent = `${length} character${length !== 1 ? "s" : ""}`;
    }

    textarea.addEventListener("input", () => {
        updateCharCounter();
        hideError();
    });

    updateCharCounter();

    /* ----- Error Handling ----- */
    function showError() {
        errorMessage.classList.remove("hidden");
        textarea.focus();
    }

    function hideError() {
        errorMessage.classList.add("hidden");
    }

    /* ----- Form Submission ----- */
    function submitForm() {
        const text = textarea.value.trim();
        if (!text) {
            showError();
            return;
        }
        hideError();
        analyzeBtn.disabled = true;
        analyzeBtn.classList.add("loading");
        form.submit();
    }

    form.addEventListener("submit", (event) => {
        const text = textarea.value.trim();
        if (!text) {
            event.preventDefault();
            showError();
            return;
        }
        hideError();
        analyzeBtn.disabled = true;
        analyzeBtn.classList.add("loading");
    });

    textarea.addEventListener("keydown", (event) => {
        if (event.ctrlKey && event.key === "Enter") {
            event.preventDefault();
            submitForm();
        }
    });

    /* ----- Confidence Visualization ----- */
    function getConfidenceTier(value) {
        if (value >= 80) return { label: "Very High", className: "tier-high" };
        if (value >= 60) return { label: "High", className: "tier-high" };
        if (value >= 40) return { label: "Moderate", className: "tier-mid" };
        return { label: "Low", className: "tier-low" };
    }

    function buildConfidenceSegments(container, value, totalSegments = 20) {
        if (!container) return;

        container.innerHTML = "";
        const pct = Math.min(100, Math.max(0, parseFloat(value) || 0));
        const activeCount = Math.round((pct / 100) * totalSegments);

        for (let i = 0; i < totalSegments; i++) {
            const segment = document.createElement("div");
            segment.className = "confidence-segment";
            container.appendChild(segment);

            if (i < activeCount) {
                setTimeout(() => {
                    segment.classList.add("active");
                }, 80 + i * 45);
            }
        }
    }

    function animateConfidenceNumber(el, targetValue, duration = 900) {
        if (!el) return;

        const target = parseFloat(targetValue) || 0;
        const start = performance.now();

        function tick(now) {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = target * eased;
            el.textContent = current.toFixed(1);

            if (progress < 1) {
                requestAnimationFrame(tick);
            } else {
                el.textContent = target.toFixed(1);
            }
        }

        requestAnimationFrame(tick);
    }

    function animateConfidenceBar(fillEl, headEl, value) {
        const pct = Math.min(100, Math.max(0, parseFloat(value) || 0));

        requestAnimationFrame(() => {
            if (fillEl) fillEl.style.width = `${pct}%`;
            if (headEl) headEl.style.left = `${pct}%`;
        });
    }

    if (resultPanel) {
        const fillEl = resultPanel.querySelector(".confidence-fill-glow");
        const headEl = resultPanel.querySelector(".confidence-head");
        const segmentsEl = document.getElementById("confidence-segments");
        const numberEl = document.getElementById("confidence-number");
        const tierEl = document.getElementById("confidence-tier");
        const confidenceValue = fillEl?.dataset.value;

        if (confidenceValue) {
            const tier = getConfidenceTier(parseFloat(confidenceValue));

            if (tierEl) {
                tierEl.textContent = tier.label;
                tierEl.classList.add(tier.className);
            }

            animateConfidenceNumber(numberEl, confidenceValue);
            buildConfidenceSegments(segmentsEl, confidenceValue);
            animateConfidenceBar(fillEl, headEl, confidenceValue);
        }

        setTimeout(() => {
            resultPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }, 300);
    }
});
