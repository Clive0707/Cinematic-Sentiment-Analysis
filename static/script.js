/**
 * MarathiSense AI — Frontend Controller
 */

document.addEventListener("DOMContentLoaded", () => {
    /* ----- Navigation Tabs Switching ----- */
    const navTabs = document.querySelectorAll(".nav-tab");
    const tabContents = document.querySelectorAll(".tab-content");

    navTabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            const targetTabId = `tab-${tab.dataset.tab}`;

            navTabs.forEach((t) => t.classList.remove("active"));
            tabContents.forEach((c) => c.classList.add("hidden"));

            tab.classList.add("active");
            const targetEl = document.getElementById(targetTabId);
            if (targetEl) {
                targetEl.classList.remove("hidden");
            }
        });
    });

    /* ----- Form Elements & Sample Presets ----- */
    const form = document.getElementById("review-form");
    const textarea = document.getElementById("review");
    const analyzeBtn = document.getElementById("analyze-btn");
    const charCounter = document.getElementById("char-counter");
    const errorMessage = document.getElementById("error-message");
    const presetChips = document.querySelectorAll(".preset-chip");

    if (textarea) {
        function updateCharCount() {
            const len = textarea.value.length;
            if (charCounter) {
                charCounter.textContent = `${len} character${len !== 1 ? "s" : ""}`;
            }
        }

        textarea.addEventListener("input", () => {
            updateCharCount();
            if (errorMessage) errorMessage.classList.add("hidden");
        });

        updateCharCount();
    }

    presetChips.forEach((chip) => {
        chip.addEventListener("click", () => {
            const sampleText = chip.dataset.sample;
            if (textarea && sampleText) {
                textarea.value = sampleText;
                textarea.dispatchEvent(new Event("input"));
                textarea.focus();
            }
        });
    });

    /* ----- Keyboard Shortcut & Form Submit ----- */
    if (textarea && form) {
        textarea.addEventListener("keydown", (e) => {
            if (e.ctrlKey && e.key === "Enter") {
                e.preventDefault();
                form.requestSubmit();
            }
        });
    }

    if (form) {
        form.addEventListener("submit", (e) => {
            const text = textarea ? textarea.value.trim() : "";
            if (!text) {
                e.preventDefault();
                if (errorMessage) errorMessage.classList.remove("hidden");
                if (textarea) textarea.focus();
                return;
            }

            if (analyzeBtn) {
                analyzeBtn.disabled = true;
                const btnText = analyzeBtn.querySelector(".btn-text");
                const btnLoader = analyzeBtn.querySelector(".btn-loader");
                if (btnText) btnText.classList.add("hidden");
                if (btnLoader) btnLoader.classList.remove("hidden");
            }
        });
    }

    /* ----- Smooth Scroll to Results if Present ----- */
    const resultsAnchor = document.getElementById("results-anchor");
    if (resultsAnchor) {
        setTimeout(() => {
            resultsAnchor.scrollIntoView({ behavior: "smooth", block: "start" });
        }, 200);
    }
});
