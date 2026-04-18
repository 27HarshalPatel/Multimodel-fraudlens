/* ═══════════════════════════════════════════════════════════════════════
   FraudLens Dashboard — Client-Side Logic
   Supports two modes: Image Scan (OCR) and Full Analysis (3-Modal)
   ═══════════════════════════════════════════════════════════════════════ */

(function () {
    'use strict';

    // ── Mode state ──────────────────────────────────────────────────
    let currentMode = 'image-scan'; // 'image-scan' | 'full-analysis'

    // ── Tab refs ────────────────────────────────────────────────────
    const tabImageScan = document.getElementById('tab-image-scan');
    const tabFullAnalysis = document.getElementById('tab-full-analysis');
    const tabIndicator = document.getElementById('tab-indicator');

    // ── Section refs ────────────────────────────────────────────────
    const imageScanSection = document.getElementById('image-scan-section');
    const fullAnalysisSection = document.getElementById('full-analysis-section');
    const resultsSection = document.getElementById('results-section');

    // ── Image Scan form refs ────────────────────────────────────────
    const scanForm = document.getElementById('image-scan-form');
    const scanSubmitBtn = document.getElementById('scan-submit-btn');
    const scanBtnLoader = document.getElementById('scan-btn-loader');
    const scanDropzone = document.getElementById('scan-dropzone');
    const scanImageInput = document.getElementById('scan-image-input');
    const scanDropzoneContent = document.getElementById('scan-dropzone-content');
    const scanPreviewWrapper = document.getElementById('scan-preview-wrapper');
    const scanImagePreview = document.getElementById('scan-image-preview');
    const scanRemoveBtn = document.getElementById('scan-remove-btn');

    // ── Full Analysis form refs ─────────────────────────────────────
    const fullForm = document.getElementById('fraud-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnLoader = document.getElementById('btn-loader');
    const dropzone = document.getElementById('dropzone');
    const imageInput = document.getElementById('image-input');
    const dropzoneContent = document.getElementById('dropzone-content');
    const fullPreviewWrapper = document.getElementById('full-preview-wrapper');
    const imagePreview = document.getElementById('image-preview');
    const fullRemoveBtn = document.getElementById('full-remove-btn');

    // ── Result refs ─────────────────────────────────────────────────
    const gaugeRing = document.getElementById('gauge-ring');
    const gaugeScore = document.getElementById('gauge-score');
    const riskBadge = document.getElementById('risk-badge');
    const attnTabular = document.getElementById('attn-tabular');
    const attnImage = document.getElementById('attn-image');
    const attnText = document.getElementById('attn-text');
    const attnTabularBar = document.getElementById('attn-tabular-bar');
    const attnImageBar = document.getElementById('attn-image-bar');
    const attnTextBar = document.getElementById('attn-text-bar');
    const branchTabular = document.getElementById('branch-tabular');
    const branchImage = document.getElementById('branch-image');
    const branchText = document.getElementById('branch-text');
    const reasonsList = document.getElementById('reasons-list');
    const rawJson = document.getElementById('raw-json');

    // ── OCR refs ────────────────────────────────────────────────────
    const ocrSection = document.getElementById('ocr-section');
    const ocrText = document.getElementById('ocr-text');
    const ocrKeywordsList = document.getElementById('ocr-keywords-list');
    const ocrKeywordsTags = document.getElementById('ocr-keywords-tags');

    /* ═══════════════════════════════════════════════════════════════════
       TAB SWITCHING
       ═══════════════════════════════════════════════════════════════════ */

    function switchMode(mode) {
        currentMode = mode;
        resultsSection.hidden = true;

        if (mode === 'image-scan') {
            tabImageScan.classList.add('mode-tab--active');
            tabFullAnalysis.classList.remove('mode-tab--active');
            tabIndicator.classList.remove('at-full');
            imageScanSection.hidden = false;
            imageScanSection.style.animation = 'none';
            imageScanSection.offsetHeight; // trigger reflow
            imageScanSection.style.animation = 'cardIn 0.4s ease-out';
            fullAnalysisSection.hidden = true;
        } else {
            tabFullAnalysis.classList.add('mode-tab--active');
            tabImageScan.classList.remove('mode-tab--active');
            tabIndicator.classList.add('at-full');
            fullAnalysisSection.hidden = false;
            fullAnalysisSection.style.animation = 'none';
            fullAnalysisSection.offsetHeight;
            fullAnalysisSection.style.animation = 'cardIn 0.4s ease-out';
            imageScanSection.hidden = true;
        }
    }

    tabImageScan.addEventListener('click', () => switchMode('image-scan'));
    tabFullAnalysis.addEventListener('click', () => switchMode('full-analysis'));

    /* ═══════════════════════════════════════════════════════════════════
       IMAGE SCAN — Dropzone
       ═══════════════════════════════════════════════════════════════════ */

    function setupDropzone(dz, input, content, previewWrapper, preview, removeBtn, onFileChange) {
        dz.addEventListener('click', (e) => {
            if (e.target === removeBtn || removeBtn.contains(e.target)) return;
            input.click();
        });

        dz.addEventListener('dragover', (e) => {
            e.preventDefault();
            dz.classList.add('dragover');
        });

        dz.addEventListener('dragleave', () => {
            dz.classList.remove('dragover');
        });

        dz.addEventListener('drop', (e) => {
            e.preventDefault();
            dz.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                input.files = e.dataTransfer.files;
                showPreview(e.dataTransfer.files[0], content, previewWrapper, preview);
                if (onFileChange) onFileChange(true);
            }
        });

        input.addEventListener('change', () => {
            if (input.files.length > 0) {
                showPreview(input.files[0], content, previewWrapper, preview);
                if (onFileChange) onFileChange(true);
            }
        });

        if (removeBtn) {
            removeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                input.value = '';
                content.hidden = false;
                previewWrapper.hidden = true;
                if (onFileChange) onFileChange(false);
            });
        }
    }

    function showPreview(file, content, previewWrapper, preview) {
        const reader = new FileReader();
        reader.onload = (e) => {
            preview.src = e.target.result;
            previewWrapper.hidden = false;
            content.hidden = true;
        };
        reader.readAsDataURL(file);
    }

    // Setup scan dropzone
    setupDropzone(
        scanDropzone, scanImageInput, scanDropzoneContent,
        scanPreviewWrapper, scanImagePreview, scanRemoveBtn,
        (hasFile) => { scanSubmitBtn.disabled = !hasFile; }
    );

    // Setup full analysis dropzone
    setupDropzone(
        dropzone, imageInput, dropzoneContent,
        fullPreviewWrapper, imagePreview, fullRemoveBtn,
        null
    );

    /* ═══════════════════════════════════════════════════════════════════
       IMAGE SCAN — Form Submission
       ═══════════════════════════════════════════════════════════════════ */

    scanForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        scanSubmitBtn.disabled = true;
        scanBtnLoader.hidden = false;

        const formData = new FormData();
        formData.append('image', scanImageInput.files[0]);

        try {
            const response = await fetch('/api/analyze-image', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.error || `Server error: ${response.status}`);
            }

            const data = await response.json();
            if (data.error) throw new Error(data.error);
            renderResults(data);
        } catch (err) {
            alert(`Image analysis failed: ${err.message}`);
            console.error('[FraudLens]', err);
        } finally {
            scanSubmitBtn.disabled = false;
            scanBtnLoader.hidden = true;
        }
    });

    /* ═══════════════════════════════════════════════════════════════════
       FULL ANALYSIS — Form Submission
       ═══════════════════════════════════════════════════════════════════ */

    fullForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        submitBtn.disabled = true;
        btnLoader.hidden = false;

        const formData = new FormData(fullForm);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();
            if (data.error) throw new Error(data.error);
            renderResults(data);
        } catch (err) {
            alert(`Analysis failed: ${err.message}`);
            console.error('[FraudLens]', err);
        } finally {
            submitBtn.disabled = false;
            btnLoader.hidden = true;
        }
    });

    /* ═══════════════════════════════════════════════════════════════════
       RENDER RESULTS (shared between both modes)
       ═══════════════════════════════════════════════════════════════════ */

    function renderResults(data) {
        resultsSection.hidden = false;
        resultsSection.style.animation = 'none';
        resultsSection.offsetHeight;
        resultsSection.style.animation = 'cardIn 0.5s ease-out';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

        // Animate gauge
        animateGauge(data.fraud_score, data.risk_level);

        // ── OCR Section ─────────────────────────────────────────────
        if (data.ocr_extracted_text && data.ocr_extracted_text.trim()) {
            ocrSection.hidden = false;
            ocrText.textContent = data.ocr_extracted_text;

            if (data.ocr_keywords && data.ocr_keywords.length > 0) {
                ocrKeywordsList.hidden = false;
                ocrKeywordsTags.innerHTML = '';
                data.ocr_keywords.forEach((kw) => {
                    const tag = document.createElement('span');
                    tag.className = 'ocr-keyword-tag';
                    tag.textContent = kw;
                    ocrKeywordsTags.appendChild(tag);
                });
            } else {
                ocrKeywordsList.hidden = true;
            }
        } else {
            ocrSection.hidden = true;
        }

        // ── Modality bars ───────────────────────────────────────────
        setTimeout(() => {
            setBar(attnTabular, attnTabularBar, data.attention_weights.tabular);
            setBar(attnImage, attnImageBar, data.attention_weights.image);
            setBar(attnText, attnTextBar, data.attention_weights.text);

            // Toggle disabled state
            const isImageScanMode = currentMode === 'image-scan';

            if (isImageScanMode) {
                // In image scan mode: tabular is always disabled
                toggleDisabled(attnTabularBar.parentElement.parentElement, false);
                toggleDisabled(attnImageBar.parentElement.parentElement, true);
                toggleDisabled(attnTextBar.parentElement.parentElement, data.attention_weights.text > 0);
            } else {
                const isImageProvided = imageInput.files && imageInput.files.length > 0;
                const isTextProvided = document.getElementById('description').value.trim().length > 0
                    || (data.ocr_extracted_text && data.ocr_extracted_text.trim());

                toggleDisabled(attnTabularBar.parentElement.parentElement, true);
                toggleDisabled(attnImageBar.parentElement.parentElement, isImageProvided);
                toggleDisabled(attnTextBar.parentElement.parentElement, isTextProvided);
            }
        }, 400);

        // ── Branch scores ───────────────────────────────────────────
        branchTabular.textContent = data.branch_scores.tabular + '%';
        branchImage.textContent = data.branch_scores.image + '%';
        branchText.textContent = data.branch_scores.text + '%';

        colorBranchScore(branchTabular, data.branch_scores.tabular);
        colorBranchScore(branchImage, data.branch_scores.image);
        colorBranchScore(branchText, data.branch_scores.text);

        // ── Risk reasons ────────────────────────────────────────────
        reasonsList.innerHTML = '';
        data.risk_reasons.forEach((reason) => {
            const li = document.createElement('li');
            li.textContent = reason;
            reasonsList.appendChild(li);
        });

        // ── Deep Analysis Explanations ──────────────────────────────
        const explanationsSection = document.getElementById('explanations-section');
        const imageExplanationCard = document.getElementById('image-explanation-card');
        const imageHeatmap = document.getElementById('image-heatmap');
        const textExplanationCard = document.getElementById('text-explanation-card');
        const textHighlights = document.getElementById('text-highlights');

        if (data.image_explanation_base64 || (data.text_attributions && data.text_attributions.length > 0)) {
            explanationsSection.hidden = false;

            if (data.image_explanation_base64) {
                imageHeatmap.src = data.image_explanation_base64;
                imageExplanationCard.hidden = false;
            } else {
                imageExplanationCard.hidden = true;
            }

            if (data.text_attributions && data.text_attributions.length > 0) {
                textHighlights.innerHTML = '';
                data.text_attributions.forEach(item => {
                    const span = document.createElement('span');

                    let word = item.word;
                    let isSubword = false;
                    if (word.startsWith('##')) {
                        word = word.substring(2);
                        isSubword = true;
                    } else if (textHighlights.childNodes.length > 0) {
                        textHighlights.appendChild(document.createTextNode(' '));
                    }

                    span.textContent = word;
                    span.className = 'highlight-token';

                    if (item.weight > 0.1) {
                        const intensity = Math.min(item.weight, 1.0);
                        span.style.backgroundColor = `rgba(239, 68, 68, ${intensity * 0.8})`;
                        span.style.color = '#fff';
                    } else if (item.weight < -0.1) {
                        const intensity = Math.min(Math.abs(item.weight), 1.0);
                        span.style.backgroundColor = `rgba(59, 130, 246, ${intensity * 0.8})`;
                        span.style.color = '#fff';
                    }

                    textHighlights.appendChild(span);
                });
                textExplanationCard.hidden = false;
            } else {
                textExplanationCard.hidden = true;
            }
        } else {
            explanationsSection.hidden = true;
        }

        // ── Raw JSON ────────────────────────────────────────────────
        rawJson.textContent = JSON.stringify(data, null, 2);
    }

    /* ═══════════════════════════════════════════════════════════════════
       GAUGE ANIMATION
       ═══════════════════════════════════════════════════════════════════ */

    function animateGauge(score, level) {
        let color;
        if (score >= 80) color = 'var(--accent-red)';
        else if (score >= 60) color = 'var(--accent-orange)';
        else if (score >= 35) color = 'var(--accent-yellow)';
        else color = 'var(--accent-green)';

        const duration = 1200;
        const startTime = performance.now();

        function tick(now) {
            const elapsed = now - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = Math.round(score * eased);

            gaugeScore.textContent = current;

            gaugeRing.style.background = `conic-gradient(
                ${color} 0%,
                ${color} ${current}%,
                rgba(255, 255, 255, 0.05) ${current}%
            )`;

            if (progress < 1) {
                requestAnimationFrame(tick);
            }
        }

        requestAnimationFrame(tick);

        riskBadge.textContent = level;
        riskBadge.className = 'risk-badge risk-badge--' + level.toLowerCase();
    }

    /* ═══════════════════════════════════════════════════════════════════
       HELPER FUNCTIONS
       ═══════════════════════════════════════════════════════════════════ */

    function setBar(valueEl, barEl, pct) {
        valueEl.textContent = pct.toFixed(1) + '%';
        barEl.style.width = pct + '%';
    }

    function toggleDisabled(barContainer, isProvided) {
        if (!isProvided) {
            barContainer.classList.add('disabled');
        } else {
            barContainer.classList.remove('disabled');
        }
    }

    function colorBranchScore(el, score) {
        if (score >= 60) el.style.color = 'var(--accent-red)';
        else if (score >= 40) el.style.color = 'var(--accent-orange)';
        else if (score >= 25) el.style.color = 'var(--accent-yellow)';
        else el.style.color = 'var(--accent-green)';
    }

})();
