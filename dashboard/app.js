/* ═══════════════════════════════════════════════════════════════════════
   FraudLens Dashboard — Client-Side Logic
   ═══════════════════════════════════════════════════════════════════════ */

(function () {
    'use strict';

    // DOM refs
    const form = document.getElementById('fraud-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnLoader = document.getElementById('btn-loader');
    const resultsSection = document.getElementById('results-section');
    const dropzone = document.getElementById('dropzone');
    const imageInput = document.getElementById('image-input');
    const dropzoneContent = document.getElementById('dropzone-content');
    const imagePreview = document.getElementById('image-preview');

    // Gauge refs
    const gaugeRing = document.getElementById('gauge-ring');
    const gaugeScore = document.getElementById('gauge-score');
    const riskBadge = document.getElementById('risk-badge');

    // Modality bars
    const attnTabular = document.getElementById('attn-tabular');
    const attnImage = document.getElementById('attn-image');
    const attnText = document.getElementById('attn-text');
    const attnTabularBar = document.getElementById('attn-tabular-bar');
    const attnImageBar = document.getElementById('attn-image-bar');
    const attnTextBar = document.getElementById('attn-text-bar');

    // Branch scores
    const branchTabular = document.getElementById('branch-tabular');
    const branchImage = document.getElementById('branch-image');
    const branchText = document.getElementById('branch-text');

    // Reasons and raw
    const reasonsList = document.getElementById('reasons-list');
    const rawJson = document.getElementById('raw-json');

    /* ── Image Upload ─────────────────────────────────────────────────── */

    dropzone.addEventListener('click', () => imageInput.click());

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            imageInput.files = files;
            showImagePreview(files[0]);
        }
    });

    imageInput.addEventListener('change', () => {
        if (imageInput.files.length > 0) {
            showImagePreview(imageInput.files[0]);
        }
    });

    function showImagePreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.hidden = false;
            dropzoneContent.hidden = true;
        };
        reader.readAsDataURL(file);
    }

    /* ── Form Submission ──────────────────────────────────────────────── */

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Show loading
        submitBtn.disabled = true;
        btnLoader.hidden = false;

        const formData = new FormData(form);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            renderResults(data);
        } catch (err) {
            alert(`Analysis failed: ${err.message}`);
            console.error('[FraudLens]', err);
        } finally {
            submitBtn.disabled = false;
            btnLoader.hidden = true;
        }
    });

    /* ── Render Results ───────────────────────────────────────────────── */

    function renderResults(data) {
        resultsSection.hidden = false;
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

        // Animate gauge
        animateGauge(data.fraud_score, data.risk_level);

        // Modality bars
        setTimeout(() => {
            setBar(attnTabular, attnTabularBar, data.attention_weights.tabular);
            setBar(attnImage, attnImageBar, data.attention_weights.image);
            setBar(attnText, attnTextBar, data.attention_weights.text);

            // Toggle disabled state for absent modalities
            toggleDisabled(attnImageBar.parentElement.parentElement, data.attention_weights.image);
            toggleDisabled(attnTextBar.parentElement.parentElement, data.attention_weights.text);
        }, 400);

        // Branch scores
        branchTabular.textContent = data.branch_scores.tabular + '%';
        branchImage.textContent = data.branch_scores.image + '%';
        branchText.textContent = data.branch_scores.text + '%';

        colorBranchScore(branchTabular, data.branch_scores.tabular);
        colorBranchScore(branchImage, data.branch_scores.image);
        colorBranchScore(branchText, data.branch_scores.text);

        // Risk reasons
        reasonsList.innerHTML = '';
        data.risk_reasons.forEach((reason) => {
            const li = document.createElement('li');
            li.textContent = reason;
            reasonsList.appendChild(li);
        });

        // Deep Analysis Explanations
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

        // Raw JSON
        rawJson.textContent = JSON.stringify(data, null, 2);
    }

    function animateGauge(score, level) {
        // Color based on score
        let color;
        if (score >= 80) color = 'var(--accent-red)';
        else if (score >= 60) color = 'var(--accent-orange)';
        else if (score >= 35) color = 'var(--accent-yellow)';
        else color = 'var(--accent-green)';

        // Animate score number
        const duration = 1200;
        const startTime = performance.now();
        const startVal = 0;

        function tick(now) {
            const elapsed = now - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
            const current = Math.round(startVal + (score - startVal) * eased);

            gaugeScore.textContent = current;

            // Update ring
            const pct = current;
            gaugeRing.style.background = `conic-gradient(
                ${color} 0%,
                ${color} ${pct}%,
                rgba(255, 255, 255, 0.05) ${pct}%
            )`;

            if (progress < 1) {
                requestAnimationFrame(tick);
            }
        }

        requestAnimationFrame(tick);

        // Risk badge
        riskBadge.textContent = level;
        riskBadge.className = 'risk-badge risk-badge--' + level.toLowerCase();
    }

    function setBar(valueEl, barEl, pct) {
        valueEl.textContent = pct.toFixed(1) + '%';
        barEl.style.width = pct + '%';
    }

    function toggleDisabled(barContainer, weight) {
        if (weight <= 0) {
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
