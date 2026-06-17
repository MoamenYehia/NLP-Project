/* ── Samples ── */
const SAMPLES = {
  positive: `Just got back from the most incredible trip to Japan! The food was absolutely mind-blowing — from the freshest sushi I've ever tasted to ramen hidden in tiny alleyways. The people were so kind and helpful. Tokyo's energy is unlike anything I've experienced. I cannot wait to go back and explore more of this wonderful country!`,
  negative: `I'm beyond frustrated with this company's customer service. I've been waiting three weeks for a refund promised within five business days. Every time I call, I'm put on hold for over an hour then transferred to someone who has no idea what's going on. This is completely unacceptable. I will never purchase from them again. Total waste of time and money.`,
  mixed: `The movie had some genuinely breathtaking cinematography and the lead actress delivered a powerhouse performance. But the script was a mess — full of plot holes and lazy dialogue that undercut every emotional moment. I left the theater unsure how to feel. Admiration for the craft, deep frustration with the story. Worth seeing for the visuals, just don't expect a satisfying ending.`,
  neutral: `The annual report shows that global average temperatures rose by 1.1 degrees Celsius above pre-industrial levels in 2023. Scientists collected data from 142 monitoring stations across six continents. The findings will be presented at the upcoming international climate conference scheduled for November. Further analysis is expected to take several months.`,
};

const EMOTION_CONFIG = {
  Joy:          { emoji: '😄', color: '#34d399' },
  Sadness:      { emoji: '😢', color: '#60a5fa' },
  Anger:        { emoji: '😠', color: '#f87171' },
  Fear:         { emoji: '😨', color: '#a78bfa' },
  Surprise:     { emoji: '😲', color: '#fbbf24' },
  Disgust:      { emoji: '🤢', color: '#6ee7b7' },
  Trust:        { emoji: '🤝', color: '#7c6af7' },
  Anticipation: { emoji: '🤩', color: '#f9a8d4' },
};

const SENTIMENT_COLORS = {
  Positive: '#34d399',
  Negative: '#f87171',
  Neutral:  '#60a5fa',
  Mixed:    '#fbbf24',
};

const POS_COLORS = ['#7c6af7','#34d399','#f87171','#fbbf24','#60a5fa','#f9a8d4'];

/* ── DOM ── */
const textInput     = document.getElementById('text-input');
const charNum       = document.getElementById('char-num');
const analyzeBtn    = document.getElementById('analyze-btn');
const loading       = document.getElementById('loading');
const errorState    = document.getElementById('error-state');
const errorMsg      = document.getElementById('error-msg');
const resultsContent = document.getElementById('results-content');

let radarChart = null;
let barChart   = null;
let posChart   = null;

/* ── Events ── */
textInput.addEventListener('input', () => { charNum.textContent = textInput.value.length; });
document.querySelectorAll('.sample-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    textInput.value = SAMPLES[btn.dataset.sample] || '';
    charNum.textContent = textInput.value.length;
  });
});
analyzeBtn.addEventListener('click', runAnalysis);

/* ── Main ── */
async function runAnalysis() {
  const text = textInput.value.trim();
  if (!text) { textInput.focus(); return; }

  setLoading(true);
  errorState.hidden    = true;
  resultsContent.hidden = true;

  try {
    const res  = await fetch('/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || 'Analysis failed');
    renderResults(data);
  } catch (err) {
    errorMsg.textContent = err.message;
    errorState.hidden    = false;
  } finally {
    setLoading(false);
  }
}

/* ── Render ── */
function renderResults(data) {
  renderBanner(data.sentiment);
  renderModelScores(data.sentiment.details, data.stats);
  renderEmotionGrid(data.emotions);
  renderCharts(data.emotions);
  renderPosChart(data.stats.pos_distribution);
  renderKeywords(data.keywords);
  document.getElementById('summary-text').textContent = data.summary;

  resultsContent.hidden = false;
  requestAnimationFrame(() => resultsContent.scrollIntoView({ behavior: 'smooth', block: 'start' }));
}

function renderBanner(sentiment) {
  const color = SENTIMENT_COLORS[sentiment.label] || '#7c6af7';
  document.getElementById('sentiment-banner').style.setProperty('--banner-color', color);
  document.getElementById('sentiment-label').textContent      = sentiment.label;
  document.getElementById('sentiment-explanation').textContent = sentiment.explanation;

  const pct    = Math.round(sentiment.score * 100);
  const offset = 314 - 314 * sentiment.score;
  document.getElementById('ring-number').textContent           = pct;
  document.getElementById('ring-fill').style.strokeDashoffset  = offset;
}

function renderModelScores(details, stats) {
  // VADER bars
  const vaderBars = document.getElementById('vader-bars');
  const vaderData = [
    { label: 'Pos', value: details.vader_pos,      color: '#34d399' },
    { label: 'Neg', value: details.vader_neg,      color: '#f87171' },
    { label: 'Neu', value: details.vader_neu,      color: '#60a5fa' },
    { label: 'Cmp', value: (details.vader_compound + 1) / 2, color: '#a78bfa', raw: details.vader_compound },
  ];
  vaderBars.innerHTML = vaderData.map(d => `
    <div class="vader-row">
      <span class="vader-lbl">${d.label}</span>
      <div class="vader-bar-bg">
        <div class="vader-bar-fill" style="width:0%;background:${d.color}" data-target="${d.value * 100}"></div>
      </div>
      <span class="vader-val">${d.raw !== undefined ? d.raw.toFixed(2) : d.value.toFixed(2)}</span>
    </div>
  `).join('');

  // Animate VADER bars
  requestAnimationFrame(() => {
    vaderBars.querySelectorAll('.vader-bar-fill').forEach(el => {
      el.style.width = el.dataset.target + '%';
    });
  });

  // TextBlob meter
  const tbMeter = document.getElementById('textblob-meter');
  const polarity   = details.textblob_polarity;    // -1 to 1
  const subjectivity = details.textblob_subjectivity; // 0 to 1
  const polPct     = ((polarity + 1) / 2) * 100;  // convert to 0-100%

  tbMeter.innerHTML = `
    <div class="tb-row">
      <div class="tb-label">
        <span>Polarity</span>
        <span style="font-family:var(--mono);font-size:0.65rem;color:var(--text-mid)">${polarity.toFixed(3)}</span>
      </div>
      <div class="tb-track">
        <div class="tb-thumb" id="tb-thumb-pol" style="left:50%"></div>
      </div>
      <div class="tb-axis"><span>−1 Neg</span><span>0</span><span>+1 Pos</span></div>
    </div>
    <div class="tb-row" style="margin-top:0.8rem">
      <div class="tb-label">
        <span>Subjectivity</span>
        <span style="font-family:var(--mono);font-size:0.65rem;color:var(--text-mid)">${subjectivity.toFixed(3)}</span>
      </div>
      <div class="tb-track">
        <div class="tb-thumb" id="tb-thumb-sub" style="left:0%;background:#fbbf24;box-shadow:0 0 0 2px rgba(251,191,36,0.3)"></div>
      </div>
      <div class="tb-axis"><span>Objective</span><span>Subjective</span></div>
    </div>
  `;

  requestAnimationFrame(() => {
    document.getElementById('tb-thumb-pol').style.left = polPct + '%';
    document.getElementById('tb-thumb-sub').style.left = (subjectivity * 100) + '%';
  });

  // Stat pills
  const pills = document.getElementById('stat-pills');
  const flesch = stats.flesch_reading_ease;
  const readLevel = flesch > 70 ? 'Easy' : flesch > 50 ? 'Medium' : 'Complex';
  pills.innerHTML = [
    { label: 'Words',       val: stats.word_count },
    { label: 'Sentences',   val: stats.sentence_count },
    { label: 'Unique words',val: stats.unique_words },
    { label: 'TTR',         val: stats.type_token_ratio.toFixed(2) },
    { label: 'Avg sent len',val: stats.avg_sentence_length.toFixed(1) },
    { label: 'Flesch',      val: `${flesch} (${readLevel})` },
  ].map(p => `
    <div class="stat-pill">
      <span class="stat-pill-label">${p.label}</span>
      <span class="stat-pill-val">${p.val}</span>
    </div>
  `).join('');
}

function renderEmotionGrid(emotions) {
  const grid = document.getElementById('emotion-grid');
  grid.innerHTML = '';
  emotions.forEach(({ name, score }) => {
    const cfg = EMOTION_CONFIG[name] || { emoji: '•', color: '#7c6af7' };
    const pct = Math.round(score * 100);
    const div = document.createElement('div');
    div.className = 'emotion-item';
    div.innerHTML = `
      <div class="emotion-top">
        <span class="emotion-name">${name}</span>
        <span class="emotion-emoji">${cfg.emoji}</span>
      </div>
      <div class="emotion-score">${pct}%</div>
      <div class="emotion-bar-bg">
        <div class="emotion-bar-fill" style="width:0%;background:${cfg.color}"></div>
      </div>
    `;
    grid.appendChild(div);
    requestAnimationFrame(() => { div.querySelector('.emotion-bar-fill').style.width = pct + '%'; });
  });
}

function renderCharts(emotions) {
  const labels = emotions.map(e => e.name);
  const values = emotions.map(e => Math.round(e.score * 100));
  const colors = emotions.map(e => (EMOTION_CONFIG[e.name] || {}).color || '#7c6af7');

  if (radarChart) radarChart.destroy();
  radarChart = new Chart(document.getElementById('radarChart'), {
    type: 'radar',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: 'rgba(124,106,247,0.12)',
        borderColor: '#7c6af7',
        borderWidth: 2,
        pointBackgroundColor: colors,
        pointRadius: 4,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { r: {
        min: 0, max: 100,
        ticks: { stepSize: 25, color: '#6b7280', font: { size: 9 }, backdropColor: 'transparent' },
        grid:        { color: '#252a38' },
        angleLines:  { color: '#252a38' },
        pointLabels: { color: '#9ca3af', font: { size: 10 } },
      }},
      plugins: { legend: { display: false } },
    },
  });

  if (barChart) barChart.destroy();
  barChart = new Chart(document.getElementById('barChart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors.map(c => c + 'cc'),
        borderColor: colors, borderWidth: 1, borderRadius: 5,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false, indexAxis: 'y',
      scales: {
        x: { min: 0, max: 100, grid: { color: '#252a38' }, ticks: { color: '#6b7280', font: { size: 9 }, callback: v => v + '%' }},
        y: { grid: { display: false }, ticks: { color: '#9ca3af', font: { size: 10 } }},
      },
      plugins: { legend: { display: false } },
    },
  });
}

function renderPosChart(posDist) {
  const labels = Object.keys(posDist);
  const values = Object.values(posDist);
  if (posChart) posChart.destroy();
  posChart = new Chart(document.getElementById('posChart'), {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: POS_COLORS.slice(0, labels.length),
        borderColor: '#13161e',
        borderWidth: 2,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'right',
          labels: { color: '#9ca3af', font: { size: 10 }, boxWidth: 10, padding: 8 },
        },
      },
      cutout: '60%',
    },
  });
}

function renderKeywords(keywords) {
  document.getElementById('keywords-list').innerHTML =
    keywords.map(kw => `<span class="keyword-tag">${kw}</span>`).join('');
}

/* ── Helpers ── */
function setLoading(on) {
  loading.hidden = !on;
  analyzeBtn.disabled = on;
  analyzeBtn.querySelector('.btn-text').textContent = on ? 'Analyzing…' : 'Analyze';
}

function clearError() { errorState.hidden = true; }
