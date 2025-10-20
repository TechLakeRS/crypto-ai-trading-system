// Crypto AI Trading Dashboard - Main JavaScript
// Using safe DOM manipulation methods

let ws = null;
let reconnectInterval = null;

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = protocol + '//' + window.location.host + '/ws';
    ws = new WebSocket(wsUrl);

    ws.onopen = function() {
        console.log('WebSocket connected');
        document.getElementById('connection-status').textContent = 'Connected';
        document.getElementById('connection-status').className = 'status-connected';
        clearInterval(reconnectInterval);
    };

    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        if (data.type === 'market_update') {
            updateMarketData(data.data);
            updateLastUpdateTime();
        }
    };

    ws.onerror = function(error) {
        console.error('WebSocket error:', error);
    };

    ws.onclose = function() {
        console.log('WebSocket disconnected');
        document.getElementById('connection-status').textContent = 'Disconnected';
        document.getElementById('connection-status').className = 'status-disconnected';
        reconnectInterval = setInterval(function() {
            connectWebSocket();
        }, 5000);
    };
}

function updateMarketData(data) {
    const grid = document.getElementById('crypto-grid');
    for (const symbol in data) {
        const info = data[symbol];
        const cardId = 'card-' + symbol.replace('/', '-');
        let card = document.getElementById(cardId);
        if (!card) {
            card = createCryptoCard(symbol, info);
            grid.appendChild(card);
        } else {
            updateCryptoCard(card, info);
        }
    }
}

function createCryptoCard(symbol, info) {
    const card = document.createElement('div');
    card.className = 'card';
    card.id = 'card-' + symbol.replace('/', '-');

    const isPositive = info.change_24h >= 0;

    const header = document.createElement('div');
    header.className = 'card-header';

    const title = document.createElement('div');
    title.className = 'card-title';
    title.textContent = symbol;

    const badge = document.createElement('div');
    badge.className = 'change-badge ' + (isPositive ? 'change-positive' : 'change-negative');
    badge.textContent = (isPositive ? '▲' : '▼') + ' ' + Math.abs(info.change_24h).toFixed(2) + '%';

    header.appendChild(title);
    header.appendChild(badge);

    const price = document.createElement('div');
    price.className = 'price ' + (isPositive ? 'price-up' : 'price-down');
    price.textContent = '$' + info.price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});

    const metrics = document.createElement('div');
    metrics.className = 'metrics';

    metrics.appendChild(createMetric('24h Volume', '$' + (info.volume / 1000000).toFixed(2) + 'M'));

    // Display sentiment (info only - not used in signals)
    const sentimentText = info.sentiment_text || 'Neutral';
    const sentimentMetric = createMetric('Sentiment', sentimentText);
    const sentimentValue = sentimentMetric.querySelector('.metric-value');
    if (sentimentText === 'Bullish') {
        sentimentValue.style.color = '#10b981';
        sentimentValue.style.fontWeight = '600';
    } else if (sentimentText === 'Bearish') {
        sentimentValue.style.color = '#ef4444';
        sentimentValue.style.fontWeight = '600';
    }
    metrics.appendChild(sentimentMetric);

    // Display technical confluences (used for signals)
    const confluences = info.confluences || 0;
    const confluenceMetric = createMetric('Confluences', confluences + '/5');
    const confluenceValue = confluenceMetric.querySelector('.metric-value');
    if (confluences >= 3) {
        confluenceValue.style.color = '#10b981';
        confluenceValue.style.fontWeight = '600';
    } else if (confluences >= 2) {
        confluenceValue.style.color = '#f59e0b';
        confluenceValue.style.fontWeight = '600';
    }
    metrics.appendChild(confluenceMetric);

    card.appendChild(header);
    card.appendChild(price);
    card.appendChild(metrics);

    return card;
}

function createMetric(label, value) {
    const metric = document.createElement('div');
    metric.className = 'metric';

    const labelEl = document.createElement('div');
    labelEl.className = 'metric-label';
    labelEl.textContent = label;

    const valueEl = document.createElement('div');
    valueEl.className = 'metric-value';
    valueEl.textContent = value;

    metric.appendChild(labelEl);
    metric.appendChild(valueEl);

    return metric;
}

function updateCryptoCard(card, info) {
    const isPositive = info.change_24h >= 0;

    const priceEl = card.querySelector('.price');
    priceEl.textContent = '$' + info.price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
    priceEl.className = 'price ' + (isPositive ? 'price-up' : 'price-down');

    const badge = card.querySelector('.change-badge');
    badge.textContent = (isPositive ? '▲' : '▼') + ' ' + Math.abs(info.change_24h).toFixed(2) + '%';
    badge.className = 'change-badge ' + (isPositive ? 'change-positive' : 'change-negative');
}

async function loadMarketData() {
    const pairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT'];
    const grid = document.getElementById('crypto-grid');
    grid.textContent = '';

    for (const pair of pairs) {
        try {
            const encodedPair = encodeURIComponent(pair);
            const response = await fetch('/api/market/' + encodedPair);
            const data = await response.json();
            const card = createCryptoCard(pair, {
                price: data.current_price,
                change_24h: data.change_24h_pct,
                volume: data.volume_24h,
                sentiment_text: data.sentiment_text,
                sentiment_score: data.sentiment_score,
                confluences: data.confluences || 0
            });
            grid.appendChild(card);
        } catch (error) {
            console.error('Error loading ' + pair + ':', error);
        }
    }

    updateLastUpdateTime();
}

async function loadSignals() {
    const container = document.getElementById('signals-container');
    container.textContent = 'Loading signals...';
    container.className = 'loading';

    try {
        const response = await fetch('/api/signals');
        const data = await response.json();
        container.textContent = '';
        container.className = '';

        if (data.signals && data.signals.length > 0) {
            data.signals.forEach(function(signal) {
                container.appendChild(createSignalCard(signal));
            });
        } else {
            const noData = document.createElement('div');
            noData.className = 'no-data';
            noData.textContent = 'No active signals at the moment. The system is analyzing the market...';
            container.appendChild(noData);
        }
    } catch (error) {
        console.error('Error loading signals:', error);
        const errorDiv = document.createElement('div');
        errorDiv.className = 'no-data';
        errorDiv.textContent = 'Failed to load signals. Please try again.';
        container.appendChild(errorDiv);
    }
}

function createSignalCard(signal) {
    const card = document.createElement('div');
    const signalClass = signal.signal_type.includes('buy') ? 'signal-buy' : 'signal-sell';
    card.className = 'signal-card ' + signalClass;

    const header = document.createElement('div');
    header.className = 'signal-header';

    const headerLeft = document.createElement('div');

    const signalType = document.createElement('div');
    signalType.className = 'signal-type';
    signalType.textContent = signal.signal_type.replace('_', ' ');

    const symbolDiv = document.createElement('div');
    symbolDiv.style.color = '#9ca3af';
    symbolDiv.style.marginTop = '5px';
    symbolDiv.textContent = signal.symbol;

    headerLeft.appendChild(signalType);
    headerLeft.appendChild(symbolDiv);

    const confluences = document.createElement('div');
    confluences.className = 'confluences';
    confluences.textContent = signal.confluences + ' Confluences';

    header.appendChild(headerLeft);
    header.appendChild(confluences);

    const confidenceSection = document.createElement('div');
    confidenceSection.style.margin = '15px 0';

    const confLabel = document.createElement('div');
    confLabel.style.display = 'flex';
    confLabel.style.justifyContent = 'space-between';
    confLabel.style.marginBottom = '5px';

    const confText = document.createElement('span');
    confText.textContent = 'Confidence';

    const confValue = document.createElement('span');
    const strong = document.createElement('strong');
    strong.textContent = (signal.confidence * 100).toFixed(0) + '%';
    confValue.appendChild(strong);

    confLabel.appendChild(confText);
    confLabel.appendChild(confValue);

    const barContainer = document.createElement('div');
    barContainer.className = 'confidence-bar';
    const barFill = document.createElement('div');
    barFill.className = 'confidence-fill';
    barFill.style.width = (signal.confidence * 100) + '%';
    barContainer.appendChild(barFill);

    confidenceSection.appendChild(confLabel);
    confidenceSection.appendChild(barContainer);

    const details = document.createElement('div');
    details.className = 'signal-details';

    details.appendChild(createMetric('Current Price', '$' + signal.current_price.toFixed(2)));
    details.appendChild(createMetric('Entry Price', '$' + signal.entry_price.toFixed(2)));
    details.appendChild(createMetric('Stop Loss', '$' + signal.stop_loss.toFixed(2)));

    const tpString = signal.take_profit.map(function(tp) {
        return tp[0] + ': $' + tp[1].toFixed(2);
    }).join(', ');
    details.appendChild(createMetric('Take Profit', tpString));
    details.appendChild(createMetric('Risk/Reward', '1:' + signal.risk_reward.toFixed(2)));
    details.appendChild(createMetric('Position Size', signal.position_size_pct.toFixed(1) + '%'));

    card.appendChild(header);
    card.appendChild(confidenceSection);
    card.appendChild(details);

    const reason = document.createElement('div');
    reason.style.marginTop = '20px';

    const reasonTitle = document.createElement('div');
    reasonTitle.style.fontWeight = '600';
    reasonTitle.style.marginBottom = '10px';
    reasonTitle.textContent = 'Primary Reason:';

    const reasonText = document.createElement('div');
    reasonText.style.color = '#d1d5db';
    reasonText.textContent = signal.primary_reason;

    reason.appendChild(reasonTitle);
    reason.appendChild(reasonText);
    card.appendChild(reason);

    if (signal.supporting_factors && signal.supporting_factors.length > 0) {
        const factors = document.createElement('div');
        factors.style.marginTop = '15px';

        const factorsTitle = document.createElement('div');
        factorsTitle.style.fontWeight = '600';
        factorsTitle.style.marginBottom = '10px';
        factorsTitle.textContent = 'Supporting Factors:';

        const factorsList = document.createElement('ul');
        factorsList.className = 'factors-list';

        signal.supporting_factors.forEach(function(factor) {
            const li = document.createElement('li');
            li.textContent = factor;
            factorsList.appendChild(li);
        });

        factors.appendChild(factorsTitle);
        factors.appendChild(factorsList);
        card.appendChild(factors);
    }

    const footer = document.createElement('div');
    footer.style.marginTop = '15px';
    footer.style.paddingTop = '15px';
    footer.style.borderTop = '1px solid rgba(255,255,255,0.1)';
    footer.style.color = '#9ca3af';
    footer.style.fontSize = '0.9em';
    footer.textContent = 'Generated: ' + new Date(signal.timestamp).toLocaleString() + ' | Priority: ' + signal.priority + '/10';
    card.appendChild(footer);

    return card;
}

function updateLastUpdateTime() {
    const now = new Date();
    document.getElementById('last-update').textContent = now.toLocaleTimeString();
}

// ===== TRIPLE MODEL FUNCTIONS =====

async function loadAllModels() {
    console.log('Loading signals from all 3 models...');
    const model1Container = document.getElementById('model1-signals');
    const model2Container = document.getElementById('model2-signals');
    const model3Container = document.getElementById('model3-signals');

    model1Container.textContent = '';
    model2Container.textContent = '';
    model3Container.textContent = '';

    const loading1 = document.createElement('div');
    loading1.className = 'loading';
    loading1.textContent = 'Loading Model 1 signals...';
    model1Container.appendChild(loading1);

    const loading2 = document.createElement('div');
    loading2.className = 'loading';
    loading2.textContent = 'Loading Model 2 signals...';
    model2Container.appendChild(loading2);

    const loading3 = document.createElement('div');
    loading3.className = 'loading';
    loading3.textContent = 'Loading Model 3 signals...';
    model3Container.appendChild(loading3);

    try {
        const response = await fetch('/api/signals/all');
        const data = await response.json();

        // Model 1 signals
        renderModelSignals(data.model1, model1Container, 1);

        // Model 2 signals
        renderModelSignals(data.model2, model2Container, 2);

        // Model 3 signals
        renderModelSignals(data.model3, model3Container, 3);

        updateLastUpdateTime();
    } catch (error) {
        console.error('Error loading all models:', error);

        model1Container.textContent = '';
        const error1 = document.createElement('div');
        error1.className = 'no-signals';
        error1.textContent = 'Failed to load Model 1 signals';
        model1Container.appendChild(error1);

        model2Container.textContent = '';
        const error2 = document.createElement('div');
        error2.className = 'no-signals';
        error2.textContent = 'Failed to load Model 2 signals';
        model2Container.appendChild(error2);

        model3Container.textContent = '';
        const error3 = document.createElement('div');
        error3.className = 'no-signals';
        error3.textContent = 'Failed to load Model 3 signals';
        model3Container.appendChild(error3);
    }
}

function renderModelSignals(modelData, container, modelNumber) {
    container.textContent = '';

    if (!modelData || !modelData.signals || modelData.signals.length === 0) {
        const noSignals = document.createElement('div');
        noSignals.className = 'no-signals';
        noSignals.textContent = 'No signals from ' + (modelData ? modelData.name : 'model');
        container.appendChild(noSignals);
        return;
    }

    modelData.signals.forEach(function(signal) {
        const signalCard = createModelSignalCard(signal, modelNumber);
        container.appendChild(signalCard);
    });
}

function createModelSignalCard(signal, modelNumber) {
    const card = document.createElement('div');
    const signalClass = signal.signal_type === 'long' ? 'model-signal-buy' : 'model-signal-sell';
    card.className = 'model-signal-card ' + signalClass;

    // Header
    const header = document.createElement('div');
    header.style.display = 'flex';
    header.style.justifyContent = 'space-between';
    header.style.alignItems = 'center';
    header.style.marginBottom = '12px';

    const signalInfo = document.createElement('div');
    const signalType = document.createElement('div');
    signalType.style.fontSize = '1.3em';
    signalType.style.fontWeight = 'bold';
    signalType.style.textTransform = 'uppercase';
    signalType.style.color = signal.signal_type === 'long' ? '#4ade80' : '#ef4444';
    signalType.textContent = signal.signal_type;

    const symbol = document.createElement('div');
    symbol.style.fontSize = '0.95em';
    symbol.style.color = '#9ca3af';
    symbol.style.marginTop = '3px';
    symbol.textContent = signal.symbol;

    signalInfo.appendChild(signalType);
    signalInfo.appendChild(symbol);

    const confidence = document.createElement('div');
    confidence.style.fontSize = '1.2em';
    confidence.style.fontWeight = '600';
    confidence.style.color = '#4facfe';
    confidence.textContent = (signal.confidence * 100).toFixed(0) + '%';

    header.appendChild(signalInfo);
    header.appendChild(confidence);
    card.appendChild(header);

    // Prices
    const prices = document.createElement('div');
    prices.style.display = 'grid';
    prices.style.gridTemplateColumns = 'repeat(3, 1fr)';
    prices.style.gap = '8px';
    prices.style.marginBottom = '12px';

    prices.appendChild(createSmallMetric('Entry', '$' + signal.entry_price.toFixed(2)));
    prices.appendChild(createSmallMetric('Stop', '$' + signal.stop_loss.toFixed(2)));

    const tpLabel = signal.take_profit && signal.take_profit.length > 0
        ? (Array.isArray(signal.take_profit[0])
            ? '$' + signal.take_profit[0][1].toFixed(2)  // For model format [["TP1", value]]
            : '$' + signal.take_profit[0].toFixed(2))     // For simple array format [value]
        : 'N/A';
    prices.appendChild(createSmallMetric('Target', tpLabel));

    card.appendChild(prices);

    // Model-specific metadata
    if ((modelNumber === 2 || modelNumber === 3) && signal.leverage) {
        const leverageDiv = document.createElement('div');
        leverageDiv.style.marginBottom = '10px';

        const leverageBadge = document.createElement('span');
        leverageBadge.className = 'leverage-badge';
        leverageBadge.textContent = signal.leverage + '× Leverage';

        leverageDiv.appendChild(leverageBadge);

        if (signal.risk_usd) {
            const riskText = document.createElement('span');
            riskText.style.marginLeft = '10px';
            riskText.style.fontSize = '0.9em';
            riskText.style.color = '#9ca3af';
            riskText.textContent = ' Risk: $' + signal.risk_usd.toFixed(2);
            leverageDiv.appendChild(riskText);
        }

        card.appendChild(leverageDiv);

        // Invalidation condition
        if (signal.invalidation_condition) {
            const invalidation = document.createElement('div');
            invalidation.className = 'invalidation-warning';
            invalidation.textContent = '⚠ ' + signal.invalidation_condition;
            card.appendChild(invalidation);
        }
    }

    // Primary reason
    const reason = document.createElement('div');
    reason.style.marginTop = '10px';
    reason.style.fontSize = '0.9em';
    reason.style.color = '#d1d5db';
    reason.textContent = signal.primary_reason;
    card.appendChild(reason);

    // Confluences indicator
    const confluencesDiv = document.createElement('div');
    confluencesDiv.style.marginTop = '10px';
    confluencesDiv.style.paddingTop = '10px';
    confluencesDiv.style.borderTop = '1px solid rgba(255,255,255,0.1)';
    confluencesDiv.style.fontSize = '0.85em';
    confluencesDiv.style.color = '#9ca3af';

    let confText;
    if (modelNumber === 1) {
        confText = signal.confluences + '/5 technical sources';
    } else if (modelNumber === 3) {
        confText = signal.confluences + '/9 confluences';
    } else {
        confText = 'Confidence: ' + (signal.confidence * 100).toFixed(0) + '%';
    }

    confluencesDiv.textContent = confText + ' | R:R ' + signal.risk_reward.toFixed(2);
    card.appendChild(confluencesDiv);

    return card;
}

function createSmallMetric(label, value) {
    const metric = document.createElement('div');
    metric.style.padding = '8px';
    metric.style.background = 'rgba(0,0,0,0.2)';
    metric.style.borderRadius = '6px';
    metric.style.textAlign = 'center';

    const labelEl = document.createElement('div');
    labelEl.style.fontSize = '0.75em';
    labelEl.style.color = '#9ca3af';
    labelEl.style.marginBottom = '4px';
    labelEl.textContent = label;

    const valueEl = document.createElement('div');
    valueEl.style.fontSize = '0.95em';
    valueEl.style.fontWeight = '600';
    valueEl.textContent = value;

    metric.appendChild(labelEl);
    metric.appendChild(valueEl);

    return metric;
}

window.onload = function() {
    console.log('Initializing dashboard...');
    connectWebSocket();
    loadMarketData();
    loadAllModels(); // Load all 3 models
    setInterval(loadAllModels, 3 * 60 * 1000); // Refresh every 3 minutes
};
