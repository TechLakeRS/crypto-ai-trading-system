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
    metrics.appendChild(createMetric('Confluences', '-'));

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
            const response = await fetch('/api/market/' + pair);
            const data = await response.json();
            const card = createCryptoCard(pair, {
                price: data.current_price,
                change_24h: data.change_24h_pct,
                volume: data.volume_24h
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

window.onload = function() {
    console.log('Initializing dashboard...');
    connectWebSocket();
    loadMarketData();
    loadSignals();
    setInterval(loadSignals, 5 * 60 * 1000);
};
