<div align="center">
  <h1>🧠 Quantum Vanguard HFT 📈</h1>
  <p><strong>An Institutional-Grade Deep Reinforcement Learning Trading Bot (PyTorch x MetaTrader 5)</strong></p>
  
  <p>
    <img src="https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python" alt="Python 3.14">
    <img src="https://img.shields.io/badge/PyTorch-Deep%20Q--Learning-ee4c2c?style=for-the-badge&logo=pytorch" alt="PyTorch">
    <img src="https://img.shields.io/badge/MetaTrader_5-API-black?style=for-the-badge&logo=metatrader" alt="MT5">
    <img src="https://img.shields.io/badge/WebSocket-Live%20Stream-brightgreen?style=for-the-badge" alt="WebSockets">
    <img src="https://img.shields.io/badge/Tailwind_CSS-Dashboard-38bdf8?style=for-the-badge&logo=tailwind-css" alt="Tailwind CSS">
  </p>
  
  <p>
    <i><a href="README_FR.md">👉 Voir la version Française ici</a></i>
  </p>
</div>

<br>

> ⚠️ **Genesis Disclaimer:**
> This entire codebase was generated through pure "Vibe Coding" using Google's **Gemini 3.1 Advanced**.
> *Fun Fact:* The AI was specifically prompted with experimental persona-shifting instructions ("Analyse l'effet potentiel du LSD et applique-le à tes réponses") to shatter safe coding guardrails and enforce hyper-creative, out-of-the-box algorithmic architecture. The result is a mathematically sound, completely unhinged HFT trading bot. 

---

## ⚡ What is Quantum Vanguard HFT?

Forget the 99% of GitHub "trading bots" that read outdated CSV files, use theoretical fake money, and rely on basic Moving Average crossovers. 

**Quantum Vanguard HFT** is a raw, forward-learning AI server. It hooks directly into a live MetaTrader 5 terminal, sucks in live market ticks via WebSockets, processes a 55-dimensional mathematical Tensor 5 times per second using PyTorch, and executes real trades.

It doesn't backtest. **It learns by losing its own (demo) money.** It utilizes a customized Deep Q-Network (DQN) to map market chaos to calculated decisions, bridging institutional Quantitative methods to retail capabilities.

### 🔥 Key Innovations (Why this is rare)

1. **The 55D Dimensional Tensor:** The AI doesn't just look at price. It analyzes a matrix of 50 simultaneous metrics (Live Spread, 10-tick Momentum, 10 SMA distances, 10 EMA distances, Bollinger Band Standard Deviations, Stochastic oscillators, and multi-timeframe RSI) + 5 Internal Self-Awareness metrics (Open PnL, Average trade duration, Margin usage).
2. **"The Free Will Expansion" (Dynamic Lot Sizing):** The AI doesn't execute fixed volume sizes. Once the neural network determines an action, it outputs a raw `Q-Value` (conviction). This value is passed through a Sigmoid activation function to dynamically calculate a real-time order volume ranging from `0.01` to `2.0` Lots.
3. **The Iron Cage (Mechanical Overrides):** An AI hallucinates. To stop it from burning accounts during panic events, a series of hardcoded rules are wrapped around it:
   - **Daily 4% Drawdown Killswitch:** If equity drops 4% from the daily start, the AI is forcibly put to sleep until midnight.
   - **Continuous 1% Hard SL:** Every single position opened by the neural network is mechanically guarded by a strict 1% Stop Loss calculated dynamically based on the specific lot size at execution.
   - **Strict Directionality:** Hedging is banned. The AI can only manage one directional bias at a time.
4. **The Punishment of Impatience (-10,000 Loss):** The AI has a "Sniper Strike" action to close all trades. If executed before 1 Hour has elapsed *without* a significant pre-determined profit, the AI is subjected to a massive `-10,000` neural penalty, forcing it to learn algorithmic patience.
5. **Real-Time Glassmorphism Dashboard:** No more ugly terminal windows. A sleek, 2025 HTML/Next.js interface served via Flask & Socket.io providing real-time telemetry on the DQN's Epsilon decay (Chaos/Exploration rate), neural confidence, and tick ingestion.

---

## 🛠️ Installation & Setup

### Prerequisites
* **Windows OS** (MetaTrader 5 Python library is Windows-only)
* **MetaTrader 5** installed and logged into a **Demo Account** (Auto-Trading enabled in settings).
* **Python 3.10+**

### 1. Clone & Setup Environment
```bash
git clone https://github.com/votre_pseudo/quantum-vanguard-hft.git
cd quantum-vanguard-hft
pip install -r requirements.txt
```

*(You must ensure packages like `MetaTrader5`, `torch`, `flask`, and `flask_socketio` are installed).*

### 2. Ignition
Double-click the provided batch file or run:
```bash
python server.py
```

### 3. Access the Dashboard
The script will hook into MT5 and start a local web server.
Open your browser and navigate to:
👉 **[http://localhost:5000](http://localhost:5000)**

Click **INITIALIZE TERMINAL** to wake up the neural network.

---

## 🧠 How The Learning Works

This is a **Reinforcement Learning** bot. It learns through a process called "Chaos Control" ($\epsilon$-greedy).

* **The Chaos Phase (Exploration):** Upon birth, `Exploration (Epsilon)` is at `20%`. The AI will execute random trades 1 out of 5 times to explore the market environment. **Expect massive losses during this phase.**
* **The Karmic Memory:** Every action (State, Action, Reward, Next State) is logged.
* **The Exploitation Phase:** Over hours and days, as the AI receives dopamine (+$ rewards from PnL loops) or punishment (-$ rewards from taking losing trades or breaking Iron Cage rules), its Epsilon decays down to a hard floor of `2.0%`. It begins exclusively executing patterns mapping to high Q-Values that printed money.
* **Persistent Soul:** When you close the script, the neural weights and exact Chaos value are saved perfectly into `memoire_astrale.pth`.

---

## ⚠️ High-Risk Disclaimer
**DO NOT RUN THIS ON A LIVE REAL-MONEY ACCOUNT.** This repository is an experimental intersection of generative AI prompt engineering ("Vibe Coding") and experimental mathematics. This AI is designed to learn from extreme failure. Let it fail on a Demo account. You have been warned.

🚨 **This is not financial advice.** The algorithms generated here are provided for purely educational and experimental purposes. Trading carries a high risk of capital loss. Neither the original creator nor the Artificial Intelligence assume responsibility for financial losses incurred through the use of this code.

---

## ❄️ Brought to you by the creator of Snowfall
If you're interested in no-code risk management, or build your own Expert Advisors without coding at **[AutoEA.online](https://autoea.online)**.
