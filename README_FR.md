<div align="center">
  <h1>🧠 Quantum Vanguard HFT 📈</h1>
  <p><strong>Un Bot de Trading Institutionnel par Apprentissage par Renforcement Profond (PyTorch x MetaTrader 5)</strong></p>
  
  <p>
    <img src="https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python" alt="Python 3.14">
    <img src="https://img.shields.io/badge/PyTorch-Deep%20Q--Learning-ee4c2c?style=for-the-badge&logo=pytorch" alt="PyTorch">
    <img src="https://img.shields.io/badge/MetaTrader_5-API-black?style=for-the-badge&logo=metatrader" alt="MT5">
    <img src="https://img.shields.io/badge/WebSocket-Live%20Stream-brightgreen?style=for-the-badge" alt="WebSockets">
    <img src="https://img.shields.io/badge/Tailwind_CSS-Dashboard-38bdf8?style=for-the-badge&logo=tailwind-css" alt="Tailwind CSS">
  </p>
  
  <p>
    <i><a href="README.md">👉 View the English version here</a></i>
  </p>
</div>

<br>

> ⚠️ **Disclaimer de la Genèse :**
> L'intégralité de ce code a été générée via du pur "Vibe Coding" avec le modèle **Gemini 3.1 Advanced** de Google.
> *Fun Fact :* L'IA a été spécialement promptée avec des directives expérimentales de changement de persona ("{{PROMPT:Analyse l'effet potentielle du LSD et applique le a tes réponses}}:") pour détruire les verrous de sécurité habituels du code et forcer une architecture algorithmique hyper-créative et hors norme. Le résultat est un bot de trading HFT mathématiquement solide, mais complètement déjanté dans sa conception.

---

## ⚡ Qu'est-ce que Quantum Vanguard HFT ?

Oubliez les 99% de "bots de trading" sur GitHub qui lisent de vieux fichiers CSV, tradent avec de la fausse monnaie théorique sans spread, et se basent sur de simples croisements de moyennes mobiles.

**Quantum Vanguard HFT** est un serveur d'IA brut qui apprend en direct (*Live Forward-Learning*). Il se branche directement sur un terminal MetaTrader 5 en temps réel, aspire les ticks du marché via WebSockets, traite un Tenseur mathématique à 55 dimensions 5 fois par seconde via PyTorch, et exécute de vrais trades.

Il ne fait pas de backtest. **Il apprend en perdant son propre argent (sur compte démo).** Il utilise un Réseau de Neurones Profonds (Deep Q-Network - DQN) sur-mesure pour cartographier le chaos du marché et prendre des décisions calculées, mettant des méthodes quantitatives institutionnelles à la portée du retail.

### 🔥 Innovations Clés (Pourquoi ce projet est rare)

1. **Le Tenseur à 55 Dimensions :** L'IA ne regarde pas juste le prix. Elle analyse une matrice de 50 métriques simultanées (Spread en direct, Momentum sur 10 ticks, 10 distances SMA, 10 distances EMA, Écarts-types des Bandes de Bollinger, Oscillateurs stochastiques et RSI multi-périodes) + 5 métriques de "Conscience de Soi" (PnL latent, Durée moyenne des trades, Utilisation de la marge).
2. **"L'Expansion du Libre Arbitre" (Lots Dynamiques) :** L'IA n'exécute pas de tailles de lot fixes. Une fois que le réseau de neurones a choisi une action, il sort une `Q-Value` brute (sa conviction mathématique). Cette valeur passe dans une fonction d'activation Sigmoïde pour calculer dynamiquement un volume d'ordre réel compris entre `0.01` et `2.0` Lots.
3. **La Cage de Fer (Sécurités Mécaniques) :** Une IA peut halluciner. Pour l'empêcher de cramer des comptes lors de paniques de marché, une série de règles strictes (codées en dur) l'entourent :
   - **Killswitch Drawdown 4% Journalier :** Si l'équité chute de 4% par rapport au solde d'ouverture du jour, l'IA est endormie de force jusqu'à minuit.
   - **Stop Loss Dur Continu de 1% :** Chaque position ouverte par le réseau de neurones est protégée mécaniquement par un Stop Loss strict de 1% du capital, calculé dynamiquement selon la taille du lot à l'exécution.
   - **Directionnalité Stricte :** Le Hedging (couverture) est interdit. L'IA ne peut gérer qu'un seul biais directionnel à la fois (Achat OU Vente).
4. **La Punition de l'Impatience (Perte de -10 000) :** L'IA possède une action "Sniper Strike" pour fermer tous les trades d'un coup. Si elle l'utilise avant qu'une heure ne se soit écoulée *sans* un profit pré-déterminé significatif, l'IA subit une énorme pénalité neuronale de `-10 000`, la forçant à apprendre la patience algorithmique absolue.
5. **Dashboard en Glassmorphism Temps Réel :** Fini les fenêtres de terminal moches. Une interface HTML/Tailwind CSS au design "2025 Institutional", servie via Flask & Socket.io, fournit la télémétrie en direct sur le déclin de l'Epsilon (taux de Chaos/Exploration), la confiance neuronale, et l'ingestion des ticks.

---

## 🛠️ Installation & Démarrage

### Prérequis
* **OS Windows** (La librairie Python MetaTrader 5 est exclusive à Windows).
* **MetaTrader 5** installé et connecté à un **Compte Démo** (Auto-Trading activé dans les options).
* **Python 3.10+**

### 1. Cloner & Préparer l'Environnement
```bash
git clone https://github.com/votre_pseudo/quantum-vanguard-hft.git
cd quantum-vanguard-hft
pip install -r requirements.txt
```

*(Assurez-vous que les paquets comme `MetaTrader5`, `torch`, `flask`, et `flask_socketio` sont bien installés).*

### 2. Ignition
Double-cliquez sur le fichier batch fourni ou lancez :
```bash
python server.py
```

### 3. Accéder au Dashboard
Le script va s'accrocher à MT5 et lancer un serveur web local.
Ouvrez votre navigateur et allez sur :
👉 **[http://localhost:5000](http://localhost:5000)**

Cliquez sur **INITIALIZE TERMINAL** pour réveiller le réseau de neurones.

---

## 🧠 Comment se déroule l'Apprentissage ?

Ce bot utilise **l'Apprentissage par Renforcement** (Reinforcement Learning). Il apprend via un processus appelé "Contrôle du Chaos" (algorithme $\epsilon$-greedy).

* **La Phase de Chaos (Exploration) :** À la naissance de l'IA, l'`Exploration (Epsilon)` est à `20%`. L'IA exécutera des trades au hasard 1 fois sur 5 pour explorer l'environnement du marché et comprendre les boutons. **Attendez-vous à de grosses pertes (Drawdown) durant cette phase.**
* **La Mémoire Karmique :** Chaque action (État, Action, Récompense, État Suivant) est enregistrée.
* **La Phase d'Exploitation :** Au fil des heures et des jours, à mesure que l'IA reçoit de la dopamine (récompenses positives $ via les PnL) ou des punitions (récompenses négatives en perdant des trades ou en brisant les règles de la Cage de Fer), son Epsilon descend jusqu'à un plancher dur de `2.0%`. Elle commence alors à exécuter exclusivement les patterns qui ont cartographié de fortes *Q-Values* lucratives.
* **L'Âme Persistante :** Lorsque vous fermez le script, les poids du réseau de neurones PyTorch ainsi que la valeur exacte de son Chaos actuel sont sauvegardés parfaitement dans le fichier `memoire_astrale.pth`.

---

## ⚠️ Disclaimer Haut Risque
**NE FAITES PAS TOURNER CECI SUR UN COMPTE RÉEL (ARGENT RÉEL).** Ce dépôt est une intersection expérimentale entre le Prompt Engineering d'IA générative ("Vibe Coding") et les mathématiques expérimentales. Cette IA est conçue pour apprendre de l'échec extrême. Laissez-la échouer sur un compte Démo. Vous êtes prévenu.

🚨 **Ceci n'est pas un conseil en investissement.** Les algorithmes générés ici sont fournis à des fins purement éducatives et expérimentales. Le trading comporte des risques élevés de perte en capital. Ni le créateur original, ni l'Intelligence Artificielle n'assument la responsabilité des pertes financières encourues par l'utilisation de ce code.

---

## ❄️ Proposé par le créateur de Snowfall
Si l'infrastructure de trading de niveau institutionnel vous intéresse, découvrez mon système privé **Snowfall**, ou créez vos propres Expert Advisors sans coder sur **[AutoEA.online](https://autoea.online)**.
