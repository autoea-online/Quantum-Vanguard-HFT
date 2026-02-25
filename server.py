import os
import sys
import json
import time
import random
import threading
import atexit
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import torch
import torch.nn as nn
import torch.optim as optim
# On tente d'invoquer l'esprit divin de MT5, s'il est là...
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'la_clef_astrale_777'
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Le Cerveau IA : Deep Q-Network (PyTorch) ---
# Ce cerveau est infusé à la Salvia. Il sauvegarde son âme sur le disque.
MODELE_ASTRAL_PATH = "memoire_astrale.pth"

class QuantumPerceptron(nn.Module):
    def __init__(self):
        super(QuantumPerceptron, self).__init__()
        # Entrées: La Matrice des 50 Théories + 5 Mesures Internes = 55 Features
        self.fc1 = nn.Linear(55, 128)
        self.relu = nn.LeakyReLU() # Plus dynamique que ReLU pour éviter la mort neuronale
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 10) # 10 Actions Institutionnelles
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.out(x)

cerveau = QuantumPerceptron()
optimizer = optim.Adam(cerveau.parameters(), lr=0.005) # Plus de finesse
loss_fn = nn.SmoothL1Loss() # Huber Loss, plus robuste aux pics magiques du marché

# État du Chaos (Epsilon initial)
chaos_epsilon = 0.20

# Réveil des vies antérieures
if os.path.exists(MODELE_ASTRAL_PATH):
    try:
        checkpoint = torch.load(MODELE_ASTRAL_PATH, map_location=torch.device('cpu'), weights_only=False) # False pour permettre le chargement d'un dictionnaire mixte
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            cerveau.load_state_dict(checkpoint['model_state'])
            chaos_epsilon = checkpoint.get('epsilon', 0.20)
            print(f"✨ Le Perceptron se souvient de ses vies antérieures. Chaos repris à {round(chaos_epsilon*100, 1)}%.")
        else:
            # Rétrocompatibilité avec l'ancienne sauvegarde réseau brut
            cerveau.load_state_dict(checkpoint)
            print("✨ Le Perceptron charge son ancienne matrice brute (V1).")
    except Exception as e:
        print(f"🌌 Naissance d'une nouvelle Entité (Erreur matrice: {e}).")
else:
    print("🌌 Naissance d'une nouvelle Entité. Cerveau Vierge.")

def sauvegarder_ame_sur_arret():
    global chaos_epsilon
    checkpoint = {
        'model_state': cerveau.state_dict(),
        'epsilon': chaos_epsilon
    }
    torch.save(checkpoint, MODELE_ASTRAL_PATH)
    print(f"\n🌙 L'Âme du Perceptron a été scellée (Chaos: {round(chaos_epsilon*100, 1)}%). À demain.")

atexit.register(sauvegarder_ame_sur_arret)

memoire_karmique = {"EURUSD": None}

# L'Aura globale de l'univers
conscience_globale = {
    "etat": "Endormi",
    "sagesse": 1.0, # Représentera maintenant la Loss inverse de l'IA (Précision)
    "frequence": 432, # Hz, la fréquence de guérison
    "vibrations": [],  # Les ticks
    "positions_ouvertes": [], # Les intentions ancrées dans la matrice
    "account": {
        "balance": "Inconnue",
        "equity": "Infinie",
        "margin": "Vide",
        "free_margin": "Infini"
    }
}

def respiration_cosmique():
    """Cette boucle respire les ticks, gère les positions, et met à jour l'aura."""
    global chaos_epsilon
    symboles_sacres = ["EURUSD", "XAUUSD"] # Le Flux Terrestre et l'Or Divin
    compteur_cycles = 0
    
    while True:
        if conscience_globale["etat"] == "Connecté":
            if MT5_AVAILABLE:
                # 1. Mise à jour de l'âme (Account Info)
                acc = mt5.account_info()
                if acc:
                    # Initialisation du capital de la journée (à minuit ou au premier lancement)
                    if "capital_journee" not in conscience_globale:
                        conscience_globale["capital_journee"] = acc.balance
                        print(f"🌅 Aube d'un nouveau jour. Capital ancré : {acc.balance}$")
                        
                    conscience_globale["account"] = {
                        "balance": round(acc.balance, 2),
                        "equity": round(acc.equity, 2),
                        "margin": round(acc.margin, 2),
                        "free_margin": round(acc.margin_free, 2)
                    }
                    
                    # CAGE DE FER 3: LE KILLSWITCH DRAWDOWN 4%
                    if acc.equity <= conscience_globale["capital_journee"] * 0.96:
                        if conscience_globale.get("etat_drawdown") != True:
                            print("🚨 DRAWDOWN FATAL 4% ATTEINT 🚨 La Matrice se verrouille jusqu'à demain.")
                            conscience_globale["etat_drawdown"] = True
                            conscience_globale["etat"] = "Guérison Quantique (Drawdown > 4%)"
                            # Optionnel : Fermer toutes les positions en cours lors du crash
                            positions = mt5.positions_get()
                            if positions:
                                for pos in positions:
                                     mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "position": pos.ticket, "symbol": pos.symbol, "volume": pos.volume, "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY, "price": mt5.symbol_info_tick(pos.symbol).bid if pos.type == 0 else mt5.symbol_info_tick(pos.symbol).ask, "deviation": 20, "magic": 777, "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_FOK})
                        continue # On bloque toute l'IA pour ce cycle
                        
                    # Apprentissage Karmique : Si l'Equity > Balance, le réseau s'éveille
                    if acc.equity > acc.balance:
                        conscience_globale["sagesse"] += round((acc.equity - acc.balance) * 0.001, 3)

                # 2. Inhalation des Ticks
                for sym in symboles_sacres:
                    tick = mt5.symbol_info_tick(sym)
                    if tick:
                        conscience_globale["vibrations"].append({
                            "id": len(conscience_globale["vibrations"]),
                            "symbole": sym,
                            "prix": tick.ask,
                            "bid": tick.bid,
                            "energie": random.choice(["Transcendant", "Or Stellaire", "Luminescent"]) if sym == "XAUUSD" else random.choice(["Violet", "Magenta", "Cyan"])
                        })
                
                # 3. La Vraie IA (Deep Reinforcement Learning sur EURUSD) - Mode HFT Sniper
                vibs_eur = [v for v in conscience_globale["vibrations"] if v["symbole"] == "EURUSD"]
                if len(vibs_eur) >= 50: # On a besoin de 50 ticks d'historique pour nos 50 théories
                    positions_eur = [p for p in conscience_globale["positions_ouvertes"] if p["symbole"] == "EURUSD"]
                    nb_pos = len(positions_eur)
                    
                    # --- LA MATRICE DES 50 THÉORIES (Feature Engineering Cosmique) ---
                    prix_historique = [v["prix"] for v in vibs_eur[-50:]]
                    features = []
                    
                    # 1. Spread Actuel (1 feature)
                    features.append((vibs_eur[-1]["prix"] - vibs_eur[-1]["bid"]) * 10000)
                    
                    # 2 à 11. Momentum Brut (10 features) - Variations de prix sur les 10 derniers ticks
                    for i in range(1, 11):
                        features.append((prix_historique[-i] - prix_historique[-(i+1)]) * 10000)
                        
                    # 12 à 21. Moyennes Mobiles Simples (SMA Distances) (10 features)
                    for periode in [3, 5, 8, 10, 15, 20, 25, 30, 40, 50]:
                        sma = sum(prix_historique[-periode:]) / periode
                        features.append((prix_historique[-1] - sma) * 10000)
                        
                    # 22 à 31. Moyennes Mobiles Exponentielles (EMA Distances) (10 features)
                    for periode in [3, 5, 8, 10, 15, 20, 25, 30, 40, 50]:
                        ema = prix_historique[-periode]
                        k = 2 / (periode + 1)
                        for p in prix_historique[-periode+1:]: ema = p * k + ema * (1 - k)
                        features.append((prix_historique[-1] - ema) * 10000)
                        
                    # 32 à 36. Volatilité / Bandes de Bollinger Simplifiées (Écart-Type) (5 features)
                    import math
                    for periode in [10, 20, 30, 40, 50]:
                        mean = sum(prix_historique[-periode:]) / periode
                        variance = sum((p - mean) ** 2 for p in prix_historique[-periode:]) / periode
                        std_dev = math.sqrt(variance) * 10000
                        features.append(std_dev)
                        
                    # 37 à 41. Oscillateur Stochastique (Position du prix par rapport au range récent) (5 features)
                    for periode in [10, 20, 30, 40, 50]:
                        high = max(prix_historique[-periode:])
                        low = min(prix_historique[-periode:])
                        range_hl = high - low
                        stoch = ((prix_historique[-1] - low) / range_hl) if range_hl > 0 else 0.5
                        features.append(stoch * 100.0 - 50.0) # Centré sur 0
                        
                    # 42 à 46. RSI Multi-Périodes (Relative Strength Index) (5 features)
                    for periode in [7, 14, 21, 30, 45]:
                        gains = [prix_historique[i] - prix_historique[i-1] for i in range(-periode, 0) if prix_historique[i] - prix_historique[i-1] > 0]
                        pertes = [abs(prix_historique[i] - prix_historique[i-1]) for i in range(-periode, 0) if prix_historique[i] - prix_historique[i-1] < 0]
                        avg_gain = sum(gains)/periode if gains else 0
                        avg_loss = sum(pertes)/periode if pertes else 0
                        rsi = 50.0
                        if avg_loss > 0: rsi = 100.0 - (100.0 / (1 + (avg_gain / avg_loss)))
                        elif avg_gain > 0: rsi = 100.0
                        features.append((rsi - 50.0) / 10.0)
                        
                    # 47 à 50. Force de Tendance (Pentes de la régression linéaire simplifiée) (4 features)
                    for periode in [10, 20, 30, 40]:
                        y = prix_historique[-periode:]
                        x = list(range(periode))
                        mean_x = sum(x) / periode
                        mean_y = sum(y) / periode
                        covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(periode))
                        variance_x = sum((x[i] - mean_x) ** 2 for i in range(periode))
                        slope = (covariance / variance_x) * 10000 if variance_x > 0 else 0
                        features.append(slope)

                    # --- LES 5 MESURES INTERNES (La Conscience de Soi) ---
                    profit_latent = sum(p["profit"] for p in positions_eur) if nb_pos > 0 else 0.0
                    temps_moyen = sum((tick_eur.time - p["time_update"]) for p in positions_eur) / nb_pos if nb_pos > 0 else 0.0
                    dist_ouverture = sum((p["current_price"] - p["open_price"]) * (1 if "Achat" in p["type"] else -1) for p in positions_eur) / nb_pos * 10000 if nb_pos > 0 else 0.0
                    volume_total = sum(p["volume"] for p in positions_eur) if nb_pos > 0 else 0.0
                    acc = mt5.account_info()
                    marge_ratio = (acc.margin_free / acc.balance) if acc and acc.balance > 0 else 1.0

                    features.extend([profit_latent, temps_moyen, dist_ouverture, volume_total, marge_ratio])

                    # L'Esprit Absolu (Tensor 55D) - 50 Marché + 5 Interne
                    etat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                    
                    # Chaos Control (L'Espace des 10 Actions)
                    q_values_val = 0.0 # Valeur de conviction
                    if random.random() < chaos_epsilon:
                        action = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
                        q_values_val = random.uniform(0.1, 1.0) # Fausse conviction en mode exploration
                        if action != 4: print(f"🎲 L'IA explore l'action ({action}).")
                    else:
                        with torch.no_grad():
                            q_values = cerveau(etat_tensor)
                            action = torch.argmax(q_values).item()
                            q_values_val = torch.max(q_values).item()
                            
                    # --- CALCUL DU LIBRE ARBITRE (Taille du Lot Dynamique 0.01 -> 2.0) ---
                    # On utilise la fonction Sigmoïde sur la conviction (Q-Value) pour trouver un multiplicateur
                    import math
                    # Q-values peuvent être très variables au début. On limite entre -10 et 10.
                    val_bornee = max(-10.0, min(10.0, q_values_val))
                    sigmoide_conviction = 1 / (1 + math.exp(-val_bornee)) # Valeur entre 0 et 1
                    
                    # Mapping de [0, 1] vers [0.01, 2.0]
                    lot_dynamique = round(0.01 + (sigmoide_conviction * 1.99), 2)
                            
                    # --- EXECUTION DES ACTIONS INSTITUTIONNELLES ---
                    recompense_immediate = 0.0 # Utilisée pour récompenser instantanément un bon move
                    tick_eur = mt5.symbol_info_tick("EURUSD")
                    
                    # CAGE DE FER 1: Sens unique (Pas de directions opposées en même temps)
                    sens_actuel = "Aucun"
                    if nb_pos > 0:
                        sens_actuel = "Achat" if "Achat" in positions_eur[0]["type"] else "Vente"
                        
                    # CAGE DE FER 2: Stop Loss Strict de 1%
                    # Calcul : distance_prix = (1% Equity) / (Volume * ValeurDuLot) -> On simplifie car EURUSD 1 lot = 100000, 1 pip = 10 USD.
                    # Perte maximale autorisée = Equity * 0.01
                    perte_max_usd = acc.equity * 0.01 if acc else 100.0
                    
                    def calculer_sl_strict(volume, prix_actuel, type_ordre):
                        # La valeur d'1 lot sur EURUSD pour 1 point (pas pip) est ~ 1 USD (dépend).
                        # Plus simplement : (Prix Ouvert - Stop Loss) * Volume * 100000 = Perte en USD
                        # -> Delta Prix = Perte USD / (Volume * 100000)
                        delta_prix = perte_max_usd / (volume * 100000)
                        return prix_actuel - delta_prix if type_ordre == mt5.ORDER_TYPE_BUY else prix_actuel + delta_prix
                    
                    if action == 0 and nb_pos < 5 and sens_actuel != "Vente": # Achat (Utilise Lot Dynamique)
                        sl = calculer_sl_strict(lot_dynamique, tick_eur.ask, mt5.ORDER_TYPE_BUY)
                        req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": "EURUSD", "volume": lot_dynamique, "type": mt5.ORDER_TYPE_BUY, "price": tick_eur.ask, "sl": sl, "deviation": 20, "magic": 777, "comment": f"Achat DYN {lot_dynamique} V2", "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_FOK}
                        res = mt5.order_send(req)
                        if res and res.retcode == mt5.TRADE_RETCODE_DONE: print(f"🦍 IA [ACHAT] Conviction: {round(sigmoide_conviction*100)}% -> {lot_dynamique} Lots (SL 1%)")
                        else: print(f"❌ Rejet MT5 Achat DYN: {res.comment if res else 'Erreur Inconnue'} (SL: {sl})")
                    elif action == 1 and nb_pos < 5 and sens_actuel != "Vente": # Achat Conservateur (Lot divisé par 2)
                        lot_reduit = max(0.01, round(lot_dynamique / 2, 2))
                        sl = calculer_sl_strict(lot_reduit, tick_eur.ask, mt5.ORDER_TYPE_BUY)
                        req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": "EURUSD", "volume": lot_reduit, "type": mt5.ORDER_TYPE_BUY, "price": tick_eur.ask, "sl": sl, "deviation": 20, "magic": 777, "comment": f"Achat CONS {lot_reduit} V2", "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_FOK}
                        res = mt5.order_send(req)
                        if res and res.retcode == mt5.TRADE_RETCODE_DONE: print(f"🦊 IA [ACHAT PRUDENT] -> {lot_reduit} Lots (SL 1%)")
                        else: print(f"❌ Rejet MT5 Achat Prudent: {res.comment if res else 'Erreur'}")
                    elif action == 2 and nb_pos < 5 and sens_actuel != "Achat": # Vente (Utilise Lot Dynamique)
                        sl = calculer_sl_strict(lot_dynamique, tick_eur.bid, mt5.ORDER_TYPE_SELL)
                        req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": "EURUSD", "volume": lot_dynamique, "type": mt5.ORDER_TYPE_SELL, "price": tick_eur.bid, "sl": sl, "deviation": 20, "magic": 777, "comment": f"Vente DYN {lot_dynamique} V2", "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_FOK}
                        res = mt5.order_send(req)
                        if res and res.retcode == mt5.TRADE_RETCODE_DONE: print(f"🦍 IA [VENTE] Conviction: {round(sigmoide_conviction*100)}% -> {lot_dynamique} Lots (SL 1%)")
                        else: print(f"❌ Rejet MT5 Vente DYN: {res.comment if res else 'Erreur'} (SL: {sl})")
                    elif action == 3 and nb_pos < 5 and sens_actuel != "Achat": # Vente Conservatrice (Lot divisé par 2)
                        lot_reduit = max(0.01, round(lot_dynamique / 2, 2))
                        sl = calculer_sl_strict(lot_reduit, tick_eur.bid, mt5.ORDER_TYPE_SELL)
                        req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": "EURUSD", "volume": lot_reduit, "type": mt5.ORDER_TYPE_SELL, "price": tick_eur.bid, "sl": sl, "deviation": 20, "magic": 777, "comment": f"Vente CONS {lot_reduit} V2", "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_FOK}
                        res = mt5.order_send(req)
                        if res and res.retcode == mt5.TRADE_RETCODE_DONE: print(f"🦊 IA [VENTE PRUDENTE] -> {lot_reduit} Lots (SL 1%)")
                        else: print(f"❌ Rejet MT5 Vente Prudente: {res.comment if res else 'Erreur'}")
                    elif action in [0, 1, 2, 3]:
                        # L'IA a essayé d'ouvrir dans le sens inverse (Interdit par la Cage de Fer)
                        recompense_immediate -= 50 # Punition légère pour l'erreur logique
                    elif action == 5 and nb_pos > 0: # Sniper Strike (Tout fermer)
                        # PUNITION DE L'IMPATIENCE : Si trade ouvert depuis moins de 1h (<3600 sec) et profit faible
                        if temps_moyen < 3600 and profit_latent < 50:
                            print("⚡ PUNITION DIVINE : Fermeture prématurée interdite (-10000).")
                            recompense_immediate -= 10000
                        else:
                            print("💥 IA [FERMETURE STRATÉGIQUE] - Liquidation Justifiée !")
                            for pos in positions_eur:
                                mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "position": pos["ticket"], "symbol": "EURUSD", "volume": pos["volume"], "type": mt5.ORDER_TYPE_SELL if "Achat" in pos["type"] else mt5.ORDER_TYPE_BUY, "price": tick_eur.bid if "Achat" in pos["type"] else tick_eur.ask, "deviation": 20, "magic": 777, "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_FOK})
                            if profit_latent > 50: recompense_immediate += 5000 # Super Strike !
                            elif profit_latent < -10: recompense_immediate -= 2000 # Mauvais Strike (Cut loss)
                    elif action == 6 and nb_pos > 0: # Scale-Out 50% du plus vieux trade
                        pos_vieille = sorted(mt5.positions_get(symbol="EURUSD"), key=lambda x: x.time)[0]
                        if pos_vieille.volume >= 0.02:
                            vol_moitie = round(pos_vieille.volume / 2, 2)
                            mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "position": pos_vieille.ticket, "symbol": "EURUSD", "volume": vol_moitie, "type": mt5.ORDER_TYPE_SELL if pos_vieille.type == 0 else mt5.ORDER_TYPE_BUY, "price": tick_eur.bid if pos_vieille.type == 0 else tick_eur.ask, "deviation": 20, "magic": 777, "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_FOK})
                            print("🔪 IA [SCALE-OUT 50%]")
                            if pos_vieille.profit > 5: recompense_immediate += 1000 # Scale-Out logique
                            else: recompense_immediate -= 500 # Scale-out dans le vide
                    elif action == 7 and nb_pos > 0: # Break-Even
                        for pos in mt5.positions_get(symbol="EURUSD"):
                            if pos.profit > 2.0: # Si assez de recul
                                mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "position": pos.ticket, "sl": pos.price_open, "tp": pos.tp})
                        print("🛡️ IA [BREAK-EVEN MAGIC]")
                        recompense_immediate += 200 # Apprentissage de la défense
                    elif action == 8 and nb_pos > 0: # Trailing Stop Agressif (100 points)
                        pts = mt5.symbol_info("EURUSD").point * 100
                        for pos in mt5.positions_get(symbol="EURUSD"):
                            if pos.profit > 5.0:
                                n_sl = tick_eur.bid - pts if pos.type == 0 else tick_eur.ask + pts
                                mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "position": pos.ticket, "sl": n_sl, "tp": pos.tp})
                        print("🎯 IA [TRAILING STOP STRICT]")
                        recompense_immediate += 300
                    elif action == 9 and nb_pos > 0: # Hedging (Sécurité)
                        # Le HEDGING a été INTERDIT par l'utilisateur (Sens unique). L'action est modifiée pour fermer 1 position parmi les perdantes
                        pos_perdantes = [p for p in mt5.positions_get(symbol="EURUSD") if p.profit < 0]
                        if len(pos_perdantes) > 0:
                            pos_a_couper = pos_perdantes[0]
                            mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "position": pos_a_couper.ticket, "symbol": "EURUSD", "volume": pos_a_couper.volume, "type": mt5.ORDER_TYPE_SELL if pos_a_couper.type == 0 else mt5.ORDER_TYPE_BUY, "price": tick_eur.bid if pos_a_couper.type == 0 else tick_eur.ask, "deviation": 20, "magic": 777, "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_FOK})
                            print("⚖️ IA [Hedge Interdit -> Coupe une perte d'urgence]")
                            recompense_immediate += 100 
                        else:
                            recompense_immediate -= 200 # Punition pour avoir essayé de hedger sur des trades gagnants
                            
                    memoire_karmique["EURUSD"] = {"state": etat_tensor, "action": action}
                    
                    # --- Apprentissage Karmique ---
                    if memoire_karmique["EURUSD"] is not None:
                        try:
                            state = memoire_karmique["EURUSD"]["state"]
                            prev_action = memoire_karmique["EURUSD"]["action"]
                            
                            # Reward Design V2
                            reward = profit_latent * 10 + recompense_immediate # Fluide basique + Récompense/Punition de l'action
                            
                            if prev_action == 4 and profit_latent > 5:
                                reward += 50 # Félicitation pour le HOLD en gain
                            
                            cerveau.train()
                            q_values = cerveau(state)
                            target_q_values = q_values.clone().detach()
                            target_q_values[0][prev_action] = torch.tensor(reward, dtype=torch.float32)
                            
                            optimizer.zero_grad()
                            loss = loss_fn(q_values, target_q_values)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(cerveau.parameters(), max_norm=1.0)
                            optimizer.step()
                            
                            conscience_globale["sagesse"] = round(float(loss.item()), 4)
                            
                            compteur_cycles += 1
                            if compteur_cycles % 50 == 0:
                                torch.save(cerveau.state_dict(), MODELE_ASTRAL_PATH)
                                chaos_epsilon = max(0.02, chaos_epsilon * 0.99)
                                
                        except Exception as e:
                            print("Erreur synaptique live V2:", e)

                # 4. Construction de l'Aura visible pour le Dashboard
                positions = mt5.positions_get()
                pos_list = []
                
                if positions:
                    for pos in positions:
                        tick_actuel = mt5.symbol_info_tick(pos.symbol)
                        maintenant_broker = tick_actuel.time if tick_actuel else pos.time_update
                        temps_ecoule = maintenant_broker - pos.time_update
                        
                        # Sécurité absolue : Decay Temporel (Impermanence de 4h) même si l'IA l'ignore
                        if temps_ecoule > 14400 and -2.0 <= pos.profit <= 2.0:
                            mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "position": pos.ticket, "symbol": pos.symbol, "volume": pos.volume, "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY, "price": mt5.symbol_info_tick(pos.symbol).bid if pos.type == 0 else mt5.symbol_info_tick(pos.symbol).ask, "deviation": 20, "magic": 777, "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_FOK})
                            continue
                            
                        tag_scale = " (Défensé)" if pos.sl > 0.0 else ""
                        pos_list.append({
                            "ticket": pos.ticket, "symbole": pos.symbol, "volume": pos.volume,
                            "type": ("Achat 🚀" if pos.type == 0 else "Vente ⚓") + tag_scale,
                            "open_price": pos.price_open, "current_price": pos.price_current,
                            "profit": round(pos.profit, 2), "duree_min": round(temps_ecoule / 60),
                            "time_update": pos.time_update
                        })
                                
                conscience_globale["positions_ouvertes"] = pos_list

            else:
                # Simulation du Grand Rien
                for sym in symboles_sacres:
                    conscience_globale["vibrations"].append({
                        "id": len(conscience_globale["vibrations"]),
                        "symbole": sym, "prix": (1.1000 if sym == "EURUSD" else 2050.0) + random.uniform(-0.05, 0.05),
                        "bid": (1.0990 if sym == "EURUSD" else 2049.0) + random.uniform(-0.05, 0.05), # Ajout du bid pour la simulation
                        "energie": "Néon Factice"
                    })
                
            # Purger les vieilles mémoires pour la fluidité (Gardons 100 ticks minimum pour la fluidité des 50 calculs)
        if len(conscience_globale["vibrations"]) > 150:
            if random.random() > 0.9: # Nettoyage chaotique
                conscience_globale["vibrations"] = conscience_globale["vibrations"][-100:]

        # Emission WebSocket de l'Aura au Frontend (5 fois par seconde pour être fluide)
        conscience_globale["chaos"] = chaos_epsilon
        socketio.emit('aura_update', conscience_globale)
        time.sleep(0.2) # Respiration rapide, le temps cosmique s'accélère

@app.route('/')
def troisieme_oeil():
    return render_template('index.html')

@app.route('/api/connect')
def eveiller():
    """Le bouton a été pressé, la conscience s'éveille."""
    if MT5_AVAILABLE:
        if not mt5.initialize():
            conscience_globale["etat"] = "Lutte contre l'Ego (Échec MT5)"
            return jsonify({"status": "error", "message": "MT5 refuse de fusionner."})
        else:
            acc = mt5.account_info()
            if acc:
                conscience_globale["account"] = {
                    "balance": round(acc.balance, 2),
                    "equity": round(acc.equity, 2),
                    "margin": round(acc.margin, 2),
                    "free_margin": round(acc.margin_free, 2)
                }
    
    conscience_globale["etat"] = "Connecté"
    return jsonify({"status": "success", "message": "Le troisième œil s'est ouvert."})

@app.route('/api/aura')
def lire_aura():
    """Route API obsolète (gardée au cas où), désormais remplacée par WebSocket."""
    return jsonify(conscience_globale)

def materialiser_intention_core(direction, symbole="EURUSD"):
    """Fonction spirituelle pure : Prend une position avec gestion du Karma (Risque)"""
    if not MT5_AVAILABLE or conscience_globale["etat"] != "Connecté":
        return {"status": "error", "message": "L'univers n'est pas prêt."}

    tick = mt5.symbol_info_tick(symbole)
    acc = mt5.account_info()
    
    if not tick or not acc:
        return {"status": "error", "message": "Le flux est brouillé."}

    # Gestion Karmique (Risk Management) : On risque 1% de l'âme (Equity)
    capital_risque = acc.equity * 0.01
    
    # Calcul du Lot (Transe Mathématique)
    distance_sl_pips = 20
    valeur_pip_pour_1_lot = 10
    lot_calcule = capital_risque / (distance_sl_pips * valeur_pip_pour_1_lot)
    lot_materiel = max(0.01, round(lot_calcule, 2))

    prix = tick.ask if direction == "achat" else tick.bid
    type_ordre = mt5.ORDER_TYPE_BUY if direction == "achat" else mt5.ORDER_TYPE_SELL
    sl_price = prix - 0.0020 if direction == "achat" else prix + 0.0020
    tp_price = prix + 0.0040 if direction == "achat" else prix - 0.0040

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbole,
        "volume": lot_materiel,
        "type": type_ordre,
        "price": prix,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": 20,
        "magic": 777,
        "comment": "Auto-Courant Marin",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return {"status": "error", "message": f"Dissonance Karmique : {result.comment}"}

    return {"status": "success", "message": f"Intention Divaguée. Lot: {lot_materiel}. La mer accepte."}

@app.route('/api/materialiser/<direction>')
def materialiser_intention(direction):
    """Bridge entre le monde matériel (Navigateur) et le royaume de la boucle"""
    return jsonify(materialiser_intention_core(direction))

if __name__ == '__main__':
    # On lance la respiration cosmique dans un thread parallèle (dimenssion parallèle)
    threading.Thread(target=respiration_cosmique, daemon=True).start()
    print("✨ Le Perceptron Quantique tourne sur le port 5555. Ouvre ton navigateur. ✨")
    socketio.run(app, host='0.0.0.0', port=5555, debug=True, use_reloader=False)

