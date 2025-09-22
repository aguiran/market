import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
import pygame
import os
from dataclasses import dataclass


@dataclass
class AgentPerturbateur:
    credibility: float  # 0..10
    scope: float        # 0..10
    impact: float       # -10..10
    probability: float  # 0..1
    timestamp_min: int  # minutes since epoch (or since env start)


class MarketEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, **kwargs):
        super().__init__()

        # Config par défaut
        self.config = {
            # Durées de frames (minutes)
            "step_minutes": 1,
            "past_far_min": 14 * 24 * 60,
            "past_mid_min": 23 * 60,
            "past_near_min": 60,
            "future_near_min": 60,
            "future_mid_min": 47 * 60,
            "future_far_min": 48 * 60,  # > 48h
            # Prix
            "price_min": 0,
            "price_max": 300000,
            # Agents
            "max_agents_per_frame": 200,
            # DQN action step (pourcent)
            "price_action_step_percent": 0.1,  # 0.1%
            # Episode
            "max_episode_steps": 5000,
            # Logging
            "log_dir": "runs",
            "log_file": "market_simulation.txt",
            # Poids d'influence
            "weights_future": {"near": 1.0, "mid": 0.5, "far": 0.2},
            "weights_past": {"present": 0.02, "near": 0.01, "mid": 0.005, "far": 0.002},
            # Terme de tendance (léger)
            "trend_coef": 0.01,
            "trend_window": 15,
            # Données externes optionnelles (30j = 43200 min)
            "use_data_files": False,
            "data_dir": "market_env/data",
            "price_file": "prices_30d.txt",
            "agents_file": "agents_30d.txt",
            # Présent choisi dans la fenêtre 30j (par défaut: 10j)
            "start_timestamp_min": 10 * 24 * 60,
        }
        self.config.update(kwargs or {})

        # Observation: 300 points passés + 7 frames agrégées (5 features chacune) + 3 contextes
        self.past_points_per_frame = 100
        obs_dim = 300 + 5 * 7 + 3
        self.observation_space = spaces.Box(
            low=np.full((obs_dim,), -np.inf, dtype=np.float32),
            high=np.full((obs_dim,), np.inf, dtype=np.float32),
            dtype=np.float32,
        )

        # Actions: delta prix en % discretisé {-5,-4.9,...,0,...,4.9,5} avec pas 0.1
        step = self.config["price_action_step_percent"]
        deltas = np.arange(-5.0, 5.0 + step, step)
        self.action_deltas = deltas / 100.0
        self.action_space = spaces.Discrete(len(self.action_deltas))

        # Etat interne
        self.t = 0  # minutes depuis le début
        self.episode_steps = 0
        self.price_true = None
        self.price_pred = None
        self.price_series = None  # si use_data_files

        # Frames d'agents (deque pour chaque frame)
        self.frames = {
            "past_far": deque(maxlen=self.config["max_agents_per_frame"]),
            "past_mid": deque(maxlen=self.config["max_agents_per_frame"]),
            "past_near": deque(maxlen=self.config["max_agents_per_frame"]),
            "present": deque(maxlen=self.config["max_agents_per_frame"]),
            "future_near": deque(maxlen=self.config["max_agents_per_frame"]),
            "future_mid": deque(maxlen=self.config["max_agents_per_frame"]),
            "future_far": deque(maxlen=self.config["max_agents_per_frame"]),
        }

        # Series de prix passée (buffer de 14 jours à la minute)
        self.history_minutes = self.config["past_far_min"]
        self.price_history = deque(maxlen=self.history_minutes)

        # Logging
        os.makedirs(self.config["log_dir"], exist_ok=True)
        self.log_path = os.path.join(self.config["log_dir"], self.config["log_file"])
        # Data files
        self.data_dir = self.config["data_dir"]
        self.price_file_path = os.path.join(self.data_dir, self.config["price_file"])
        self.agents_file_path = os.path.join(self.data_dir, self.config["agents_file"])
        os.makedirs(self.data_dir, exist_ok=True)

        # Render
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.window_size = (900, 300)

    # --- Utils de génération ---
    def _rand_agent(self, ts):
        return AgentPerturbateur(
            credibility=float(np.random.uniform(0, 10)),
            scope=float(np.random.uniform(0, 10)),
            impact=float(np.random.uniform(-10, 10)),
            probability=float(np.random.uniform(0, 1)),
            timestamp_min=int(ts),
        )

    def _seed_frames_random(self):
        for key in self.frames.keys():
            self.frames[key].clear()
            n = np.random.randint(0, self.config["max_agents_per_frame"] + 1)
            for _ in range(n):
                self.frames[key].append(self._rand_agent(self.t))

    # --- Data files generation/loading ---
    def _generate_data_files(self):
        total = 30 * 24 * 60  # 30 jours en minutes
        # Agents: ~2000 aléatoires répartis sur 30 jours
        num_agents = 2000
        rng = np.random.default_rng()
        with open(self.agents_file_path, "w") as f:
            f.write("timestamp_min,credibility,scope,impact,probability\n")
            for _ in range(num_agents):
                ts = int(rng.integers(0, total))
                cred = float(rng.uniform(0, 10))
                scope = float(rng.uniform(0, 10))
                impact = float(rng.uniform(-10, 10))
                prob = float(rng.uniform(0, 1))
                f.write(f"{ts},{cred:.6f},{scope:.6f},{impact:.6f},{prob:.6f}\n")

        # Prix synthétique influencé par ces agents (futur dominant)
        prices = np.zeros(total, dtype=np.float64)
        prices[0] = float(np.random.uniform(1000, 2000))
        # Charger agents rapides en mémoire pour calcul
        agents_by_ts = {}
        with open(self.agents_file_path, "r") as f:
            next(f)
            for line in f:
                ts, cred, scope, imp, prob = line.strip().split(",")
                ts = int(ts)
                agents_by_ts.setdefault(ts, []).append((float(cred), float(scope), float(imp), float(prob)))
        for i in range(1, total):
            prev = prices[i-1]
            ar = 0.98 * prev + 0.02 * float(np.random.uniform(self.config["price_min"], self.config["price_max"]))
            # influence des agents à horizon court (proxy): utiliser agents avec ts==i (effet au moment i)
            eff = 0.0
            for (cred, scope, imp, prob) in agents_by_ts.get(i, []):
                eff += prob * imp * (1.0 + cred/10.0)
            prices[i] = float(np.clip(ar + eff, self.config["price_min"], self.config["price_max"]))

        with open(self.price_file_path, "w") as f:
            f.write("minute,price\n")
            for i, p in enumerate(prices):
                f.write(f"{i},{p:.6f}\n")

    def _load_data_files(self):
        if not (os.path.exists(self.price_file_path) and os.path.exists(self.agents_file_path)):
            self._generate_data_files()
        # Charger prix
        data = np.loadtxt(self.price_file_path, delimiter=",", skiprows=1)
        # data shape (N,2): minute, price
        self.price_series = data[:, 1].astype(np.float64)
        # Charger agents
        self.frames_clear()
        self.all_agents = []
        with open(self.agents_file_path, "r") as f:
            next(f)
            for line in f:
                ts, cred, scope, imp, prob = line.strip().split(",")
                ag = AgentPerturbateur(
                    credibility=float(cred), scope=float(scope), impact=float(imp), probability=float(prob), timestamp_min=int(ts)
                )
                self.all_agents.append(ag)

    def frames_clear(self):
        for key in self.frames.keys():
            self.frames[key].clear()

    # Migration des frames futures/passes par fenêtres glissantes
    def _migrate_frames(self):
        step = self.config["step_minutes"]
        # On déplace les frontières en minutes: la future_near gagne +step, le prend à future_mid, etc.
        # Ici on se contente de déplacer des agents dont le timestamp tombe désormais dans la nouvelle fenêtre.
        def frame_bounds():
            a = self.config
            # bornes relatives au présent t (min inclus, max exclus)
            return {
                "past_far": (-a["past_far_min"], -a["past_mid_min"] - a["past_near_min"]),
                "past_mid": (-a["past_mid_min"] - a["past_near_min"], -a["past_near_min"]),
                "past_near": (-a["past_near_min"], 0),
                "present": (0, 0),
                "future_near": (0, a["future_near_min"]),
                "future_mid": (a["future_near_min"], a["future_near_min"] + a["future_mid_min"]),
                "future_far": (a["future_near_min"] + a["future_mid_min"], np.inf),
            }

        bounds = frame_bounds()

        def place_frame(agent):
            dt = agent.timestamp_min - self.t
            for k, (lo, hi) in bounds.items():
                if lo <= dt < hi:
                    return k
            return "future_far"

        all_agents = []
        for key in self.frames:
            all_agents.extend(list(self.frames[key]))
            self.frames[key].clear()
        # Replacer
        for ag in all_agents:
            k = place_frame(ag)
            if len(self.frames[k]) < self.frames[k].maxlen:
                self.frames[k].append(ag)

    def _update_agent_probabilities(self):
        for key in self.frames:
            for ag in self.frames[key]:
                ag.probability = float(np.clip(ag.probability + np.random.normal(0, 0.02), 0.0, 1.0))

    def _true_price_next(self):
        # Modèle simple: AR(1) + influence agents futurs pondérée
        if len(self.price_history) == 0:
            prev = float(np.random.uniform(1000, 2000))
        else:
            prev = self.price_history[-1]

        ar = 0.98 * prev + 0.02 * float(np.random.uniform(self.config["price_min"], self.config["price_max"]))

        # Poids des frames
        wf = self.config["weights_future"]
        wp = self.config["weights_past"]

        def frame_effect(frame, w):
            eff = 0.0
            for ag in frame:
                eff += ag.probability * ag.impact * (1.0 + ag.credibility / 10.0)
            return w * eff

        # Futur (principal)
        eff_future = (
            frame_effect(self.frames["future_near"], wf["near"]) +
            frame_effect(self.frames["future_mid"], wf["mid"]) +
            frame_effect(self.frames["future_far"], wf["far"]) 
        )

        # Présent + passé (faible)
        eff_past = (
            frame_effect(self.frames["present"], wp["present"]) +
            frame_effect(self.frames["past_near"], wp["near"]) +
            frame_effect(self.frames["past_mid"], wp["mid"]) +
            frame_effect(self.frames["past_far"], wp["far"]) 
        )

        # Terme de tendance très léger
        trend = 0.0
        if len(self.price_history) > self.config["trend_window"]:
            window = self.config["trend_window"]
            p0 = self.price_history[-window]
            p1 = self.price_history[-1]
            trend = self.config["trend_coef"] * (p1 - p0) / max(1e-6, window)

        eff_total = eff_future + eff_past + trend

        # Convertir l'effet en variation de prix (échelle arbitraire)
        next_price = float(np.clip(ar + eff_total, self.config["price_min"], self.config["price_max"]))
        return next_price

    def _observation(self):
        # 300 points passés uniformément échantillonnés sur C/B/A (100 chacun)
        a = self.config
        def sample_history(duration_min):
            idxs = np.linspace(max(0, len(self.price_history) - duration_min), len(self.price_history) - 1, self.past_points_per_frame)
            idxs = np.clip(idxs.astype(int), 0, len(self.price_history) - 1)
            return np.array([self.price_history[i] if len(self.price_history)>0 else 0.0 for i in idxs], dtype=np.float32)

        past_near = sample_history(a["past_near_min"]) if len(self.price_history)>0 else np.zeros(self.past_points_per_frame, dtype=np.float32)
        past_mid = sample_history(a["past_near_min"] + a["past_mid_min"]) if len(self.price_history)>0 else np.zeros(self.past_points_per_frame, dtype=np.float32)
        past_far = sample_history(a["past_far_min"]) if len(self.price_history)>0 else np.zeros(self.past_points_per_frame, dtype=np.float32)

        def agg(frame):
            if len(frame) == 0:
                return np.array([0, 0, 0, 0, 0], dtype=np.float32)
            probs = np.array([ag.probability for ag in frame], dtype=np.float32)
            impacts = np.array([ag.impact for ag in frame], dtype=np.float32)
            creds = np.array([ag.credibility for ag in frame], dtype=np.float32)
            scopes = np.array([ag.scope for ag in frame], dtype=np.float32)
            return np.array([
                float(np.sum(probs * impacts)),
                float(np.max(np.abs(impacts))),
                float(np.mean(creds)),
                float(np.mean(scopes)),
                float(len(frame) / self.config["max_agents_per_frame"]),
            ], dtype=np.float32)

        frames_order = ["past_far","past_mid","past_near","present","future_near","future_mid","future_far"]
        aggs = np.concatenate([agg(self.frames[k]) for k in frames_order]).astype(np.float32)

        price_now = self.price_history[-1] if len(self.price_history)>0 else 0.0
        price_now_norm = (price_now - self.config["price_min"]) / max(1.0, (self.config["price_max"] - self.config["price_min"]))
        err_prev = 0.0 if self.price_pred is None or price_now==0 else abs(self.price_pred - price_now) / max(1e-6, price_now)
        deriv = 0.0 if len(self.price_history)<2 else (self.price_history[-1] - self.price_history[-2])

        obs = np.concatenate([
            past_far, past_mid, past_near,
            aggs,
            np.array([price_now_norm, err_prev, deriv], dtype=np.float32)
        ]).astype(np.float32)
        return obs

    # --- Gym API ---
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Charger données si demandé
        if self.config["use_data_files"]:
            self._load_data_files()
        self.t = int(self.config.get("start_timestamp_min", 10 * 24 * 60))
        self.episode_steps = 0
        self.price_history.clear()

        if self.config["use_data_files"] and self.price_series is not None:
            # Remplir l'historique avec les minutes précédant t
            start_hist = max(0, self.t - self.price_history.maxlen)
            for i in range(start_hist, self.t + 1):
                self.price_history.append(float(self.price_series[i]))
            self.price_true = self.price_history[-1]
            self.price_pred = self.price_true
            # Placer agents selon timestamp
            self.frames_clear()
            for ag in self.all_agents:
                dt = ag.timestamp_min - self.t
                # place via _migrate logique: reutiliser place_frame
            self._migrate_frames()
        else:
            # Seed prix initial sur 2 jours aléatoire
            init_len = self.config["past_near_min"]
            base = float(np.random.uniform(1000, 2000))
            for i in range(init_len):
                base = 0.98 * base + 0.02 * float(np.random.uniform(self.config["price_min"], self.config["price_max"]))
                self.price_history.append(base)
            self.price_true = self.price_history[-1]
            self.price_pred = self.price_true
            self._seed_frames_random()

        # Log header
        with open(self.log_path, "w") as f:
            f.write("t,price_true,price_pred,err,agents_past_far,agents_past_mid,agents_past_near,agents_present,agents_future_near,agents_future_mid,agents_future_far\n")

        if self.render_mode == "human":
            self.render()
        return self._observation(), {}

    def step(self, action):
        self.episode_steps += 1
        # Appliquer action
        delta = self.action_deltas[int(action)]
        self.price_pred = float(np.clip(self.price_pred * (1.0 + delta), self.config["price_min"], self.config["price_max"]))

        # Avancer le temps
        self.t += self.config["step_minutes"]

        # Migration des agents entre frames selon timestamp relatif
        self._migrate_frames()

        # Mise à jour probas agents
        self._update_agent_probabilities()

        # Prix réel suivant
        if self.config["use_data_files"] and self.price_series is not None:
            idx = min(self.t, len(self.price_series)-1)
            self.price_true = float(self.price_series[idx])
        else:
            self.price_true = self._true_price_next()
        self.price_history.append(self.price_true)

        # Erreur et récompense
        err = abs(self.price_pred - self.price_true) / max(1e-6, self.price_true)
        if err <= 0.01:
            reward = 2.0
        elif err <= 0.03:
            reward = 0.5
        else:
            reward = -10.0

        # Terminaisons
        terminated = err > 0.05
        truncated = self.episode_steps >= self.config["max_episode_steps"]

        # Log
        with open(self.log_path, "a") as f:
            f.write(
                f"{self.t},{self.price_true:.6f},{self.price_pred:.6f},{err:.6f},{len(self.frames['past_far'])},{len(self.frames['past_mid'])},{len(self.frames['past_near'])},{len(self.frames['present'])},{len(self.frames['future_near'])},{len(self.frames['future_mid'])},{len(self.frames['future_far'])}\n"
            )
        if self.render_mode == "human":
            self.render()
        return self._observation(), float(reward), bool(terminated), bool(truncated), {}

    def render(self):
        if self.render_mode != "human":
            return
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("MarketEnv - Present HUD")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((245, 245, 245))

        # Text metrics
        font = pygame.font.Font(None, 24)
        price_now = self.price_history[-1] if len(self.price_history)>0 else 0.0
        err = 0.0 if price_now == 0 else abs(self.price_pred - price_now)/max(1e-6, price_now)
        lines = [
            f"Step: {self.episode_steps}  t(min): {self.t}",
            f"Price true: {price_now:.2f}",
            f"Price pred: {float(self.price_pred):.2f}",
            f"Error %: {err*100:.2f}",
            f"Agents present: {len(self.frames['present'])}",
            f"Future near/mid/far: {len(self.frames['future_near'])}/{len(self.frames['future_mid'])}/{len(self.frames['future_far'])}",
            f"Past near/mid/far: {len(self.frames['past_near'])}/{len(self.frames['past_mid'])}/{len(self.frames['past_far'])}",
        ]
        for i, text in enumerate(lines):
            surf = font.render(text, True, (20, 20, 20))
            canvas.blit(surf, (12, 12 + i*24))

        # Simple present gauge for agents influence
        def present_influence():
            eff = 0.0
            for ag in self.frames['present']:
                eff += ag.probability * ag.impact * (1.0 + ag.credibility/10.0)
            return eff
        eff = present_influence()
        midx, midy = 700, 150
        pygame.draw.rect(canvas, (220, 220, 220), (midx-100, midy-10, 200, 20))
        val = max(-1.0, min(1.0, eff/100.0))
        color = (200, 50, 50) if val>0 else (50, 120, 220)
        width = int(100*abs(val))
        if val >= 0:
            pygame.draw.rect(canvas, color, (midx, midy-10, width, 20))
        else:
            pygame.draw.rect(canvas, color, (midx-width, midy-10, width, 20))
        gauge_label = font.render("Present influence", True, (20,20,20))
        canvas.blit(gauge_label, (midx-80, midy-36))

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])


