"""10x10 environment + Q-Learning with SGDRegressor (incremental learning)."""
import random
import numpy as np
from sklearn.linear_model import SGDRegressor

GRID = [
    "Aoo#oooD#o",
    "o#oo#o#ooo",
    "oDo#oDoo#o",
    "ooo#o#oDoo",
    "o#oooDo#oo",
    "oooD#ooo#o",
    "o#oo#o#ooD",
    "oDo#ooo#oo",
    "o#oooo#ooo",
    "ooooDoDooT",
]
ROWS, COLS = len(GRID), len(GRID[0])
ACTIONS = ["Up", "Down", "Left", "Right"]
MOVES = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}
CELL_NAMES = {"A": "Start", "T": "Goal", "o": "Path", "#": "Wall", "D": "Danger"}

REWARDS = {"normal": -1, "invalid": -5, "wall": -5, "danger": -10, "goal": 100}
CONFIG = {"episodes": 800, "max_steps": 200, "gamma": 0.95, "epsilon": 1.0,
          "epsilon_min": 0.05, "epsilon_decay": 0.992, "eta0": 0.1}


def find(ch):
    return next((r, c) for r in range(ROWS) for c in range(COLS) if GRID[r][c] == ch)

START, GOAL = find("A"), find("T")


def counts():
    flat = "".join(GRID)
    return {ch: flat.count(ch) for ch in "ATo#D"}


class Environment:
    def __init__(self, max_steps):
        self.max_steps = max_steps

    def reset(self):
        self.state, self.steps = START, 0
        return self.state

    def step(self, action):
        """Returns (next_state, cell_type, reward, done)."""
        self.steps += 1
        r, c = self.state
        dr, dc = MOVES[action]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < ROWS and 0 <= nc < COLS):
            cell, reward, nxt = "Invalid", REWARDS["invalid"], self.state
        elif GRID[nr][nc] == "#":
            cell, reward, nxt = "Wall", REWARDS["wall"], self.state
        else:
            nxt = (nr, nc)
            ch = GRID[nr][nc]
            cell = {"D": "Danger", "T": "Goal"}.get(ch, "Path")
            reward = {"D": REWARDS["danger"], "T": REWARDS["goal"]}.get(ch, REWARDS["normal"])
        self.state = nxt
        done = nxt == GOAL or self.steps >= self.max_steps
        return nxt, cell, reward, done


class QAgent:
    """One SGDRegressor per action; input = one-hot state (r*COLS+c)."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.models = {a: SGDRegressor(learning_rate="constant", eta0=cfg["eta0"],
                                       fit_intercept=False, alpha=0.0, random_state=42)
                       for a in ACTIONS}
        for m in self.models.values():          # initialize (predicts 0)
            m.partial_fit(np.zeros((1, ROWS * COLS)), [0.0])

    @staticmethod
    def x(s):
        v = np.zeros((1, ROWS * COLS)); v[0, s[0] * COLS + s[1]] = 1; return v

    def q(self, s):
        i = s[0] * COLS + s[1]   # same as predict() with one-hot input, but much faster
        return np.array([self.models[a].coef_[i] for a in ACTIONS])

    def act(self, s, eps):
        if random.random() < eps:
            return random.choice(ACTIONS)                      # exploration
        return ACTIONS[int(np.argmax(self.q(s)))]              # exploitation

    def update(self, s, a, r, s2, done):
        target = r if done else r + self.cfg["gamma"] * self.q(s2).max()
        self.models[a].partial_fit(self.x(s), [target])


def train_and_evaluate(cfg=None):
    cfg = {**CONFIG, **(cfg or {})}
    random.seed(42); np.random.seed(42)
    env, agent, eps = Environment(cfg["max_steps"]), QAgent(cfg), cfg["epsilon"]
    successes, rewards = 0, []
    for _ in range(cfg["episodes"]):
        s, done, total = env.reset(), False, 0
        while not done:
            a = agent.act(s, eps)
            s2, _, r, done = env.step(a)
            agent.update(s, a, r, s2, done)
            s, total = s2, total + r
        successes += s == GOAL
        rewards.append(total)
        eps = max(cfg["epsilon_min"], eps * cfg["epsilon_decay"])

    # Evaluation without exploration (epsilon = 0)
    s, done, steps, path, total = env.reset(), False, [], [START], 0
    while not done:
        a = ACTIONS[int(np.argmax(agent.q(s)))]
        s2, cell, r, done = env.step(a)
        steps.append(dict(step=len(steps) + 1, state=s, action=a, next_state=s2, cell=cell, reward=r))
        total += r; path.append(s2); s = s2

    qtable = []
    for r_ in range(ROWS):
        for c_ in range(COLS):
            if GRID[r_][c_] != "#":
                qtable.append(dict(state=(r_, c_), q=[round(float(v), 2) for v in agent.q((r_, c_))]))
    return dict(
        cfg=cfg, episodes=cfg["episodes"], successes=int(successes),
        success_pct=round(100 * successes / cfg["episodes"], 2),
        avg_reward=round(float(np.mean(rewards)), 2), final_epsilon=round(eps, 4),
        steps=steps, n_moves=len(steps), eval_reward=total, goal_reached=s == GOAL,
        path=set(path), path_len=len(set(path)), qtable=qtable,
        danger_hits=sum(1 for t in steps if t["cell"] == "Danger"),
    )
