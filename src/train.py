from tqdm import tqdm
import numpy as np
import pickle

from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed
from evaluate import evaluate_HIV

from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
GAMMA = 0.95
NB_ACTIONS = env.action_space.n


class ProjectAgent:
    def __init__(self) -> None:
        self.Qvalue = None
        self.gamma = GAMMA
        self.nb_actions = NB_ACTIONS
        self.S = None
        self.A = None
        self.R = None
        self.S2 = None

    def _greedy_action(self,s):
        Qsa = []
        for a in range(self.nb_actions):
            sa = np.append(s,a).reshape(1, -1)
            Qsa.append(self.Qvalue.predict(sa))
        return np.argmax(Qsa)

    def _rf_fqi(self, iterations, disable_tqdm=True):
        nb_samples = self.S.shape[0]
        SA = np.append(self.S,self.A,axis=1)
        for iter in tqdm(range(iterations), disable=disable_tqdm):
            if iter==0:
                value=self.R.copy()
            else:
                Q2 = np.zeros((nb_samples,self.nb_actions))
                for a2 in range(self.nb_actions):
                    A2 = a2*np.ones((nb_samples,1))
                    S2A2 = np.append(self.S2,A2,axis=1)
                    Q2[:,a2] = Q.predict(S2A2)
                max_Q2 = np.max(Q2,axis=1)
                value = self.R + self.gamma*max_Q2
            Q = RandomForestRegressor(50, max_depth=15,n_jobs=-1)
            Q.fit(SA,value)
        self.Qvalue = Q

    def _aux_multiproc(self, env, epsilon):
        s, _ = env.reset()
        res = np.empty((200,14))
        for i in range(200):
            res[i,:6] = s
            a = self.act(s,use_random=np.random.rand() < epsilon)
            s, r, _, _, _ = env.step(a)
            res[i,6:12] = s
            res[i, 12] = a
            res[i, 13] = r
        return res


    def _collect_samples(self,env, nb_samples, epsilon):
        result = Parallel(n_jobs=10)(delayed(self._aux_multiproc)(env, epsilon) for _ in range(nb_samples//200))
        for res in result:
            S = res[:,:6]
            S2 = res[:,6:12]
            A = (res[:,12]).reshape(-1,1)
            R = res[:,13]
            self.S = S if self.S is None else np.vstack([self.S,S])
            self.S2 = S2 if self.S2 is None else np.vstack([self.S2,S2])
            self.A = A if self.A is None else np.vstack([self.A,A])
            self.R = R if self.R is None else np.append(self.R,R)

    def train(self, env, nb_train, nb_samples, nb_iter):
        score = 0
        for i in range(nb_train):
            print(i)
            epsilon = 1 if i==0 else 1/(i*i)
            self._collect_samples(env, nb_samples, epsilon)
            self._rf_fqi(nb_iter)
            score2 = evaluate_HIV(agent=self, nb_episode=1)
            print(score2, end=" ")
            if score2 > score:
                score = score2
            else:
                self.Qvalue = self.old_Qvalue
            print(score)
            self.old_Qvalue = self.Qvalue
            agent.save(f"model_{i}.pkl")

    def act(self, observation, use_random=False):
        if use_random:
            action = np.random.randint(self.nb_actions)
        else:
            action = self._greedy_action(observation)
        return action

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.Qvalue, f)

    def load(self):
        with open("model.pkl", "rb") as f:
            self.Qvalue = pickle.load(f)

if __name__ == "__main__":
    nb_samples, nb_iter = 4000, 200
    nb_train = 20
    agent = ProjectAgent()
    agent.train(env,nb_train, nb_samples, nb_iter)
    agent.save("model.pkl")

