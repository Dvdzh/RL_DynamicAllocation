import CustomEnv
import CustomAgent
import matplotlib.pyplot as plt
import numpy as np
import random


def train(CustomAgent: CustomAgent, CustomEnv: CustomEnv, episodes=10_000, max_steps=100, learning_rate=0.01, gamma=0.8, epsilon=0.7, alpha=0, trial=None):
    if var_optuna:
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.5)
        gamma = trial.suggest_float('gamma', 0., 0.8)
        # epsilon = trial.suggest_float('epsilon', 0.4, 3)
    q_table, dictio = CustomAgent.q_table, CustomAgent.dictio
    q_table_min = []
    q_table_max = []
    CustomEnv = CustomEnv()
    cout_liste = []
    for i in range(episodes):
        obs = CustomEnv.reset()
        # print("Episode {} commence à {}".format(i,obs))
        for j in range(max_steps):
            # pour tracer le q_table min
            q_table_min.append(q_table.min())
            q_table_max.append(q_table.max())
            # choix de l'action, avec politique epsilon greedy
            p = np.random.random()
            if p < epsilon:
                action = np.argmax(q_table[dictio.get(obs)])
            else:
                action = random.randint(0, K)
            # on allume le nombre nécessaire de vm
            old_obs = obs
            obs, cout = CustomEnv.step(action)
            cout_liste.append(cout)
            att = q_table[dictio.get(old_obs)][action]
            # on modifie la q_table avec la q_valeur
            q_value = (1 - alpha) * q_table[dictio.get(old_obs)][action] + learning_rate * \
                (-cout + gamma * np.max(q_table[dictio.get(obs)]
                                        ) - q_table[dictio.get(old_obs)][action])
            q_table[dictio.get(old_obs)][action] = q_value
    plt.plot(q_table_min, label="q_table_min")
    plt.plot(q_table_max, label="q_table_max")
    plt.legend()
    # plt.show()
    if var_save:
        np.savez("weight", q_table=q_table)
    return CustomAgent, sum(cout_liste)/len(cout_liste)


if __name__ == '__main__':
    agent, env = CustomAgent(), CustomEnv()
    if var_train:
        agent = train(agent, env, episodes=100)
    if var_optuna:
        def train_opt(trial):
            agent, env = Agent(), Env()
            _, cout = train(agent, env, episodes=1_000, trial=trial)
            return cout
        study = optuna.create_study()
        study.optimize(train_opt, n_trials=100, n_jobs=1)
        print(study.best_params)
        #trial.suggest_int('n_hidden', 1, 3)
