from CustomEnv import CustomEnv
from CustomAgent import CustomAgent
import matplotlib.pyplot as plt
import argparse
import random
import numpy as np
import optuna
from affichage import affichage
from matplotlib.animation import FuncAnimation  # test
import time


def train(agent: CustomAgent, env: CustomEnv,
          episodes=10_000, max_steps=100,
          learning_rate=0.01, gamma=0.8, epsilon=0.7, alpha=0,
          trial=None, bool_save=True, bool_optuna=False):
    start = time.time()
    if bool_optuna:
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.5)
        gamma = trial.suggest_float('gamma', 0., 0.8)
        # epsilon = trial.suggest_float('epsilon', 0.4, 3)
    q_table, dictio = agent.q_table, agent.dictio
    q_table_min = []
    q_table_max = []
    cout_liste = []
    for i in range(episodes):
        obs = env.reset()
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
                action = random.randint(0, env.param["K"])
            # on allume le nombre nécessaire de vm
            old_obs = obs
            obs, cout = env.step(action)
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
    plt.show()
    if bool_save:
        np.savez("weight", q_table=q_table)
    print("Train executed in {} ms".format(1000*(time.time()-start)))
    return agent, sum(cout_liste)/len(cout_liste)


def test(agent: CustomAgent, env: CustomEnv, episodes=4, max_steps_test=100, bool_load=False):
    q_table, dictio = agent.q_table, agent.dictio
    if bool_load:
        data = np.load("weight.npz")
        q_table = data['q_table']
    for i in range(episodes):
        obs = env.reset()
        arrivee = []
        personne_en_file = []
        vm_actif = []
        liste_cout = []
        for j in range(max_steps_test):
            # avance
            action = np.argmax(q_table[dictio.get(obs)])
            obs, cout = env.step(action)
            # print la liste
            arrivee.append(env.arrivee)
            personne_en_file.append(env.personne_en_file)
            vm_actif.append(env.vm_actif)
            liste_cout.append(env.cout)
        # plt plot
        fig1 = plt.subplot(211)
        plt.plot(personne_en_file, label="personne en file")
        plt.plot(arrivee, label="arrivee")
        plt.plot(vm_actif, label="vm_actif")
        plt.axhline(y=20, color='gray', label="personne max")
        fig2 = plt.subplot(212)
        plt.plot(liste_cout, label="coût")
        fig1.legend()
        fig2.legend()
        plt.show()
        print("Arrivee moyenne :", sum(arrivee)/len(arrivee))
        print("Cout moyen : ", sum(liste_cout)/len(liste_cout))
        print("Personne en file moyen : ", sum(
            personne_en_file)/len(personne_en_file))


def test_anim(agent: CustomAgent, env: CustomEnv, episodes=1, max_steps_test=100, bool_load=False):
    q_table, dictio = agent.q_table, agent.dictio
    if bool_load:
        data = np.load("weight.npz")
        q_table = data['q_table']
    for i in range(episodes):
        arrivee = []
        personne_en_file = []
        vm_actif = []
        liste_cout = []
        x = []
        fig = plt.figure()
        ln1, = plt.plot([], [], 'r')
        ln2, = plt.plot([], [], 'b')
        ln3, = plt.plot([], [], 'g')
        ln4, = plt.plot([], [], 'y')

        obs = [env.reset()]

        def update(frame):
            action = np.argmax(q_table[dictio.get(obs[-1])])
            obs.append(env.step(action)[0])
            x.append(frame)
            liste_cout.append(env.cout)
            vm_actif.append(env.vm_actif)
            arrivee.append(env.arrivee)
            personne_en_file.append(env.personne_en_file)
            plt.xlim([-1, x[-1]+1])
            plt.ylim([-1, liste_cout[-1]+1])
            ln1.set_data(x, liste_cout)
            ln2.set_data(x, vm_actif)
            ln3.set_data(x, arrivee)
            ln4.set_data(x, personne_en_file)
            return ln1, ln2, ln3, ln4,

        ani = FuncAnimation(fig, update, frames=100, interval=200)
        plt.show()


if __name__ == '__main__':
    epilog = "Goooood train !"
    parser = argparse.ArgumentParser(prog='DynamicAllocation',
                                     description="Simule des machines virtuelles",
                                     epilog=epilog)
    parser.add_argument("--test", type=int, help="Test de l'agent",
                        nargs="+", metavar=("nb_episode", "nb_max_step"))
    parser.add_argument("--test_anim", help="Test de l'agent avec animation",
                        action="store_true")

    parser.add_argument("--train", type=int, help="active train avec config",
                        nargs="+", metavar=("nb_episode", "nb_max_step"))
    parser.add_argument("--optuna", type=int, help="Cherche les bons hyperparamètres",
                        nargs="+", metavar=("n_trails, nb_jobs"))

    parser.add_argument("-s", "--save", help="active save weight, fonctionne que si --train, ne fonctionne pas si --optuna",
                        action="store_true")
    parser.add_argument("-l", "--load", help="active load weight, fonctionne que si --test ",
                        action="store_true")

    parser.add_argument("-tt", "--train_test", help="Train puis test de l'agent",
                        action="store_true")
    parser.add_argument("-a", "--affichage", help="Affiche la q_table",
                        action="store_true")
    args = parser.parse_args()

    env = CustomEnv()
    env.set_K(1)  # pour affichage
    env.set_B(10)
    agent = CustomAgent(env)

    if args.train:
        train(agent, env, episodes=args.train[0],
              max_steps=args.train[1], bool_save=args.save)
    if args.optuna:
        def train_opt(trial):
            _, cout = train(agent, env, episodes=args.optuna[0],
                            trial=args.optuna[1], bool_save=False)
            return cout
        study = optuna.create_study()
        study.optimize(train_opt, n_trials=10, n_jobs=1)
        print(study.best_params)
    if args.test:
        test(agent, env, bool_load=args.load,
             episodes=args.train[0], max_steps_test=args.test[1])
    if args.train_test:
        train(agent, env, episodes=100,
              max_steps=100, bool_save=args.save)
        test(agent, env, bool_load=args.load, max_steps_test=100)
    if args.affichage:
        affichage(env.param["B"], env.param["K"]*env.param["B"])
    if args.test_anim:
        test_anim(agent, env, bool_load=args.load,
                  max_steps_test=100)
