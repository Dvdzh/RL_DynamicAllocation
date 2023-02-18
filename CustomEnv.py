import random


class CustomEnv():
    def __init__(self):
        self.vm_actif = 0
        self.personne_en_file = 0
        self.cout = 0
        self.arrivee = 0
        self.param = {"B": 10, "K": 10, "b": 2, "d": 1,
                      "arrivee_pe": 10, "Cf": 1, "Ca": 1, "Cd": 1, "Ch": 1}

    def step(self, action: int):  # retoure observation, cout
        self.arrivee = self.generer_arrive()
        self.personne_en_file = min(
            self.param["B"], max(0, self.personne_en_file + self.arrivee - action * self.param["d"]))
        self.cout = self.calc_cout(
            self.personne_en_file, action, self.vm_actif)
        self.vm_actif = action
        return (self.personne_en_file, self.vm_actif), self.cout

    def reset(self):
        self.vm_actif = 0
        self.personne_en_file = 0
        self.cout = 0
        return (self.personne_en_file, self.vm_actif)

    def calc_cout(self, personne: int, action: int, vm_actif: int):
        return action*self.param["Cf"] + personne*self.param["Ch"] + max((action - vm_actif)*self.param["Ca"], (vm_actif - action)*self.param["Cd"])

    def generer_arrive(self, normal=False) -> int:
        if normal:
            return int(np.random.normal(5, 2, 1)[0])
        return random.randint(0, self.param["arrivee_pe"])

    def set_B(self, val):
        self.param["B"] = val

    def set_K(self, val):
        self.param["K"] = val

    def set_b(self, val):
        self.param["b"] = val

    def set_d(self, val):
        self.param["d"] = val

    def set_arrivee_pe(self, val):
        self.param["arrivee_pe"] = val

    def set_Cf(self, val):
        self.param["Cf"] = val

    def set_Ca(self, val):
        self.param["Ca"] = val

    def set_Cd(self, val):
        self.param["Cd"] = val

    def set_Ch(self, val):
        self.param["Ch"] = val
