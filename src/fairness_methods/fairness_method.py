class FairnessMethod:
    def __init__(self, configs, dataset):
        self.configs = configs
        self.dataset = dataset
        self.set_defaults_specifics()

    def set_paths_specifics(self, out_path):
        pass

    def set_paths_runs_specifics(self, i):
        pass

    def set_defaults_specifics(self):
        pass

    def generate_fair_data(self, data_train, data_test):
        return data_train, data_test