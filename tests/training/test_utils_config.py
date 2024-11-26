import unittest
from training.utils.config import load_config
import yaml
import tempfile
import os


class TestLoadConfig(unittest.TestCase):
    def setUp(self):
        # Crée un fichier temporaire avec des données YAML pour les tests
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
        self.sample_config = {
            "paths": {"data": "data/dataset.csv", "model": "models/trained_model.pkl"},
            "training_params": {"batch_size": 32, "epochs": 100},
        }

        # Écrit le contenu YAML dans le fichier temporaire
        with open(self.temp_file.name, "w") as file:
            yaml.dump(self.sample_config, file)

    def test_load_config(self):
        # Appelle la fonction à tester
        config = load_config(self.temp_file.name)

        # Vérifie que le fichier a été chargé correctement
        self.assertEqual(config, self.sample_config)

    def tearDown(self):
        # Supprime le fichier temporaire après le test
        os.remove(self.temp_file.name)


if __name__ == "__main__":
    unittest.main()
