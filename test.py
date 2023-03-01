import unittest
import finetune.load_model as load_model
# ----------------------------------------
import torchvision.models as models


class finetune_TestCase(unittest.TestCase):
    """测试name_function.py"""

    def test_net_arch(self):
        """能够正确的处理像janis joplin这样的姓名吗"""
        model = load_model.load_model(models.alexnet(), num_classes=10)
        self.assertEqual(model.classifier[6].out_features, 10)

if __name__ == '__main':
    unittest.main()
