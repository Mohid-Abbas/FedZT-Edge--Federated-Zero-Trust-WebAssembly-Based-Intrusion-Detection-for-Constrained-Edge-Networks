import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.layer4_crypto import CryptoEnvelope

class TestCryptoLayer(unittest.TestCase):
    def test_encryption_and_decryption(self):
        crypto = CryptoEnvelope(enabled=True)
        payload = {"round": 5, "weights": [0.12, -0.45, 0.88, 1.05]}
        
        env = crypto.pack_and_encrypt(payload)
        self.assertTrue(env["crypto_enabled"])
        self.assertGreater(len(env["ciphertext"]), 0)
        self.assertGreater(len(env["signature"]), 0)
        
        decrypted_payload, valid_sig, latency = crypto.decrypt_and_verify(env)
        self.assertTrue(valid_sig)
        self.assertEqual(decrypted_payload["round"], 5)
        self.assertEqual(decrypted_payload["weights"], payload["weights"])

    def test_tamper_detection(self):
        crypto = CryptoEnvelope(enabled=True)
        payload = {"action": "ALLOW", "user_id": 42}
        env = crypto.pack_and_encrypt(payload)
        
        # Tamper with the sha256 or ciphertext
        env["sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"
        decrypted, valid, _ = crypto.decrypt_and_verify(env)
        self.assertFalse(valid)

    def test_disabled_crypto_ablation(self):
        crypto = CryptoEnvelope(enabled=False)
        payload = {"data": [1, 2, 3]}
        env = crypto.pack_and_encrypt(payload)
        self.assertFalse(env["crypto_enabled"])
        self.assertEqual(env["overhead_bytes"], 0)
        
        decrypted, valid, _ = crypto.decrypt_and_verify(env)
        self.assertTrue(valid)
        self.assertEqual(decrypted["data"], [1, 2, 3])

if __name__ == "__main__":
    unittest.main()
