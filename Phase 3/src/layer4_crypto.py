"""
Layer 4: Cryptographic Protection (FedZT-Edge Cross-Cutting Layer)
Implements AES-256 (EAX) payload encryption, RSA-2048 digital signatures,
and SHA-256 integrity checks on model updates, alerts, and policies.
Measures cryptographic latency and byte overhead for Ablation (e).
"""

import time
import hashlib
import json
import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

class CryptoEnvelope:
    def __init__(self, key_size=256, enabled=True):
        self.enabled = enabled
        self.aes_key = AESGCM.generate_key(bit_length=key_size)
        self.aesgcm = AESGCM(self.aes_key)
        
        # RSA-2048 keypair
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()

    def pack_and_encrypt(self, payload_dict):
        """
        Serializes, hashes (SHA-256), signs (RSA-2048), and encrypts (AES-256-GCM/EAX).
        """
        start_time = time.perf_counter()
        raw_bytes = json.dumps(payload_dict, default=str).encode('utf-8')
        raw_size = len(raw_bytes)
        
        if not self.enabled:
            return {
                "ciphertext": raw_bytes,
                "nonce": b"",
                "signature": b"",
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "overhead_bytes": 0,
                "latency_ms": (time.perf_counter() - start_time) * 1000.0,
                "crypto_enabled": False
            }
            
        # 1. SHA-256 integrity hash
        digest = hashlib.sha256(raw_bytes).digest()
        sha256_hex = digest.hex()
        
        # 2. RSA-2048 signature
        signature = self.private_key.sign(
            digest,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        # 3. AES-256 encryption
        nonce = os.urandom(12)
        ciphertext = self.aesgcm.encrypt(nonce, raw_bytes, None)
        
        total_size = len(ciphertext) + len(nonce) + len(signature)
        overhead_bytes = total_size - raw_size
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        
        return {
            "ciphertext": ciphertext,
            "nonce": nonce,
            "signature": signature,
            "sha256": sha256_hex,
            "overhead_bytes": overhead_bytes,
            "latency_ms": elapsed_ms,
            "crypto_enabled": True
        }

    def decrypt_and_verify(self, envelope):
        """
        Decrypts, checks SHA-256 hash, and verifies RSA signature.
        """
        start_time = time.perf_counter()
        if not envelope.get("crypto_enabled", True):
            raw_bytes = envelope["ciphertext"]
            payload = json.loads(raw_bytes.decode('utf-8'))
            return payload, True, (time.perf_counter() - start_time) * 1000.0
            
        # 1. AES Decrypt
        decrypted_bytes = self.aesgcm.decrypt(envelope["nonce"], envelope["ciphertext"], None)
        
        # 2. Verify SHA-256
        digest = hashlib.sha256(decrypted_bytes).digest()
        if digest.hex() != envelope["sha256"]:
            return None, False, (time.perf_counter() - start_time) * 1000.0
            
        # 3. Verify RSA Signature
        try:
            self.public_key.verify(
                envelope["signature"],
                digest,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            valid_sig = True
        except Exception:
            valid_sig = False
            
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        payload = json.loads(decrypted_bytes.decode('utf-8'))
        return payload, valid_sig, elapsed_ms
