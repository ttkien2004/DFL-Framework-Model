import os
import pickle
import torch
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

class CryptoUtils:
    @staticmethod
    def generate_rsa_keypair():
        """Tạo cặp khóa RSA 2048-bit (Dùng cho setup ủy ban)"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        public_key = private_key.public_key()
        return private_key, public_key

    @staticmethod
    def hybrid_encrypt(data_obj, public_key):
        """
        Mã hóa lai:
        1. Tạo Session Key (AES) ngẫu nhiên.
        2. Mã hóa Data bằng AES-GCM.
        3. Mã hóa Session Key bằng RSA Public Key.
        """
        # 1. Serialize dữ liệu (Tensor -> Bytes)
        # Chuyển về CPU để tránh lỗi pickle device
        if isinstance(data_obj, torch.Tensor):
            data_bytes = pickle.dumps(data_obj.cpu())
        else:
            data_bytes = pickle.dumps(data_obj)

        # 2. Tạo khóa AES (Session Key) 32 bytes = 256 bit
        session_key = os.urandom(32)
        iv = os.urandom(12) # Initialization Vector cho GCM

        # 3. Mã hóa dữ liệu bằng AES-GCM
        cipher = Cipher(algorithms.AES(session_key), modes.GCM(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data_bytes) + encryptor.finalize()
        auth_tag = encryptor.tag

        # 4. Mã hóa Session Key bằng RSA Public Key
        encrypted_session_key = public_key.encrypt(
            session_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        return {
            "enc_session_key": encrypted_session_key,
            "iv": iv,
            "ciphertext": ciphertext,
            "auth_tag": auth_tag
        }

    @staticmethod
    def hybrid_decrypt(encrypted_pkg, private_key):
        """
        Giải mã lai:
        1. Giải mã Session Key bằng RSA Private Key.
        2. Giải mã Data bằng AES-GCM.
        """
        try:
            # 1. Trích xuất các thành phần
            enc_session_key = encrypted_pkg['enc_session_key']
            iv = encrypted_pkg['iv']
            ciphertext = encrypted_pkg['ciphertext']
            auth_tag = encrypted_pkg['auth_tag']

            # 2. Giải mã Session Key
            session_key = private_key.decrypt(
                enc_session_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )

            # 3. Giải mã dữ liệu bằng AES-GCM
            cipher = Cipher(algorithms.AES(session_key), modes.GCM(iv, auth_tag), backend=default_backend())
            decryptor = cipher.decryptor()
            decrypted_bytes = decryptor.update(ciphertext) + decryptor.finalize()

            # 4. Deserialize
            return pickle.loads(decrypted_bytes)

        except Exception as e:
            print(f"[Crypto] Decryption failed: {e}")
            return None