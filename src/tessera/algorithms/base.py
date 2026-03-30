from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple
from dataclasses import dataclass


@dataclass
class KEMKeyPair:
    public_key: bytes
    secret_key: bytes


@dataclass  
class KEMEncapsulation:
    ciphertext: bytes
    shared_secret: bytes


class KEM(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def public_key_size(self) -> int:
        pass

    @property
    @abstractmethod
    def secret_key_size(self) -> int:
        pass

    @property
    @abstractmethod
    def ciphertext_size(self) -> int:
        pass

    @property
    @abstractmethod
    def shared_secret_size(self) -> int:
        pass

    @abstractmethod
    def keygen(self) -> Tuple[bytes, bytes]:
        pass

    @abstractmethod
    def encaps(self, pk: bytes) -> Tuple[bytes, bytes]:
        pass

    @abstractmethod
    def decaps(self, sk: bytes, ct: bytes) -> bytes:
        pass

    def keygen_keypair(self) -> KEMKeyPair:
        pk, sk = self.keygen()
        return KEMKeyPair(public_key=pk, secret_key=sk)

    def encaps_result(self, pk: bytes) -> KEMEncapsulation:
        ct, ss = self.encaps(pk)
        return KEMEncapsulation(ciphertext=ct, shared_secret=ss)


@dataclass
class SignatureKeyPair:
    public_key: bytes
    secret_key: bytes


class Signature(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def public_key_size(self) -> int:
        pass

    @property
    @abstractmethod
    def secret_key_size(self) -> int:
        pass

    @property
    @abstractmethod
    def signature_size(self) -> int:
        pass

    @abstractmethod
    def keygen(self) -> Tuple[bytes, bytes]:
        pass

    @abstractmethod
    def sign(self, sk: bytes, message: bytes) -> bytes:
        pass

    @abstractmethod
    def verify(self, pk: bytes, message: bytes, signature: bytes) -> bool:
        pass

    def keygen_keypair(self) -> SignatureKeyPair:
        pk, sk = self.keygen()
        return SignatureKeyPair(public_key=pk, secret_key=sk)
