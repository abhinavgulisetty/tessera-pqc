from dataclasses import dataclass


@dataclass(frozen=True)
class KyberParams:
    name: str
    n: int
    q: int
    k: int
    eta1: int
    eta2: int
    du: int
    dv: int

    @property
    def public_key_bytes(self) -> int:
        return 32 + self.k * self.n * 12 // 8

    @property
    def secret_key_bytes(self) -> int:
        return self.k * self.n * 12 // 8 + self.public_key_bytes + 32 + 32

    @property
    def ciphertext_bytes(self) -> int:
        return self.k * self.n * self.du // 8 + self.n * self.dv // 8

    @property
    def shared_secret_bytes(self) -> int:
        return 32


KYBER_512 = KyberParams(
    name="Kyber-512",
    n=256,
    q=3329,
    k=2,
    eta1=3,
    eta2=2,
    du=10,
    dv=4,
)

KYBER_768 = KyberParams(
    name="Kyber-768",
    n=256,
    q=3329,
    k=3,
    eta1=2,
    eta2=2,
    du=10,
    dv=4,
)

KYBER_1024 = KyberParams(
    name="Kyber-1024",
    n=256,
    q=3329,
    k=4,
    eta1=2,
    eta2=2,
    du=11,
    dv=5,
)


def get_kyber_params(variant: str) -> KyberParams:
    variants = {
        "512": KYBER_512,
        "768": KYBER_768,
        "1024": KYBER_1024,
        "kyber512": KYBER_512,
        "kyber768": KYBER_768,
        "kyber1024": KYBER_1024,
    }
    key = variant.lower().replace("-", "").replace("_", "")
    if key not in variants:
        raise ValueError(f"Unknown Kyber variant: {variant}")
    return variants[key]
