from dataclasses import dataclass


@dataclass(frozen=True)
class DilithiumParams:
    name: str
    n: int
    q: int
    d: int
    tau: int
    challenge_entropy: int
    gamma1: int
    gamma2: int
    k: int
    l: int
    eta: int
    beta: int
    omega: int

    @property
    def public_key_bytes(self) -> int:
        return 32 + self.k * self.n * 10 // 8

    @property
    def secret_key_bytes(self) -> int:
        seedbytes = 32
        crhbytes = 64
        trbytes = 64
        eta_packed = (self.l + self.k) * (self.n * (4 if self.eta == 2 else 3) // 8)
        t0_packed = self.k * self.n * 13 // 8
        return seedbytes * 2 + trbytes + eta_packed + t0_packed

    @property
    def signature_bytes(self) -> int:
        z_packed = self.l * self.n * (17 + (self.gamma1 == (1 << 19))) // 8
        h_packed = self.omega + self.k
        return 32 + z_packed + h_packed

    @property
    def root_of_unity(self) -> int:
        return 1753


DILITHIUM2 = DilithiumParams(
    name="Dilithium2",
    n=256,
    q=8380417,
    d=13,
    tau=39,
    challenge_entropy=192,
    gamma1=2**17,
    gamma2=(8380417 - 1) // 88,
    k=4,
    l=4,
    eta=2,
    beta=78,
    omega=80,
)

DILITHIUM3 = DilithiumParams(
    name="Dilithium3",
    n=256,
    q=8380417,
    d=13,
    tau=49,
    challenge_entropy=225,
    gamma1=2**19,
    gamma2=(8380417 - 1) // 32,
    k=6,
    l=5,
    eta=4,
    beta=196,
    omega=55,
)

DILITHIUM5 = DilithiumParams(
    name="Dilithium5",
    n=256,
    q=8380417,
    d=13,
    tau=60,
    challenge_entropy=257,
    gamma1=2**19,
    gamma2=(8380417 - 1) // 32,
    k=8,
    l=7,
    eta=2,
    beta=120,
    omega=75,
)


def get_dilithium_params(variant: str) -> DilithiumParams:
    variants = {
        "2": DILITHIUM2,
        "3": DILITHIUM3,
        "5": DILITHIUM5,
        "dilithium2": DILITHIUM2,
        "dilithium3": DILITHIUM3,
        "dilithium5": DILITHIUM5,
    }
    key = variant.lower().replace("-", "").replace("_", "")
    if key not in variants:
        raise ValueError(f"Unknown Dilithium variant: {variant}")
    return variants[key]
