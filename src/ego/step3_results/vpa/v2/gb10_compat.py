"""GB10(sm_121) 호환 패치 — nvrtc JIT 리덕션 우회.

증상: `nvrtc: error: invalid value for --gpu-architecture (-arch)`.
원인: 이 장비의 GPU는 compute capability **12.1**인데 설치된 torch가 담고 있는 아키텍처는
`sm_80/90/100/120` 뿐이다(`torch.cuda.get_arch_list()`). 사전 컴파일된 커널(matmul 등)은
정상 동작하지만, **런타임 nvrtc JIT로 생성되는 리덕션 커널**은 `compute_121`을 인자로 받아
CUDA 12.8 nvrtc가 거부한다.

Qwen3-VL 경로에서 이 JIT를 밟는 것은 `prod` 계열뿐이다(이미지 grid_thw 같은 아주 작은
정수 텐서). 따라서 **prod만 CPU로 우회**하면 되고, 연산량이 미미해 속도 영향은 없다.
정확도에는 영향이 없다(동일 정수 연산).

torch가 sm_121 빌드로 갱신되면 이 모듈은 불필요해진다 — `torch.cuda.get_arch_list()`에
`sm_121`이 보이면 패치를 건너뛴다.
"""
from __future__ import annotations


def apply() -> bool:
    """필요할 때만 패치. 적용했으면 True."""
    import torch

    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(0)
    if f"sm_{major}{minor}" in torch.cuda.get_arch_list():
        return False  # 네이티브 지원 — 패치 불필요

    _prod = torch.Tensor.prod
    _tprod = torch.prod

    def prod_cpu(self, *a, **kw):
        if self.is_cuda:
            return _prod(self.cpu(), *a, **kw).to(self.device)
        return _prod(self, *a, **kw)

    def tprod_cpu(input, *a, **kw):  # noqa: A002 - torch API 시그니처 유지
        if hasattr(input, "is_cuda") and input.is_cuda:
            return _tprod(input.cpu(), *a, **kw).to(input.device)
        return _tprod(input, *a, **kw)

    torch.Tensor.prod = prod_cpu
    torch.prod = tprod_cpu
    return True
