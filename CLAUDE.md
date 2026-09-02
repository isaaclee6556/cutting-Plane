# Structural Feasibility Pump — Project Context

Source: Cornuéjols & Lee (2018) structural cuts for `conv({0,1}^n \ X)`, applied
incrementally inside a feasibility-pump-style heuristic.

---

## 현재 상태 (2026-09-02 기준)

**Milestone 1–6 전부 완료. 203 tests, all passing.**

```
python -m pytest          # 203 passed
python experiments/benchmark.py   # 성능 비교표 출력
```

---

## 프로젝트 구조

```
research_cutting_plane/
├── CLAUDE.md                     ← 이 파일 (Claude Code 자동 로드)
├── structural_fp.md              ← 원본 알고리즘 스펙 (참고용)
├── pyproject.toml
├── structural_fp/                ← Python 패키지
│   ├── __init__.py               ← 모든 public API export
│   ├── bitops.py                 ← M1: 비트 연산 기본 도구
│   ├── structure_index.py        ← M1: StructureIndex + INSERT 알고리즘
│   ├── cuts.py                   ← M2: 심볼릭 컷 클래스 7종
│   ├── cut_pool.py               ← M3: efficacy 랭킹 + 지배 pruning
│   ├── io.py                     ← M4: MILP 인스턴스 팩토리 함수들
│   └── fp_loop.py                ← M4~6: 메인 FP 루프 (Gurobi 연동)
├── tests/
│   ├── test_bitops.py            ← 16 tests
│   ├── test_structure_index.py   ← 30 tests
│   ├── test_cuts.py              ← 42 tests
│   ├── test_cut_pool.py          ← 29 tests
│   ├── test_fp_loop.py           ← 33 tests (plain FP)
│   ├── test_fp_loop_cuts.py      ← 28 tests (FP + structural cuts)
│   └── test_perturbation.py      ← 25 tests (cycle-breaking)
└── experiments/
    ├── benchmark.py              ← plain vs cuts vs cuts+perturb 비교
    └── results/                  ← CSV 출력 디렉터리 (gitignored)
```

---

## 환경 세팅

```bash
cd ~/Desktop/research_cutting_plane
source .venv/bin/activate    # Python 3.9.7, Gurobi 12.0.3 (academic license)
```

패키지는 editable install 되어 있음 (`pip install -e .`).

---

## 핵심 API

```python
from structural_fp import *

# --- MILP 인스턴스 생성 ---
model = make_covering_milp(n=8)
model = make_random_covering_milp(n=20, m=10, seed=3)
model = load_mps("path/to/file.mps")

# --- Feasibility Pump 실행 ---
# Plain FP (M4)
res = feasibility_pump(model)

# FP + structural cuts (M5)
res = feasibility_pump(model, use_structural_cuts=True)

# FP + cuts + cycle-breaking perturbation (M6)
res = feasibility_pump(
    model,
    use_structural_cuts=True,
    max_perturbations=10,
    perturb_scale=0.3,
    random_seed=42,
    verbose=True,
)

# --- 결과 확인 ---
res.feasible       # bool
res.solution       # dict {var_name: int_value} or None
res.iterations     # int
res.reason         # "feasible" | "cycle" | "lp_infeasible" | "max_iter"
res.cuts_added     # int (structural cuts added to LP)
res.perturbations  # int (cycle-breaking perturbations applied)

# --- StructureIndex 직접 사용 ---
idx = StructureIndex(n=3)
idx.insert(0b011)
cuts = cuts_from_index(idx)     # list[SymbolicCut]
selected = select_cuts(cuts, x_hat={0: 0.3, 1: 0.7, 2: 0.5}, k=30)
```

---

## 주요 설계 결정 (구현 중 확정)

### 비트마스크 인덱싱
- 정수 변수 이름을 sorted() 기준으로 정렬한 순서가 비트 위치와 대응
- `sorted_ivars[i]` ↔ bit i in bitmask
- `fp_loop.py`의 x_hat_idx 변환: `{i: x_hat[sorted_ivars[i]] for i in range(n)}`

### Feasibility Checker 분리
- projection LP (`relax`)와 feasibility checker (`_FeasChecker`) 를 별도 모델로 유지
- checker는 원래 제약만 포함 (structural cuts 없음) → 원본 MILP 실현 가능성만 판단
- projection LP에는 cuts가 누적되어 점점 타이트해짐

### CubeCut 엣지 케이스
- n=3이고 free={0,1,2}이면 coeffs가 비어서 trivially tight (`0 >= 1`): 수학적으로 올바름
- 테스트에서 명시적으로 검증됨

### dominance_prune에서 id() 사용
- SymbolicCut은 frozen dataclass → 동등성 비교가 복잡
- `dominated: set[int]` (id 기반)로 지배 여부 추적

### Perturbation 설계
- cycle 감지 시: `visited.clear()` + 목적함수에 uniform noise 주입
- 노이즈 크기: base_coeff(±1) + Uniform(-scale, scale)
- `continue`로 정상 projection 건너뛰고 바로 다음 iter

### cubes dict 구조
- `cubes: dict[tuple[Point, frozenset], frozenset]`
- key = `(min_corner, frozenset_of_3_dirs)`
- value = frozenset of all 8 corners

---

## 마일스톤별 완성 내용

### M1: bitops + StructureIndex (46 tests)
- `Point = int` bitmask 표현
- `flip`, `flip2`, `bit`, `popcount`, `bits` 연산
- INSERT 알고리즘 5단계: edge → square → star → tulip/cube → propeller
- `Square(base, dirs)` frozen dataclass

### M2: cuts (42 tests)
- 7개 컷 클래스: `VertexCut`, `EdgeCut`, `SquareCut`, `StarCut`, `CubeCut`, `TulipCut`, `PropellerCut`
- `_delta_to_x()`: delta-space → x-space 변환
- `cuts_from_index(idx)`: StructureIndex → list[SymbolicCut]

### M3: cut_pool (29 tests)
- `efficacy(cut, x_hat)`: (rhs - a^T x̂) / ‖a‖₂
- `filter_by_violation()`, `top_k_by_efficacy()`
- `dominance_prune()`: Cube > Square > Edge > Vertex 계층
- `select_cuts()`: 3단계 파이프라인

### M4: fp_loop — Plain FP (33 tests)
- `FPResult` dataclass
- `_FeasChecker`: bound 고정/해제로 원본 MILP feasibility 판단
- `feasibility_pump()`: relax → round → check → cycle → projection

### M5: fp_loop — structural cuts 통합 (28 tests)
- `use_structural_cuts=True` 플래그
- `_to_bitmask()`, `_add_cut_to_relax()`
- cuts는 projection LP에만 누적 (checker는 원본 유지)

### M6: perturbation + benchmark (25 tests)
- `_set_perturbed_objective()`: 노이즈 추가 projection
- `max_perturbations`, `perturb_scale`, `random_seed` 파라미터
- `experiments/benchmark.py`: plain / cuts / cuts+perturb 3-way 비교표

---

## Benchmark 결과 요약

```
Instance               Plain   Cuts   Cuts+Perturb
rnd_cover_n16_m8         2C      2C        19✓       ← perturbation으로 cycle 탈출
rnd_cover_n20_m10        2C      2C        12✓
cycle_trap               2C      1L         1L        ← cut이 LP를 infeasible로
covering_*               1✓      1✓         1✓        ← 쉬운 인스턴스는 1iter
```

---

## 다음으로 할 수 있는 작업

### 실험 확장
- MIPLIB 실제 인스턴스로 benchmark (`.mps` 파일 다운로드 후 `load_mps()` 사용)
- 더 어려운 랜덤 인스턴스 생성 (constraint density, n 크기 조정)
- cuts+perturbation이 실제로 iteration을 줄이는 케이스 분석

### 알고리즘 개선
- `max_perturbations` 자동 조정 (adaptive perturbation)
- 누적 cut 수가 많아질 때 오래된 cut 제거 (cut aging)
- Gurobi MIP solver와 warm-start 결합
- 일반 정수 변수(non-binary) 지원 확장

### 논문 재현
- Cornuéjols & Lee (2018) 논문의 실험 테이블 재현
- 논문 원본: `s10107-017-1226-4.pdf` (프로젝트 루트에 있음, gitignored)

---

## 원본 알고리즘 스펙

상세 알고리즘 명세는 `structural_fp.md` 참조.
