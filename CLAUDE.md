# Structural Feasibility Pump — Claude Code Context

Source: Cornuéjols & Lee (2018), `conv({0,1}^n \ X)` structural cuts inside a feasibility pump.
진행 상황·다음 작업은 `PROGRESS.md` 참조.

---

## 환경

```bash
cd ~/Desktop/research_cutting_plane
source .venv/bin/activate    # Python 3.9.7 / Gurobi 12.0.3 (academic license)
python -m pytest             # 전체 테스트 (203 tests, all pass)
```

---

## 파일 구조

```
structural_fp/
  bitops.py          Point=int bitmask, flip/popcount/bits
  structure_index.py StructureIndex + INSERT (edge→square→star→tulip/cube→propeller)
  cuts.py            심볼릭 컷 7종 (Vertex/Edge/Square/Star/Cube/Tulip/Propeller)
  cut_pool.py        efficacy 랭킹 + dominance pruning + select_cuts()
  io.py              MILP 인스턴스 팩토리 (make_covering_milp 등)
  fp_loop.py         feasibility_pump() 메인 루프 (Gurobi 연동)
tests/               pytest 파일들
experiments/
  benchmark.py       plain FP vs FP+cuts vs FP+cuts+perturb 비교표
```

---

## 핵심 API

```python
from structural_fp import *

# 인스턴스 생성
model = make_covering_milp(n=8)
model = make_random_covering_milp(n=20, m=10, seed=3)
model = load_mps("path/to/file.mps")

# FP 실행 (세 가지 모드)
res = feasibility_pump(model)                                        # plain
res = feasibility_pump(model, use_structural_cuts=True)              # + cuts
res = feasibility_pump(model, use_structural_cuts=True,              # + cuts
                       max_perturbations=10, random_seed=42,         # + perturb
                       verbose=True)

# FPResult 필드
res.feasible       # bool
res.solution       # dict {var_name: int} or None
res.iterations     # int
res.reason         # "feasible" | "cycle" | "lp_infeasible" | "max_iter"
res.cuts_added     # int
res.perturbations  # int
```

---

## 주요 설계 결정 (코드 수정 시 반드시 숙지)

- **비트마스크 인덱싱**: `sorted(int_vars)[i]` ↔ bit i. x_hat 변환: `{i: x_hat[sorted_ivars[i]] for i in range(n)}`
- **Checker 분리**: projection LP(`relax`)와 feasibility checker(`_FeasChecker`)는 별도 모델. checker는 원본 제약만 포함(cuts 없음).
- **CubeCut n=3**: free={0,1,2}이면 coeffs 비어 `0 >= 1` → trivially tight, 수학적으로 올바름.
- **dominance_prune**: frozen dataclass 동등성 복잡 → `dominated: set[int]` (id() 기반).
- **cubes dict**: `key=(min_corner, frozenset_3_dirs)`, `value=frozenset_8_corners`.
- **Perturbation**: cycle 시 `visited.clear()` + uniform noise 주입 후 `continue` (정상 projection 건너뜀).
