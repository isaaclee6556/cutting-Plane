# 진행 상황 (Progress)

마지막 업데이트: 2026-09-02

---

## 완료된 마일스톤

| 마일스톤 | 내용 | 파일 | 테스트 |
|---|---|---|---|
| M1 | bitops + StructureIndex + INSERT 알고리즘 | `bitops.py`, `structure_index.py` | 46 tests |
| M2 | 심볼릭 컷 클래스 7종 + to_inequality() | `cuts.py` | 42 tests |
| M3 | Cut pool (efficacy, violation filter, dominance prune) | `cut_pool.py` | 29 tests |
| M4 | Gurobi 연동 + plain FP 루프 + FPResult | `fp_loop.py`, `io.py` | 33 tests |
| M5 | Structural cuts 루프 통합 | `fp_loop.py` 확장 | 28 tests |
| M6 | Cycle-breaking perturbation + benchmark 스크립트 | `fp_loop.py`, `experiments/benchmark.py` | 25 tests |

**총 203 tests, all passing.**

---

## Benchmark 결과 (2026-09-02)

```
Instance               Plain   Cuts   Cuts+Perturb(max=10)
covering_*               1✓      1✓        1✓          쉬운 인스턴스
rnd_cover_n16_m8         2C      2C        19✓         perturbation으로 cycle 탈출
rnd_cover_n20_m10        2C      2C        12✓         perturbation으로 cycle 탈출
cycle_trap               2C      1L        1L          cut이 LP를 infeasible로 만들어 즉시 종료
```

- `✓` = feasible solution 발견
- `C` = cycle 감지로 종료
- `L` = LP infeasible로 종료 (컷이 원래 fractional LP를 infeasible로 만듦)

---

## Git 상태

- branch: `main`
- commits: 2개 (초기 구현 + CLAUDE.md 업데이트)
- remote: 아직 없음 (GitHub push 미완료)
  - 사용자 GitHub: `isaaclee6556`
  - 목표 레포: `github.com/isaaclee6556/research_cutting_plane`
  - push 방법: `git remote add origin https://github.com/isaaclee6556/research_cutting_plane.git && git push -u origin main`

---

## 다음으로 할 수 있는 작업

### 우선순위 높음
- [ ] GitHub push (remote 연결 + `git push -u origin main`)
- [ ] MIPLIB 실제 인스턴스로 benchmark
  - `.mps` 파일 다운로드 후 `load_mps("path.mps")` 사용
  - 추천 소규모 인스턴스: `p0033`, `p0040`, `air03`

### 알고리즘 개선
- [ ] **Cut aging**: 누적 cut이 많아질 때 오래된 cut 제거 (LP 속도 유지)
- [ ] **Adaptive perturbation**: 같은 z가 반복될수록 perturb_scale 자동 증가
- [ ] **일반 정수 변수 지원**: 현재 binary만 StructureIndex 지원, 일반 정수 변수는 rounding만
- [ ] **Warm-start 활용**: Gurobi MIP solver와 feasibility pump 결합

### 실험 및 논문 재현
- [ ] Cornuéjols & Lee (2018) 논문 실험 테이블 재현
  - 논문 원본: 프로젝트 루트 `s10107-017-1226-4.pdf` (gitignored)
- [ ] Plain FP vs FP+cuts iteration 수 통계적 비교 (더 많은 인스턴스)
- [ ] Cycling이 자주 발생하는 인스턴스 특성 분석

### 코드 정리
- [ ] `experiments/results/` 디렉터리 생성 및 CSV 자동 저장
- [ ] `make_random_covering_milp`보다 realistic한 인스턴스 생성기 추가

---

## 주요 구현 메모

### StructureIndex.cubes 구조
```python
cubes: dict[tuple[Point, frozenset], frozenset]
# key = (min_corner, frozenset({dir_a, dir_b, dir_c}))
# value = frozenset of all 8 corners
# 주의: for dirs in idx.cubes.values() — values는 frozenset (tuple 아님)
```

### cuts_from_index의 cubes 순회
```python
for (min_corner, dirs), _ in idx.cubes.items():
    result.append(CubeCut(z=min_corner, free=dirs, n=n))
```

### _FeasChecker bound 리셋
```python
# is_feasible() 호출 후 반드시 bound 리셋 + lp.update() 필요
# 안 하면 다음 호출에서 이전 bound가 유지됨
```
