이 저장소는 [*SALSA VERDE: a machine learning attack on Learning With Errors with sparse small secrets*](https://arxiv.org/abs/2306.11641)의 결과를 재현하기 위한 코드를 담고 있습니다. 이 공격은 Transformer를 이용해 LWE 샘플 `(\mathbf{a}, b)`로부터 secret을 복구합니다. 또한 이 저장소의 코드는 [*SALSA PICANTE: a Machine Learning Attack on LWE with Binary Secrets*](https://arxiv.org/abs/2303.04178)의 공격을 실행하는 데에도 사용할 수 있습니다. 성능 면에서는 VERDE 공격이 PICANTE 공격을 엄밀하게 상회합니다.

## 빠른 시작

__설치__: 먼저 GPU가 최소 1개 있는 머신에 저장소를 클론하세요. 이후 `conda create --name lattice_env --file requirements.txt`로 필요한 conda 환경을 만들고, `conda activate lattice_env`로 활성화하면 됩니다.

__데이터 다운로드__: 바로 실험을 시작할 수 있도록, 전처리가 끝난 데이터셋을 함께 제공합니다. 이 데이터셋은 `n=256`, `log_2 q=20`인 sparse binary secret 실험에 사용할 수 있습니다. 데이터는 [이 링크](https://dl.fbaipublicfiles.com/verde/n256_logq20_binary_for_release.tar.gz)에서 받을 수 있습니다. 데이터 폴더에는 아래 파일들이 들어 있습니다.
- `orig_A.npy`와 `orig_b.npy`: 원본 4n개의 LWE 샘플
- `params.pkl`: 데이터셋 파라미터 파일입니다. Transformer가 이 파일을 읽으므로, 이 실험에서는 LWE 파라미터(`N`, `Q`, `sigma`)를 별도로 지정할 필요가 없습니다.
- `secret.npy`: 여러 개의 secret이 저장된 파일
- `test.prefix`와 `train.prefix`: 전처리된 테스트/학습 세트. 각 줄은 `{a} ; {b}` 형태의 문자열입니다.

__첫 실험__: 준비가 끝났다면 다음 명령을 실행하세요.
```bash
python3 train.py --reload_data /path/to/data --secret_seed 3 --hamming 30 --input_int_base 105348 --share_token 64 --optimizer adam_warmup,lr=0.00001,warmup_updates=1000,warmup_init_lr=0.00000001
```
이 명령은 전처리된 데이터셋(`n=256`, `log_2 q=20`, `h=30`) 위에서 모델을 학습합니다. 이 설정에서 사용하는 입력 인코딩 base와 share token은 VERDE Appendix A.1의 Table 9에 정리되어 있고, 모델 구조는 논문 Section 2에 설명되어 있습니다. 이 모델은 NVIDIA Quadro GV100 32GB 한 장에서 무난하게 동작합니다. 보통 epoch당 약 2시간 정도 걸리며, secret이 복구된다면 대체로 초반 epoch에서 성공합니다. 첫 시도에서 실패하더라도 다른 `secret_seed`(범위 `0-9`)나 다른 Hamming weight(범위 `3-40`)로 다시 시도해 볼 수 있습니다. 모든 공격이 첫 실행에서 성공하는 것은 아닙니다.

__조절 가능한 파라미터__:  
파라미터는 자유롭게 바꿔볼 수 있지만, 기본 학습 파라미터는 `train.py`와 함께 제공되는 `params.pkl` 안에 기본값으로 정의되어 있습니다. 현재 코드베이스는 seq2seq 모델만 지원하며, 논문 Section 7에서 실험한 encoder-only 모델은 지원하지 않습니다.
- 모델 구조 파라미터 (`src/train.py`에 정의)
  - `enc_emb_dim`: encoder embedding 차원
  - `dec_emb_dim`: decoder embedding 차원
  - `n_enc_layers`: encoder 레이어 수
  - `n_dec_layers`: decoder 레이어 수
  - `n_enc_heads`: encoder attention head 수
  - `n_dec_heads`: decoder attention head 수
  - `enc_loops`: encoder 반복 횟수 (Universal Transformer 파라미터)
  - `dec_loops`: decoder 반복 횟수 (Universal Transformer 파라미터)
- 학습 파라미터
  - `epoch_size`: epoch당 사용할 LWE 샘플 수
  - `batch_size`: batch당 사용할 LWE 샘플 수
- LWE 문제 파라미터
  - `N`: lattice 차원
  - `Q`: LWE 문제의 소수 modulus
  - `sigma`: LWE에 사용하는 오류 분포의 표준편차
  - `secret_type`: secret 각 좌표를 어떤 분포에서 뽑을지 지정합니다. 코드상으로는 `binary`, `ternary`, `gaussian`, `binomial`을 지원합니다. 다만 전체 secret recovery는 현재 `binary`와 `ternary`에 대해서만 구현되어 있습니다.
  - `hamming`: LWE secret의 0이 아닌 좌표 개수
  - `input_int_base`: Transformer 입력 정수 인코딩 base
  - `output_int_base`: Transformer 출력 정수 인코딩 base

__slurm으로 sweep 실행__: 저희는 클러스터에서 JSON 파일을 slurm이 읽도록 해서 여러 실험을 분산 실행합니다. JSON 안의 리스트에 원소를 더 넣고, 예를 들어 `hamming: [30, 35]`처럼 설정한 뒤 적절한 파서를 붙이면 로컬 환경에서도 sweep을 구성할 수 있습니다.

__결과 분석__: 여러 실험 결과를 한꺼번에 분석하고 싶다면 `./notebooks/LatticeMLReader.ipynb`를 사용할 수 있습니다. 이 노트북은 지정한 실험의 로그 파일을 파싱하고, 부가적인 분석 정보도 함께 제공합니다.

__직접 데이터 생성__: 직접 reduced data를 만들어 다른 공격을 돌리고 싶다면 아래 순서대로 진행하면 됩니다.
- 먼저 `[0, q)` 범위의 정수를 원소로 가지는 원본 LWE 샘플 행렬을 생성합니다. 아래 명령은 기본적으로 `4N x N` shape의 `orig_A.npy`를 씁니다.
```bash
python generate.py --step origA --lwe true --N 256 --Q 842779 --dump_path <output folder>
```
- 다음으로 lattice reduction 전처리를 실행합니다.
```bash
python generate.py --timeout 432000 --N 256 --Q 842779 --lll_delta 0.99 --float_type dd --bkz_block_size 35 --threshold 0.435 --threshold2 0.5 --use_polish true --step RA_tiny2 --reload_data <path of orig_A.npy>
```
공격하려는 설정에 맞게 `N`, `Q` 등을 바꿔도 됩니다.
- 이 단계가 끝나면 reduced matrix를 담은 디렉터리가 생성되고, 그 안에 `params.pkl`과 `data.prefix`가 생깁니다.
- 마지막으로 reduced `(A, b)` 데이터셋을 만듭니다.
```bash
python generate.py --min_hamming 16 --max_hamming 25 --num_workers 1 --num_secret_seeds 5 --step Ab --secret_type binary --epoch_size 1000000 --reload_size 1000000 --reload_data <path of directory for the reduced matrices>
```
이 명령은 Hamming weight가 `16`부터 `25`까지인 binary secret을 각 weight마다 5개씩, 총 50개 만들어 데이터셋을 생성합니다.

이제 공격을 돌릴 수 있는 reduced matrix 세트가 준비됐습니다. 위에서 제공한 학습 명령은 `reload_data` 경로만 직접 만든 데이터셋으로 바꾸면 그대로 사용할 수 있습니다.

__성공한 end-to-end 예시 (N=5, Q=17, h=2, 10k preprocessing)__: 아래 명령들은 `lattice_env`에서 실제로 성공한 작은 sanity-check 파이프라인입니다. 논문 스케일의 `n=256` 설정이 아니라, 전체 LWE 공격 흐름을 빠르게 재현해보는 작은 예제입니다.

1. 환경 활성화:
```bash
source /home/yu_mcc/miniconda3/etc/profile.d/conda.sh
conda activate lattice_env
```

2. 원본 LWE 샘플 생성. 정확히 `4N x N` 크기의 `orig_A.npy`를 씁니다.
```bash
python generate.py \
  --cpu true \
  --step origA \
  --lwe true \
  --N 5 \
  --Q 17 \
  --num_orig_samples 20 \
  --dump_path /home/yu_mcc/PQC_salsa/runs \
  --exp_name n5_q17_h2_10k \
  --exp_id origA_4n
```
기대 결과:
- `runs/n5_q17_h2_10k/origA_4n/orig_A.npy`가 생성됨
- shape은 `(20, 5)`
- 원소는 `[0, 16]` 범위에 있음
- 현재 `origA` 구현은 `ctypes`로 C 표준 라이브러리의 `rand()`를 호출하지 않습니다. `src/generate/lwe.py`에서 NumPy `RandomState`를 사용해 `[0, q)`의 iid uniform 정수를 생성합니다.

3. 저장한 `orig_A.npy`로 reduction 단계를 실행합니다. 이 작은 설정에서는 행 재정렬, `lll_penalty=10`, adaptive BKZ 파라미터, `use_polish=false` 조합이 안정적으로 동작했습니다. `N=5`여도 downstream `Ab` 단계가 `epoch_size=10000`에서 `test.prefix`와 `train.prefix`를 모두 만들 수 있도록 `m=10`을 유지합니다.
```bash
python generate.py \
  --cpu true \
  --step RA_tiny2 \
  --reload_data /home/yu_mcc/PQC_salsa/runs/n5_q17_h2_10k/origA_4n/orig_A.npy \
  --N 5 \
  --Q 17 \
  --m 10 \
  --num_workers 1 \
  --epoch_size 10000 \
  --timeout 30 \
  --float_type double \
  --lll_penalty 10 \
  --algo BKZ2.0 \
  --lll_delta 0.96 \
  --bkz_block_size 5 \
  --algo2 BKZ2.0 \
  --lll_delta2 0.99 \
  --bkz_block_size2 8 \
  --threshold 0.95 \
  --threshold2 0.90 \
  --use_polish false \
  --dump_path /home/yu_mcc/PQC_salsa/runs \
  --exp_name n5_q17_h2_10k \
  --exp_id ra_main
```
기대 결과:
- `runs/n5_q17_h2_10k/ra_main/data.prefix`가 생성됨
- `wc -l runs/n5_q17_h2_10k/ra_main/data.prefix` 결과가 `100000`

4. Hamming weight 2인 sparse binary secret에 대한 reduced `(A, b)` 샘플 생성:
```bash
python generate.py \
  --cpu true \
  --step Ab \
  --reload_data /home/yu_mcc/PQC_salsa/runs/n5_q17_h2_10k/ra_main \
  --min_hamming 2 \
  --max_hamming 2 \
  --num_secret_seeds 1 \
  --secret_type binary \
  --sigma 3 \
  --epoch_size 10000 \
  --reload_size 100000 \
  --num_workers 1 \
  --dump_path /home/yu_mcc/PQC_salsa/runs \
  --exp_name n5_q17_h2_10k \
  --exp_id ab_main
```
기대 결과:
- `runs/n5_q17_h2_10k/ab_main/train.prefix`와 `test.prefix`가 생성됨
- `runs/n5_q17_h2_10k/ab_main/secret.npy`가 생성됨
- secret shape은 `(5, 1)`이고 Hamming weight는 `2`
- 로컬 실행 기준으로 `test.prefix`는 `10000`줄, `train.prefix`는 `134790`줄이었습니다.

5. GPU에서 secret recovery 실행. 아래 명령은 로컬에서 실제로 성공한 모델 설정을 그대로 사용합니다. evaluator가 secret을 복구했다고 판단하면 조기 종료하도록 설계된 학습입니다.
```bash
python train.py \
  --cpu false \
  --reload_data /home/yu_mcc/PQC_salsa/runs/n5_q17_h2_10k/ab_main \
  --secret_seed 0 \
  --hamming 2 \
  --input_int_base 3 \
  --share_token 1 \
  --dump_path /home/yu_mcc/PQC_salsa/runs \
  --exp_name n5_q17_h2_10k \
  --exp_id train_gpu \
  --batch_size 256 \
  --epoch_size 12000 \
  --max_epoch 5 \
  --eval_size 2048 \
  --distinguisher_size 256 \
  --enc_emb_dim 256 \
  --dec_emb_dim 256 \
  --n_enc_layers 1 \
  --n_dec_layers 2 \
  --n_enc_heads 4 \
  --n_dec_heads 4 \
  --n_cross_heads 4 \
  --enc_loops 1 \
  --dec_loops 2 \
  --dropout 0 \
  --attention_dropout 0 \
  --optimizer adam_warmup,lr=0.0001,warmup_updates=200,warmup_init_lr=0.000001 \
  --num_workers 0 \
  --beam_size 1 \
  --max_output_len 4
```
기대 결과:
- `runs/n5_q17_h2_10k/train_gpu/train.log`에 `cpu: False`가 찍힘
- evaluator 로그에 `Distinguisher Method emd: all bits in secret have been recovered!`가 나타남
- evaluator 로그에 `Direct: all bits in secret have been recovered!`가 나타남
- evaluator 로그에 `CA: all bits in secret have been recovered!`가 나타남
- evaluator 로그에 `Found secret match - ending experiment.`가 나타남
- 종료 전 `runs/n5_q17_h2_10k/train_gpu/checkpoint.pth`가 저장됨

참고:
- 이 toy 설정은 보통 epoch `0` 평가 단계에서 바로 성공하므로 실행 시간이 짧습니다.
- `input_int_base=3`와 `share_token=1`은 작은 `Q=17` 예제 전용입니다. 논문 스케일의 `n=256, q=842779` 설정에서는 다른 인코딩 값을 사용합니다.
- 즉시 조기 종료되는 대신 더 긴 학습 곡선을 보고 싶다면, 짧은 로그를 실패로 해석하기보다 문제 난이도를 올려서 실험하세요.

__가우시안 설정 정의 (실제 성공한 경로 기준, N=5, Q=17, h=3)__: 아래 명령들은 로컬에서 실제로 support recovery가 성공했던 `data_n5q17_pathcheck` Gaussian 파이프라인을 그대로 재현합니다. 위의 짧은 binary sanity check보다, 작은 `q=17`에서 Gaussian support recovery를 다시 확인하고 싶을 때 적합한 설정입니다.

1. 환경을 활성화하고 `fpylll`가 준비되어 있는지 확인:
```bash
source /home/yu_mcc/miniconda3/etc/profile.d/conda.sh
conda activate lattice_env
conda install -y -c conda-forge fpylll
```

2. 원본 LWE 샘플 행렬을 수동으로 생성합니다. 수학적 분포는 `generate.py --step origA`와 같지만, 성공했던 로컬 run과 동일하게 수동 생성 방식을 사용합니다.
```bash
mkdir -p data_n5q17_pathcheck
python - <<'PY'
import numpy as np

A = np.random.randint(0, 17, size=(20, 5), dtype=np.int64)
np.save("data_n5q17_pathcheck/orig_A.npy", A)
print("orig_A.npy saved", A.shape)
PY
```
기대 결과:
- `data_n5q17_pathcheck/orig_A.npy`가 생성됨
- shape은 `(20, 5)`
- 원소는 `[0, 16]` 범위에 있음
- 현재 저장소 기준으로 이 예시는 `ctypes`로 C 표준 라이브러리의 `rand()`를 부르는 방식이 아니라, NumPy의 정수 난수 생성기를 사용합니다.
- `generate.py --step origA`도 동일하게 NumPy 기반으로 `[0, q)`에서 iid uniform 정수를 샘플링합니다. 즉, 두 방식은 현재 코드 기준으로 같은 분포의 `A`를 만들며, 차이는 주로 재현성 설정과 출력 경로 관리에 있습니다.

3. 같은 작은 파라미터 reduction 스타일로 `RA_tiny2` 실행:
```bash
python generate.py --step RA_tiny2 --dump_path ./data_n5q17_pathcheck/ra \
  --reload_data ./data_n5q17_pathcheck/orig_A.npy \
  --timeout 120 \
  --N 5 --Q 17 \
  --lll_delta 0.99 --float_type double \
  --bkz_block_size 10 --threshold 0.8 --threshold2 0.6 \
  --num_workers 1 --epoch_size 10000
```
기대 결과:
- `data_n5q17_pathcheck/ra/debug/` 아래에 새 디렉터리가 생성됨
- reduced data는 `data_n5q17_pathcheck/ra/debug/<run_id>/data.prefix`에 기록됨
- 로컬 성공 run에서는 `data_n5q17_pathcheck/ra/debug/dsm9ksxqyc/data.prefix`가 생성되었고, `wc -l` 결과는 `50000`이었습니다.

4. Hamming weight `3`인 sparse Gaussian secret에 대한 reduced `(A, b)` 샘플 생성:
```bash
python generate.py --step Ab --dump_path ./data_n5q17_pathcheck/ab \
  --reload_data ./data_n5q17_pathcheck/ra/debug/* \
  --secret_dir ./data_n5q17_pathcheck/ab \
  --min_hamming 3 --max_hamming 3 --num_secret_seeds 5 \
  --secret_type gaussian --sigma 3 \
  --epoch_size 20000 --reload_size 20000 \
  --num_workers 1 --timeout 120
```
기대 결과:
- `data_n5q17_pathcheck/ab/debug/` 아래에 새 디렉터리가 생성됨
- `data_n5q17_pathcheck/ab/debug/<run_id>/train.prefix`와 `test.prefix`가 생성됨
- `data_n5q17_pathcheck/ab/debug/<run_id>/secret.npy`가 생성됨
- 로컬 성공 run에서는 `data_n5q17_pathcheck/ab/debug/1426x4dp23/train.prefix`가 `78664`줄, `test.prefix`가 `10000`줄이었습니다.
- `num_secret_seeds=5`이므로 `secret.npy` shape은 `(5, 5)`입니다.
- 같은 run에서 `secret_seed=0`의 secret은 `[-3, 2, 0, 0, 6]`, support는 `[1, 1, 0, 0, 1]`이었습니다.

5. GPU에서 Gaussian 공격 학습 시작:
```bash
python train.py --cpu false --reload_data ./data_n5q17_pathcheck/ab/debug/* \
  --dump_path ./data_n5q17_pathcheck/train_run \
  --hamming 3 --secret_seed 0 \
  --epoch_size 100 --max_epoch 9999999 \
  --batch_size 512 --num_workers 0 \
  --input_int_base 17 --share_token 1 --output_int_base 17 \
  --eval_size 100 --debug
```
기대 결과:
- `data_n5q17_pathcheck/train_run/debug/` 아래에 새 디렉터리가 생성됨
- `data_n5q17_pathcheck/train_run/debug/debug_<run_id>/train.log`가 생성됨
- 평가가 끝날 때마다 `data_n5q17_pathcheck/train_run/debug/debug_<run_id>/checkpoint.pth`가 갱신됨
- `tail -n 40 data_n5q17_pathcheck/train_run/debug/debug_*/train.log`를 통해 validation loss와 evaluator 출력을 확인할 수 있음
- 로컬 성공 run에서는 `data_n5q17_pathcheck/train_run/debug/debug_50519924/train.log`에서 epoch `286` 평가 시 `Direct: all bits in secret have been recovered!`가 나타났고, 직후 `Found secret match - ending experiment.`로 종료되었습니다.
- 같은 성공 epoch의 `valid_xe_loss`는 약 `2.320574`였습니다.

참고:
- `--output_int_base`는 채팅이나 PDF에서 복사한 유니코드 대시가 아니라, 일반 ASCII 하이픈을 사용해야 합니다.
- 이 Gaussian 설정은 튜닝이 많이 된 논문 스케일 실험이라기보다, 재현과 디버깅을 위한 설정에 더 가깝습니다. `input_int_base=17`, `share_token=1`, `epoch_size=100`, `--debug` 조합은 내부 상태를 보기 쉽게 유지하기 위한 선택입니다.
- 작은 `q=17` Gaussian 설정에서는 `train_acc1/2`, `valid_acc1/2`가 오래 `0.0`으로 남아도 `Direct` probing으로 support recovery가 성공할 수 있습니다. 즉 이 설정에선 `ACC`보다 `valid_xe_loss`와 evaluator의 `Direct / Distinguisher / CA` 로그를 더 중요하게 해석해야 합니다.
- 로컬 성공 run `debug_50519924`도 epoch `286`에서 support recovery가 성공했지만, 같은 epoch의 `train_acc1/2`, `valid_acc1/2`는 모두 `0.0`이었습니다.
- 같은 데이터셋으로 다시 학습을 올리면 항상 같은 epoch에 성공한다는 보장은 없습니다. 작은 toy Gaussian 설정은 seed와 샘플 인스턴스 영향이 큽니다.
- `eval_size`가 `distinguisher_size`보다 작으면, 현재 저장소는 사용 가능한 evaluation set 크기에 맞춰 distinguisher sample 수를 자동으로 줄이도록 되어 있어 이 작은 Gaussian 설정이 깨지지 않게 처리합니다.

위의 두 preprocessing 단계를 slurm으로 돌리고 싶다면 `slurm_params` 폴더에 있는 `create_n256_data_step1.json`과 `create_n256_data_step2.json`을 참고하면 됩니다. 이 파일들은 `sbatch` 같은 slurm 스케줄링 도구를 설정할 때 좋은 예시가 됩니다.

## 인용

이 저장소를 인용할 때는 아래 BibTeX를 사용해 주세요.

```
@inproceedings{li2023salsa,
  title={SALSA VERDE: a machine learning attack on Learning With Errors with sparse small secrets},
  author={Li, Cathy and Wenger, Emily and Allen-Zhu, Zeyuan and Charton, Francois and Lauter, Kristin},
  booktitle={Advances in Neural Information Processing Systems},
  volume={37},
  year={2023}
}
```

## 라이선스

SALSA VERDE의 라이선스는 `LICENSE` 파일을 따릅니다. 기여 방법은 [CONTRIBUTING](CONTRIBUTING.md) 문서를 참고해 주세요.
