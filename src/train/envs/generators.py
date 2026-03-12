# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import circulant
from logging import getLogger

logger = getLogger()

class Generator(ABC):
    def __init__(self, params):
        self.N = params.N
        self.Q = params.Q
        self.secret = params.secret #비밀 벡터 s의 원소는 0과 1로 이루어져 있고, 이 값이 1이면 랜덤행렬 
        self.hamming = params.hamming #해밍 무게 값이 크면 비밀 벡터의 1의 개수가 많아진다. 비밀 벡터의 1의 개수가 많아지면 랜덤행렬 c와의 내적에서 1이 곱해지는 원소가 많아져서 b의 값이 더 다양해진다. 반대로 해밍 무게가 작으면 비밀 벡터의 1의 개수가 적어져서 b의 값이 덜 다양해진다.
        self.sigma = params.sigma #표준편차 시그마 값이 작으면 0근처에 오밀조밀하게 모인다 크면 분포가 넓어진다

    @abstractmethod
    def generate(self):
        pass

    @abstractmethod
    def evaluate(self, src, tgt, hyp):
        pass

class RLWE(Generator):
    def __init__(self, params):
        super().__init__(params)
        self.correctQ = params.correctQ
        self.q2_correction = np.vectorize(self.q2_correct)

    def generate(self, rng, train = True):
        # sample a uniformly from Z_q^n
        a = self.gen_a_row(rng, 0, self.Q)

        # do the circulant:
        c = self.compute_circulant(a)

        b = self.compute_b(c, rng)
       
        return c, b # return shapes NxN, N
    
    # a만드는 함수임
    def gen_a_row(self, rng, minQ, maxQ): # 1차원 벡터생성 a0 a1 a2 a3 ... an-1, n개 원소, 0~Q-1 사이의 정수로 채워짐
        a = rng.randint(minQ, maxQ, size=self.N, dtype=np.int64)
        return a

    def compute_circulant(self, a): # 랜덤행렬 A계산 함수, NxN 행렬로 반환 이건 그냥 랜덤행렬임
        c = circulant(a) # circulant 행렬로 a로 부터 c를 선언
        tri = np.triu_indices(self.N, 1) # 1차원 벡터 a를 한칸식 민다
        c[tri] *= -1 # c의 upper triangular 부분을 -1로 곱한다. (a0 a1 a2 a3) -> (a0 -a3 -a2 -a1)
        if self.correctQ: # correctQ가 발동이 될때 이거 모듈러 q를 보정하기 위함임 이유는 인공지능에서 0을 기준으로 양 쪽으로 퍼져있는경우 더 잘학습한다고 함 작은 노이즈라는 의미가 보존됨
            c = self.q2_correction(c) # c의 원소들을 -Q/2 ~ Q/2 사이로 보정한다. ex) Q=10이면, 0~9 사이의 원소들을 -5~4 사이로 보정

        c = c % self.Q # c의 원소들을 0~Q-1 사이로 보정한다. ex) Q=10이면, -5~4 사이의 원소들을 0~9 사이로 보정

        assert (np.min(c) >= 0) and (np.max(c) < self.Q) # 데이터 검증용 c가 0~Q-1 사이의 원소로만 이루어져 있는지 그리고 c의 최대값이 Q보다 작은지 검증

        return c # c반환 이게 결국
        
    def compute_b(self, c, rng): # b = c * s + e mod q, s는 secret, e는 노이즈 ## 식 완성하는 함수임
        if self.sigma > 0: #시그마가 0보다 크면 노이즈가 존재한다는 의미임
            e = np.int64(rng.normal(0, self.sigma, size = self.N).round()) # 랜덤 노이즈 e를 만드는 함수인데 N이 256이고 시그마가 3.0인 경우 -3에서 3사이 작은 숫자들로 채워짐
            b = (np.inner(c, self.secret) + e) % self.Q # 연립방정식으로 이뤄진 최종 수식 b = c * s + e mod q
            b = np.inner(c, self.secret) % self.Q # 노이즈가 없는 경우 그냥 결합

        if self.correctQ: # Q 보정을 할경우 식 전체를 -Q/2 ~ Q/2 사이로 보정한다. ex) Q=10이면, 0~9 사이의 원소들을 -5~4 사이로 보정
            b = self.q2_correction(b)
        return b

    def q2_correct(self, x): # 보정 함수의 조건에 따른 작동 함수 이게 있는 이유는 q2_correction이 발동될때 호출이 되고 해당 함수를 벡터화 해서 위 b = self.q2_correction(b)의 작동을 담당함 
                             #즉 의미는 q2_correction(b)자체가 q2_correct함수의 벡터화된 버전임 그래서 q2_correction이 발동될때 b의 원소 하나하나에 대해서 q2_correct함수가 작동하는 형태임
        if x <= -self.Q/2:
            x = x+self.Q # x가 -Q나누기 2 보다 작을 경우 x에 Q를 더해서 보정한다
        elif x >= self.Q/2: 
            x = x-self.Q # 만약 클 경우 Q나누기 2보다 크면 x에서 Q를 빼서 보정한다
        return x

    def evaluate(self, src, tgt, hyp): # hyp는 모델이 예측한 값, tgt는 실제값, src는 입력값
        return 1 if hyp == tgt else 0 # 완전히 일치할 경우 1 아니면 0

    def get_difference(self, tgt, hyp): # hyp와 tgt의 차이를 계산하는 함수, 이 함수는 hyp와 tgt가 0~Q-1 사이의 정수로 이루어져 있다고 가정하고 작동함
        diff = (hyp[0]-tgt[0]) % self.Q # hyp와 tgt의 차이를 계산하는데, 이때 차이는 모듈로 Q로 계산됨. 예를 들어, Q=10이고 hyp=2, tgt=8인 경우 diff는 (2-8) mod 10 = -6 mod 10 = 4가 됨. 이렇게 모듈로 연산을 하면 hyp와 tgt의 차이가 항상 0~Q-1 사이의 값으로 표현됨
        if diff > self.Q // 2:
            return abs(diff - self.Q)
        return diff

    def evaluate_bitwise(self, tgt, hyp):
        return [int(str(e1)==str(e2)) for e1,e2 in zip(tgt,hyp)]
