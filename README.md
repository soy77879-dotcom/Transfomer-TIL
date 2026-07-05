# Transformer TIL

Transformer를 공부하면서 정리한 내용이다. 단순히 구조 이름을 외우기보다, 왜 Transformer가 등장했고 Attention이 어떤 방식으로 문장을 처리하는지 이해하는 데 초점을 두었다.

## 1. Transformer란?

Transformer는 2017년 Google의 논문 *Attention Is All You Need*에서 제안된 모델 구조다. 이름 그대로 핵심은 Attention이다. 이전의 자연어 처리 모델은 RNN, LSTM, GRU처럼 단어를 순서대로 읽는 구조를 많이 사용했지만, Transformer는 순환 구조를 사용하지 않는다. 대신 문장 안의 단어들이 서로 얼마나 관련 있는지를 Attention으로 계산한다.

기존 Seq2Seq 모델은 인코더가 입력 문장을 하나의 벡터로 압축하고, 디코더가 그 벡터를 바탕으로 출력 문장을 만드는 방식이었다. 이 방식은 직관적이지만 문장이 길어질수록 문제가 생긴다. 긴 문장의 정보를 하나의 벡터에 모두 담기 어렵고, 앞쪽 단어의 정보가 뒤쪽으로 갈수록 약해질 수 있다.

Attention은 이런 한계를 줄이기 위해 등장했다. 디코더가 출력 단어를 만들 때 입력 문장의 어느 부분을 더 봐야 하는지 직접 계산하는 방식이다. Transformer는 여기서 더 나아가 RNN을 보조하던 Attention을 모델의 중심으로 가져왔다. 그래서 Transformer를 이해하려면 “순서대로 읽는 모델”에서 “단어 사이 관계를 한 번에 계산하는 모델”로 관점을 바꾸는 것이 중요하다.

## 2. 왜 Transformer가 필요했을까?

RNN 계열 모델은 문장을 앞에서 뒤로 순차적으로 처리한다. 이 방식은 문장 순서를 자연스럽게 반영할 수 있다는 장점이 있다. 하지만 단어를 하나씩 처리해야 하므로 병렬 연산이 어렵고, 긴 문장에서는 멀리 떨어진 단어 사이의 관계를 잘 잡기 힘들다.

예를 들어 다음 문장을 생각해볼 수 있다.

```text
그 동물은 길을 건너지 않았다. 왜냐하면 그것은 너무 피곤했기 때문이다.
```

여기서 “그것”은 “길”이 아니라 “동물”을 가리킨다. 사람은 문맥상 바로 알 수 있지만, 모델 입장에서는 문장 안의 여러 단어 중 어떤 단어와 연결되는지 계산해야 한다. Transformer의 Self-Attention은 이런 관계를 직접 계산한다. 각 단어가 문장 안의 다른 단어들을 모두 참고하면서 자신의 의미를 다시 만든다.

이 점이 Transformer의 가장 큰 특징이다. 단어를 차례대로 읽는 대신, 모든 단어 쌍의 관계를 한 번에 본다. 덕분에 병렬 처리가 가능하고 긴 거리 의존성도 더 잘 다룰 수 있다.

## 3. 전체 구조

Transformer는 RNN을 쓰지 않지만 Seq2Seq의 큰 틀은 유지한다. 즉, 인코더와 디코더로 구성된다.

```text
입력 문장
  -> 토큰화
  -> 임베딩
  -> 포지셔널 인코딩
  -> 인코더
  -> 디코더
  -> 선형 변환
  -> Softmax
  -> 출력 단어 예측
```

원 논문에서는 인코더 6개와 디코더 6개를 쌓은 구조를 사용했다. 각각의 인코더와 디코더는 완전히 다른 모양이 아니라 같은 형태의 층을 반복해서 쌓은 구조다. 인코더는 입력 문장을 문맥이 반영된 벡터로 바꾸고, 디코더는 그 정보를 참고해 출력 문장을 생성한다.

Transformer에서 자주 나오는 하이퍼파라미터는 다음과 같다.

| 기호 | 의미 |
| --- | --- |
| `d_model` | 단어 임베딩과 각 층 출력의 벡터 차원 |
| `num_layers` | 인코더와 디코더 층의 개수 |
| `num_heads` | Multi-Head Attention에서 사용하는 head 개수 |
| `d_ff` | Feed Forward Network 내부 차원 |
| `dropout` | 과적합을 줄이기 위한 비율 |
| `vocab_size` | 모델이 다루는 토큰 개수 |
| `max_position` | 처리할 수 있는 최대 문장 길이 |

논문에서는 대표적으로 `d_model = 512`, `num_heads = 8`, `num_layers = 6`, `d_ff = 2048`을 사용했다. 숫자 자체보다 중요한 점은 모든 토큰이 같은 차원의 벡터로 표현되고, 여러 Attention head가 서로 다른 관점에서 문장을 본다는 것이다.

## 4. 입력은 어떻게 처리되는가?

### 4.1 임베딩

모델은 문자를 그대로 이해하지 못한다. 그래서 먼저 단어 또는 토큰을 숫자 벡터로 바꾼다. 이 과정을 임베딩이라고 한다. 예를 들어 “나는 학생이다”라는 문장을 토큰 단위로 나누면, 각 토큰은 `d_model` 차원의 벡터가 된다.

하지만 임베딩만으로는 단어의 위치를 알 수 없다. RNN은 단어를 순서대로 입력받기 때문에 위치 정보가 자연스럽게 생기지만, Transformer는 문장의 모든 토큰을 한 번에 처리한다. 그래서 “어떤 단어가 몇 번째에 있는지”를 따로 알려줘야 한다.

### 4.2 포지셔널 인코딩

포지셔널 인코딩은 단어의 위치 정보를 벡터로 만들어 임베딩에 더하는 방법이다. Transformer는 입력 임베딩에 위치 벡터를 더한 값을 인코더와 디코더에 넣는다.

대표적인 식은 다음과 같다.

```text
PE(pos, 2i)     = sin(pos / 10000^(2i / d_model))
PE(pos, 2i + 1) = cos(pos / 10000^(2i / d_model))
```

`pos`는 단어의 위치이고, `i`는 벡터 차원의 인덱스다. 짝수 차원에는 sine, 홀수 차원에는 cosine을 사용한다. 이렇게 하면 각 위치마다 고유한 패턴이 생기고, 모델은 단어의 순서와 상대적인 거리 정보를 활용할 수 있다.

## 5. Attention의 핵심 아이디어

Attention은 현재 단어를 이해할 때 문장 안의 어떤 단어를 더 참고할지 정하는 과정이다. Transformer에서는 이 과정을 Query, Key, Value라는 세 가지 벡터로 설명한다.

| 이름 | 의미 |
| --- | --- |
| Query(Q) | 현재 단어가 찾고자 하는 정보 |
| Key(K) | 각 단어가 가진 검색 기준 |
| Value(V) | 실제로 가져올 정보 |

조금 더 쉽게 말하면 Query는 질문, Key는 각 단어의 색인, Value는 실제 내용에 가깝다. 어떤 Query가 들어오면 모든 Key와의 유사도를 계산하고, 유사도가 높은 단어의 Value를 더 많이 반영한다.

Transformer의 기본 Attention은 Scaled Dot-Product Attention이다.

```text
Attention(Q, K, V) = softmax((QK^T) / sqrt(d_k))V
```

과정은 다음과 같이 정리할 수 있다.

1. Query와 Key를 곱해 단어 간 유사도 점수를 구한다.
2. 점수가 너무 커지지 않도록 `sqrt(d_k)`로 나눈다.
3. Softmax로 점수를 확률처럼 바꾼다.
4. 이 가중치를 Value에 곱해 문맥 벡터를 만든다.

여기서 중요한 부분은 `QK^T`다. 이 연산을 통해 문장 안의 모든 단어 쌍에 대한 관련도를 한 번에 계산한다. 그래서 Transformer는 순차 처리 없이도 전체 문맥을 볼 수 있다.

## 6. Self-Attention

Self-Attention은 같은 문장 안에서 Query, Key, Value를 모두 만드는 Attention이다. 즉, 한 문장의 단어들이 서로를 바라보며 자신의 표현을 업데이트한다.

입력 행렬을 `X`라고 하면 Q, K, V는 다음처럼 만들어진다.

```text
Q = XW_Q
K = XW_K
V = XW_V
```

여기서 Q, K, V가 같은 입력에서 나온다는 점이 중요하다. 다만 완전히 같은 벡터를 그대로 쓰는 것은 아니다. 각각 다른 가중치 행렬 `W_Q`, `W_K`, `W_V`를 통과하기 때문에 서로 다른 역할을 하는 벡터가 된다.

Self-Attention을 거치면 각 단어는 단순히 자기 자신의 의미만 갖는 것이 아니라, 주변 단어와의 관계가 반영된 의미를 갖게 된다. 예를 들어 대명사가 어떤 명사를 가리키는지, 동사가 어떤 주어와 연결되는지 같은 정보를 표현에 포함할 수 있다.

## 7. Multi-Head Attention

Attention을 한 번만 수행하면 문장을 한 가지 관점에서만 보게 된다. Multi-Head Attention은 Attention을 여러 개의 head로 나누어 병렬로 수행한다.

각 head는 서로 다른 관계를 볼 수 있다. 어떤 head는 주어와 동사의 관계를 볼 수 있고, 다른 head는 대명사와 명사의 관계를 볼 수 있다. 또 다른 head는 단어의 위치나 구문적 연결을 볼 수도 있다.

흐름은 다음과 같다.

1. 입력에서 여러 벌의 Q, K, V를 만든다.
2. 각 head가 독립적으로 Attention을 계산한다.
3. head들의 결과를 이어 붙인다.
4. 다시 선형 변환을 거쳐 최종 출력 벡터를 만든다.

```text
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
```

Multi-Head Attention은 같은 문장을 여러 관점에서 읽는 효과를 만든다. 이 덕분에 Transformer는 문장의 의미 관계를 더 풍부하게 표현할 수 있다.

## 8. 인코더

인코더는 입력 문장을 문맥이 반영된 벡터 표현으로 바꾼다. 하나의 인코더 층은 두 개의 핵심 부분으로 구성된다.

1. Multi-Head Self-Attention
2. Position-wise Feed Forward Network

각 부분 뒤에는 Add & Norm이 붙는다. Add는 Residual Connection이고, Norm은 Layer Normalization이다.

```text
입력
  -> Multi-Head Self-Attention
  -> Add & Norm
  -> Position-wise FFNN
  -> Add & Norm
  -> 출력
```

인코더의 Self-Attention에서는 입력 문장의 모든 단어가 서로를 참고한다. 예를 들어 문장에 토큰이 5개 있다면, 각 토큰은 자기 자신을 포함한 5개 토큰 전체와의 관련도를 계산한다. 그 결과 각 단어 벡터는 문맥을 반영한 표현으로 바뀐다.

이후 Position-wise Feed Forward Network가 적용된다.

```text
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

이 FFNN은 각 위치의 벡터에 독립적으로 적용된다. Attention이 단어 사이 관계를 섞어주는 역할이라면, FFNN은 각 단어 표현을 더 비선형적으로 변환해 표현력을 높이는 역할을 한다.

Residual Connection은 서브층의 입력을 출력에 더하는 방식이다.

```text
출력 = LayerNorm(x + Sublayer(x))
```

이 구조는 깊은 모델을 안정적으로 학습시키는 데 도움이 된다. Layer Normalization은 벡터의 분포를 정리해 학습을 더 안정적으로 만든다.

## 9. 디코더

디코더는 인코더의 출력 정보를 참고해 문장을 생성한다. 하나의 디코더 층은 세 부분으로 구성된다.

1. Masked Multi-Head Self-Attention
2. Encoder-Decoder Multi-Head Attention
3. Position-wise Feed Forward Network

```text
출력 문장 입력
  -> Masked Multi-Head Self-Attention
  -> Add & Norm
  -> Encoder-Decoder Attention
  -> Add & Norm
  -> Position-wise FFNN
  -> Add & Norm
  -> 다음 단어 예측
```

### 9.1 Masked Self-Attention

디코더는 출력 문장을 생성하는 역할을 한다. 이때 아직 생성하지 않은 미래 단어를 보면 안 된다. 예를 들어 “나는 학교에 간다”를 만드는 중에 “학교에”를 예측하는 단계라면, 뒤에 나올 “간다”를 참고하면 학습 규칙이 깨진다.

그래서 디코더의 첫 번째 Attention에는 Look-ahead Mask를 적용한다. 이 마스크는 현재 위치보다 뒤에 있는 단어를 보지 못하게 만든다.

```text
예측 위치: 3번째 단어
참고 가능: 1번째, 2번째, 3번째 단어
참고 불가: 4번째 이후 단어
```

이렇게 하면 학습할 때는 문장을 한 번에 넣어 병렬 처리를 하면서도, 실제 생성 과정처럼 이전 단어만 참고하도록 만들 수 있다.

### 9.2 Encoder-Decoder Attention

디코더의 두 번째 Attention은 Encoder-Decoder Attention이다. 여기서는 Q는 디코더에서 오고, K와 V는 인코더의 출력에서 온다.

```text
Q = 디코더의 이전 서브층 출력
K = 인코더의 최종 출력
V = 인코더의 최종 출력
```

이 Attention은 디코더가 출력 단어를 만들 때 입력 문장의 어느 부분을 참고할지 정한다. 번역 모델이라면 현재 생성 중인 단어가 원문 문장의 어떤 단어와 연결되는지 계산하는 부분이라고 볼 수 있다.

## 10. Transformer의 세 가지 Attention

Transformer에는 Attention이 여러 번 등장한다. 위치에 따라 역할이 조금씩 다르다.

| 위치 | Attention 종류 | Query | Key | Value | 마스크 |
| --- | --- | --- | --- | --- | --- |
| 인코더 | Self-Attention | 입력 문장 | 입력 문장 | 입력 문장 | Padding Mask |
| 디코더 첫 번째 서브층 | Masked Self-Attention | 출력 문장 | 출력 문장 | 출력 문장 | Look-ahead Mask |
| 디코더 두 번째 서브층 | Encoder-Decoder Attention | 디코더 출력 | 인코더 출력 | 인코더 출력 | Padding Mask |

Padding Mask는 패딩 토큰을 Attention 계산에서 제외하기 위해 사용한다. Look-ahead Mask는 디코더가 미래 단어를 보지 못하게 하기 위해 사용한다.

## 11. 전체 동작 과정

Transformer가 번역을 수행한다고 가정하면 흐름은 다음과 같다.

1. 입력 문장을 토큰으로 나눈다.
2. 각 토큰을 임베딩 벡터로 바꾼다.
3. 포지셔널 인코딩을 더해 위치 정보를 넣는다.
4. 인코더의 Self-Attention이 입력 문장 내부의 단어 관계를 계산한다.
5. 인코더의 FFNN이 각 단어 표현을 한 번 더 변환한다.
6. 여러 인코더 층을 지나며 문맥 표현이 정교해진다.
7. 디코더는 시작 토큰을 입력받아 출력을 만들기 시작한다.
8. Masked Self-Attention으로 이미 생성된 단어들만 참고한다.
9. Encoder-Decoder Attention으로 입력 문장의 관련 부분을 참고한다.
10. FFNN과 선형 변환을 거쳐 다음 단어의 확률 분포를 만든다.
11. 가장 가능성이 높은 단어를 선택하거나 샘플링해 다음 출력으로 사용한다.
12. 종료 토큰이 나올 때까지 이 과정을 반복한다.

## 12. 장점

Transformer의 장점은 분명하다.

- RNN보다 병렬 처리가 쉬워 학습 속도가 빠르다.
- 긴 문장에서도 멀리 떨어진 단어 사이 관계를 직접 계산할 수 있다.
- Self-Attention으로 문맥 정보를 효과적으로 반영한다.
- Multi-Head Attention으로 다양한 관점의 의미 관계를 학습할 수 있다.
- 번역, 요약, 질의응답, 문장 생성 등 여러 NLP 작업에 적용할 수 있다.

이 구조는 이후 BERT, GPT, T5 같은 모델의 기반이 되었다. BERT는 Transformer의 인코더 구조를 중심으로 사용하고, GPT는 디코더 구조를 중심으로 사용한다.

## 13. 한계

물론 Transformer가 모든 문제를 해결한 것은 아니다.

- Self-Attention은 모든 단어 쌍의 관계를 계산하므로 문장이 길어질수록 계산량이 크게 증가한다.
- 단어 순서를 직접 처리하지 않기 때문에 위치 정보를 따로 넣어야 한다.
- 큰 모델일수록 많은 데이터와 연산 자원이 필요하다.
- Attention weight가 항상 사람이 해석하기 쉬운 의미를 보여주는 것은 아니다.

특히 Self-Attention의 계산 복잡도는 시퀀스 길이를 `n`이라고 할 때 `O(n^2)`이다. 긴 문서를 처리할 때 메모리와 계산 비용이 커지는 이유가 여기에 있다.

## 14. 핵심 정리

Transformer는 RNN 없이 Attention만으로 시퀀스를 처리하는 인코더-디코더 기반 모델이다. 문장을 순서대로 읽는 대신, 문장 안의 모든 단어가 서로를 참고하도록 만든다. 이때 단어 순서 정보는 포지셔널 인코딩으로 보완한다.

가장 중요한 개념은 Self-Attention이다. Self-Attention은 각 단어가 문장 안의 다른 단어들과 얼마나 관련 있는지 계산하고, 그 결과를 이용해 문맥이 반영된 표현을 만든다. Multi-Head Attention은 이 과정을 여러 관점에서 동시에 수행해 표현력을 높인다.

인코더는 입력 문장을 문맥 벡터로 바꾸고, 디코더는 그 벡터를 참고해 출력 문장을 생성한다. 디코더에서는 미래 단어를 보지 못하도록 Look-ahead Mask를 사용하고, Encoder-Decoder Attention을 통해 입력 문장의 정보를 가져온다.

정리하면 Transformer는 “현재 단어를 이해하기 위해 문장 안의 어떤 단어를 얼마나 봐야 하는가”를 계산하는 모델이다. 이 아이디어가 현대 NLP와 생성형 AI 모델의 기반이 되었다.

## 15. 회고

Transformer를 처음 볼 때는 Q, K, V라는 이름 때문에 어렵게 느껴졌다. 하지만 Query는 찾는 것, Key는 비교 기준, Value는 실제로 가져올 정보라고 생각하니 훨씬 이해하기 쉬웠다.

또 하나 중요한 점은 Transformer가 단순히 Attention을 “추가한” 모델이 아니라, Attention을 중심으로 전체 구조를 다시 설계한 모델이라는 점이다. RNN처럼 순서대로 읽지 않아도 문맥을 파악할 수 있다는 아이디어가 이후 BERT와 GPT 같은 모델로 이어졌다.

앞으로 BERT나 GPT를 공부할 때도 인코더, 디코더, Self-Attention, Masking, Positional Encoding 개념이 계속 등장하므로 이 내용을 기준점으로 삼으면 좋다.
