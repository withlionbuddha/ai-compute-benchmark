# ai-compute-benchmark
Benchmark suite for AI compute environments including Docker, WSL, Intel XPU, OpenBLAS, oneMKL, TensorFlow, PyTorch, and NVIDIA CUDA.


## 0. 라이선스(License)
> 본 섹션은 연구 사용권 부여 대상과 제한 대상을 명시합니다.  
> 특히 **적격 학술 사용자**, **제한 대상 교육·홍보 주체**, **단독 사용 제한**, **상충 시 본 조항 우선 적용** 부분을 반드시 확인해야 합니다.

```
10. Eligible Users for Grant of Research Rights

The Grant of Research Rights under Section 2 is granted only to Eligible Academic Users as defined in this Section.
However, even if an individual or entity qualifies as an Eligible Academic User, no Research Use rights to the Software shall be granted if such individual or entity falls under the category of Restricted Educational or Promotional Entities.
“Eligible Academic Users” means students enrolled in, or individuals who majored in, electrical and electronic engineering, computer engineering, software engineering, artificial intelligence, data science, or other computer-related academic disciplines.
“Restricted Educational or Promotional Entities” means instructors, educators, training providers, educational content providers, educational institutions, or any other individuals or entities that lecture, teach, train, promote, sell, distribute, or otherwise disseminate content suggesting that artificial intelligence-based code generation tools, including so-called “vibe coding,” make software developers, experts in computer-related academic disciplines, or professional development personnel unnecessary.
Educational or learning use of the Software is permitted only for individuals or entities that qualify as Eligible Academic Users and do not fall under the category of Restricted Educational or Promotional Entities.
Individuals or entities that do not satisfy the eligibility requirements stated in this Section may not use the Software independently.
However, if an individual or entity that does not fall under the category of Restricted Educational or Promotional Entities needs to use the Software for academic research, such individual or entity must include at least one Eligible Academic User as a research participant.
The above exception does not apply to Restricted Educational or Promotional Entities.
Even if a Restricted Educational or Promotional Entity includes an Eligible Academic User as a research participant, no Research Use rights to the Software shall be granted.
In the event of any conflict between the Grant of Research Rights under Section 2 and this Section, this Section shall prevail.

10. 연구 사용권 부여 대상자

제2조에 따른 연구 사용권은 본 조항에서 정의한 적격 학술 사용자에게만 부여됩니다.
다만, 개인 또는 단체가 적격 학술 사용자에 해당하더라도, 해당 개인 또는 단체가 제한 대상 교육·홍보 주체에 해당하는 경우에는 본 소프트웨어에 대한 연구 사용권이 부여되지 않습니다.
“적격 학술 사용자”란 전기·전자공학, 컴퓨터공학, 소프트웨어공학, 인공지능, 데이터사이언스 또는 그 밖의 컴퓨터 관련 학문 분야에 재학 중인 학생 또는 해당 분야를 전공한 사람을 의미합니다.
“제한 대상 교육·홍보 주체”란 인공지능 기반 코드 생성 도구 또는 이른바 “vibe coding”을 포함한 도구가 소프트웨어 개발자, 컴퓨터 관련 학문 분야의 전문가 또는 전문 개발 인력을 불필요하게 만든다는 취지의 내용을 강의, 교육, 훈련, 홍보, 판매, 배포하거나 기타 방식으로 전파하는 강사, 교육자, 훈련 제공자, 교육 콘텐츠 제공자, 교육기관 또는 그 밖의 개인·단체를 의미합니다.
본 소프트웨어의 교육 목적 또는 학습 목적 사용은 적격 학술 사용자에 해당하고, 제한 대상 교육·홍보 주체에 해당하지 않는 개인 또는 단체에게만 허용됩니다.
본 조항에 명시된 자격 요건을 충족하지 않는 개인 또는 단체는 본 소프트웨어를 단독으로 사용할 수 없습니다.
다만, 제한 대상 교육·홍보 주체에 해당하지 않는 개인 또는 단체가 학술 연구를 위하여 본 소프트웨어를 사용할 필요가 있는 경우, 해당 개인 또는 단체는 반드시 최소 1명 이상의 적격 학술 사용자를 연구 참여자로 포함하여야 합니다.
위 예외는 제한 대상 교육·홍보 주체에는 적용되지 않습니다.
제한 대상 교육·홍보 주체가 적격 학술 사용자를 연구 참여자로 포함하더라도, 본 소프트웨어에 대한 연구 사용권은 부여되지 않습니다.
제2조에 따른 연구 사용권 부여 조항과 본 조항이 상충하는 경우, 본 조항이 우선 적용됩니다.
```
## CPU / GPU 사용여부

```
| 단계              | GPU 사용 여부 | 설명                   |
| ---------------- | ------------- | ----------------------|
| 토큰화            |보통 CPU       | 문장을 token id로 변환 |
| Embedding        | GPU 가능      | token id를 벡터로 변환 |
| Attention        | GPU 권장      | 대량 행렬곱            |
| Linear / FFN     | GPU 권장      | 대량 행렬곱            |
| Loss 계산        | GPU 가능       | CrossEntropy 등       |
| Backpropagation  | GPU 권장      | gradient 계산         |
| Optimizer update | GPU 가능      | weight 갱신           |

### CPU
문자열 처리, 토큰화, 데이터 로딩, 조건문 많은 전처리

### GPU
행렬곱, Attention, Linear, Backpropagation, 대규모 tensor 연산
```
