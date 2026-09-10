# GR-LLMs: критический обзор и воспроизводимый учебный синтез

> Zhen Yang, Haitao Lin, Jiawei Xue, Ziji Zhang, *GR-LLMs: Recent Advances in Generative Recommendation Based on Large Language Models*, 2025.  
> [Статья на arXiv](https://arxiv.org/abs/2507.06507) · [PDF](https://arxiv.org/pdf/2507.06507)

## 0. Главный вывод

GR-LLMs систематизирует переход от каскада recall → pre-rank → rank к генеративным рекомендательным моделям на базе LLM-подобных sequence models. Авторы группируют работы по месту в production pipeline, способу кодирования item-ов и training strategy. Сильная сторона survey — компактная карта быстро растущей области и явный акцент на industrial constraints: latency, огромный каталог, cold start, multimodality и alignment. Слабая — отсутствие систематического search protocol, единого benchmark и количественного meta-analysis; утверждения разных работ нельзя читать как честный leaderboard.

Поскольку в статье **нет собственного Model Core, dataset split или нового loss**, `poc.ipynb` реализует прозрачный **учебный синтез** трёх семейств из survey:

1. token-based recall через residual-quantized semantic IDs (линия TIGER/OneRec);
2. autoregressive decoder с next-semantic-token CE;
3. contrastive representation alignment и DPO-подобная preference alignment;
4. наше дополнительное улучшение — catalog-constrained beam search по prefix trie, исключающий несуществующие item codes.

Это не «реализация GR-LLMs model» — такой модели у авторов нет — и не копия одной из процитированных работ.

## Подробный обзор содержания статьи

### Паспорт публикации и тип работы

Статья написана Zhen Yang, Haitao Lin, Jiawei Xue и Ziji Zhang из AMAP, Alibaba Group и опубликована на arXiv в 2025 году. PDF занимает 13 страниц: примерно восемь страниц основного текста и пять страниц библиографии. Это компактный narrative survey, сфокусированный преимущественно на работах 2023–2025 годов и industrial cases. В статье нет собственного dataset, новой модели или экспериментальной секции; её научный результат — систематизация быстро меняющегося поля.

По формату работа заметно отличается от классического systematic review. Авторы не описывают поисковые базы, ключевые запросы, критерии включения и процедуру оценки качества публикаций. Поэтому слово *comprehensive* следует понимать как широкий экспертный обзор выбранных авторами направлений, а не как гарантию полного покрытия всей литературы.

В статье одна основная схема — **Figure 1** с традиционной воронкой recall → pre-ranking → ranking — и две сводные таблицы:

- **Table 1** сопоставляет learning targets и post-training strategies GR-методов;
- **Table 2** перечисляет multimodal LLM recommenders, типы данных и объекты рекомендации.

### Главный исследовательский вопрос survey

Авторы пытаются ответить не на вопрос «какая одна GR-модель лучшая», а на четыре более практичных вопроса:

1. Где LLM-based generative recommendation располагается относительно MLR, DLR и традиционного каскада?
2. Как generative models применяются в recall, ranking и end-to-end recommendation?
3. Какие training и serving решения делают их пригодными для индустрии?
4. Какие проблемы — scaling, data quality, cold start и унификация задач — остаются открытыми?

В результате taxonomy строится сразу по нескольким осям: **место в pipeline**, **форма представления item**, **training stage**, **источник знания** и **тип выхода**. Это полезно, но порождает пересечения: одна работа может одновременно быть token-based recall, multi-stage model и end-to-end system.

### Как устроен рассказ статьи

Логика survey движется от общего к прикладному:

1. Introduction описывает смену трёх поколений recommendation algorithms и объясняет, почему GR рассматривается как новая парадигма.
2. Preliminaries сводит LLM к next-token objective и напоминает устройство традиционного каскада.
3. Application Settings сортирует работы по recall, rank и end-to-end deployment.
4. Main Considerations and Challenges пересортировывает те же и дополнительные работы по training strategy, inference efficiency и cold-start/world-knowledge mechanisms.
5. Future Directions выделяет model scaling, data cleaning и one-model-for-all.
6. Conclusion и Limitations кратко фиксируют границы самого survey.

Таким образом, Section 3 отвечает на вопрос **«где использовать GR?»**, а Section 4 — **«как обучить и обслуживать GR?»**. Это две ортогональные классификации, и их полезно читать вместе.

### Раздел 1 — Introduction

Введение описывает три технологические эпохи. **Machine-learning recommendation** опирается на collaborative filtering, content similarity и matrix factorization; её ограничения — sparsity, cold start и ручные признаки. **Deep-learning recommendation** учит нелинейные представления, но в промышленности всё равно зависит от множества handcrafted features и каскада специализированных моделей. **Generative recommendation** переносит задачу в sequence-generation framework и обещает масштабировать capacity более естественно.

После исторического контекста авторы перечисляют наиболее заметные линии развития:

- SASRec как ранний autoregressive Transformer для next-item prediction;
- HSTU как large-scale action-sequence encoder;
- TIGER и semantic IDs через RQ-VAE для уменьшения output vocabulary;
- OneRec с MoE и DPO для end-to-end list generation;
- MTGR, который возвращает важные DLR/cross-features;
- EGA-V1/V2 для единой модели рекламного ranking;
- URM как universal recommendation learner.

Затем формулируются ожидаемые преимущества GR: explainability, novelty/creativity, унификация системы и scaling. Важно, что это неоднородный список. Explanation относится прежде всего к language-generating models, тогда как HSTU или semantic-ID decoder сам по себе не порождает человекочитаемое объяснение. Survey местами объединяет свойства разных подклассов GR под общей вывеской.

### Раздел 2 — Preliminaries

**Section 2.1** задаёт стандартную autoregressive objective $P(x_t\mid x_{<t};\theta)$ и напоминает переход LLM от чистого текста к изображениям, аудио и видео. Раздел намеренно короткий: attention equations, parameter-efficient tuning и decoding algorithms не разбираются. Для авторов важен общий принцип — произвольные modalities можно представить последовательностью и обучать next-token prediction.

**Section 2.2** и Figure 1 описывают каскад. Исходный corpus порядка миллионов объектов последовательно сокращается recall, pre-ranking и ranking до выдаваемого списка. Такая архитектура экономит compute, но каждая стадия создаёт ceiling для следующей: потерянный на recall item невозможно восстановить ranker-ом. Дополнительная проблема — objective mismatch, когда recall оптимизирует hit rate, ranker CTR, а продукту нужна долгосрочная utility.

Авторы используют этот каскад как координатную систему всего дальнейшего survey. GR может либо улучшать одну стадию, сохраняя pipeline, либо пытаться заменить каскад целиком.

### Раздел 3 — Application Settings of GR

#### Section 3.1: Recall

Recall делится на три семейства.

**Prompt-based methods** передают LLM текстовое описание пользователя. LLMTreeRec последовательно резюмирует интерес, выбирает категории и спускается по item tree; параметры LLM остаются frozen. SyNeg использует LLM не как retriever, а как генератор hard negatives для обучения другой модели. Эти подходы хорошо используют world knowledge, но требуют entity linking и не гарантируют, что сгенерированное имя соответствует доступному catalog item.

**Token-based methods** превращают действия в специальные tokens. HSTU формулирует recall как generative transduction; KuaiFormer сжимает раннюю, среднюю и свежую историю на разных уровнях и, по survey, внедрён на платформе с 400 млн DAU; URM добавляет objective tokens, чтобы одна модель выполняла разные retrieval intents. Это collaborative GR: знание появляется из behavioral logs, а не обязательно из pretraining на языке.

**Embedding-based methods** используют pretrained vision/text encoders как feature generators. В MoRec item embeddings поступают в знакомые DSSM или SASRec. Такое решение не заменяет каскад, но предлагает низкорисковый способ перенести open-world semantics и cold-start signal в существующую инфраструктуру.

#### Section 3.2: Rank

Survey противопоставляет **generative architectures** и **hybrid integration**.

К первой группе относятся GR/HSTU, action-oriented GenRank и dual-flow DFGR. GenRank рассматривает items как positional context и фокусируется на генерации action labels; это сокращает sequence length. DFGR разделяет real и fake flows, затем объединяет action-type tokens, чтобы моделировать heterogeneous behavior и не повторять дорогие вычисления.

Hybrid models сохраняют DLRM serving path. LEARN строит LLM-derived content embeddings и адаптирует их к collaborative space через twin-tower; HLLM последовательно применяет ITEM LLM и USER LLM; SRP4CTR переносит self-supervised sequential knowledge в CTR model через cross-attention; MTGR соединяет HSTU-like backbone с raw и cross features. Появление MTGR — важная коррекция раннего оптимизма: полное удаление production features может быть слишком дорогим по quality.

#### Section 3.3: End-to-end recommendation

End-to-end модель должна не просто вернуть candidates, а выдать ранжированный список и потенциально заменить recall/pre-rank/rank. OneRec генерирует session-wise lists и использует iterative reward-model/DPO alignment. OneSug переносит list-wise preference alignment на query suggestions. EGA-V1/V2 вводят auction-aware preference/reward training для рекламы.

Академические S-DPO, RosePO и SPRec исследуют качество preference data: несколько одновременно показанных negatives, hard negatives по popularity/semantic similarity и self-replay собственных предсказаний. Этот блок показывает, как поле переходит от next-item likelihood к list utility. Одновременно возрастает риск exposure bias: то, что item не был выбран, ещё не означает отрицательное предпочтение.

### Раздел 4 — Main Considerations and Challenges

#### Section 4.1: Training pipelines

Авторы делят training на single-stage и multi-stage. В **single-stage** HSTU/URM оптимизируют next-item/action CE, KuaiFormer — in-batch contrastive similarity, MTGR — CTR binary classification. Это простой deployment path, но обычно одна objective соответствует только одной стадии.

В **representation-based multi-stage** сначала обучаются user/item representations, после чего embeddings поступают в downstream DLR. HLLM, LEARN и LUM используют contrastive alignment; QARM дополнительно квантует embedding в semantic ID, который можно подать как ranking feature. LUM описан как трёхэтапный pipeline: next-item pretraining, contrastive representation learning, затем recall/ranking model.

В **model-based multi-stage** меняется сама policy: next-item pretraining создаёт базовую способность генерации, затем DPO/reward alignment улучшает ranking/list behavior. К этой ветви относятся OneRec, OneSug и EGA. Table 1 является краткой картой раздела, но не сообщает datasets, model sizes и качества — это таблица методов, а не leaderboard.

#### Section 4.2: Inference efficiency

Раздел систематизирует три способа ускорения.

1. **Sequence compression:** GenRank убирает отдельные action tokens, DFGR объединяет interaction flows, KuaiFormer агрегирует старые события.
2. **Architecture changes:** HSTU заменяет softmax attention pointwise aggregation; EGA-V1/RecFormer использует cluster-aware attention.
3. **Output/decoding compression:** semantic IDs через RQ-VAE, multi-token prediction и приближённый Top-K уменьшают огромный item vocabulary.

Для ranking отдельно обсуждаются M-FALCON и mask MTGR: несколько candidates оцениваются в одном pass, но не видят друг друга. Survey верно выделяет candidate scoring как отдельную проблему от autoregressive recall. При этом не приводится единая таблица p95/p99, QPS или GPU cost, поэтому сравнить реальную эффективность методов по самому survey нельзя.

#### Section 4.3: Cold start and world knowledge

Авторы различают два механизма. **Information augmentation** добавляет generated embeddings/knowledge к входу: SAID строит item embeddings из текста, CSRec объединяет metadata и commonsense. **Model reasoning** поручает LLM непосредственно вывести рекомендацию из prompt; пример — LLM-Rec.

World knowledge может помочь новому item до накопления кликов, но требует адаптации к collaborative domain. LC-Rec объединяет Llama semantics с interaction signals; RAG может добавить актуальные domain facts. Далее раздел расширяется до multimodality: изображения, видео, речь и аудио. Table 2 перечисляет InteraRec, I-LLMRec, NoteLLM-2 и TALKPLAY. NoteLLM-2 использует image+text для notes, TALKPLAY — audio/lyrics/tags для music, InteraRec — web-page screenshots для commerce.

Survey сообщает отдельные business/public результаты, например рост exposure у NoteLLM-2 и оценку TALKPLAY на Million Playlist Dataset, но не нормализует baseline, split и serving budget. Эти примеры доказывают реализуемость multimodal recommendation, а не превосходство одного architecture class.

### Раздел 5 — Future Directions

**Model scaling.** Авторы считают scaling естественным продолжением HSTU/GenRank/MTGR, но признают, что большинство новых GR ограничено моделями порядка 0.x–1.xB, а выигрыш гораздо больших systems изучен слабо. Главный конфликт — больше parameters и длиннее history против online latency.

**Data cleaning.** Behavioral logs не имеют аналога грамматической корректности текста. Bots, accidental clicks, stale sessions, biased exposure и смешение modalities делают quality assessment отдельной задачей. Авторы предлагают discriminative filtering и quality-conditioned training, но конкретную метрику «валидности последовательности» не вводят.

**One model for all.** URM рассматривается как шаг к единой модели для recommendation, search, multiple scenarios, objectives и long-tail. Task/objective tokens могут заменить отдельные architectures, но survey не разбирает подробно gradient interference, routing capacity и blast radius единой production model.

### Раздел 6 и Limitations

Conclusion повторяет три результата survey: preliminaries/application map, industrial considerations и future directions. Отдельная секция Limitations признаёт две проблемы: быстрое появление новых работ и невозможность раскрыть все технические детали из-за лимита страниц.

Это честное, но неполное самоограничение. Авторы не обсуждают отсутствие systematic search protocol, риск publication/industry bias и несопоставимость online claims. Эти пункты добавлены в критический анализ ниже.

### Карта наиболее часто обсуждаемых работ

| Работа в survey | Место в системе | Ключевая идея в изложении авторов |
|---|---|---|
| HSTU / GR | Recall и rank | Action/item transduction, pointwise attention, scaling |
| TIGER | Token-based recall | RQ-VAE semantic IDs вместо огромного atomic vocabulary |
| KuaiFormer | Recall | Иерархическое сжатие early/middle/recent history |
| URM | Recall / universal learner | Objective tokens и единый input-output format |
| MoRec | Embedding-based recall | Pretrained vision/text item embeddings в DSSM/SASRec |
| GenRank | Rank | Action-oriented sequence, item как positional context |
| DFGR | Rank | Real/fake flows и объединение interaction tokens |
| LEARN / HLLM | Hybrid rank | LLM representations как адаптируемые DLR features |
| MTGR | Hybrid/generative rank | HSTU-like sequence model плюс raw/cross features |
| OneRec | End-to-end list | Semantic IDs, MoE, iterative DPO alignment |
| OneSug | End-to-end query suggestion | List-wise preference alignment |
| EGA-V1/V2 | End-to-end ads | Auction-aware preference/reward alignment |
| QARM / LUM | Multi-stage | Quantized IDs или contrastive embeddings для downstream DLR |
| SAID / CSRec / LC-Rec | Cold start | Content/world knowledge плюс collaborative adaptation |
| NoteLLM-2 / TALKPLAY / InteraRec | Multimodal | Image, text, audio или screenshots как recommendation signal |

Таблица показывает, что survey не продвигает один dominant design. Скорее он описывает continuum: от LLM как offline feature generator до модели, полностью заменяющей online cascade.

### Что является собственным вкладом авторов survey

| Элемент | Собственный вклад GR-LLMs? | Комментарий |
|---|---|---|
| Новая architecture/loss | Нет | Формулы и methods принадлежат процитированным работам |
| Деление MLR → DLR → GR | Синтез | Используется как историческая рамка |
| Recall: prompt/token/embedding | Да, как taxonomy | Полезная классификация способа применения LLM |
| Rank: generative/hybrid | Да, как taxonomy | Отражает степень отказа от DLRM |
| Single-/multi-stage training map | Да, как synthesis | Сведена в Table 1 |
| Inference/cold-start discussion | Синтез | Объединяет разрозненные engineering patterns |
| Scaling/data cleaning/one model for all | Авторская позиция | Research agenda, а не экспериментально доказанный результат |

### Как правильно интерпретировать доказательства survey

Survey смешивает четыре уровня evidence:

- результаты публичных datasets из исходных работ;
- proprietary offline metrics;
- online lifts, часто без абсолютной базы и confidence intervals;
- концептуальные ожидания авторов survey.

Их нельзя складывать в общий рейтинг. Например, 6.4% роста exposure мультимодальной модели, HR@K semantic retriever и CTR lift ranking model измеряют разные stages и business objectives. Корректный вывод статьи — «существует несколько успешно опробованных design patterns», а не «LLM-based GR гарантированно превосходит DLR на X процентов».

## 1. Проблематика и эволюция парадигм

### 1.1. Что решает survey

Область GR развивалась быстрее, чем устоялись термины и протоколы сравнения. Одинаковым словом *generative recommendation* называют prompt-only выдачу названий, next-item prediction по ID, генерацию semantic IDs, LLM embeddings внутри DLRM и end-to-end генерацию ранжированного списка. Survey создаёт taxonomy и обсуждает, как эти варианты встраиваются в real-world систему.

### 1.2. MLR → DLR → GR

| Парадигма | Представление | Выход | Сильная сторона | Основное ограничение |
|---|---|---|---|---|
| MLR | ручные признаки, user–item matrix | score | дёшево и интерпретируемо | sparsity, cold start, feature engineering |
| DLR | embeddings + engineered crosses | CTR/relevance score | нелинейные interactions, зрелый serving | сложный каскад, objective mismatch, слабое scaling |
| GR | токены поведения, semantic IDs, text/multimodal embeddings | token/item/list | единый sequence objective, transfer и scaling | decoding cost, invalid IDs, alignment и governance |

Каскад эффективен: из порядка $10^6$ items recall оставляет $10^4$, pre-rank — $10^3$, rank — $10^2$. Но ошибка ранней стадии необратима, а каждая стадия оптимизирует proxy objective. End-to-end GR обещает убрать error propagation и objective mismatch, но должна сохранить стоимость каскада — главный нерешённый компромисс.

### 1.3. Почему «LLM» здесь шире текстовой модели

Survey использует LLM-based GR для нескольких классов:

- pretrained text/multimodal LLM как frozen reasoner или encoder;
- Transformer-подобная большая autoregressive модель, обученная на action/item tokens;
- гибрид: LLM representation поступает в обычный DLRM;
- end-to-end decoder генерирует item ID, semantic ID или список.

Поэтому наличие GPT-подобного decoder ещё не даёт world knowledge: если модель обучена только на item IDs, это large sequence model, но её семантика целиком collaborative.

## 2. Таксономия методов

### 2.1. Recall

**Prompt-based.** История и профиль превращаются в natural-language prompt, frozen LLM сначала резюмирует интерес, затем предлагает категории/items (LLMTreeRec). Плюсы — zero/few-shot и объяснения; минусы — hallucination, нестабильность prompt, дорогой decoder и необходимость entity resolution.

**Token-based.** Поведение кодируется токенами, задача — next-token/item prediction. HSTU использует raw item/action stream; KuaiFormer иерархически сжимает раннюю/среднюю/свежую историю; URM добавляет objective tokens; TIGER/RQ-VAE заменяет огромный item vocabulary несколькими semantic code tokens.

**Embedding-based.** LLM/vision encoder строит content embeddings, которые поступают в DSSM, SASRec или другой retriever (MoRec). Это самый безопасный migration path: knowledge улучшает cold start, а ANN-serving остаётся прежним.

### 2.2. Ranking

**Generative architecture.** Последовательность напрямую порождает action/CTR target: GR/HSTU, action-oriented GenRank, dual-flow DFGR. Плюс — общая история и scaling; риск — scoring десятков тысяч candidates и calibration generative logits.

**Hybrid integration.** Frozen/trainable LLM строит признаки для DLRM. LEARN сохраняет open-world knowledge и адаптирует его twin-tower; HLLM использует ITEM LLM и USER LLM; SRP4CTR передаёт sequential knowledge через cross-attention; MTGR возвращает raw/cross features в HSTU-like generative backbone. Гибрид менее элегантен, зато проще внедряется и показывает, что отказ от production crosses может ухудшать качество.

### 2.3. End-to-end recommendation

OneRec генерирует session-wise list и затем проходит iterative reward-model/DPO alignment; OneSug применяет list-wise preference alignment; EGA-V1/V2 оптимизируют auction/business reward для рекламы. Академические S-DPO, RosePO и SPRec усложняют preference negatives. Главная идея: next-item CE учит правдоподобие, но не обязательно порядок, diversity или долгосрочный reward; post-training с preference signal приближает model policy к продуктовой цели.

## 3. Общая архитектура LLM-based GR

Survey не задаёт единую архитектуру, поэтому ниже — нормализованный pipeline, полезный для сравнения работ.

### 3.1. Item tokenizer

**Atomic ID:** один item = один token. Простой lookup, но vocabulary $|\mathcal I|$ может быть миллиардным, новые items требуют расширения таблицы, а похожие items не делят статистику.

**Text ID:** title/description токенизируются обычным tokenizer. Есть world knowledge и cold start, но строка не гарантирует однозначный catalog entity.

**Semantic ID:** content/collaborative vector $e_i\in\mathbb R^D$ квантуется последовательностью кодов. Для residual quantization:

$$
r_i^{(0)}=e_i,\quad c_i^{(m)}=\arg\min_{k<K}\|r_i^{(m)}-C_{m,k}\|_2^2,
$$

$$
r_i^{(m+1)}=r_i^{(m)}-C_{m,c_i^{(m)}},\qquad
\hat e_i=\sum_{m=0}^{M-1} C_{m,c_i^{(m)}}.
$$

Один item становится tuple $(c_i^{(0)},\ldots,c_i^{(M-1)})$. Vocabulary сокращается с $|\mathcal I|$ до $M\cdot K$, а общие префиксы переносят сигнал между похожими items. Цена — quantization error и collision: разные items могут получить одинаковый tuple. Production-решение требует collision suffix или tie-breaker.

### 3.2. Autoregressive model и размерности

Пусть batch $B$, длина flattened context $L$, hidden size $d$, heads $H$, code depth $M$, codebook size $K$. История из $S$ items превращается примерно в $L=S\cdot M$ code tokens плюс task/category/prompt tokens.

$$
X_0=E_{token}[x]+E_{level}[\ell]+E_{pos}[p]in\mathbb R^{B\times L\times d}.
$$

Decoder-only Transformer с causal mask даёт $H_L\in\mathbb R^{B\times L\times d}$. На каждом из $M$ шагов output head предсказывает один из $K$ кодов target item:

$$
p(c_m\mid x_{history},c_{<m})=\operatorname{softmax}(W_mh_m+b_m).
$$

Генерация semantic ID — не natural-language generation: decoder должен пройти допустимый путь catalog tree и затем разрешить код в реальный item.

### 3.3. Training objectives из survey

| Семейство/пример | Target | Objective |
|---|---|---|
| HSTU, URM | next item/action | cross-entropy; часто sampled negatives |
| KuaiFormer | user–item similarity | in-batch contrastive loss |
| MTGR | CTR | binary cross-entropy |
| QARM, HLLM, LEARN, LUM | representation alignment | InfoNCE/contrastive, затем downstream DLR |
| OneRec | next item/list + preferences | CE pretraining, iterative DPO |
| OneSug | query/item sequence | list-wise preference alignment |
| EGA-V1/V2 | CTR/item sequence | auction/reward-based alignment |

**Token CE:**

$$
\mathcal L_{CE}=-\frac1M\sum_{m=1}^{M}\log \pi_\theta(c_m^+\mid h,c_{<m}^+).
$$

**InfoNCE** для user representation $u$ и target content vector $v^+$:

$$
\mathcal L_{NCE}=-\log\frac{\exp(\operatorname{sim}(u,v^+)/\tau)}{\sum_{v\in\mathcal B}\exp(\operatorname{sim}(u,v)/\tau)}.
$$

**DPO** для preferred $y^+$ и rejected $y^-$ относительно frozen reference $\pi_{ref}$:

$$
\mathcal L_{DPO}=-\log\sigma\left(\beta\left[\log\frac{\pi_\theta(y^+|h)}{\pi_{ref}(y^+|h)}-\log\frac{\pi_\theta(y^-|h)}{\pi_{ref}(y^-|h)}\right]\right).
$$

DPO не заменяет корректный logged-feedback design: exposure bias означает, что unexposed item — не доказанный negative.

## 4. Training pipeline и serving

### Single-stage

Одна фаза оптимизирует next-item/action CE, in-batch contrastive retrieval или CTR BCE. Проще и дешевле, но objective остаётся proxy для list quality и долгосрочного reward.

### Multi-stage

- **Representation-based:** contrastive pretraining → embeddings как features DLRM. Хороший migration path и дешёвый online serving.
- **Model-based:** CE pretraining → preference/reward alignment. Потенциально end-to-end, но чувствительно к quality reward model, off-policy logs и reward hacking.

### Как ускоряют inference

1. Сжимают историю: action-oriented sequence, hierarchical aggregation, real/fake flow.
2. Меняют attention: HSTU pointwise aggregation, clustered attention/RecFormer.
3. Сжимают output: semantic IDs, RQ-VAE, multi-token prediction.
4. Избегают полного Top-K: probabilistic sampling/matrix decomposition; при ranking — M-FALCON-like shared history и candidate masks.
5. На практике добавляются KV-cache, prefix cache, quantization, speculative decoding и ANN shortlist, хотя конкретный набор зависит от модели.

## 5. Результаты, метрики и валидация

### 5.1. Чего в статье нет

GR-LLMs **не проводит собственного эксперимента** и не содержит единой таблицы quality/latency по общему dataset и split. Table 1 сравнивает objectives/strategies, Table 2 — modalities/targets. Поэтому нельзя приписывать survey SOTA или вычислять средний uplift по несопоставимым цитатам.

Из отдельных кейсов авторы отмечают deployment KuaiFormer на платформе с 400 млн DAU, online gains OneRec, 6.4% рост exposure у NoteLLM-2 и результаты TALKPLAY на Million Playlist Dataset. Без общих baseline, traffic definition, variance и latency это evidence of feasibility, а не строгий рейтинг методов.

### 5.2. Какая offline проверка нужна

**Split:** global temporal cutoff или leave-last-session-out. Для cold-item evaluation items должны появиться только после cutoff, а content encoder/tokenizer обучается без их interaction labels. Vocabulary, codebooks, popularity и hard negatives fit-ятся только на train.

**Retrieval:** full-catalog Recall/HR@K, NDCG@K, MRR; sampled-negative результат сообщается отдельно. Semantic-ID generation дополнительно требует valid-code rate, collision rate и exact-item resolution rate.

**Ranking:** log-loss/Normalized Entropy, AUC/GAUC и calibration error. AUC оценивает порядок, log-loss — качество вероятностей; GAUC не даёт heavy users полностью доминировать.

**List и продукт:** NDCG, coverage, diversity, novelty, serendipity, popularity bias, repetition rate, constraint violation. Для generative list важны position duplicates и catalog validity.

**Система:** p50/p95/p99 latency, tokens/s, QPS/GPU, HBM, cache hit, energy/request и денежная стоимость на тысячу рекомендаций. Quality без serving budget — неполное сравнение.

**Онлайн:** CTR/watch time/conversion вместе с retention, hide/report, diversity и safety guardrails; SRM-check, заранее выбранная primary metric и доверительный интервал.

### 5.3. Leakage и confounders

- Text LLM мог видеть reviews, titles и popularity будущего test item при pretraining; это transductive world knowledge, его нужно декларировать.
- Random interaction split переносит будущие предпочтения пользователя в train.
- RQ codebook на всём каталоге может использовать test-item collaborative embeddings.
- In-batch «negative» может быть релевантным, но не показанным item; exposure propensity не равна preference.
- Candidate sampling резко завышает HR/NDCG относительно full catalog.
- Offline gain content model может объясняться более свежими metadata, а не архитектурой.

## 6. Критический анализ survey

### Что сделано хорошо

- Полезное разделение по production stages: recall, rank, end-to-end.
- Одновременно рассмотрены architecture, training pipeline, inference, cold start и multimodality.
- Авторы не сводят область к prompt engineering: token/embedding/hybrid методы занимают центральное место.
- Отмечена важность semantic IDs, cross-features, preference alignment и efficiency — именно там ломается industrial deployment.
- Future directions конкретны: scaling, data cleaning и one-model-for-all.

### Что ограничивает выводы

1. **Нет systematic-review methodology:** не описаны поисковые базы, запросы, период, inclusion/exclusion и оценка качества работ.
2. **Нет общей формализации taxonomy:** «LLM», «GR» и «end-to-end» охватывают слишком разные вычислительные графы; boundary местами размыта.
3. **Нет нормализованной evidence table:** datasets, catalog size, negatives, parameter count, latency и absolute online metrics не сведены.
4. **Сильная индустриальная selection bias:** многие результаты закрыты, опубликованы авторами систем и не имеют независимой репликации.
5. **Мало safety/governance:** privacy, memorization, filter bubbles, prompt injection через item text, explanation faithfulness, fairness и environmental cost почти не разобраны.
6. **Недостаточно retrieval-constrained decoding:** hallucinated/nonexistent items и collision semantic IDs — центральная проблема, но survey не даёт единого evaluation protocol.
7. **Быстрое устаревание:** сами авторы признают, что динамичная область и лимит страниц не позволяют охватить все новые работы и технические детали.

## 7. Применимость в индустрии

### Матрица выбора

| Ситуация | Практичный первый выбор | Почему |
|---|---|---|
| Много cold items, богатые descriptions/images | frozen content encoder → two-tower/DLRM | быстрый ANN, контролируемый rollout |
| Длинные action histories, мало metadata | token GR/HSTU-like encoder | collaborative sequence signal |
| Миллионы+ items, нужен generative retrieval | semantic IDs + constrained decoding | малый vocabulary, валидный каталог |
| Зрелый DLRM и строгий p99 SLA | hybrid LLM features | небольшой serving risk |
| Нужен единый ranked list и есть reward logs | CE pretrain + preference alignment | list objective, но нужен сильный guardrail |
| Малые данные/CPU serving | SASRec/two-tower/GBDT | LLM-scale не окупится |

### Production architecture

Разумная highload схема остаётся условно двухскоростной: offline/nearline LLM кодирует metadata и обновляет semantic/item embeddings; дешёвый online sequence model генерирует semantic prefixes или query vector; prefix trie/ANN возвращает только существующие items; компактный ranker применяет policy/business constraints; большой LLM вызывается асинхронно для explanation или редких сложных запросов.

Обязательны fallback на популярное/обычный retriever, versioned tokenizer/codebook, deterministic entity resolution, incremental index update, tombstone удалённых items, quota по latency/cost, moderation metadata и мониторинг invalid/duplicate rate.

### Open questions из статьи

- **Scaling:** эффекты выше 0.x–1.xB в большинстве новых GR работ ещё недостаточно проверены; длинная история конфликтует с online latency.
- **Data cleaning:** у behavioral sequence нет аналога grammaticality; нужны quality scoring, discriminative filtering и data curriculum.
- **One model for all:** objective/task tokens могут объединить recommendation, search, scenarios и targets, но конфликт градиентов и единая failure domain растут.
- **Multimodality/cold start:** world knowledge помогает, но требует domain adaptation, freshness и защиты от contamination.

## 8. Дополнительное улучшение и PoC

### Почему выбран именно такой PoC

Semantic-ID decoder пересекает наиболее важные ветви survey: token-based recall, content knowledge, catalog compression и generative output. На синтетическом каталоге можно честно проверить весь путь без внешнего pretrained LLM и многогигабайтного dataset.

### Pipeline ноутбука

1. Создать item catalog с category/content embeddings и пользовательские sessions; последние взаимодействия оставить для temporal validation.
2. Реализовать residual vector quantization несколькими Torch k-means codebooks; измерить reconstruction error и collisions.
3. Превратить history items в semantic tokens с level embeddings; добавить category/task prompt token.
4. Обучить маленький decoder-only Transformer на target code CE и InfoNCE alignment.
5. Показать DPO loss на preferred/rejected next items и выполнить короткую alignment stage относительно frozen reference.
6. Декодировать greedy и constrained beam search по catalog prefix trie.
7. Сравнить valid-item rate, HR@K, NDCG@K, coverage и latency с popularity baseline.
8. Провести smoke tests: shapes, causal invariance, finite gradients, trie не выдаёт несуществующий код.

### Наше улучшение: catalog-constrained beam search

На шаге $m$ разрешены только коды, продолжающие хотя бы один catalog tuple с уже выбранным prefix:

$$
\mathcal A(c_{<m})=\{k:\exists i\in\mathcal I,\ (c_i^{(0)},\ldots,c_i^{(m)})=(c_{<m},k)\}.
$$

Перед top-k logits всех $k\notin\mathcal A$ становятся $-\infty$. После $M$ шагов каждый beam соответствует существующему semantic ID. Это даёт 100% **code validity**, но не обязательно 100% exact-item uniqueness из-за collisions; collisions разрешаются популярностью/точным content similarity. В production вместо Python trie нужен компактный FSA/GPU mask или hierarchical index.

### Что PoC не доказывает

- Tiny Transformer не является LLM и не обладает внешним world knowledge.
- Torch k-means — учебный аналог residual quantization, не полноценный jointly trained RQ-VAE.
- Синтетический lift не переносится на индустрию.
- Один DPO step демонстрирует механику, а не доказывает alignment quality.
- CPU latency не моделирует GPU batching, KV-cache, network и ANN.

## 9. Вопросы для защиты

**Почему survey можно реализовать, если у него нет метода?** Нельзя честно реализовать несуществующий «метод GR-LLMs». Можно реализовать representative pipeline, явно указав, какие паттерны синтезированы из taxonomy, и проверить ключевые риски.

**Зачем semantic ID, если можно один item token?** Tuple из малых codebooks уменьшает output vocabulary и делит статистику между похожими items. Плата — больше decoding steps, collisions и необходимость constrained resolution.

**Почему CE недостаточно для ranking?** CE максимизирует likelihood следующего observed item; exposure bias и позиционные эффекты не равны предпочтению, а top-list utility может зависеть от порядка, diversity и долгосрочного reward. Отсюда DPO/reward alignment.

**Constrained decoding ухудшает novelty?** Он запрещает только несуществующие codes, но не новые комбинации внутри актуального catalog. Novelty относительно истории сохраняется; полностью новый item войдёт после обновления trie/codebook mapping.

**Что быстрее внедрить в highload?** Frozen LLM/content embeddings как side features. End-to-end autoregressive list generator имеет наибольший потенциальный выигрыш и наибольший serving/validation risk.

## 10. Итог

GR-LLMs полезен как карта design space, но не как источник единого SOTA-рецепта. Самый надёжный вывод survey: генеративный recommender — это сочетание представления item, sequence model, objective, constrained serving и evaluation, а не просто замена DLRM на большой Transformer. Индустриально разумно начинать с content embeddings или дополнительного retrieval source, измерять full-catalog temporal quality и только затем переходить к semantic decoding и preference-aligned end-to-end lists.
