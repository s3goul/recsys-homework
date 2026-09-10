# Actions Speak Louder than Words: разбор статьи и план воспроизведения

> Jiaqi Zhai et al., *Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations*, ICML 2024.  
> [Статья на arXiv](https://arxiv.org/abs/2402.17152) · [публикация PMLR](https://proceedings.mlr.press/v235/zhai24a.html) · [авторский код](https://github.com/facebookresearch/generative-recommenders)  


## 0. Короткий ответ: что именно предложено

Авторы Meta заменяют промышленный DLRM с сотнями ручных признаков на **Generative Recommender (GR)**: действия пользователя, объекты и медленно меняющиеся категориальные признаки объединяются в одну временную последовательность, а retrieval и ranking формулируются как causal sequential transduction. Ядро GR — **Hierarchical Sequential Transduction Unit (HSTU)**. Это не «обычный Transformer побольше»: HSTU использует ненормированную pointwise SiLU-attention, gate, относительный позиционно-временной bias и упрощённый блок из двух линейных проекций.

Главная инженерная идея статьи шире архитектуры: генеративное обучение амортизирует один проход по истории на много target-ов; Stochastic Length сокращает дорогие длинные истории; ragged kernels уменьшают padding; M-FALCON переиспользует вычисления истории при ranking большого числа кандидатов. Именно сочетание постановки, блока и системных оптимизаций позволило авторам сообщить о моделях до 1.5 трлн параметров и production A/B lift до 12.4%.

## Подробный обзор содержания статьи

### Паспорт публикации и заявленный масштаб

Работа подготовлена исследователями Meta AI и опубликована на ICML 2024. PDF занимает 26 страниц: основная аргументация завершается выводами и impact statement, после библиографии следует большое техническое приложение. Это важно для чтения: в первых разделах авторы формулируют новую парадигму и показывают основные результаты, а значительная часть определений, сравнений с академическими sequential recommenders, деталей Stochastic Length и M-FALCON вынесена в appendices.

Статья одновременно является работой трёх типов:

- **методологической** — ranking и retrieval переписываются как generative sequential transduction;
- **архитектурной** — предлагается новый encoder HSTU;
- **системной** — описываются sparsity, управление activation/embedding memory и amortized candidate scoring.

Из-за этого заголовок про «trillion-parameter transducers» отражает только верхний уровень результата. Центральный вопрос статьи сформулирован глубже: можно ли заменить многолетний стек DLRM одним масштабируемым последовательным представлением действий пользователя и при этом сохранить production latency?

### Как построена аргументация авторов

Статья следует цепочке «новая постановка → специализированная архитектура → оптимизация вычислений → многоуровневая проверка»:

1. Авторы показывают, почему добавление compute к DLRM не гарантирует улучшения и почему recommendation data нельзя механически обработать обычным LLM.
2. Затем они преобразуют heterogeneous feature space в единую временную ось и показывают, что обе основные стадии recommender system допускают causal generative formulation.
3. После смены постановки вводится HSTU, который должен лучше обычного Transformer работать с динамическими ID и интенсивностью повторяющихся действий.
4. Далее авторы устраняют практические ограничения длинных последовательностей: padding, $N^2$ attention, activation memory, огромные embedding tables и повторное scoring одной истории для тысяч candidates.
5. Наконец, гипотезы проверяются последовательно на synthetic data, публичных datasets, закрытом streaming benchmark, system benchmarks и online A/B.

Такая структура сильнее типичной статьи «новый блок + одна offline-таблица»: каждый эксперимент отвечает на отдельный тезис. Однако наиболее важные production результаты остаются закрытыми и не могут быть независимо воспроизведены.

### Раздел 1 — Introduction

Введение начинает не с архитектуры, а с диагноза индустрии. DLRM использует огромные high-cardinality embeddings, численные счётчики, cross-features и несколько специализированных сетей, однако quality часто насыщается при увеличении вычислений. **Figure 1** показывает рост годового training compute у DLRM и внедрённых GR-моделей. Это иллюстрация масштаба deployment, а не контролируемый causal experiment: точки соответствуют разным поколениям систем.

Авторы выделяют три препятствия для «LLM-момента» в рекомендациях:

- признаки не образуют готовую дискретную структуру, подобную тексту;
- словарь ID имеет миллиардную размерность и меняется в streaming режиме;
- событий настолько много, что наивный full-history Transformer вычислительно невозможен.

Затем вводится ключевой тезис статьи: **user actions — самостоятельная modality для generative modeling**. В отличие от работ, которые превращают item description в текстовый prompt, здесь основной носитель информации — реальные последовательности показов и реакций. В конце Introduction перечислены четыре результата: GR formulation, HSTU, M-FALCON/system efficiency и демонстрация scaling law до 1.5T parameters.

### Раздел 2 — Recommendation as Sequential Transduction Tasks

Это концептуально главный раздел статьи. **Section 2.1** и **Figure 2** показывают переход от feature snapshot к temporal representation. В DLRM каждый impression порождает самостоятельный пример с актуальным снимком сотен признаков. В GR item/action events образуют основной поток; редкие изменения языка, региона или подписок сжимаются и добавляются в него; часто меняющиеся численные статистики предлагается восстанавливать моделью из causal history.

Рисунок 2 полезно читать сверху вниз. Слева одна и та же история многократно материализуется в DLRM training examples. Справа последовательность кодируется один раз, а supervision возникает в нескольких позициях. Поэтому унификация признаков одновременно меняет **данные**, **задачу** и **стоимость обучения**.

В **Section 2.2** авторы вводят формальный transduction mapping $x_i\mapsto y_i$ и в **Table 1** задают две схемы:

- в retrieval положительная реакция превращает следующий item в generative target, а negative/no-label позиции маскируются;
- в ranking items и actions чередуются, так что representation позиции $\Phi_i$ уже target-aware, но ещё не содержит ответ $a_i$.

Это отличие от обычного SASRec существенно. SASRec по прошлым положительным items предсказывает следующий item; GR моделирует совместный поток того, **что система показала**, и того, **как пользователь отреагировал**. В приложении B авторы записывают цель ещё шире как моделирование joint distribution $p(\Phi_0,a_0,\ldots,\Phi_{n-1},a_{n-1})$.

**Section 2.3** объясняет generative training. Один causal pass даёт loss сразу в нескольких временных точках и амортизирует encoder cost по targets. Sampling пользователей с вероятностью, обратно пропорциональной длине истории, не позволяет heavy users полностью определять compute budget. Этот раздел связывает новую probabilistic formulation с практической возможностью обучаться на гораздо большем числе действий.

### Раздел 3 — HSTU и системный co-design

Раздел 3 начинается с трёх уравнений HSTU и **Figure 3**, где сложный DLRM слева заменён стеком одинаковых HSTU blocks справа. Авторы утверждают, что один блок способен объединить функции трёх классов DLRM-компонентов:

- attention pooling выполняет feature extraction;
- `A(X)V(X) ⊙ U(X)` моделирует feature interactions;
- gate `U(X)` и output projection выполняют conditional transformation, частично аналогичную MoE.

В **Section 3.1** pointwise aggregated attention мотивируется интенсивностью предпочтения. Softmax отвечает преимущественно на вопрос «какие прошлые элементы важнее относительно других», но нормировка мешает различить один релевантный event и сотню похожих events. Synthetic Dirichlet-process experiment в Table 2 специально моделирует non-stationary vocabulary и rich-get-richer pattern. Удаление softmax даёт большой выигрыш, поэтому synthetic test служит проверкой inductive bias, а не общим benchmark recommendation quality.

**Section 3.2** переходит от model quality к sparsity. Реальные batch-и состоят из историй разной длины, а padded dense attention платит за максимальную длину. Авторы используют ragged grouped GEMMs и предлагают Stochastic Length: длинные histories случайно сокращаются с управляемой степенью $\alpha$. Смысл SL — не просто augmentation. Он непосредственно изменяет распределение training cost, позволяя чаще обучаться на больших maximum context без постоянной оплаты полного квадрата длины.

**Section 3.3** разбирает память. HSTU оставляет две крупные linear projections вместо стандартной комбинации Q/K/V/O и двух FFN-проекций, не хранит attention matrix для backward и допускает recomputation. Отдельно рассматриваются embedding tables: для словаря 10 млрд ID, размерности 512 и Adam в fp32 наивная оценка достигает 60 TB. Row-wise AdamW и вынос optimizer state в DRAM уменьшают HBM cost с 12 до 2 bytes на embedding value. Это одна из причин, почему «1.5T параметров» нельзя интерпретировать как 1.5T dense Transformer weights.

**Section 3.4** вводит M-FALCON. Основная идея — history одна, а ranking candidates много. Candidates объединяются в microbatch, получают одну logical position, видят history, но не видят друг друга. Cache сохраняет history states между microbatches. В результате статья рассматривает serving не как обычное token-by-token language decoding, а как специализированное parallel target scoring.

### Раздел 4 — лестница экспериментов

Экспериментальная часть намеренно разделяет проверяемые утверждения.

**Section 4.1.1** спрашивает: полезен ли HSTU как sequential encoder вне закрытой системы? Для этого используются MovieLens-1M, MovieLens-20M и Amazon Books, full-corpus HR/NDCG и сильный SASRec recipe. HSTU и HSTU-large улучшают результаты, особенно на sparse Books.

**Section 4.1.2** спрашивает: сохраняется ли эффект при one-pass industrial streaming? Авторы фиксируют небольшие configurations encoder-а, сравнивают Transformer, Transformer++, HSTU и ablations на ranking/retrieval задачах и отмечают нестабильность Transformer ranking training. Здесь показывается вклад pointwise attention и relative positional-temporal bias.

**Section 4.2** проверяет стоимость. Figure 4 связывает $\alpha$, sparsity и изменение NE. Figure 5 сравнивает wall-clock HSTU с FlashAttention2 Transformer при длинах 1024–8192. Это именно encoder/system benchmark при заданной configuration, а не end-to-end latency всей рекомендательной платформы.

**Section 4.3** отвечает на главный production вопрос. Tables 6–7 сравнивают GR с зрелыми DLRM для retrieval и ranking; Figure 6 показывает QPS при M-FALCON; Figure 7 сопоставляет quality с training compute. Авторы подчёркивают характерный crossover: в low-compute режиме DLRM может быть сильнее за счёт ручных признаков, но затем насыщается, тогда как GR продолжает улучшаться. На этом наблюдении основано заявление о scaling law для рекомендаций.

### Разделы 5–6 и Impact Statement

Related Work отделяет предложенный подход от трёх соседних направлений: классических sequential recommenders, generative retrieval по sub-item tokens и LLM-for-recommendation через text/world knowledge. Авторы прямо отмечают, что текстовые LLM-подходы в low-data regime перспективны, но на large-scale collaborative tasks тогда ещё не превосходили сильные методы на MovieLens. Это помогает не смешивать HSTU/GR с prompt-based рекомендациями.

В Conclusion повторяется тезис о foundation model для recommendation/search/ads. В Impact Statement авторы предполагают, что отказ от большого числа вручную собранных признаков может улучшить privacy, а long-horizon sequential objectives — уменьшить давление в сторону clickbait. Это **ожидаемые эффекты**, а не измеренные результаты статьи: privacy leakage, long-term welfare и carbon footprint отдельными экспериментами не оценивались.

### Что содержат приложения A–H

Приложения занимают почти половину PDF и существенно дополняют основной текст:

| Appendix | Содержание | Зачем читать |
|---|---|---|
| A | Полная таблица обозначений и shapes `Q, K, V, U, A` | Уточняет, что `d_qk` и `d_v` могут отличаться, а bias может разделяться между heads |
| B | История GRU4Rec/SASRec/BERT4Rec и production DLRM; формальное сравнение постановок | Объясняет, почему GR не равен обычному next-item recommender |
| C | Генератор synthetic Dirichlet-process data | Показывает, какую именно гипотезу проверяет pointwise attention |
| D | Дополнительные public baselines GRU4Rec и BERT4Rec | Делает сравнение шире одной версии SASRec |
| E | Описание сильных production DLRM baselines | Фиксирует DIN/DCN/MMoE, число feature groups и ограничение конфиденциальности |
| F | Варианты subsequence selection, sparsity на 30/60/90 днях, length extrapolation | Показывает, что weighted sampling лучше greedy/random и что SL сравнивался с RoPE/extrapolation |
| G | Sparse grouped GEMMs и fused relative bias | Объясняет источник speedup и recomputation trade-off |
| H | Псевдокод M-FALCON, masks, microbatching, cache и throughput | Даёт implementable описание candidate isolation |

Особенно важен Appendix B: авторы сопоставляют GR с GRU4Rec, SASRec, BERT4Rec, DIN, BST, TWIN и TransAct по входу, target-awareness, architecture и training procedure. Без этого приложения легко ошибочно свести вклад статьи к «ещё одному Transformer для next-item prediction».

### Навигатор по ключевым рисункам и таблицам

| Элемент статьи | Что показывает | Корректная интерпретация |
|---|---|---|
| Figure 1 | Рост training compute DLRM/GR по годам | Мотивация и deployment history, не доказательство качества |
| Figure 2 | Feature snapshots DLRM против единого event stream GR | Главная смена data/training representation |
| Table 1 | Ranking и retrieval как transduction | Формальное ядро статьи |
| Figure 3 | DLRM modules против стека HSTU | Архитектурное упрощение, а не равенство функций при любом масштабе |
| Table 2 | Softmax/pointwise attention на synthetic stream | Проверка inductive bias при non-stationary vocabulary |
| Tables 4–5 | Public и industrial encoder comparisons | Quality и ablations до full production scale |
| Figures 4–5 | SL quality/sparsity и encoder efficiency | Cost-quality trade-off HSTU |
| Tables 6–7 | Offline/online GR против DLRM | Главный product evidence, но на закрытых задачах |
| Figure 7 | Quality против compute | Эмпирический scaling trend, не универсальный закон |

### Утверждения статьи и соответствующие доказательства

| Утверждение авторов | Evidence в статье | Что остаётся неизвестным |
|---|---|---|
| Pointwise attention лучше подходит dynamic action streams | Synthetic DP test и industrial ablations | Поведение на других типах drift и noise |
| HSTU сильнее обычных sequential encoders | MovieLens/Amazon + GRU4Rec/BERT4Rec appendix | Variance по seeds и единый tuning budget раскрыты частично |
| Generative training эффективнее impression-level | Анализ сложности и session-level emission | Полный end-to-end accounting feature pipeline |
| GR превосходит production DLRM | Offline tables и online A/B | Определения E/C metrics, confidence intervals, traffic split |
| GR масштабируется лучше DLRM | Figure 7 на трёх порядках compute | Одновременно меняются width, depth, history и negatives |
| Сложную модель можно обслуживать в том же budget | HSTU benchmark и M-FALCON QPS | Переносимость без специализированных kernels/caches |

## 1. Проблематика и контекст

### 1.1. Почему классический промышленный DLRM упирается в потолок

Типичный DLRM получает тысячи разнородных сигналов:

- high-cardinality ID: item, creator, topic, community, locale;
- dense-статистики: CTR, частоты, отношения, экспоненциально затухающие счётчики;
- ручные cross-features и отдельные модули feature interaction;
- разные головы и MoE/PLE для нескольких бизнес-целей.

Это даёт сильный inductive bias при малом compute, но создаёт четыре проблемы: признаки дорого проектировать и синхронизировать; признаки и embedding-таблицы быстро устаревают; независимые impression-level примеры многократно пересчитывают одну историю; увеличение FLOPs DLRM в экспериментах статьи быстро перестаёт улучшать качество.

### 1.2. Эволюция идеи

1. Matrix factorization и two-tower учат статические user/item embeddings, но слабо моделируют порядок событий.
2. GRU4Rec, SASRec и BERT4Rec превращают **последовательность item ID** в задачу next-item prediction. Это сильная академическая постановка, но она обычно теряет тип действия, время, контекст и target-aware ranking.
3. DIN/TWIN-подобные production-модели добавляют target-aware attention поверх ручных признаков, но остаются частью сложного DLRM-конвейера.
4. GR последовательнизирует почти всё важное и решает retrieval и ranking одним causal encoder-ом. При длине истории, стремящейся к бесконечности, модель в принципе может восстановить счётчики из сырых событий, хотя на конечной истории это лишь приближение.

### 1.3. Три заявленных вызова

- У heterogeneous features нет естественной структуры, аналогичной словам и предложениям.
- Каталог содержит до миллиардов ID и непрерывно меняется; полный softmax и target-aware scoring дороги.
- Пользовательских событий на несколько порядков больше, чем токенов в типичном LLM pretraining, а история может достигать $10^5$ событий.

## 2. Данные: как DLRM-признаки становятся последовательностью

Пусть взаимодействие $i$ состоит из объекта $\Phi_i$, действия $a_i$ и времени $t_i$. Основной поток строится из item/action-событий. Медленно меняющиеся категориальные ряды — язык, город, подписки — сжимаются: в каждом непрерывном сегменте одинакового состояния сохраняется первая запись, затем потоки merge-sort-ятся по времени.

Часто меняющиеся dense counters напрямую не материализуются. Предполагается, что достаточно выразительная causal target-aware модель научится их приближать по исходным категориям и действиям. Это экономит feature engineering, но переносит цену в длину истории и capacity модели.

### Ranking как transduction

В упрощённом виде вход:

$$
x=(\Phi_0,a_0,\Phi_1,a_1,\ldots,\Phi_{n-1},a_{n-1}),
$$

а supervision определена только на позициях item:

$$
y=(a_0,\varnothing,a_1,\varnothing,\ldots,a_{n-1},\varnothing).
$$

Перед предсказанием $a_i$ encoder уже видит target $\Phi_i$ и всю допустимую историю. Поэтому один causal pass вычисляет target-aware representation каждого impression без утечки будущего.

### Retrieval как transduction

Пара $(\Phi_i,a_i)$ рассматривается как один engagement token, а положительные действия задают следующий релевантный item:

$$
p(\Phi_{i+1}^{+}\mid \Phi_{\le i},a_{\le i}).
$$

Если действие отрицательное, target может быть $\varnothing$. На практике миллиардный softmax заменяют sampled softmax / negatives, MIPS или иерархическим декодированием.

### Почему generative training дешевле

При impression-level обучении история длины $n_i$ кодируется для многих targets отдельно. Для self-attention суммарная цена в обозначениях статьи имеет порядок

$$
\sum_i n_i(n_i^2d+n_id_{ff}d).
$$

В генеративном режиме один проход даёт loss на многих позициях. При sampling пользователя с вероятностью $s_u(n_i)\propto 1/n_i$ авторы снижают порядок до $O(N^2d+Nd^2)$ вместо $O(N^3d+N^2d^2)$, где $N=\max_i n_i$.

## 3. Архитектура HSTU

### 3.1. Один слой и размерности

Пусть $X\in\mathbb{R}^{B\times N\times d}$, число heads — $H$, размер головы $d_h=d/H$.

**Pointwise projection** одной fused-линейной операцией:

$$
[U,V,Q,K]=\mathrm{Split}(\mathrm{SiLU}(XW_1+b_1)).
$$

В учебной реализации $U,V,Q,K\in\mathbb{R}^{B\times H\times N\times d_h}$. В production конфигурации размеры value/gate и query/key не обязаны совпадать.

**Spatial aggregation**:

$$
S_{b,h,i,j}=\frac{Q_{b,h,i,:}K_{b,h,j,:}^{\top}}{\sqrt{d_h}}+r^{h}_{p(i-j),t(t_i-t_j)},
$$

$$
A=\mathrm{SiLU}(S)\odot M_{causal},\qquad Z=AV.
$$

Ключевое отличие от Transformer — **нет softmax по $j$**. Softmax стирает абсолютную массу релевантной истории: одинаковый результат может получиться от одного сильного и ста умеренных прошлых сигналов. Ненормированная сумма сохраняет intensity, а LayerNorm после pooling стабилизирует масштаб.

**Pointwise transformation и residual**:

$$
Y=X+W_2\left(\mathrm{LayerNorm}(Z)\odot U\right)+b_2.
$$

Gate $U$ делает блок близким по смыслу к SwiGLU/MoE gating. В статье $f_1$ и $f_2$ — по одной linear layer: меньше activation memory и легче fusion, чем attention + отдельный FFN в Transformer.

### 3.2. Relative attention bias

$r_{p,t}$ кодирует одновременно относительную позицию и реальный time gap. Позиция отвечает за recency в числе событий, время различает, например, десять кликов за минуту и десять кликов за месяц. В ноутбуке реализованы learned positional buckets и head-specific штраф к $\log(1+\Delta t)$; это прозрачная аппроксимация идеи, а не копия закрытого production kernel.

### 3.3. Loss

Для `K_a` действий ranking head выдаёт вектор логитов `z_act[i]` размерности `K_a`, а retrieval head — `z_item[i]` размерности `|V|`. Mask исключает padding и позиции без target:

$$
\mathcal L_{rank}=-\frac{1}{|M_a|}\sum_{i\in M_a}\log p(a_i\mid x_{\le i}),
$$

$$
\mathcal L_{ret}=-\frac{1}{|M_r|}\sum_{i\in M_r}\log p(\Phi_{i+1}^{+}\mid x_{\le i}),
$$

$$
\mathcal L=\mathcal L_{rank}+\lambda\mathcal L_{ret}.
$$

Статья использует multi-task production objectives; точные веса и определения закрыты. Поэтому PoC использует понятные CE-losses, а не выдаёт их за точную production-конфигурацию.

## 4. Системные оптимизации

### 4.1. Ragged attention и Stochastic Length

Истории крайне неравномерны, поэтому padding до общего $N$ тратит память впустую. Авторы применяют fused ragged kernels. Дополнительно Stochastic Length (SL) случайно прореживает длинные последовательности так, чтобы ожидаемая attention-сложность стала $O(N^\alpha d)$, $1<\alpha\le 2$. При $\alpha=1.6$ и $N=4096$ в статье часто остаётся 776 токенов; 64–84% sparsity почти не ухудшает Normalized Entropy.

В ноутбуке сохранены последние события и случайная часть далёкой истории. Это **концептуальный regularizer**, не буквальная реализация формулы SL и не GPU speedup: dense PyTorch всё равно строит квадратную матрицу сокращённой последовательности.

### 4.2. M-FALCON для ranking

При $m$ кандидатах наивный target-aware encoder $m$ раз пересчитывает одну историю. M-FALCON объединяет $b_m$ кандидатов с общей history, меняет mask и relative bias так, чтобы кандидаты не видели друг друга, и переиспользует history/KV:

$$
O(b_mn^2d)\rightarrow O((n+b_m)^2d)\approx O(n^2d),\quad b_m\ll n.
$$

Все $m$ объектов обрабатываются микробатчами. В PoC проверяется важный инвариант: score кандидата в microbatch совпадает с его score при одиночном запуске (в `eval()`), то есть нет cross-candidate leakage.

## 5. Эксперименты и результаты статьи

### 5.1. Схемы валидации

| Режим | Данные | Протокол | Метрики |
|---|---|---|---|
| Synthetic streaming | Dirichlet Process, non-stationary vocabulary | один проход | HR@10, HR@50 |
| Public sequential | MovieLens-1M, MovieLens-20M, Amazon Books | full shuffle, multi-epoch; тот же setup, что SASRec (2023) | HR@K, NDCG@K по всему корпусу |
| Industrial streaming | закрытые данные Meta, более 100 млрд DLRM-equivalent examples | one-pass, streaming | ranking: Normalized Entropy; retrieval: log-perplexity / HR; online A/B |
| Efficiency | H100, `N = 1024…8192` | одинаковые базовые размеры | wall-clock, HBM, QPS |

**Физический смысл метрик.** HR@K отвечает, попал ли релевантный item в top-K, но не различает позиции внутри K. NDCG@K дисконтирует низкие позиции и потому лучше отражает видимость верхушки выдачи. Perplexity измеряет уверенность next-item distribution. Normalized Entropy — log-loss, нормированный на энтропию label: меньше — лучше, а сравнение устойчивее к разной base rate. Онлайн engagement/conversion — конечная бизнес-проверка, но статья не раскрывает определения задач E/C.

### 5.2. Public datasets: HSTU-large против SASRec

| Dataset | HR@10 | HR@50 | HR@200 | NDCG@10 | NDCG@200 |
|---|---:|---:|---:|---:|---:|
| ML-1M, SASRec | .2853 | .5474 | .7528 | .1603 | .2498 |
| ML-1M, HSTU-large | **.3294 (+15.5%)** | **.5935 (+8.4%)** | **.7839 (+4.1%)** | **.1893 (+18.1%)** | **.2771 (+10.9%)** |
| ML-20M, SASRec | .2906 | .5499 | .7655 | .1621 | .2521 |
| ML-20M, HSTU-large | **.3567 (+22.8%)** | **.6149 (+11.8%)** | **.8076 (+5.5%)** | **.2106 (+30.0%)** | **.2971 (+17.9%)** |
| Books, SASRec | .0292 | .0729 | .1400 | .0156 | .0350 |
| Books, HSTU-large | **.0469 (+60.6%)** | **.1066 (+46.2%)** | **.1876 (+33.9%)** | **.0257 (+65.8%)** | **.0508 (+45.1%)** |

HSTU меньшего размера тоже превосходит SASRec; scaling даёт дополнительный выигрыш. В малом industrial ablation HSTU получил retrieval log-perplexity 3.978 против 4.069 у Transformer; vanilla Transformer дал NaN на ranking задачах. Авторы сообщают 1.5–2× меньший wall-clock и 50% меньше HBM в этом сравнении.

### 5.3. Production результаты

| Задача | DLRM offline | GR offline | Online lift GR |
|---|---:|---:|---:|
| Retrieval HR@100 | 29.0% | 36.9% (new source) | E-task +6.2%; replace source +5.1% |
| Retrieval HR@500 | 55.5% | 62.4% (new source) | C-task +5.0%; replace source +1.9% |
| Ranking NE, E-task | .4982 | **.4845** | **+12.4%** |
| Ranking NE, C-task | .7842 | **.7645** | **+4.4%** |

HSTU на длине 8192 был в 5.3–15.2 раза быстрее FlashAttention2-based Transformer в зависимости от режима. M-FALCON позволил обслуживать GR с 285× большим числом FLOPs при 1.50–2.99× большем throughput относительно production DLRM в показанной конфигурации. Эти отношения нельзя переносить на произвольное железо и batch profile.

## 6. Критический анализ

### Сильные стороны

- Постановка согласует training и serving: каждое прошлое действие становится supervision, а не только последняя метка сессии.
- Сравнение включает synthetic mechanism test, public datasets, ablations, system benchmarks и production A/B.
- DLRM-baseline не учебный: это многолетняя production-система, а не заведомо слабая модель.
- Авторы честно показывают low-compute режим, где ручные признаки DLRM могут быть лучше.
- Architecture/system co-design важнее одной «магической» формулы attention.

### Ограничения валидности

1. **Невоспроизводимый главный результат.** Закрыты данные, определения E/C-task, sampling, feature freshness, абсолютный traffic split, доверительные интервалы и большая часть гиперпараметров 1.5T-модели.
2. **Неодинаковая цена признаков.** GR использует более длинную raw history, а accounting FLOPs, feature generation, embedding lookup, network traffic и preprocessing раскрыт не полностью.
3. **Scaling law не доказывает причинность.** При масштабировании совместно меняются depth, width, sequence length, negatives и embedding tables. 1.5T sparse embedding parameters нельзя напрямую сравнивать с 1.5T dense LLM.
4. **Public protocol оптимистичен для индустрии.** Full-shuffle/multi-epoch нарушает one-pass streaming realism; Books особенно sparse, а относительный +65.8% соответствует небольшому абсолютному NDCG@10: .0156→.0257.
5. **Недостаточно beyond-accuracy метрик.** Нет системного анализа diversity, calibration, fairness, popularity bias, safety и долгосрочного user welfare.
6. **«Счётчики больше не нужны» условно.** При конечном окне history, delayed feedback и privacy retention нужная статистика может быть невосстановима. Позднее семейство MTGR фактически возвращает часть raw/cross features.
7. **Efficiency зависит от специализированных kernels.** Наивный PyTorch HSTU не воспроизводит throughput статьи; без ragged fusion, caching и продуманного batching преимущество может исчезнуть.

### Data leakage и честная проверка

- Split должен идти по event time; нельзя случайно перемешивать будущие события одного пользователя в train.
- Item/creator statistics, negative pool и vocabulary строятся только по train cutoff.
- Target item виден в ranking input намеренно, но его действие и будущие действия должны быть causal-masked.
- Full-catalog HR/NDCG нельзя сравнивать с метриками на 100 sampled negatives.
- A/B требует pre-registered primary metric, guardrails, SRM-check, CUPED/стратификации и доверительного интервала; одного процента lift недостаточно.

## 7. Применимость в highload

### Когда внедрять

GR/HSTU разумен, если есть миллиарды событий, длинная история, высокая стоимость feature engineering, стабильная GPU-инфраструктура и измеримый plateau DLRM. Для небольшого каталога или десятков миллионов событий SASRec/two-tower + компактный ranker обычно дадут лучшее отношение качество/стоимость.

### Реалистичный migration plan

1. Ввести единый versioned event schema: item, action, event/request time, surface, immutable context.
2. Начать с GR как дополнительного retrieval source; оставить DLRM fallback.
3. Валидировать temporal full-catalog offline и shadow-serving latency/p95/p99, GPU memory, cache hit rate.
4. Добавить SL только после ablation по длине/активности пользователей; отдельно контролировать heavy users.
5. Для ranking внедрить candidate isolation tests и microbatch/KV cache; не допускать взаимного влияния candidates.
6. Canary → небольшой A/B → ramp-up с quality, revenue, diversity, complaint и safety guardrails.
7. Контролировать drift новых item/action tokens, delayed feedback и retraining cadence.

Главные production риски: высокая GPU-стоимость, p99 latency, динамическое шардирование embedding tables, cache invalidation, деградация cold users, сложная отладка raw sequences и концентрация нескольких стадий в одной точке отказа.

## 8. Что именно реализовано в `poc.ipynb`

Ноутбук не запускает авторский репозиторий и не притворяется trillion-scale воспроизведением. С нуля на PyTorch реализованы:

- генератор синтетических session data с temporal split;
- interleaving item/action и causal target masks;
- HSTU layer: fused $U,V,Q,K$, pointwise SiLU-attention, positional + time bias, post-pooling LayerNorm, gate и residual;
- shared encoder с action-ranking и next-item retrieval heads;
- masked multi-task CE, Normalized Entropy, HR@K и NDCG@K;
- Transformer baseline с теми же embeddings/heads;
- Stochastic Length как дополнительный regularizer;
- M-FALCON-inspired candidate microbatch и unit-test отсутствия cross-candidate leakage;
- shape/causality/finite-gradient smoke tests и CPU latency benchmark.

### Граница воспроизведения

| Компонент | В статье | В PoC |
|---|---|---|
| Последовательная постановка | production heterogeneous streams | item/action/time synthetic streams |
| Attention | custom fused ragged CUDA/Triton-like kernels | dense, читаемый PyTorch; stabilizing pre-LayerNorm перед `f1` |
| Relative bias | production positional + temporal | learned position + log-time gap |
| Objectives | закрытый multi-task набор | action CE + next-item CE |
| Retrieval | sampled/MIPS/hierarchical | полный softmax по малому каталогу |
| M-FALCON | KV cache, custom mask, tens of thousands candidates | корректность microbatch scoring |
| Scale | до 1.5T параметров, H100 fleet | тысячи параметров, CPU/GPU |

## 9. Вопросы для защиты

**Почему не softmax?** Он нормирует массу attention до единицы и теряет сигнал «сколько» релевантных событий найдено; pointwise activation сохраняет intensity, а LayerNorm стабилизирует агрегат.

**Не возникает ли утечка из target item?** В ranking target item обязан быть виден: предсказывается реакция на него. Утечкой было бы видеть саму реакцию или будущую историю.

**Почему 1.5T не означает цену GPT-3 на каждый запрос?** Большая часть параметров может находиться в sparse embeddings; M-FALCON и caches амортизируют encoder по candidates. Но memory/network cost всё равно реален и в статье не сводится к одной цифре FLOPs.

**Можно ли считать +65.8% революцией?** Это относительный lift на sparse Amazon Books; абсолютный NDCG@10 вырос с .0156 до .0257. Сильный результат, но production вывод требует абсолютных метрик, latency и A/B.

**Заменит ли GR все ручные признаки?** Только если нужная статистика наблюдаема в сохранённой последовательности и модель/окно достаточно велики. На практике полезен постепенный hybrid migration и обязательные ablations.

## 10. Итог

Главный вклад статьи — не просто HSTU, а переход от набора независимо обучаемых DLRM-признаков и targets к единому causal потоку действий. Работа убедительно показывает, что recommendation quality может масштабироваться с compute, но её strongest claims опираются на закрытую инфраструктуру. Практический вывод: сначала доказать ценность последовательной постановки и генеративного обучения на своей temporal validation, затем инвестировать в ragged kernels, caching и крупный scale.
