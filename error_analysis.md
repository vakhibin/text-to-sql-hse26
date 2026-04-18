# Анализ ошибок: Spider dev (`best_results_cheap_gen.json`)

Документ сгенерирован по файлу результатов: `outputs/best_results_cheap_gen.json` (дешёвый генераторный стек; точные имена моделей смотрите в `.env` / логах прогона — в JSON они не дублируются).

**Идентификатор прогона:** `benchmark_run_id = spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52`.

### Интерпретация результатов

По этому прогону **775 / 1034** примеров дают совпадение результата выполнения с эталоном (**EX**). Ещё **222** примера попадают в основной «семантический» зазор: SQL формально исполняется, но ответ неверен. **37** примеров помечены полем `error_message` — это либо инфраструктурные/схемные сбои, либо краевые случаи драйвера SQLite. Доля **EM (~36%)** заметно ниже EX: даже при верном результате запрос часто отличается от золотого SQL (альтернативные планы, лишние `JOIN`, другой порядок проекций).

Во всех таблицах ниже для примеров указаны:
- **`trace_id`** — стабильный идентификатор внутри этого JSON (поле `trace_id`);
- **`index`** — индекс объекта в массиве `predictions` (0..1033), удобен для прямого перехода в редакторе.

---

## 1. Сводка по прогону

| Метрика | Значение |
|---|---:|
| Execution accuracy (EX) | 74.95% |
| Exact match (EM) | 35.78% |
| Всего примеров | 1034 |
| `valid_predictions` (непустой SQL) | 1028 |
| `metrics.errors` (записей с `error_message`) | 37 |
| Время оценки | 5302.4 s (~5.13 s/пример) |
| Стоимость LLM | $30.5970 (~$0.0296/пример) |

**Интерпретация:** 6 примеров завершились таймаутом конвейера (660 s) и дали пустой `predicted_sql` — они входят в `metrics.errors` и объясняют расхождение `1034 - valid_predictions = 6`.

Остальные **31** запись с `error_message` содержат непустой SQL, но итог всё равно считается ошибкой исполнения/валидации.

Отдельно: **222** примера имеют исполняемый SQL без поля `error_message`, но результат не совпал с эталоном (**semantic / logic mismatch** по EX). На них см. раздел 7.

---

## 2. Классификация «жёстких» ошибок конвейера

Все **37** случаев с непустым `error_message` разбиты на группы:

| Группа | Описание | Кол-во |
|---|---|---:|
| **G1** | Таймаут бенчмарка (660 s), SQL не получен | 6 |
| **G2** | Отказ `schema_validation` до выполнения в SQLite | 20 |
| **G3** | SQLite: ошибка декодирования UTF-8 при чтении строки результата | 2 |
| **G4** | SQLite: `no such column` (валидация схемы не отловила до exec) | 9 |

---

## 3. G1 — таймаут бенчмарка

Типичные причины: слишком тяжёлый запрос к LLM, сетевые задержки, либо чрезмерная глубина графа на сложных вопросах. Все 6 случаев — **пустой `predicted_sql`**.

| index | trace_id | db_id | Кратко ошибка |
|---:|---|---|---|
| 29 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:29` | `concert_singer` | benchmark_timeout: example exceeded 660s |
| 93 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:93` | `car_1` | benchmark_timeout: example exceeded 660s |
| 94 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:94` | `car_1` | benchmark_timeout: example exceeded 660s |
| 139 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:139` | `car_1` | benchmark_timeout: example exceeded 660s |
| 273 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:273` | `employee_hire_evaluation` | benchmark_timeout: example exceeded 660s |
| 294 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:294` | `employee_hire_evaluation` | benchmark_timeout: example exceeded 660s |

**Примеры вопросов (для поиска в JSON):**

- [29] What are the names of the stadiums without any concerts?
- [93] How many models does each car maker produce? List maker full name, id and the number.
- [94] What is the full name of each car maker, along with its id and how many models it produces?
- [139] What is the maximum accelerate for different number of cylinders?
- [273] Return the name, location and district of all shops in descending order of number of products.
- [294] Find the districts in which there are both shops selling less than 3000 products and shops selling more than 10000 products.

## 4. G2 — schema_validation (до SQLite)

Генератор или рефайнер выдали SQL, который **не проходит детерминированную проверку схемы** (неизвестные таблицы/алиасы/столбцы). Частые паттерны в этом прогоне: несуществующие имена таблиц (`car`, `models`, `courses`, …), путаница с регистром/именованием (`countrylanguage` vs реальное имя в SQLite), ошибки в цепочке `JOIN` и псевдонимах (`T1.Name` при другом наборе столбцов).

**Подгруппы по базам (для точечной доработки промптов / few-shot):**

- **`car_1`:** выдуманные сущности «одной строкой» (`car`, `models`, `car_models`) и неверные столбцы у псевдонимов (`T2.Maker` вместо фактических полей `car_makers` / `car_names`).
- **`concert_singer`:** несоответствие имён столбцов в `JOIN` (`Name`, `theme` vs реальные идентификаторы в схеме).
- **`student_transcripts_tracking`:** перенос «университетской» схемы (`section`, `takes`, `courses`) вместо таблиц Spider.
- **`world_1`:** неверное имя/алиас для таблицы языков (`countrylanguage` — типичная ошибка к регистру и кавычкам в SQLite).
- **`dog_kennels`:** ошибка привязки стоимости к псевдониму (`T2.cost` / `T3.cost`).

| index | trace_id | db_id | Кратко ошибка |
|---:|---|---|---|
| 33 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:33` | `concert_singer` | schema_validation: unknown column 'T1.Name' |
| 34 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:34` | `concert_singer` | schema_validation: unknown column 'T1.Name'; unknown column 'T1.theme' |
| 104 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:104` | `car_1` | schema_validation: unknown column 'T1.ModelId'; unknown column 'T1.Year' |
| 120 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:120` | `car_1` | schema_validation: unknown table 'car' |
| 152 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:152` | `car_1` | schema_validation: unknown table 'models'; unknown table or alias 'T1' for column 'Model'; unknow... |
| 155 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:155` | `car_1` | schema_validation: unknown table 'car' |
| 156 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:156` | `car_1` | schema_validation: unknown table 'car' |
| 157 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:157` | `car_1` | schema_validation: unknown column 'T2.Id' |
| 158 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:158` | `car_1` | schema_validation: unknown column 'cars_data.Model' |
| 168 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:168` | `car_1` | schema_validation: unknown table 'car_models'; unknown table or alias 'T2' for column 'maker_id';... |
| 171 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:171` | `car_1` | schema_validation: unknown column 'T2.Maker' |
| 172 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:172` | `car_1` | schema_validation: unknown column 'T2.Maker' |
| 543 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:543` | `student_transcripts_tracking` | schema_validation: unknown table 'section'; unknown table 'takes' |
| 545 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:545` | `student_transcripts_tracking` | schema_validation: unknown table 'courses'; unknown table 'student_course_registrations'; unknown... |
| 548 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:548` | `student_transcripts_tracking` | schema_validation: unknown table 'Course'; unknown table or alias 'T1' for column 'crs_name'; unk... |
| 818 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:818` | `world_1` | schema_validation: unknown table or alias 'countrylanguage' for column 'CountryCode' |
| 819 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:819` | `world_1` | schema_validation: unknown table or alias 'countrylanguage' for column 'CountryCode' |
| 821 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:821` | `world_1` | schema_validation: unknown table or alias 'countrylanguage' for column 'CountryCode' |
| 937 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:937` | `dog_kennels` | schema_validation: unknown column 'T3.cost' |
| 945 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:945` | `dog_kennels` | schema_validation: unknown column 'T2.cost' |

**Примеры вопросов (для поиска в JSON):**

- [33] Show the name and theme for all concerts and the number of singers in each concert.
- [34] What are the names, themes, and number of singers for each and every concert?
- [104] What are the different models for the cards produced after 1980?
- [120] What is the minimu weight of the car with 8 cylinders produced in 1974?
- [152] What are the different models created by either the car maker General Motors or weighed more than 3500?
- [155] What is the horsepower of the car with the largest accelerate?
- [156] What is the horsepower of the car with the greatest accelerate?
- [157] For model volvo, how many cylinders does the car with the least accelerate have?
- … и ещё 12 строк в таблице выше.

## 5. G3 — UTF-8 decode при выборке

Ошибка возникает на стороне **движка/драйвера** при чтении текста из колонки (битая кодировка в БД). Это не логическая ошибка SQL: запрос может быть синтаксически и схематически корректен.

| index | trace_id | db_id | Кратко ошибка |
|---:|---|---|---|
| 455 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:455` | `wta_1` | (sqlite3.OperationalError) Could not decode to UTF-8 column 'last_name' with text 'Treyes Albarra... |
| 456 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:456` | `wta_1` | (sqlite3.OperationalError) Could not decode to UTF-8 column 'first_name \|\| ' ' \|\| last_name' ... |

**Примеры вопросов (для поиска в JSON):**

- [455] List the first and last name of all players in the order of birth date.
- [456] What are the full names of all players, sorted by birth date?

## 6. G4 — `no such column` после валидации

Валидатор схемы пропустил запрос, но SQLite сообщил об отсутствии столбца. Возможные причины: **расхождение правил имён** (регистр, кавычки), столбец в другой таблице, чем предполагал генератор, или ограничения статического анализа.

Наблюдение по этому прогону: для `car_1` несколько ошибок связаны с тем, что столбец (**`year`**, **`horsepower`**, **`MPG`**) взят не из той таблицы, где он реально есть в Spider (золотые запросы часто тянут год из `CARS_DATA` и т.д.). Это указывает на **ошибку линковки таблиц**, а не только на опечатку в имени.

| index | trace_id | db_id | Кратко ошибка |
|---:|---|---|---|
| 103 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:103` | `car_1` | (sqlite3.OperationalError) no such column: Year |
| 131 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:131` | `car_1` | (sqlite3.OperationalError) no such column: horsepower |
| 133 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:133` | `car_1` | (sqlite3.OperationalError) no such column: MPG |
| 137 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:137` | `car_1` | (sqlite3.OperationalError) no such column: Model |
| 138 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:138` | `car_1` | (sqlite3.OperationalError) no such column: edispl |
| 166 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:166` | `car_1` | (sqlite3.OperationalError) no such column: Model |
| 293 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:293` | `employee_hire_evaluation` | (sqlite3.OperationalError) no such column: Number_of_Products |
| 516 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:516` | `student_transcripts_tracking` | (sqlite3.OperationalError) no such column: department_id |
| 572 | `spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:572` | `student_transcripts_tracking` | (sqlite3.OperationalError) no such column: enr_id |

**Примеры вопросов (для поиска в JSON):**

- [103] Which distinct car models are the produced after 1980?
- [131] What is the maximum horsepower and the make of the car models with 3 cylinders?
- [133] Which model saves the most gasoline? That is to say, have the maximum miles per gallon.
- [137] What is the average edispl of the cars of model volvo?
- [138] What is the average edispl for all volvos?
- [166] For all of the 4 cylinder cars, which model has the most horsepower?
- [293] Which district has both stores with less than 3000 products and stores with more than 10000 products?
- [516] For each department id, what is the name of the department with the most number of degrees?
- … и ещё 1 строк в таблице выше.

---

## 7. Семантические промахи (EX=false, `error_message=null`)

**Всего:** 222 примера — SQL выполнился, но результат не совпал с эталоном.

**Топ `db_id` по числу таких промахов:**

| db_id | Число примеров |
|---|---:|
| `student_transcripts_tracking` | 38 |
| `car_1` | 32 |
| `flight_2` | 22 |
| `world_1` | 22 |
| `dog_kennels` | 21 |
| `employee_hire_evaluation` | 14 |
| `cre_Doc_Template_Mgt` | 10 |
| `tvshow` | 10 |
| `concert_singer` | 9 |
| `network_1` | 9 |
| `orchestra` | 8 |
| `wta_1` | 7 |
| `poker_player` | 5 |
| `pets_1` | 3 |
| `museum_visit` | 3 |
| `battle_death` | 3 |
| `course_teach` | 2 |
| `voter_1` | 2 |
| `real_estate_properties` | 2 |

**Полный список `trace_id`** (для выборочной проверки в JSON):

```text
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:20
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:21
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:22
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:23
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:24
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:37
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:38
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:43
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:44
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:65
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:66
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:72
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:97
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:99
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:100
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:101
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:102
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:109
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:110
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:111
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:112
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:115
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:116
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:119
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:121
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:122
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:125
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:126
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:127
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:132
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:134
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:142
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:143
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:144
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:145
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:146
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:150
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:151
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:163
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:164
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:165
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:167
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:175
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:176
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:198
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:201
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:202
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:203
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:204
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:215
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:217
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:225
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:226
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:227
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:228
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:231
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:232
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:233
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:235
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:237
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:239
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:247
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:248
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:249
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:252
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:258
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:267
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:268
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:269
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:270
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:271
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:272
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:274
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:275
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:276
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:284
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:291
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:292
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:295
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:296
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:309
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:347
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:348
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:349
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:350
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:352
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:354
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:356
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:361
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:362
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:387
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:388
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:419
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:420
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:427
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:447
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:448
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:452
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:469
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:470
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:471
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:472
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:493
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:496
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:500
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:512
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:515
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:517
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:518
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:519
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:520
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:523
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:524
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:526
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:528
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:529
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:530
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:533
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:534
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:535
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:536
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:538
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:539
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:540
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:542
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:544
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:549
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:550
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:554
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:561
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:563
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:564
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:566
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:569
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:570
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:573
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:574
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:575
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:576
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:578
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:579
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:580
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:582
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:587
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:613
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:614
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:629
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:630
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:637
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:641
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:642
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:645
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:646
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:649
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:650
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:651
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:652
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:674
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:687
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:698
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:705
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:725
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:754
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:755
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:756
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:760
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:761
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:772
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:773
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:774
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:777
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:778
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:779
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:781
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:783
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:784
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:785
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:798
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:813
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:816
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:817
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:820
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:830
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:831
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:832
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:833
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:844
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:845
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:851
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:859
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:867
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:883
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:897
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:904
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:905
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:906
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:907
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:908
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:917
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:921
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:924
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:925
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:929
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:942
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:943
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:944
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:946
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:947
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:954
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:955
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:958
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:960
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:961
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:967
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:969
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:979
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:983
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:989
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:998
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:999
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:1032
spider-dev-083571eb-0f32-4d0f-a39a-5aeb23c20a52:1033
```

---

## 8. Рекомендации по группам (приоритет — доработка существующих узлов)

### G1 — таймауты

- **Runner / конфиг:** поднять `example_timeout_seconds` для полного Spider или сделать его **зависимым от сложности** (из decomposer); отдельно ограничить число LLM-вызовов на пример.
- **Пайплайн:** при приближении к лимиту — **ранний выход** с лучшим доступным кандидатом (частичный результат), чтобы не «висеть» до конца окна.
- **Инфра:** снизить `concurrency` при нестабильной сети или длинных ответах модели.
- *(Новый компонент не обязателен.)*

### G2 — schema_validation

- **Генератор + промпт:** жёстче требовать **только идентификаторы из `filtered_schema` / mSchema**; запретить «универсальные» имена вроде `car`, `courses`, если их нет в схеме.
- **Query sketcher:** явно выводить **список допустимых имён таблиц и ключевых столбцов**; при неуверенности — маркер риска, а не выдуманный идентификатор.
- **Selector:** не отбрасывать релевантные таблицы для многословных вопросов (особенно `student_transcripts_tracking`, `world_1`).
- **sql_schema_validator / refiner:** расширить **канонизацию имён** SQLite (кавычки, регистр, `countryLanguage` и т.д.) и детерминированные замены по словарю из загруженной схемы.
- **Value linker:** подсказки значений уже есть; добавить **подсказки имён столбцов** из схемы для частых путаниц.

### G3 — UTF-8 в результатах

- **Слой выполнения (evaluation / SQLAlchemy):** задать политику для «битых» строк (`text_factory`, суррогатные символы) так, чтобы сравнение с эталоном было **детерминированным** и не падало на чтении.
- **Данные:** при необходимости нормализовать проблемные строки в копии БД только для бенчмарка (осторожно с воспроизводимостью Spider).
- Это **не** задача для отдельного LLM-агента.

### G4 — столбцы не пойманы до exec

- **Усилить sql_schema_validator:** проверять не только наличие таблицы, но и **реальные имена столбцов** с учётом правил SQLite для безкавычных идентификаторов.
- **Согласовать с sqlglot:** если парсер и SQLite расходятся в нормализации имён — добавить постобработку или предупреждение.
- **Генератор:** для баз вроде `car_1` и `employee_hire_evaluation` добавить **few-shot с корректными именами** (`Number_of_Products` vs фактические поля и т.д.) — через существующий механизм few-shot, без нового сервиса.

### Семантические промахи (раздел 7)

- **Judge / cheap-path:** не пропускать судью там, где структура кандидатов близка, но семантика различается (самый большой пласт EX).
- **Self-consistency:** несколько независимых генераций и голосование по **совпадению результатов выполнения**.
- **Рефайнер:** использовать не только синтаксис, но и **контрпример** (первые строки gold не обязательно — но можно сравнивать кардинальность и типы).
- **Query sketcher + decomposer:** лучше декомпозиция сложных вопросов и явные шаги агрегации.

---

## 9. Замечания по предупреждениям (`warnings`)

В этом файле предупреждения не классифицированы как «ошибки», но по семантическим промахам чаще встречаются: `execution_filter`, `query_sketcher`, `schema_validation_warning`. Имеет смысл отдельно коррелировать тип предупреждения с EX на подвыборке.
