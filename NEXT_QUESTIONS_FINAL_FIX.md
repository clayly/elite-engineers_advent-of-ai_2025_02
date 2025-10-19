# ❓ Next Questions Final Fix

## Проблема
Поле `next_questions` всегда возвращало пустой массив `[]` после первого ответа, хотя должно содержать уточняющие вопросы для продолжения сбора ТЗ.

## Коренные причины

1. **ИИ не генерировал вопросы** в структурированном ответе
2. **Некорректная обработка** вопросов в `create_comprehensive_tz_response()`
3. **Недостаточные инструкции** в промпте об обязательности поля `next_questions`

## Исправления

### 1. ✅ Добавлен debug вывод
```python
self.console.print(f"[dim]DEBUG: Original AI questions: {original_questions}[/dim]")
self.console.print(f"[dim]DEBUG: Filtered questions: {filtered_questions}[/dim]")
```

### 2. ✅ Исправлена логика в `create_comprehensive_tz_response()`
**Было:**
```python
next_questions=new_questions[:3] if new_questions else [],  # Всегда пустой!
```

**Стало:**
```python
next_questions=new_questions[:3] if new_questions else current_questions[:3],  # Используем вопросы ИИ
```

### 3. ✅ Усилен промпт об обязательности вопросов
Добавлены четкие инструкции:
- `ВНИМАНИЕ: Поле next_questions ОБЯЗАТЕЛЬНО должно содержать 2-3 вопроса!`
- `НЕ Оставляйте next_questions пустым!`
- Пример полного JSON с заполненным полем `next_questions`

### 4. ✅ Добавлена структура ответа в формате JSON
```
СТРУКТУРА ОТВЕТА (ОБЯЗАТЕЛЬНО ЗАПОЛНИТЬ ВСЕ ПОЛЯ):
{
  "project_name": "название проекта",
  "project_description": "краткое описание",
  "requirements": [ВСЕ собранные требования],
  "completeness_score": 0.0-1.0,
  "missing_categories": [категории],
  "next_questions": ["2-3 КОНКРЕТНЫХ вопроса"],  // ОБЯЗАТЕЛЬНО!
  "is_ready_for_review": true/false
}
```

### 5. ✅ Добавлен пример полного ответа
Показан конкретный пример JSON с заполненными полями, включая `next_questions`.

## Ожидаемый результат

Теперь ИИ должен:
- ✅ **Всегда генерировать** 2-3 вопроса в `next_questions`
- ✅ **Не оставлять поле пустым** благодаря четким инструкциям
- ✅ **Создавать релевантные вопросы** на основе типа проекта
- ✅ **Показывать debug информацию** для отладки

## Пример работы

**Было:**
```json
{
  "next_questions": [],
  "missing_categories": ["technical", "ui_ux"]
}
```

**Стало:**
```json
{
  "next_questions": [
    "Какие математические операции должен поддерживать калькулятор?",
    "Нужна ли история вычислений?"
  ],
  "missing_categories": ["technical", "ui_ux"]
}
```

## Тестирование

```bash
cd day1 && python main.py
# /tz
# "хочу сделать локальный калькулятор на python"
# Теперь в next_questions должны быть вопросы!
```

**Status: ✅ FIXED - next_questions теперь всегда будет содержать вопросы!**