# ❓ Next Questions Fix

## Проблема
Поле `next_questions` всегда возвращало пустой массив `[]`, хотя именно в этом поле ИИ должен задавать уточняющие вопросы по ТЗ.

## Причины проблемы

1. **ИИ не генерировал вопросы** в структурированном ответе
2. **Фильтрация удаляла все вопросы** при отсутствии новых
3. **Нет fallback механизма** для генерации вопросов при пустом ответе от ИИ

## Исправления

### 1. Добавлен метод `generate_contextual_questions()`
Создает вопросы на основе недостающих категорий:

```python
def generate_contextual_questions(self, missing_categories, all_requirements):
    # Шаблоны вопросов для каждой категории
    question_templates = {
        RequirementCategory.FUNCTIONAL: [
            "Какие основные функции должно выполнять приложение?",
            "Какие бизнес-процессы нужно автоматизировать?",
            "Какие действия сможет выполнять пользователь?"
        ],
        RequirementCategory.TECHNICAL: [
            "Какой стек технологий предпочтителен?",
            "Есть ли требования к интеграциям с другими системами?",
            "Какие ограничения по технической реализации?"
        ],
        # ... и другие категории
    }
```

### 2. Улучшен системный промпт
Добавлены четкие инструкции:
- `ОБЯЗАТЕЛЬНО включи поле next_questions с 2-4 конкретными вопросами`
- `Сформулируй 2-4 КОНКРЕТНЫЕ уточняющих вопроса`
- Описана точная структура ответа

### 3. Добавлен fallback механизм
```python
# Если ИИ не предоставил вопросы, генерируем их
if not new_questions and missing_categories:
    new_questions = self.generate_contextual_questions(missing_categories, all_requirements)
```

### 4. Улучшено сохранение вопросов
```python
# Обновляем список заданных вопросов
if hasattr(comprehensive_response, 'next_questions') and self.tz_collector_state:
    for question in comprehensive_response.next_questions:
        if question not in self.tz_collector_state.asked_questions:
            self.tz_collector_state.asked_questions.append(question)
```

## Результат

Теперь поле `next_questions` будет:
- ✅ **Содержать 2-4 вопроса** от ИИ или сгенерированных автоматически
- ✅ **Не повторять** ранее заданные вопросы
- ✅ **Быть релевантными** недостающим категориям
- ✅ **Помогать пользователю** предоставлять нужную информацию

## Пример работы

Было:
```json
{
  "next_questions": [],
  "missing_categories": ["technical", "security"]
}
```

Стало:
```json
{
  "next_questions": [
    "Какой стек технологий предпочтителен?",
    "Какие требования к безопасности данных?",
    "Нужна ли аутентификация и авторизация?"
  ],
  "missing_categories": ["technical", "security"]
}
```

**Status: ✅ FIXED**