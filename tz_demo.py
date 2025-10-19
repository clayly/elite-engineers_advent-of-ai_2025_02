#!/usr/bin/env python3
"""
Demonstration of Technical Specification Collector integration
This shows how the TZ collector works within the existing chat agent
"""

import sys
import os

# Add the day1 directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'day1'))

try:
    from main import TechnicalSpecification, RequirementCategory, Requirement, TZCollectorState
    from pydantic import ValidationError
    import json

    print("🎯 Технический спецификатор интегрирован успешно!")
    print("=" * 50)

    # Test 1: Create individual requirements
    print("\n📋 Test 1: Создание требований")

    req1 = Requirement(
        id="REQ-001",
        category=RequirementCategory.FUNCTIONAL,
        title="Пользовательская аутентификация",
        description="Система должна поддерживать вход по логину и паролю",
        priority="high",
        acceptance_criteria=[
            "Пользователь может ввести логин и пароль",
            "Система проверяет корректность данных",
            "При успешной аутентификации пользователь перенаправляется в личный кабинет"
        ]
    )

    print(f"✓ Требование создано: {req1.title}")

    # Test 2: Create Technical Specification
    print("\n📄 Test 2: Создание технического задания")

    tz_spec = TechnicalSpecification(
        project_name="Веб-приложение для управления задачами",
        project_description="Система для создания и отслеживания задач проекта",
        requirements=[req1],
        completeness_score=0.3,
        missing_categories=[
            RequirementCategory.NON_FUNCTIONAL,
            RequirementCategory.TECHNICAL,
            RequirementCategory.UI_UX
        ],
        next_questions=[
            "Какие требования к производительности системы?",
            "Какой стек технологий следует использовать?",
            "Какой дизайн интерфейса предусмотрен?"
        ],
        is_ready_for_review=False
    )

    print(f"✓ ТЗ создано: {tz_spec.project_name}")
    print(f"  - Полнота: {tz_spec.completeness_score:.0%}")
    print(f"  - Требований: {len(tz_spec.requirements)}")
    print(f"  - К заполнению: {len(tz_spec.missing_categories)} категорий")

    # Test 3: TZ Collector State
    print("\n🔄 Test 3: Состояние коллектора")

    state = TZCollectorState(
        phase="gathering",
        project_type="web_application",
        current_category=RequirementCategory.FUNCTIONAL,
        requirements_count=1,
        should_complete=False
    )

    print(f"✓ Состояние: фаза '{state.phase}', требований {state.requirements_count}")

    # Test 4: JSON serialization
    print("\n📦 Test 4: Сериализация в JSON")

    tz_dict = tz_spec.model_dump()
    print(f"✓ ТЗ сериализовано в JSON (размер: {len(json.dumps(tz_dict, ensure_ascii=False))} символов)")

    print("\n" + "=" * 50)
    print("🎉 Все тесты пройдены! Интеграция ТЗ коллектора успешна.")
    print("\n💡 Как использовать:")
    print("1. Запустите: python day1/main.py")
    print("2. Введите: /tz для режима сбора ТЗ")
    print("3. Или просто опишите ваш проект - режим определится автоматически")

except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("Установите зависимости: pip install langchain-openai pydantic rich python-dotenv")
except ValidationError as e:
    print(f"❌ Ошибка валидации: {e}")
except Exception as e:
    print(f"❌ Неожиданная ошибка: {e}")