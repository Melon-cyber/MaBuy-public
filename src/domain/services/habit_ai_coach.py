from typing import Dict, Any, List, Optional
import datetime


class HabitAICoachService:
    """
    Domain Service for generating Quantitative and AI-driven coaching insights
    for user habits, academic focus (SAT/IELTS/ML), and discipline consistency
    across Week (7d), Month (30d), and Year (365d) horizons.
    """

    @classmethod
    def generate_ai_coaching_insight(cls, stats: Dict[str, Any], period: str = "week", lang: str = "ru") -> Dict[str, Any]:
        """
        Synthesizes an intelligent quant-driven assessment, velocity score, and actionable advice.
        """
        period_clean = period.lower().strip()
        completion_rate = float(stats.get("completion_rate", 0.0))
        consistency_score = float(stats.get("consistency_score", 0.0))
        total_minutes = int(stats.get("total_minutes_logged", 0))
        total_hours = round(total_minutes / 60.0, 1)
        streak_current = int(stats.get("streak_current", 0))
        streak_best = int(stats.get("streak_best", 0))
        best_habit = stats.get("best_habit", {})
        weakest_habit = stats.get("weakest_habit", {})
        category_breakdown = stats.get("category_breakdown", {})

        # Compute Performance Grade
        if completion_rate >= 85 and consistency_score >= 80:
            grade = "A+ (Elite Master)" if lang != "ru" else "A+ (Элитный Титан)"
            momentum = "🔥 Неудержимый рост (Hyper-Growth)" if lang == "ru" else "🔥 Hyper-Growth Momentum"
            readiness_index = 95
        elif completion_rate >= 70 and consistency_score >= 65:
            grade = "A (Strong Discipline)" if lang != "ru" else "A (Железный фокус)"
            momentum = "⚡️ Стабильный прогресс (High Momentum)" if lang == "ru" else "⚡️ High Momentum"
            readiness_index = 82
        elif completion_rate >= 50:
            grade = "B (Good Baseline)" if lang != "ru" else "B (Рабочий темп)"
            momentum = "📈 Набирает обороты (Building Momentum)" if lang == "ru" else "📈 Building Momentum"
            readiness_index = 68
        else:
            grade = "C (Needs Focus)" if lang != "ru" else "C (Требует перезагрузки)"
            momentum = "⚠️ Риск потери фокуса (Low Momentum)" if lang == "ru" else "⚠️ Low Momentum"
            readiness_index = 45

        # Academic Readiness Projection (SAT 1500+ & Top Universities / NU)
        sat_stats = category_breakdown.get("academic", {}) or category_breakdown.get("sat", {})
        code_stats = category_breakdown.get("code", {})
        sat_done = sat_stats.get("completed", 0)
        code_done = code_stats.get("completed", 0)

        if lang == "ru":
            if period_clean == "week":
                title = "⚡️ Тактический разбор недели"
                summary = (
                    f"За последние 7 дней закрыто {stats.get('total_completed_tasks', 0)} задач "
                    f"с дисциплиной {consistency_score:.0f}%. Вложено <b>{total_hours} ч</b> чистого фокуса."
                )
                if completion_rate >= 75:
                    recommendation = (
                        "Отличный недельный спринт! Темп позволяет легко держать стрик. "
                        "На следующей неделе добавь 15 минут к решению сложных задач SAT Math (модуль 2)."
                    )
                else:
                    weak_title = weakest_habit.get("title", "отстающие задачи")
                    recommendation = (
                        f"Главная точка роста на ближайшие 7 дней — подтянуть «{weak_title}». "
                        "Поставь таймер на утро сразу после пробуждения, пока когнитивный ресурс на максимуме."
                    )
            elif period_clean == "month":
                title = "🗓 Стратегический отчет за 30 дней"
                summary = (
                    f"За месяц набрано {stats.get('xp_earned', 0)} XP. Общее время глубокой работы: <b>{total_hours} часов</b>. "
                    f"Стрик удержан на отметке {streak_current} дней (рекорд: {streak_best} дней)."
                )
                if sat_done > 15:
                    sat_eval = "Математика SAT и академическая база укрепляются с опережением графика."
                else:
                    sat_eval = "Рекомендуется усилить регулярность тренировок SAT Math (минимум 5 дней в неделю)."

                recommendation = (
                    f"{sat_eval} Баланс между кодом ({code_done} спринтов) и здоровьем держит продуктивность стабильной. "
                    f"Индекс готовности к экзаменам: <b>{readiness_index}%</b>."
                )
            else: # year
                title = "🏆 Годовая макро-аналитика & Прогрессия Фаундера"
                summary = (
                    f"Суммарный объем инвестированного времени за год: <b>{total_hours} часов</b>. "
                    f"Выполнено {stats.get('total_completed_tasks', 0)} фокусных блоков."
                )
                recommendation = (
                    f"Компаундинг привычек работает на полную мощность. Накопленные часы в коде и академических дисциплинах "
                    f"формируют фундаментальное преимущество для поступления в Назарбаев Университет и топ мировых вузов. "
                    f"Прогнозируемый балл SAT: <b>1520–1560</b>, IELTS: <b>8.0+</b>."
                )
        else: # en
            if period_clean == "week":
                title = "⚡️ Weekly Tactical AI Review"
                summary = f"Completed {stats.get('total_completed_tasks', 0)} focus tasks with {consistency_score:.0f}% consistency ({total_hours} hrs logged)."
                recommendation = "Keep momentum high! Consider dedicating morning deep work sprints to SAT Math Module 2 hard questions."
            elif period_clean == "month":
                title = "🗓 Monthly Strategic AI Report"
                summary = f"Earned {stats.get('xp_earned', 0)} XP across {total_hours} total deep work hours. Current streak: {streak_current} days."
                recommendation = f"Discipline index stands at {readiness_index}%. Compounding daily efforts directly accelerates target admission readiness."
            else:
                title = "🏆 Annual Founder Mastery Projection"
                summary = f"Invested {total_hours} deep work hours into academic & coding mastery over the year."
                recommendation = "Long-term trajectory demonstrates top 1% discipline. Projected SAT Score: 1520–1560, IELTS: 8.0+."

        return {
            "period": period_clean,
            "grade": grade,
            "momentum": momentum,
            "readiness_index": readiness_index,
            "title": title,
            "summary": summary,
            "recommendation": recommendation,
            "total_hours": total_hours,
            "completion_rate": completion_rate,
            "consistency_score": consistency_score
        }
