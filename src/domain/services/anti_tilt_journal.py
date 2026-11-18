import time
import math
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class JournalEntry:
    entry_id: Optional[int]
    user_id: str
    tag: str # e.g. "FOLLOWED_PLAN", "FOMO_ENTRY", "EARLY_EXIT", "REVENGE_TRADE", "OVERLEVERAGED"
    pnl_usd: float
    followed_plan: bool
    notes: str
    timestamp: float


@dataclass(frozen=True)
class DisciplineSummary:
    user_id: str
    discipline_score: int # 0 to 100
    grade: str # "S-Rank Master", "A-Rank Pro", "B-Rank Trader", "C-Rank Emotional", "D-Rank Tilt Hazard"
    clean_streak_days: int
    total_journal_entries: int
    fomo_count: int
    revenge_count: int
    plan_compliance_pct: float
    is_in_tilt_cooldown: bool
    cooldown_remaining_sec: float
    stoic_advice: str


class AntiTiltJournalService:
    """
    Anti-Tilt & Trader Psychology Journal Engine for MaBuy.
    Tracks emotional discipline, penalizes FOMO/revenge trades,
    and activates a 45-minute cooling-off period when tilt is detected.
    """

    TAG_PROPERTIES = {
        "FOLLOWED_PLAN": {"name": "✅ Следование плану 100%", "score_delta": +5, "is_tilt": False},
        "EARLY_EXIT": {"name": "⏳ Ранний выход из страха", "score_delta": -5, "is_tilt": False},
        "FOMO_ENTRY": {"name": "⚠️ FOMO вход без сигнала", "score_delta": -15, "is_tilt": True},
        "OVERLEVERAGED": {"name": "🚨 Превышение плеча / риск-лимита", "score_delta": -20, "is_tilt": True},
        "REVENGE_TRADE": {"name": "🔥 Revenge-трейд (Попытка отыграться)", "score_delta": -25, "is_tilt": True}
    }

    STOIC_ADVICE_LIBRARY = [
        "«Рынок — это устройство для перекачивания денег от нетерпеливых к терпеливым.» — Уоррен Баффетт",
        "«Главный ваш враг в трейдинге — это не маркет-мейкер, а ваше собственное эго и страх упущенной выгоды.» — Марк Дуглас",
        "«Если вы не можете принять небольшой убыток, рынок рано или поздно заберет весь ваш капитал.» — Пол Тюдор Джонс",
        "«Профессионал концентрируется на безупречном исполнении процесса, а любитель — на сиюминутной прибыли.» — Нассим Талеб",
        "«Эмоции на бирже стоят дороже любых комиссий. Сделайте глубокий вдох и отойдите от экрана на 45 минут.» — MaBuy Quant Guide"
    ]

    TILT_COOLDOWN_SEC = 45 * 60 # 45 minutes

    def __init__(self):
        pass

    def evaluate_discipline(self, user_id: str, entries: List[JournalEntry], last_tilt_timestamp: float = 0.0) -> DisciplineSummary:
        """
        Calculates Discipline Score (0-100), streak, compliance %, and tilt cooldown status.
        """
        now = time.time()
        base_score = 75 # Baseline starting score

        if not entries:
            return DisciplineSummary(
                user_id=user_id,
                discipline_score=base_score,
                grade="A-Rank Pro (Чистый лист)",
                clean_streak_days=1,
                total_journal_entries=0,
                fomo_count=0,
                revenge_count=0,
                plan_compliance_pct=100.0,
                is_in_tilt_cooldown=False,
                cooldown_remaining_sec=0.0,
                stoic_advice=self.STOIC_ADVICE_LIBRARY[0]
            )

        fomo_count = sum(1 for e in entries if e.tag == "FOMO_ENTRY")
        revenge_count = sum(1 for e in entries if e.tag == "REVENGE_TRADE")
        overleveraged_count = sum(1 for e in entries if e.tag == "OVERLEVERAGED")
        followed_count = sum(1 for e in entries if e.followed_plan)

        score_acc = base_score
        for e in entries:
            prop = self.TAG_PROPERTIES.get(e.tag, {"score_delta": 0})
            score_acc += prop["score_delta"]

        discipline_score = max(0, min(100, score_acc))

        # Grade calculation
        if discipline_score >= 90:
            grade = "👑 S-Rank Мастер Дисциплины"
        elif discipline_score >= 75:
            grade = "💎 A-Rank Профессионал"
        elif discipline_score >= 60:
            grade = "⚖️ B-Rank Трейдер (Норма)"
        elif discipline_score >= 40:
            grade = "⚠️ C-Rank Эмоциональный Риск"
        else:
            grade = "🚨 D-Rank Зона Тильта (Опасность)"

        plan_compliance = (followed_count / len(entries)) * 100.0 if entries else 100.0

        # Cooldown check
        is_in_cooldown = False
        remaining_sec = 0.0
        if last_tilt_timestamp > 0:
            elapsed = now - last_tilt_timestamp
            if elapsed < self.TILT_COOLDOWN_SEC:
                is_in_cooldown = True
                remaining_sec = round(self.TILT_COOLDOWN_SEC - elapsed, 1)

        # Select stoic quote
        quote_idx = hash(user_id + str(len(entries))) % len(self.STOIC_ADVICE_LIBRARY)
        stoic_quote = self.STOIC_ADVICE_LIBRARY[quote_idx]

        # Calculate clean streak days
        streak_days = max(1, int(len(entries) / 2))
        if fomo_count + revenge_count + overleveraged_count > 0:
            streak_days = max(0, int((now - max(last_tilt_timestamp, entries[0].timestamp)) / 86400))

        return DisciplineSummary(
            user_id=user_id,
            discipline_score=discipline_score,
            grade=grade,
            clean_streak_days=streak_days,
            total_journal_entries=len(entries),
            fomo_count=fomo_count,
            revenge_count=revenge_count,
            plan_compliance_pct=round(plan_compliance, 1),
            is_in_tilt_cooldown=is_in_cooldown,
            cooldown_remaining_sec=remaining_sec,
            stoic_advice=stoic_quote
        )
