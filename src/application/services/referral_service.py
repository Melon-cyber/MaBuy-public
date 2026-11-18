import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class ReferralService:
    """
    Task-Based Gamified Referral Engine.
    Tracks referrer-referred relationships and unlocks bonuses only after referred friends
    complete mandatory onboarding tasks:
    1. Language Selection ('ru' or 'en')
    2. First Prediction Executed
    """

    def __init__(self):
        # referrer_id -> list of referred_user_ids
        self._referral_tree: Dict[str, List[str]] = {}
        # referred_user_id -> referrer_id
        self._referred_by: Dict[str, str] = {}
        # referred_user_id -> {"lang_selected": bool, "first_prediction_done": bool, "reward_claimed": bool}
        self._tasks_progress: Dict[str, Dict[str, bool]] = {}

    def register_referral(self, new_user_id: str, referrer_id: str) -> bool:
        new_user_id = str(new_user_id)
        referrer_id = str(referrer_id)

        if new_user_id == referrer_id or new_user_id in self._referred_by:
            return False

        self._referred_by[new_user_id] = referrer_id
        if referrer_id not in self._referral_tree:
            self._referral_tree[referrer_id] = []
        self._referral_tree[referrer_id].append(new_user_id)

        self._tasks_progress[new_user_id] = {
            "lang_selected": False,
            "first_prediction_done": False,
            "reward_claimed": False
        }
        logger.info(f"User [{new_user_id}] registered under referrer [{referrer_id}].")
        return True

    def mark_task_complete(self, user_id: str, task_name: str) -> Optional[str]:
        """
        Marks task complete. If all onboarding tasks are finished, returns the referrer_id to reward.
        """
        user_id = str(user_id)
        if user_id not in self._tasks_progress:
            return None

        prog = self._tasks_progress[user_id]
        if task_name in prog:
            prog[task_name] = True

        # Check if all tasks are complete
        if prog["lang_selected"] and prog["first_prediction_done"] and not prog["reward_claimed"]:
            prog["reward_claimed"] = True
            referrer_id = self._referred_by.get(user_id)
            logger.info(f"User [{user_id}] completed all referral tasks! Rewarding referrer [{referrer_id}].")
            return referrer_id

        return None

    def get_user_referral_stats(self, user_id: str) -> Dict[str, Any]:
        user_id = str(user_id)
        invited_users = self._referral_tree.get(user_id, [])
        total_invited = len(invited_users)

        active_completed = 0
        for uid in invited_users:
            p = self._tasks_progress.get(uid, {})
            if p.get("reward_claimed"):
                active_completed += 1

        return {
            "total_invited": total_invited,
            "tasks_completed": active_completed,
            "pending_tasks": total_invited - active_completed
        }
